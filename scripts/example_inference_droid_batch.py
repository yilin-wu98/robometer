#!/usr/bin/env python3
"""
Client script for the RBM eval server. No robometer dependency.

Sends a video (or .npy/.npz frames) and task instruction to a running eval server,
then saves per-frame progress and success predictions plus an optional plot.

The script supports DROID-style videos where three camera views (right, left, wrist)
are concatenated horizontally into a single wide frame. Use --view to select which
view to pass to the reward model.

Example:
  # Start the server first (in another terminal):
  #   uv run python robometer/evals/eval_server.py --config_path=robometer/configs/config.yaml --host=0.0.0.0 --port=8000

  # Single-view video:
  python scripts/example_inference_droid.py --eval-server-url http://localhost:8000 --video /path/to/video.mp4 --task "Pick up the red block"

  # Three-view concatenated video (right | left | wrist), use left camera:
  python scripts/example_inference_droid.py --eval-server-url http://localhost:8000 --video /path/to/video.mp4 --task "Pick up the red block" --view left

  # Batch: walk data_root/videos/<split>/ and annotations/<split>/, write annotation_rewards/<split>/:
  python scripts/example_inference_droid_batch.py --data-root /path/to/dataset --split files --view right
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

import matplotlib.pyplot as plt


def create_combined_progress_success_plot(
    progress_pred: np.ndarray,
    num_frames: int,
    success_binary: Optional[np.ndarray] = None,
    success_probs: Optional[np.ndarray] = None,
    success_labels: Optional[np.ndarray] = None,
    is_discrete_mode: bool = False,
    title: Optional[str] = None,
    loss: Optional[float] = None,
    pearson: Optional[float] = None,
) -> Any:
    """Create a combined plot with progress, success binary, and success probabilities.

    This function creates a unified plot with 1 subplot (progress only) or 3 subplots
    (progress, success binary, success probs), similar to the one used in compile_results.py.

    Args:
        progress_pred: Progress predictions array
        num_frames: Number of frames
        success_binary: Optional binary success predictions
        success_probs: Optional success probability predictions
        success_labels: Optional ground truth success labels
        is_discrete_mode: Whether progress is in discrete mode (deprecated, kept for compatibility)
        title: Optional title for the plot (if None, auto-generated from loss/pearson)
        loss: Optional loss value to display in title
        pearson: Optional pearson correlation to display in title

    Returns:
        matplotlib Figure object
    """
    # Determine if we should show success plots
    has_success_binary = success_binary is not None and len(success_binary) == len(progress_pred)

    if has_success_binary:
        # Three subplots: progress, success (binary), success_probs
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
        ax = axs[0]  # Progress subplot
        ax2 = axs[1]  # Success subplot (binary)
        ax3 = axs[2]  # Success probs subplot
    else:
        # Single subplot: progress only
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax2 = None
        ax3 = None

    # Plot progress
    ax.plot(progress_pred, linewidth=2)
    ax.set_ylabel("Progress")

    # Build title
    if title is None:
        title_parts = ["Progress"]
        if loss is not None:
            title_parts.append(f"Loss: {loss:.3f}")
        if pearson is not None:
            title_parts.append(f"Pearson: {pearson:.2f}")
        title = ", ".join(title_parts)
    fig.suptitle(title)

    # Set y-limits and ticks (always continuous since discrete is converted before this function)
    ax.set_ylim(0, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(y_ticks)

    # Setup success binary subplot
    if ax2 is not None:
        ax2.step(range(len(success_binary)), success_binary, where="post", linewidth=2, label="Predicted", color="blue")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_binary):
            ax2.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
            )
        ax2.set_ylabel("Success (Binary)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yticks([0, 1])
        ax2.legend()

    # Setup success probs subplot if available
    if ax3 is not None and success_probs is not None:
        ax3.plot(range(len(success_probs)), success_probs, linewidth=2, label="Success Prob", color="purple")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_probs):
            ax3.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
                linestyle="--",
            )
        ax3.set_ylabel("Success Probability")
        ax3.set_ylim(-0.05, 1.05)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.legend()

    plt.tight_layout()
    return fig


def extract_frames(video_path: str, fps: float = 1.0) -> np.ndarray:
    """Extract frames from video file as numpy array (T, H, W, C).

    Supports both local file paths and URLs (e.g., HuggingFace Hub URLs).
    Uses the provided ``fps`` to control how densely frames are sampled from
    the underlying video; there is no additional hard cap on the number of frames.

    Args:
        video_path: Path to video file or URL
        fps: Frames per second to extract (default: 1.0)

    Returns:
        numpy array of shape (T, H, W, C) containing extracted frames, or None if error
    """
    if video_path is None:
        raise ValueError("video_path is None")

    if isinstance(video_path, tuple):
        video_path = video_path[0]

    # Check if it's a URL or local file
    is_url = video_path.startswith(("http://", "https://"))
    is_local_file = os.path.exists(video_path) if not is_url else False

    if not is_url and not is_local_file:
        raise FileNotFoundError(video_path)

    try:
        import decord  # type: ignore

        # decord.VideoReader can handle both local files and URLs
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)

        # Determine native FPS; fall back to a reasonable default if unavailable
        try:
            native_fps = float(vr.get_avg_fps())
        except Exception:
            native_fps = 1.0

        # If user-specified fps is invalid or None, default to native fps
        if fps is None or fps <= 0:
            fps = native_fps

        # Compute how many frames we want based on desired fps
        # num_frames ≈ total_duration * fps = total_frames * (fps / native_fps)
        if native_fps > 0:
            desired_frames = int(round(total_frames * (fps / native_fps)))
        else:
            desired_frames = total_frames

        # Clamp to [1, total_frames]
        desired_frames = max(1, min(desired_frames, total_frames))

        # Evenly sample indices to match the desired number of frames
        if desired_frames == total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int).tolist()

        frames_array = vr.get_batch(frame_indices).asnumpy()  # Shape: (T, H, W, C)
        del vr
        return frames_array
    except Exception as e:
        raise RuntimeError(f"Error extracting frames from {video_path}: {e}")


_DROID_VIEW_ORDER = ["right", "left", "wrist"]


def crop_view_from_concatenated(frames: np.ndarray, view: str) -> np.ndarray:
    """Crop a single camera view from a horizontally concatenated three-view frame array.

    Expects frames concatenated in order: right | left | wrist (equal widths).

    Args:
        frames: Array of shape (T, H, W*3, C) where W*3 is three equal-width views.
        view: One of "right", "left", or "wrist".

    Returns:
        Array of shape (T, H, W, C) for the selected view.
    """
    if view not in _DROID_VIEW_ORDER:
        raise ValueError(f"--view must be one of {_DROID_VIEW_ORDER}, got '{view}'")
    total_width = frames.shape[2]
    if total_width % 3 != 0:
        raise ValueError(
            f"Frame width {total_width} is not divisible by 3; "
            "expected a horizontally concatenated right|left|wrist video."
        )
    view_width = total_width // 3
    idx = _DROID_VIEW_ORDER.index(view)
    start = idx * view_width
    return frames[:, :, start : start + view_width, :]


def load_frames_input(
    video_or_array_path: str,
    *,
    fps: float = 1.0,
) -> np.ndarray:
    """
    Load frames from a video file (path or URL) or from a .npy/.npz file.

    Video: uses decord; fps controls sampling density.
    .npy/.npz: expects array shape (T, H, W, C) or (T, C, H, W); for .npz uses
    key 'arr_0' or the first array.

    Returns:
        Frames as uint8 array (T, H, W, C). Raises on failure.
    """
    if video_or_array_path.endswith(".npy"):
        frames_array = np.load(video_or_array_path)
    elif video_or_array_path.endswith(".npz"):
        with np.load(video_or_array_path, allow_pickle=False) as npz:
            if "arr_0" in npz:
                frames_array = npz["arr_0"].copy()
            else:
                frames_array = next(iter(npz.values())).copy()
    else:
        frames_array = extract_frames(video_or_array_path, fps=fps)

    if frames_array is None or frames_array.size == 0:
        raise RuntimeError("Could not extract frames from input.")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    if frames_array.ndim == 4:
        if frames_array.shape[1] in (1, 3) and frames_array.shape[-1] not in (1, 3):
            frames_array = frames_array.transpose(0, 2, 3, 1)

    return frames_array


def _numpy_to_npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def build_multipart_payload(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Convert a list of sample dicts into:
      - files: mapping for requests.post(files=...)
      - data: mapping for requests.post(data=...) with sample_{i} JSON strings

    Numpy arrays inside trajectory fields are moved to .npy blobs and replaced by
    {"__numpy_file__": <file_key>} references.
    """
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}

    numpy_fields = ["frames", "lang_vector", "video_embeddings"]

    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))  # make JSON-serializable shell
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})

        for field in numpy_fields:
            val = traj.get(field, None)
            if val is None:
                continue

            # torch.Tensor -> numpy (if torch is available)
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()

            if isinstance(val, np.ndarray):
                file_key = f"sample_{i}_trajectory_{field}"
                files[file_key] = _numpy_to_npy_file_tuple(val, f"{file_key}.npy")
                traj_copy[field] = {"__numpy_file__": file_key}
            else:
                traj_copy[field] = val

        # Keep a helpful frames_shape (as list of ints) if present
        if "frames_shape" in traj_copy and isinstance(traj_copy["frames_shape"], (tuple, list)):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]

        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)

    return files, data


def post_evaluate_batch_npy(
    eval_server_url: str,
    samples: List[Dict[str, Any]],
    timeout_s: float = 120.0,
    use_frame_steps: bool = False,
) -> Dict[str, Any]:
    files, data = build_multipart_payload(samples)
    data["use_frame_steps"] = "true" if use_frame_steps else "false"
    url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"
    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def extract_rewards_from_server_output(outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse server JSON into per-frame progress and success probability arrays.

    Returns:
        progress_array: Per-frame progress (reward) predictions for the first sample.
        success_array: Per-frame success probabilities, or empty array if not in response.
    """
    outputs_progress = outputs.get("outputs_progress")
    if outputs_progress is None:
        raise ValueError("No `outputs_progress` in server response")
    progress_pred = outputs_progress.get("progress_pred", [])

    if progress_pred and len(progress_pred) > 0:
        progress_array = np.array(progress_pred[0], dtype=np.float32)
    else:
        progress_array = np.array([], dtype=np.float32)

    outputs_success = outputs.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    if success_probs and len(success_probs) > 0:
        success_array = np.array(success_probs[0], dtype=np.float32)
    else:
        success_array = np.array([], dtype=np.float32)

    return progress_array, success_array


def make_progress_sample(
    frames: np.ndarray,
    task: str,
    sample_id: str,
    subsequence_length: int,
) -> Dict[str, Any]:
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(frames.shape),
            "task": task,
            "id": sample_id,
            "metadata": {"subsequence_length": int(subsequence_length)},
            "video_embeddings": None,
        },
    }


def compute_rewards_per_frame(
    eval_server_url: str,
    video_frames: np.ndarray,
    task: str,
    timeout_s: float = 120.0,
    use_frame_steps: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Send the full trajectory to the eval server and get per-frame progress and success.

    Args:
        use_frame_steps: If True, server expands into frame-step sub-samples (0:1, 0:2, ...)
            and aggregates; can improve alignment with training. If False, one forward pass on
            the full trajectory (subsampled to fixed frames on the server).

    Returns:
        progress: Per-frame progress (reward) predictions.
        success_probs: Per-frame success probabilities (empty if model has no success head).
    """
    T = int(video_frames.shape[0])
    sample = make_progress_sample(
        frames=video_frames,
        task=task,
        sample_id="0",
        subsequence_length=T,
    )
    outputs = post_evaluate_batch_npy(
        eval_server_url, [sample], timeout_s=timeout_s, use_frame_steps=use_frame_steps
    )
    return extract_rewards_from_server_output(outputs)


def _setup_curve_ax(ax: Any, n_frames: int, ylabel: str, ylim: Tuple[float, float], yticks: List[float]) -> Any:
    """Configure a time-series subplot and return an initialised (line, vline) pair."""
    ax.set_xlim(0, max(n_frames - 1, 1))
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("Frame", fontsize=8)
    ax.set_yticks(yticks)
    ax.tick_params(labelsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    (line,) = ax.plot([], [], linewidth=2)
    vline = ax.axvline(x=0, color="tomato", linewidth=1.2, linestyle="--", alpha=0.8)
    return line, vline


def create_progress_video(
    rewards: np.ndarray,
    output_path: str,
    success_probs: Optional[np.ndarray] = None,
    success_threshold: float = 0.5,
    display_views: Optional[Dict[str, np.ndarray]] = None,
    fps: float = 4.0,
    title: str = "",
) -> None:
    """Render an animated MP4 with all three reward plots growing frame-by-frame.

    Layout:
      - Top row   : three subplots — Progress | Success (binary) | Success (prob).
                    Success plots are shown only when *success_probs* is provided.
      - Bottom row: three camera views (Right | Left | Wrist) at the current frame.
                    Shown only when *display_views* is provided.

    Args:
        rewards: Per-frame progress predictions, shape (T,).
        output_path: Destination .mp4 path.
        success_probs: Optional per-frame success probabilities, shape (T,).
        success_threshold: Threshold for binarising success_probs (default 0.5).
        display_views: Dict mapping "right"/"left"/"wrist" to (T, H, W, C) uint8 arrays.
        fps: Output video frame rate.
        title: Figure super-title.
    """
    import matplotlib.animation as animation

    n_frames = len(rewards)
    has_views = display_views is not None and all(v in display_views for v in _DROID_VIEW_ORDER)
    has_success = success_probs is not None and len(success_probs) == n_frames
    success_binary = (success_probs > success_threshold).astype(np.int32) if has_success else None

    n_rows = 2 if has_views else 1
    fig = plt.figure(figsize=(15, 7 if has_views else 3.5))
    gs = fig.add_gridspec(
        n_rows, 3,
        height_ratios=([1.3, 1] if has_views else [1]),
        hspace=0.45,
        wspace=0.35,
    )

    # Top row: three reward subplots
    ax_prog = fig.add_subplot(gs[0, 0])
    ax_succ_bin = fig.add_subplot(gs[0, 1])
    ax_succ_prob = fig.add_subplot(gs[0, 2])

    if title:
        fig.suptitle(title, fontsize=11)

    line_prog, vline_prog = _setup_curve_ax(
        ax_prog, n_frames, "Progress", (0, 1), [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    line_prog.set_color("steelblue")

    line_succ_bin, vline_succ_bin = _setup_curve_ax(
        ax_succ_bin, n_frames, "Success (binary)", (-0.05, 1.05), [0, 1]
    )
    line_succ_bin.set_color("royalblue")
    ax_succ_bin.set_title("Success (binary)", fontsize=9)

    line_succ_prob, vline_succ_prob = _setup_curve_ax(
        ax_succ_prob, n_frames, "Success prob", (-0.05, 1.05), [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    line_succ_prob.set_color("mediumpurple")
    ax_succ_prob.set_title("Success prob", fontsize=9)

    # Bottom row: camera views
    im_handles: List[Any] = []
    if has_views:
        ax_views = [fig.add_subplot(gs[1, i]) for i in range(3)]
        for i, vname in enumerate(_DROID_VIEW_ORDER):
            ax = ax_views[i]
            ax.axis("off")
            ax.set_title(vname.capitalize(), fontsize=10, pad=3)
            im = ax.imshow(display_views[vname][0])
            im_handles.append(im)

    all_artists: List[Any] = [
        line_prog, vline_prog,
        line_succ_bin, vline_succ_bin,
        line_succ_prob, vline_succ_prob,
        *im_handles,
    ]

    def _update(t: int) -> List[Any]:
        xs = np.arange(t + 1)

        line_prog.set_data(xs, rewards[: t + 1])
        vline_prog.set_xdata([t])

        if has_success:
            line_succ_bin.set_data(xs, success_binary[: t + 1])
            line_succ_prob.set_data(xs, success_probs[: t + 1])
        vline_succ_bin.set_xdata([t])
        vline_succ_prob.set_xdata([t])

        if has_views:
            for i, vname in enumerate(_DROID_VIEW_ORDER):
                im_handles[i].set_data(display_views[vname][t])

        return all_artists

    ani = animation.FuncAnimation(fig, _update, frames=n_frames, interval=1000.0 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
    ani.save(output_path, writer=writer)
    plt.close(fig)


def _numeric_stem_key(p: Path) -> int:
    try:
        return int(p.stem)
    except ValueError:
        return hash(p.stem)


def list_videos(val_dir: Path) -> List[Path]:
    """Same file discovery and sort order as scripts/batch_val_reward_annotations.py (bash reward batch)."""
    out: List[Path] = []
    for pat in ("*.mp4", "*.webm", "*.avi", "*.mov"):
        out.extend(val_dir.glob(pat))
    return sorted(out, key=_numeric_stem_key)


def annotation_input_path(data_root: Path, video_path: Path, split: str) -> Path:
    rel = video_path.relative_to(data_root / "videos" / split)
    return (data_root / "annotations" / split / rel).with_suffix(".json")


def annotation_reward_output_path(data_root: Path, video_path: Path, split: str) -> Path:
    rel = video_path.relative_to(data_root / "videos" / split)
    return (data_root / "annotation_rewards" / split / rel).with_suffix(".json")


def process_episode(
    *,
    video_path: Path,
    annotation_file: Path,
    ann_output_path: Path,
    npy_output_dir: Path,
    eval_server_url: str,
    fps: float,
    timeout_s: float,
    use_frame_steps: bool,
    success_threshold: float,
    view: Optional[str],
    save_plot: bool,
    task_override: str,
    reward_npy_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load video + annotation, run server, write npy + enriched JSON (+ optional plot)."""
    annotation: Optional[Dict[str, Any]] = None
    if annotation_file.is_file():
        with open(annotation_file, "r", encoding="utf-8") as f:
            annotation = json.load(f)

    if task_override == "":
        if annotation is None:
            raise FileNotFoundError(f"No --task and annotation missing: {annotation_file}")
        task = annotation["texts"][0]
        print(f"Inferred task: {task}")
    else:
        task = task_override

    all_frames = load_frames_input(str(video_path), fps=float(fps))

    if view is not None:
        frames = crop_view_from_concatenated(all_frames, view)
        print(f"Using '{view}' view for inference — shape: {frames.shape}")
    else:
        frames = all_frames
        print(f"Inference frame shape: {frames.shape}")

    rewards, success_probs = compute_rewards_per_frame(
        eval_server_url=eval_server_url,
        video_frames=frames,
        task=task,
        timeout_s=timeout_s,
        use_frame_steps=use_frame_steps,
    )

    # npy_output_dir.mkdir(parents=True, exist_ok=True)
    # if reward_npy_path is not None:
    #     out_path = reward_npy_path
    #     base_stem = out_path.stem.removesuffix("_rewards") if out_path.stem.endswith("_rewards") else out_path.stem
    # else:
    #     base_stem = video_path.stem
    #     out_path = npy_output_dir / f"{base_stem}_rewards.npy"
    # np.save(str(out_path), rewards)
    # success_path = out_path.parent / f"{base_stem}_success_probs.npy"
    # np.save(str(success_path), success_probs)
    base_stem = video_path.stem

    ann_out_path: Optional[Path] = None
    if annotation is not None:
        video_length: Optional[int] = None
        for key in ("video_length", "num_frames", "length"):
            if annotation.get(key) is not None:
                video_length = int(annotation[key])
                break

        def _interp_to_video_length(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0 or video_length is None or video_length == arr.size:
                return arr
            sampled_pos = np.linspace(0, video_length - 1, arr.size)
            target_pos = np.arange(video_length, dtype=np.float64)
            return np.interp(target_pos, sampled_pos, arr).astype(np.float32)

        rewards_ann = _interp_to_video_length(rewards)
        success_ann = _interp_to_video_length(success_probs)

        if video_length is not None and rewards_ann.size != video_length:
            print(f"  WARNING: could not align reward length {rewards_ann.size} to video_length {video_length}")
        else:
            print(
                f"  Rewards interpolated: {rewards.size} sampled → {rewards_ann.size} annotation frames"
                if rewards.size != rewards_ann.size
                else f"  Rewards length matches video_length ({rewards_ann.size})"
            )

        annotation["reward_progress"] = [float(x) for x in rewards_ann.tolist()] if rewards_ann.size > 0 else []
        annotation["reward_success"] = [float(x) for x in success_ann.tolist()] if success_ann.size > 0 else []
        annotation["reward_binary"] = (
            [int(x > float(success_threshold)) for x in success_ann.tolist()] if success_ann.size > 0 else []
        )
        ann_output_path.parent.mkdir(parents=True, exist_ok=True)
        ann_out_path = ann_output_path
        with open(ann_out_path, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2)
            f.write("\n")
        print(f"Annotation saved → {ann_out_path}")

    summary: Dict[str, Any] = {
        "video": str(video_path),
        "num_frames": int(frames.shape[0]),
        "out_annotation_json": str(ann_out_path) if ann_out_path else None,
        "reward_progress_min": float(np.min(rewards)) if rewards.size else None,
        "reward_progress_max": float(np.max(rewards)) if rewards.size else None,
        "reward_progress_mean": float(np.mean(rewards)) if rewards.size else None,
        "reward_success_mean": float(np.mean(success_probs)) if success_probs.size else None,
    }

    if save_plot:
        show_success = success_probs.size > 0 and success_probs.size == rewards.size
        success_binary = (success_probs > float(success_threshold)).astype(np.int32) if show_success else None
        fig = create_combined_progress_success_plot(
            progress_pred=rewards,
            num_frames=int(frames.shape[0]),
            success_binary=success_binary,
            success_probs=success_probs if show_success else None,
            title=f"Progress/Success — {video_path.name}",
        )
        plot_path = ann_out_path.parent / f"{base_stem}_progress_success.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        summary["out_plot_png"] = str(plot_path)
        print(f"Plot saved → {plot_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get per-frame progress and success predictions from an RBM eval server.",
        epilog=(
            "Batch mode: --data-root <root> walks videos/<split>/ and annotations/<split>/, "
            "writes annotation_rewards/<split>/…  Single mode: --video …"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-server-url",
        type=str,
        default="http://localhost:8000",
        help="Eval server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Dataset root containing videos/<split>/ and annotations/<split>/. "
            "When set, processes every video and writes enriched JSON to annotation_rewards/<split>/."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Subfolder under videos/ and annotations/ (default: val for single-video layout; use 'files' for batch)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path or URL to one video (or .npy/.npz). Omit when using --data-root.",
    )
    parser.add_argument("--task", type=str, default="", help="Task instruction (default: read from annotation texts[0])")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second when sampling from video (default: 1.0)",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP request timeout in seconds (default: 120)")
    parser.add_argument(
        "--use-frame-steps",
        action="store_true",
        help="If set, server uses frame-step expansion (0:1, 0:2, ...) and aggregates; can improve reward alignment",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary success curve in the plot (default: 0.5)",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        choices=_DROID_VIEW_ORDER,
        help=(
            "Camera view when the video is a horizontally concatenated three-view clip "
            "(right | left | wrist). If omitted, the full frame is passed as-is."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="[Single mode only] Output path for rewards .npy (default: <video_stem>_rewards.npy next to video)",
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a PNG with three reward subplots next to the .npy outputs",
    )
    parser.add_argument(
        "--skip-missing-annotation",
        action="store_true",
        help="[Batch mode] Skip videos with no matching annotations/<split>/… .json instead of failing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="[Batch mode] Process at most N videos (-1 = all, default)",
    )
    args = parser.parse_args()

    if args.data_root is not None and args.video is not None:
        parser.error("Pass either --data-root (batch) or --video (single), not both.")
    if args.data_root is None and args.video is None:
        parser.error("Required: --video for one file, or --data-root for batch.")

    if args.data_root is not None:
        data_root: Path = args.data_root.resolve()
        split: str = args.split
        video_root = data_root / "videos" / split
        if not video_root.is_dir():
            raise FileNotFoundError(f"Video directory not found: {video_root}")

        videos = list_videos(video_root)
        if args.limit is not None and args.limit >= 0:
            videos = videos[: args.limit]

        ok, failed = 0, 0
        for vp in videos:
            ann_in = annotation_input_path(data_root, vp, split)
            ann_out = annotation_reward_output_path(data_root, vp, split)
            if not ann_in.is_file():
                msg = f"No annotation for {vp.name} (expected {ann_in})"
                if args.skip_missing_annotation:
                    print(f"SKIP: {msg}")
                    continue
                print(f"ERROR: {msg}")
                failed += 1
                continue
            try:
                print(f"--- {vp.relative_to(video_root)} ---")
                process_episode(
                    video_path=vp,
                    annotation_file=ann_in,
                    ann_output_path=ann_out,
                    npy_output_dir=ann_out.parent,
                    eval_server_url=args.eval_server_url,
                    fps=float(args.fps),
                    timeout_s=float(args.timeout_s),
                    use_frame_steps=bool(args.use_frame_steps),
                    success_threshold=float(args.success_threshold),
                    view=args.view,
                    save_plot=bool(args.save_plot),
                    task_override=args.task,
                )
                ok += 1
            except Exception as e:
                print(f"FAIL: {vp.name}: {e}")
                failed += 1

        print(json.dumps({"processed_ok": ok, "failed": failed, "total": len(videos)}, indent=2))
        return

    # Single-video mode (legacy layout: …/annotations/<split>/<stem>.json)
    video_path = Path(args.video)
    if not str(args.video).startswith(("http://", "https://")) and not video_path.exists():
        raise FileNotFoundError(video_path)

    annotation_file = video_path.parent.parent.parent / "annotations" / args.split / f"{video_path.stem}.json"
    npy_dir = Path(args.out).parent if args.out is not None else video_path.parent
    reward_npy = Path(args.out) if args.out is not None else None
    ann_output_path = video_path.with_suffix(".json")

    summary = process_episode(
        video_path=video_path,
        annotation_file=annotation_file,
        ann_output_path=ann_output_path,
        npy_output_dir=npy_dir,
        eval_server_url=args.eval_server_url,
        fps=float(args.fps),
        timeout_s=float(args.timeout_s),
        use_frame_steps=bool(args.use_frame_steps),
        success_threshold=float(args.success_threshold),
        view=args.view,
        save_plot=bool(args.save_plot),
        task_override=args.task,
        reward_npy_path=reward_npy,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

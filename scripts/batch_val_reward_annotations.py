#!/usr/bin/env python3
"""
For each validation video under world_model_data_preprocessed_v2/videos/val:
  - Load the clip (left | right | wrist concatenated horizontally), take the *right* camera.
  - Run the RBM eval server (same protocol as scripts/example_inference_droid.py) to get
    per-frame success probabilities.
  - Copy the matching JSON from annotations/val to annotations_reward/val and add
    "reward": [ ... ] (success prob per frame; length == video frames == video_length).

Requires a running eval server, e.g.:
  uv run python robometer/evals/eval_server.py --config_path=robometer/configs/config.yaml \\
      --host=0.0.0.0 --port=8000

Example:
  uv run python scripts/batch_val_reward_annotations.py \\
      --data-root /data/yilin/world_model_data_preprocessed_v2 \\
      --eval-server-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

# Reuse inference helpers from example_inference_droid (same repo).
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import example_inference_droid as eid  # noqa: E402

TASK_INSTRUCTIONS = {
    "pour": "Pour the cup of beans into the blue bowl without spilling",
    "marker": "put the marker in the cup",
    "towel": "put the red towel in the basket",
    "stack": "pick up the blue bowl and stack on top of the other bowl",
    "bag":"put the bag of chips on the green plate"}

def crop_horizontal_view(
    frames: np.ndarray,
    view: str,
    concat_order: Sequence[str],
) -> np.ndarray:
    """Crop one horizontal strip from a 3-way concatenated video."""
    if view not in concat_order:
        raise ValueError(f"view must be one of {list(concat_order)}, got {view!r}")
    total_width = frames.shape[2]
    if total_width % 3 != 0:
        raise ValueError(
            f"Frame width {total_width} is not divisible by 3; expected 3 concatenated views."
        )
    w = total_width // 3
    idx = list(concat_order).index(view)
    start = idx * w
    return frames[:, :, start : start + w, :]


def _task_from_annotation(obj: Dict[str, Any]) -> str:
    # episode_id_orig = obj.get("episode_id_orig")
    
    for key in ("task", "instruction", "lang_instruction", "caption", "texts"):
        v = obj.get(key)
        # breakpoint()
        # for key, value in TASK_INSTRUCTIONS.items():
        #     if key in episode_id_orig: 
        #         print(f"Found task instruction for {episode_id_orig}: {value}",)
        #         v = value 
        if isinstance(v, list):
            v = v[0]
        if isinstance(v, str) and v.strip():
            return v.strip()
    # raise KeyError(
        # "Could not find a task string in annotation (tried task, instruction, lang_instruction, caption, text)"
    # )
    return ""


def _video_length_from_annotation(obj: Dict[str, Any]) -> int:
    for key in ("video_length", "num_frames", "length"):
        v = obj.get(key)
        if v is not None:
            return int(v)
    raise KeyError("Annotation has no video_length / num_frames / length")


def _numeric_stem_key(p: Path) -> int:
    """Sort by the integer in the filename stem (e.g. '42.mp4' -> 42)."""
    try:
        return int(p.stem)
    except ValueError:
        return hash(p.stem)


def list_videos(val_dir: Path) -> List[Path]:
    out: List[Path] = []
    for pat in ("*.mp4", "*.webm", "*.avi", "*.mov"):
        out.extend(val_dir.glob(pat))
    return sorted(out, key=_numeric_stem_key)


def annotation_path_for_video(data_root: Path, video_path: Path, split: str) -> Path:
    rel = video_path.relative_to(data_root / "videos" / split)
    return data_root / "annotations" / split / rel.with_suffix(".json")


def reward_out_path(data_root: Path, ann_path: Path, split: str) -> Path:
    rel = ann_path.relative_to(data_root / "annotations" / split)
    return data_root / "annotations_reward" / split / rel


def run_one(
    *,
    video_path: Path,
    ann_path: Path,
    out_json_path: Path,
    eval_server_url: str,
    concat_order: Sequence[str],
    view: str,
    fps: float,
    timeout_s: float,
    use_frame_steps: bool,
    overwrite: bool,
    temporal_downsample: int = 1,
    save_plot: bool = False,
) -> None:
    with open(ann_path, encoding="utf-8") as f:
        ann: Dict[str, Any] = json.load(f)

    task = _task_from_annotation(ann)
    expected_len = _video_length_from_annotation(ann)

    all_frames = eid.load_frames_input(str(video_path), fps=fps)
    # .copy() makes view_frames an independent allocation so all_frames (3× wider)
    # can be freed immediately instead of staying alive for the rest of the function.
    view_frames = crop_horizontal_view(all_frames, view=view, concat_order=concat_order).copy()
    del all_frames

    t_orig = int(view_frames.shape[0])

    if t_orig != expected_len:
        raise ValueError(
            f"Frame count mismatch for {video_path.name}: video has {t_orig} frames after load, "
            f"but annotation {ann_path.name} has video_length={expected_len}"
        )

    # Temporal downsampling: sample every N-th frame for inference to cut the
    # number of frame-step sub-samples (and peak VRAM) by ~N×.
    # Always include the last frame so interpolation covers the full range.
    if temporal_downsample > 1:
        infer_positions = np.unique(
            np.concatenate([np.arange(0, t_orig, temporal_downsample), [t_orig - 1]])
        )
        infer_frames = view_frames[infer_positions]  # fancy index → always a copy
        del view_frames
    else:
        infer_positions = np.arange(t_orig)
        infer_frames = view_frames
        del view_frames  # same object as infer_frames; remove the extra name

    t_infer = len(infer_positions)
    print(
        f"  [{video_path.name}] fps={fps} view={view} frames_orig={t_orig} "
        f"frames_infer={t_infer} (downsample×{temporal_downsample}) "
        f"use_frame_steps={use_frame_steps} task={task!r}"
    )

    progress, success_probs = eid.compute_rewards_per_frame(
        eval_server_url=eval_server_url,
        video_frames=infer_frames,
        task=task,
        timeout_s=timeout_s,
        use_frame_steps=use_frame_steps,
    )

    # Frame arrays are no longer needed — free before validation, JSON write,
    # and plot so each trajectory starts the next phase with minimal footprint.
    del infer_frames
    gc.collect()

    if success_probs.size == 0:
        raise RuntimeError(f"No success_probs returned for {video_path}; does the model expose a success head?")

    if int(success_probs.shape[0]) != t_infer:
        raise ValueError(
            f"Success length mismatch for {video_path.name}: got {success_probs.shape[0]} probs, expected {t_infer}. "
            f"Try --use-frame-steps or inspect the eval server / collator max frames."
        )

    if progress.size > 0 and int(progress.shape[0]) != t_infer:
        raise ValueError(
            f"Progress length mismatch for {video_path.name}: got {progress.shape[0]} values, expected {t_infer}."
        )

    # Interpolate sparse rewards back to every original frame.
    if temporal_downsample > 1:
        orig_positions = np.arange(t_orig, dtype=np.float64)
        success_probs = np.interp(orig_positions, infer_positions.astype(np.float64), success_probs).astype(np.float32)
        if progress.size > 0:
            progress = np.interp(orig_positions, infer_positions.astype(np.float64), progress).astype(np.float32)

    ann["reward"] = [float(x) for x in success_probs.tolist()]
    ann["reward_progress"] = [float(x) for x in progress.tolist()] if progress.size > 0 else []
    ann["reward_binary"] = [int(x > 0.5) for x in success_probs.tolist()]

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    if out_json_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_json_path} (pass --overwrite)")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)
        f.write("\n")

    if save_plot:
        import matplotlib.pyplot as plt

        show_success = success_probs.size > 0 and success_probs.size == (progress.size if progress.size > 0 else success_probs.size)
        success_binary = (success_probs > 0.5).astype(int) if show_success else None
        fig = eid.create_combined_progress_success_plot(
            progress_pred=progress if progress.size > 0 else success_probs,
            num_frames=t_orig,
            success_binary=success_binary,
            success_probs=success_probs if show_success else None,
            title=f"{video_path.stem} — {task[:80]}",
        )
        plot_out_path = out_json_path.with_suffix(".png")
        fig.savefig(str(plot_out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  plot → {plot_out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Batch success-prob rewards for val videos + annotation JSON copies.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/yilin/world_model_data_preprocessed_v2"),
        help="Dataset root containing video/, annotations/, annotations_reward/",
    )
    p.add_argument("--split", type=str, default="val", help="Subfolder under video/ and annotations/ (default: val)")
    p.add_argument(
        "--eval-server-url",
        type=str,
        default="http://localhost:8000",
        help="RBM eval server base URL",
    )
    p.add_argument(
        "--concat-order",
        type=str,
        default="right,left,wrist",
        help="Comma-separated order of horizontal panels (default: right,left,wrist)",
    )
    p.add_argument(
        "--view",
        type=str,
        default="right",
        help="Which panel to feed to the reward model (default: right)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help=(
            "Frames per second to sample from each video (default: 5.0). "
            "Set to 0 for native fps. "
            "To match example_inference_droid.py defaults, use --fps 1.0."
        ),
    )
    p.add_argument("--timeout-s", type=float, default=600.0, help="HTTP timeout per video (default: 600)")
    # p.set_defaults(use_frame_steps=True)
    p.add_argument(
        "--use-frame-steps",
        dest="use_frame_steps",
        action="store_true",
        help="Use frame steps (subsequences) for reward alignment and policy ranking evaluations. True = generate subsequences (0:frame_step, 0:2*frame_step, etc.), False = use whole trajectory.",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing annotations_reward JSON files")
    p.add_argument(
        "--temporal-downsample",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Sample every N-th frame for inference and interpolate rewards back to the full frame count "
            "(default: 1 = no downsampling). Reduces peak VRAM by ~N× when --use-frame-steps is set. "
            "E.g. --temporal-downsample 4 sends ~25%% of frames to the server."
        ),
    )
    p.add_argument(
        "--save-plot",
        action="store_true",
        help="Save a PNG plot alongside each JSON with three subplots: progress, success (binary), and success probability curves.",
    )
    # p.add_argument("--limit", type=int, default=120, help="Process at most N videos (default: 120; set -1 for all)")
    p.add_argument(
        "--copy-only-missing-annotations",
        action="store_true",
        help="If set, skip videos with no matching annotations/ file instead of failing",
    )
    args = p.parse_args()

    data_root: Path = args.data_root
    split: str = args.split
    val_dir = data_root / "videos" / split
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Video directory not found: {val_dir}")

    concat_order = tuple(s.strip() for s in args.concat_order.split(",") if s.strip())
    if len(concat_order) != 3 or len(set(concat_order)) != 3:
        raise ValueError("--concat-order must list exactly 3 distinct view names")

    videos = list_videos(val_dir)
    # if args.limit is not None and args.limit >= 0:
    #     videos = videos[: args.limit]

    ok, failed = 0, 0
    for vp in videos:
        ann_p = annotation_path_for_video(data_root, vp, split)
        out_p = reward_out_path(data_root, ann_p, split)

        if not ann_p.is_file():
            msg = f"No annotation for {vp.name} (expected {ann_p})"
            if args.copy_only_missing_annotations:
                print(f"SKIP: {msg}")
                continue
            print(f"ERROR: {msg}")
            failed += 1
            continue

        try:
            out_p.parent.mkdir(parents=True, exist_ok=True)
            run_one(
                video_path=vp,
                ann_path=ann_p,
                out_json_path=out_p,
                eval_server_url=args.eval_server_url,
                concat_order=concat_order,
                view=args.view,
                fps=float(args.fps),
                timeout_s=float(args.timeout_s),
                use_frame_steps=bool(args.use_frame_steps),
                overwrite=True,
                temporal_downsample=int(args.temporal_downsample),
                save_plot=args.save_plot,
            )
            print(f"OK: {vp.name} -> {out_p.relative_to(data_root)} ({out_p})")
            ok += 1
        except Exception as e:
            print(f"FAIL: {vp.name}: {e}")
            failed += 1

    print(json.dumps({"processed_ok": ok, "failed": failed, "total": len(videos)}, indent=2))


if __name__ == "__main__":
    main()

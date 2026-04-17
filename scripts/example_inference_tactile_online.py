#!/usr/bin/env python3
"""
Online (incremental) reward inference client for the RBM eval server.

Unlike example_inference_tactile.py which sends the whole video at once, this
script simulates an online setting: at each timestep i it appends the frame at
position 8*i to a growing buffer [frame_0, frame_8, ..., frame_{8*i}] and asks
the server for the reward of the *latest* frame in that buffer.  The collected
scalar rewards are then plotted.

Example:
  # Start the server first (in another terminal):
  #   uv run python robometer/evals/eval_server.py \\
  #       --config_path=robometer/configs/config.yaml --host=0.0.0.0 --port=8000

  python scripts/example_inference_tactile_online.py \\
      --eval-server-url http://localhost:8000 \\
      --video /path/to/video.mp4 \\
      --task "Pick up the red block" \\
      --step 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Re-use helpers from the sibling script – no robometer dependency needed.
from example_inference_tactile import (
    _DROID_VIEW_ORDER,
    extract_rewards_from_server_output,
    load_frames_input,
    make_progress_sample,
    post_evaluate_batch_npy,
)


def compute_online_rewards(
    eval_server_url: str,
    all_frames: np.ndarray,
    task: str,
    step: int = 8,
    timeout_s: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Incrementally infer rewards by growing the frame buffer one step at a time.

    At timestep i the buffer contains frames at raw indices [0, step, 2*step, …, i*step].
    The server is called once per timestep and we record the reward predicted for
    the *last* frame in the buffer (i.e. the newly appended frame).

    Args:
        eval_server_url: Base URL of the running eval server.
        all_frames: Full trajectory as a uint8 array of shape (T, H, W, C).
        task: Language instruction for the trajectory.
        step: Frame stride; the buffer grows by one frame every ``step`` raw frames.
        timeout_s: HTTP request timeout.

    Returns:
        online_rewards: Shape (N,) – one reward scalar per online timestep.
        online_success: Shape (N,) – one success-prob scalar per timestep (0s if absent).
        frame_indices: The raw frame indices that were appended at each timestep.
    """
    T = all_frames.shape[0]
    # Build the sequence of raw frame indices: 0, step, 2*step, …, up to T-1
    raw_indices = list(range(0, T, step))

    online_rewards: List[float] = []
    online_success: List[float] = []

    for i, raw_idx in enumerate(raw_indices):
        # Buffer = frames at indices [0, step, ..., raw_idx]
        buffer_indices = raw_indices[: i + 1]
        # buffer_indices = [0, i+1]
        
        buffer = all_frames[buffer_indices]  # (i+1, H, W, C)
        ## get the last 4 frames
        ## subsample 4 frames from the buffer_indices
        # indices = np.linspace(0, len(buffer_indices) - 1, 4, dtype=int)
        # print(f"Indices: {indices}")
        # buffer = all_frames[np.array(buffer_indices)[indices]]
        # buffer = all_frames[buffer_indices[::4]]
        # buffer = all_frames[buffer_indices[-4:]]

        subsequence_length = len(buffer)
        start_time = time.time()
        sample = make_progress_sample(
            frames=buffer,
            task=task,
            sample_id=str(i),
            subsequence_length=subsequence_length,
        )

        outputs = post_evaluate_batch_npy(
            eval_server_url,
            [sample],
            timeout_s=timeout_s,
            use_frame_steps=False,
        )
        progress_array, success_array = extract_rewards_from_server_output(outputs)
        end_time = time.time()
        print(f"Time taken for server call: {end_time - start_time} seconds")
        # The reward for the *latest* frame is the last element of the prediction.
        if progress_array.size > 0:
            online_rewards.append(float(progress_array[-1]))
        else:
            online_rewards.append(float("nan"))

        if success_array.size > 0:
            online_success.append(float(success_array[-1]))
        else:
            online_success.append(0.0)

        print(
            f"  step {i:4d} / {len(raw_indices) - 1}  "
            f"raw_frame={raw_idx:5d}  "
            f"buffer_len={len(buffer):4d}  "
            f"reward={online_rewards[-1]:.4f}  "
            f"success_prob={online_success[-1]:.4f}"
        )

    return (
        np.array(online_rewards, dtype=np.float32),
        np.array(online_success, dtype=np.float32),
        raw_indices,
    )


_CURVE_COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]


def plot_online_rewards(
    all_rewards: List[np.ndarray],
    all_success: List[np.ndarray],
    all_frame_indices: List[List[int]],
    tasks: List[str],
    out_path: Optional[str] = None,
) -> None:
    """Plot per-timestep online rewards for one or more tasks on the same axes.

    Args:
        all_rewards: One array of shape (N,) per task.
        all_success: One array of shape (N,) per task (empty/zeros if absent).
        all_frame_indices: Raw frame indices for each task's x-axis.
        tasks: Task label strings (used in the legend).
        out_path: If given, save the figure here; otherwise show interactively.
    """
    has_success = any(
        s is not None and np.any(s != 0) for s in all_success
    )
    n_plots = 2 if has_success else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4), squeeze=False)

    # Use the longest frame_indices list for tick labelling
    longest = max(all_frame_indices, key=len)
    max_len = len(longest)
    tick_every = max(1, max_len // 10)
    tick_pos = list(range(0, max_len, tick_every))
    tick_labels = [str(longest[p]) for p in tick_pos]

    ax = axes[0, 0]
    ax.set_xlim(0, max(max_len - 1, 1))
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Progress (reward)", fontsize=10)
    ax.set_xlabel("Timestep  [raw frame index]", fontsize=9)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
    ax.set_title("Online reward (latest frame)", fontsize=10)

    if has_success:
        ax2 = axes[0, 1]
        ax2.set_xlim(0, max(max_len - 1, 1))
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_ylabel("Success probability", fontsize=10)
        ax2.set_xlabel("Timestep  [raw frame index]", fontsize=9)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
        ax2.set_title("Online success probability", fontsize=10)

    for idx, (rewards, success, frame_indices, task) in enumerate(
        zip(all_rewards, all_success, all_frame_indices, tasks)
    ):
        color = _CURVE_COLORS[idx % len(_CURVE_COLORS)]
        xs = list(range(len(frame_indices)))
        label = task if task else f"Task {idx + 1}"
        ax.plot(xs, rewards, linewidth=2, color=color, marker="o", markersize=3, label=label)
        if has_success and success is not None and np.any(success != 0):
            axes[0, 1].plot(xs, success, linewidth=2, color=color, marker="o", markersize=3, label=label)

    ax.legend(fontsize=8, loc="upper left")
    if has_success:
        axes[0, 1].legend(fontsize=8, loc="upper left")

    fig.suptitle("Online incremental reward inference", fontsize=11)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved → {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Online incremental reward inference: at each timestep i the buffer "
            "[frame_0, frame_step, ..., frame_{step*i}] is sent to the eval server "
            "and the reward for the latest frame is recorded."
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
        "--video",
        type=str,
        required=True,
        help="Path or URL to a video, or a .npy/.npz file with frames (T,H,W,C)",
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=[""],
        help="One or more task instructions. Pass multiple to overlay reward curves in the same plot.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second when sampling from video (default: 1.0)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=8,
        help="Frame stride: append one frame every STEP raw frames (default: 8)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output stem for saved files (default: derived from video path)",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        choices=_DROID_VIEW_ORDER,
        help=(
            "Camera view to crop from a horizontally concatenated right|left|wrist video. "
            "If omitted, the full frame is used."
        ),
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    stem = video_path.stem

    # Resolve empty task string from annotation file
    tasks: List[str] = args.task
    if len(tasks) == 1 and tasks[0] == "":
        annotation_folder = video_path.parent.parent.parent / "annotations"
        annotation_file = annotation_folder / "val" / f"{stem}.json"
        if not annotation_file.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_file}")
        with open(annotation_file, "r") as f:
            annotation = json.load(f)
        tasks = [annotation["texts"][0]]
        print(f"Inferred task: {tasks[0]}")

    print(f"Loading frames from {args.video} …")
    all_frames = load_frames_input(str(args.video), fps=float(args.fps))
    print(f"Loaded {all_frames.shape[0]} frames  shape={all_frames.shape}")

    if args.view is not None:
        from example_inference_tactile import crop_view_from_concatenated
        all_frames = crop_view_from_concatenated(all_frames, args.view)
        print(f"Cropped to view '{args.view}': {all_frames.shape}")

    # Run inference for each task and collect results
    all_rewards: List[np.ndarray] = []
    all_success: List[np.ndarray] = []
    all_frame_indices: List[List[int]] = []
    all_elapsed: List[float] = []

    for task_idx, task in enumerate(tasks):
        print(f"\n[Task {task_idx + 1}/{len(tasks)}] '{task}'")
        print(f"Starting online inference with step={args.step} …")
        t0 = time.time()
        online_rewards, online_success, frame_indices = compute_online_rewards(
            eval_server_url=args.eval_server_url,
            all_frames=all_frames,
            task=task,
            step=args.step,
            timeout_s=float(args.timeout_s),
        )
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s  ({len(frame_indices)} server calls)")

        all_rewards.append(online_rewards)
        all_success.append(online_success)
        all_frame_indices.append(frame_indices)
        all_elapsed.append(elapsed)

        # Save per-task arrays
        safe_task = task[:40].replace(" ", "_").replace("/", "-")
        out_stem = args.out if args.out is not None else str(
            video_path.with_name(stem + f"_{safe_task}_online")
        )
        out_stem_path = Path(out_stem)
        out_stem_path.parent.mkdir(parents=True, exist_ok=True)

        rewards_path = out_stem_path.with_suffix(".npy")
        success_path = Path(str(out_stem_path) + "_success_probs.npy")
        np.save(str(rewards_path), online_rewards)
        np.save(str(success_path), online_success)
        print(f"Rewards saved → {rewards_path}")
        print(f"Success probs saved → {success_path}")

    # Combined plot (all tasks on the same axes)
    safe_all = "_vs_".join(t[:20].replace(" ", "_").replace("/", "-") for t in tasks)
    plot_stem = args.out if args.out is not None else str(
        video_path.with_name(stem + f"_{safe_all}_online")
    )
    plot_path = Path(str(plot_stem) + "_reward_plot.png")
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    plot_online_rewards(
        all_rewards=all_rewards,
        all_success=all_success,
        all_frame_indices=all_frame_indices,
        tasks=tasks,
        out_path=str(plot_path),
    )

    summary = {
        "video": str(video_path),
        "tasks": tasks,
        "step": args.step,
        "num_raw_frames": int(all_frames.shape[0]),
        "total_elapsed_s": round(sum(all_elapsed), 2),
        "plot_png": str(plot_path),
        "per_task": [
            {
                "task": task,
                "num_timesteps": len(fi),
                "elapsed_s": round(el, 2),
                "reward_min": float(np.nanmin(r)) if r.size else None,
                "reward_max": float(np.nanmax(r)) if r.size else None,
                "reward_mean": float(np.nanmean(r)) if r.size else None,
            }
            for task, r, fi, el in zip(tasks, all_rewards, all_frame_indices, all_elapsed)
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

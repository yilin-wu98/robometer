#!/usr/bin/env python3
"""
Fast online (streaming) reward inference client for the RBM eval server.

Unlike ``example_inference_tactile_online.py`` which re-sends the full growing
frame buffer on every server call, this script uses the *streaming* session API:

  1. ``POST /stream/create``  — create a server-side session (once per trajectory).
  2. ``POST /stream/{id}/append`` — send ONE new frame per timestep; the server
     caches the ViT patch embeddings for all previous frames and only runs the
     vision encoder on the new frame.  The LM still attends to the full sequence.
  3. ``DELETE /stream/{id}``  — clean up when done.

This gives the same reward quality as the slow (full re-run) version but with
significantly lower latency per step when trajectories are long, because the ViT
cost is amortized over the session lifetime.

Example:
  # Start the streaming server first (in another terminal):
  #   uv run python robometer/evals/eval_server_streaming.py \\
  #       model_path=robometer/Robometer-4B --host=0.0.0.0 --port=8001

  python scripts/example_inference_tactile_online_fast.py \\
      --eval-server-url http://localhost:8001 \\
      --video /path/to/video.mp4 \\
      --task "Pick up the red block" \\
      --step 8
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import requests


# Re-use helpers from the sibling scripts
from example_inference_tactile import (
    _DROID_VIEW_ORDER,
    load_frames_input,
)
from example_inference_tactile_online import (
    _CURVE_COLORS,
    plot_online_rewards,
)


# ---------------------------------------------------------------------------
# Streaming client helpers
# ---------------------------------------------------------------------------

def stream_create(eval_server_url: str, task: str, timeout_s: float = 30.0) -> str:
    """Create a streaming session on the server.

    Returns the session_id string.
    """
    url = f"{eval_server_url.rstrip('/')}/stream/create"
    resp = requests.post(url, json={"task": task}, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    session_id: str = data["session_id"]
    return session_id


def stream_append_frame(
    eval_server_url: str,
    session_id: str,
    frame_np: np.ndarray,
    timeout_s: float = 120.0,
) -> Tuple[float, float]:
    """Send one frame to the server and return ``(reward, success_prob)``.

    Args:
        eval_server_url: Base URL of the streaming eval server.
        session_id: Session ID returned by :func:`stream_create`.
        frame_np: uint8 numpy array of shape ``(H, W, C)``.
        timeout_s: HTTP request timeout.

    Returns:
        ``(reward, success_prob)`` – float scalars in ``[0, 1]``.
    """
    url = f"{eval_server_url.rstrip('/')}/stream/{session_id}/append"

    # Serialise the frame as a .npy binary blob
    buf = io.BytesIO()
    np.save(buf, frame_np)
    buf.seek(0)

    resp = requests.post(
        url,
        files={"frame": ("frame.npy", buf, "application/octet-stream")},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["reward"]), float(data["success_prob"])


def stream_evaluate_candidate(
    eval_server_url: str,
    session_id: str,
    frame_np: np.ndarray,
    timeout_s: float = 120.0,
) -> Tuple[float, float]:
    """Score a candidate frame WITHOUT adding it to the session history.

    The session's frame buffer and KV cache are unchanged after this call.
    Use it to rank N predicted frames at each policy query without advancing
    the rolling session history.

    Args:
        eval_server_url: Base URL of the streaming eval server.
        session_id: Session ID returned by :func:`stream_create`.
        frame_np: uint8 numpy array of shape ``(H, W, C)``.
        timeout_s: HTTP request timeout.

    Returns:
        ``(reward, success_prob)`` – float scalars in ``[0, 1]``.
    """
    url = f"{eval_server_url.rstrip('/')}/stream/{session_id}/evaluate_candidate"

    buf = io.BytesIO()
    np.save(buf, frame_np)
    buf.seek(0)

    resp = requests.post(
        url,
        files={"frame": ("frame.npy", buf, "application/octet-stream")},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return float(data["reward"]), float(data["success_prob"])


def stream_evaluate_candidates_batch(
    eval_server_url: str,
    session_id: str,
    frames_np: np.ndarray,
    timeout_s: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate N candidate frames in one batched server call.

    Sends all N frames to ``/stream/{id}/evaluate_candidates``.  The server
    runs a single batched ViT forward and a single batched LM forward (with
    the history KV cache expanded to batch size N), which is much faster than
    N sequential ``/evaluate_candidate`` calls.

    The session history is **not** modified.

    Args:
        eval_server_url: Base URL of the streaming eval server.
        session_id: Session ID returned by :func:`stream_create`.
        frames_np: uint8 numpy array of shape ``(N, H, W, C)``.
        timeout_s: HTTP request timeout.

    Returns:
        ``(rewards, success_probs)`` – float32 arrays of shape ``(N,)``.
    """
    url = f"{eval_server_url.rstrip('/')}/stream/{session_id}/evaluate_candidates"

    buf = io.BytesIO()
    np.save(buf, frames_np)
    buf.seek(0)

    resp = requests.post(
        url,
        files={"frames": ("frames.npy", buf, "application/octet-stream")},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return (
        np.array(data["rewards"], dtype=np.float32),
        np.array(data["success_probs"], dtype=np.float32),
    )


def stream_delete(eval_server_url: str, session_id: str, timeout_s: float = 10.0) -> None:
    """Delete a streaming session on the server."""
    url = f"{eval_server_url.rstrip('/')}/stream/{session_id}"
    try:
        resp = requests.delete(url, timeout=timeout_s)
        resp.raise_for_status()
    except Exception as exc:
        # Non-fatal — session will be evicted by server-side timeout anyway.
        print(f"  [warn] Could not delete session {session_id}: {exc}")


# ---------------------------------------------------------------------------
# Core streaming inference loop
# ---------------------------------------------------------------------------

def compute_online_rewards_streaming(
    eval_server_url: str,
    all_frames: np.ndarray,
    task: str,
    step: int = 8,
    timeout_s: float = 120.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Incrementally infer rewards using the streaming session API.

    At timestep i the server receives frame ``i*step`` (one new frame per call).
    The server caches the ViT patch embeddings from all previous frames and only
    runs the vision encoder on the new frame, making each call much faster than
    re-sending the full buffer.

    Args:
        eval_server_url: Base URL of the running streaming eval server.
        all_frames: Full trajectory as a uint8 array of shape ``(T, H, W, C)``.
        task: Language instruction for the trajectory.
        step: Frame stride; one frame is sent per ``step`` raw frames.
        timeout_s: HTTP request timeout.

    Returns:
        online_rewards: Shape ``(N,)`` – one reward scalar per streaming step.
        online_success: Shape ``(N,)`` – one success-prob scalar per step.
        frame_indices: The raw frame indices sent at each step.
    """
    T = all_frames.shape[0]
    raw_indices = list(range(0, T, step))
    N = len(raw_indices)

    # Create a server-side session
    session_id = stream_create(eval_server_url, task, timeout_s=timeout_s)
    print(f"  Streaming session created: {session_id}")

    online_rewards: List[float] = []
    online_success: List[float] = []

    try:
        for i, raw_idx in enumerate(raw_indices):
            frame = all_frames[raw_idx]  # (H, W, C)
            t0 = time.time()

            reward, success_prob = stream_append_frame(
                eval_server_url, session_id, frame, timeout_s=timeout_s
            )

            elapsed = time.time() - t0
            online_rewards.append(reward)
            online_success.append(success_prob)

            print(
                f"  step {i:4d} / {N - 1}  "
                f"raw_frame={raw_idx:5d}  "
                f"reward={reward:.4f}  "
                f"success_prob={success_prob:.4f}  "
                f"elapsed={elapsed:.3f}s"
            )
    finally:
        # Always clean up the session
        stream_delete(eval_server_url, session_id)
        print(f"  Streaming session deleted: {session_id}")

    return (
        np.array(online_rewards, dtype=np.float32),
        np.array(online_success, dtype=np.float32),
        raw_indices,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast online reward inference: streams one frame per timestep to the "
            "eval server.  The server caches ViT patch embeddings so only the new "
            "frame is encoded each step — same reward quality, much lower per-step cost."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-server-url",
        type=str,
        default="http://localhost:8001",
        help="Streaming eval server base URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to a video, or a .npy/.npz file with frames (T, H, W, C)",
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=[""],
        help=(
            "One or more task instructions. Pass multiple to overlay reward curves "
            "in the same plot."
        ),
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
        help="Frame stride: one frame is sent per STEP raw frames (default: 8)",
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
            "Camera view to crop from a horizontally concatenated right|left|wrist "
            "video.  If omitted, the full frame is used."
        ),
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    stem = video_path.stem

    # Resolve empty task string from annotation file (same logic as sibling scripts)
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

    all_rewards: List[np.ndarray] = []
    all_success: List[np.ndarray] = []
    all_frame_indices: List[List[int]] = []
    all_elapsed: List[float] = []

    for task_idx, task in enumerate(tasks):
        print(f"\n[Task {task_idx + 1}/{len(tasks)}] '{task}'")
        print(f"Starting streaming inference with step={args.step} …")
        t0 = time.time()

        online_rewards, online_success, frame_indices = compute_online_rewards_streaming(
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
            video_path.with_name(stem + f"_{safe_task}_online_fast")
        )
        out_stem_path = Path(out_stem)
        out_stem_path.parent.mkdir(parents=True, exist_ok=True)

        rewards_path = out_stem_path.with_suffix(".npy")
        success_path = Path(str(out_stem_path) + "_success_probs.npy")
        np.save(str(rewards_path), online_rewards)
        np.save(str(success_path), online_success)
        print(f"Rewards saved → {rewards_path}")
        print(f"Success probs saved → {success_path}")

    # Combined plot
    safe_all = "_vs_".join(t[:20].replace(" ", "_").replace("/", "-") for t in tasks)
    plot_stem = args.out if args.out is not None else str(
        video_path.with_name(stem + f"_{safe_all}_online_fast")
    )
    plot_path = Path(str(plot_stem) + "_reward_plot.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)

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
        "mode": "streaming (ViT-cached)",
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

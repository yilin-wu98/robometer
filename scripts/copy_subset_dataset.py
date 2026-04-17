#!/usr/bin/env python3
"""
Copy videos 0-119 and their reward annotations into a new dataset folder.

Source layout:
  SRC_ROOT/video/val/0.mp4 ... 119.mp4
  SRC_ROOT/annotations_reward/val/0.json ... 119.json

Destination layout:
  DST_ROOT/video/val/0.mp4 ... 119.mp4
  DST_ROOT/annotations/val/0.json ... 119.json   <- annotations_reward renamed to annotations

Example:
  python scripts/copy_subset_dataset.py
  python scripts/copy_subset_dataset.py \\
      --src /data/yilin/world_model_data_preprocessed_v2 \\
      --dst /data/yilin/world_model_data_preprocessed_ours \\
      --split val --start 0 --end 119
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def numeric_stem(p: Path) -> int:
    try:
        return int(p.stem)
    except ValueError:
        return -1


def list_videos(video_dir: Path) -> list[Path]:
    out: list[Path] = []
    for pat in ("*.mp4", "*.webm", "*.avi", "*.mov"):
        out.extend(video_dir.glob(pat))
    return sorted(out, key=numeric_stem)


def main() -> None:
    p = argparse.ArgumentParser(description="Copy a numbered video subset + reward annotations to a new dataset root.")
    p.add_argument(
        "--src",
        type=Path,
        default=Path("/data/yilin/world_model_data_preprocessed_v2"),
        help="Source dataset root",
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/yilin/world_model_data_preprocessed_ours"),
        help="Destination dataset root",
    )
    p.add_argument("--split", type=str, default="val", help="Split subfolder (default: val)")
    p.add_argument("--start", type=int, default=0, help="First video index inclusive (default: 0)")
    p.add_argument("--end", type=int, default=119, help="Last video index inclusive (default: 119)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files")
    args = p.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    split: str = args.split
    indices = set(range(args.start, args.end + 1))

    src_video_dir = src / "videos" / split
    src_ann_dir = src / "annotations_reward" / split
    dst_video_dir = dst / "videos" / split
    dst_ann_dir = dst / "annotations" / split

    for d in (src_video_dir, src_ann_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Source directory not found: {d}")

    dst_video_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)

    videos = [v for v in list_videos(src_video_dir) if numeric_stem(v) in indices]
    if not videos:
        raise RuntimeError(f"No videos with indices {args.start}-{args.end} found in {src_video_dir}")

    ok_video = ok_ann = skip_video = skip_ann = missing_ann = 0

    for vp in videos:
        idx = numeric_stem(vp)

        # --- video ---
        dst_vp = dst_video_dir / vp.name
        if dst_vp.exists() and not args.overwrite:
            print(f"SKIP (exists): {dst_vp}")
            skip_video += 1
        else:
            shutil.copy2(vp, dst_vp)
            print(f"VIDEO  {idx:4d}: {vp} -> {dst_vp}")
            ok_video += 1

        # --- annotation ---
        src_ap = src_ann_dir / vp.with_suffix(".json").name
        dst_ap = dst_ann_dir / src_ap.name
        if not src_ap.is_file():
            print(f"MISSING annotation for {vp.name}: expected {src_ap}")
            missing_ann += 1
            continue
        if dst_ap.exists() and not args.overwrite:
            print(f"SKIP (exists): {dst_ap}")
            skip_ann += 1
        else:
            shutil.copy2(src_ap, dst_ap)
            print(f"ANN    {idx:4d}: {src_ap} -> {dst_ap}")
            ok_ann += 1

    summary = {
        "videos_copied": ok_video,
        "videos_skipped": skip_video,
        "annotations_copied": ok_ann,
        "annotations_skipped": skip_ann,
        "annotations_missing": missing_ann,
        "dst_video_dir": str(dst_video_dir),
        "dst_ann_dir": str(dst_ann_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

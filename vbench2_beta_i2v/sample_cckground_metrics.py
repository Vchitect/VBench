"""
VBench-I2V Metric Evaluation
==============================
Model-agnostic: expects a folder of already-generated videos named
    {prompt}-{index}.mp4
matching the image filename stems from vbench2_i2v_full_info.json.

Usage (from VBench root):
    cd c:\\workspace\\world\\VBench
    python vbench2_beta_i2v/sample_cckground_metrics.py \\
        --videos_path  ./out/videos \\
        --dimensions   i2v_background i2v_subject \\
        --output_path  ./evaluation_results

Or from vbench2_beta_i2v dir:
    python sample_cckground_metrics.py \\
        --videos_path ./out/videos
"""

import argparse
import json
import os
import re
import sys

# ── Project roots ─────────────────────────────────────────────────────────────
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_VBENCH_ROOT = os.path.dirname(_THIS_DIR)

sys.path.insert(0, _VBENCH_ROOT)

# ── Default paths ─────────────────────────────────────────────────────────────
_INFO_JSON = os.path.join(_THIS_DIR, "vbench2_i2v_full_info.json")
_CROP_BASE = os.path.join(_THIS_DIR, "data", "crop")


def _safe(prompt):
    return re.sub(r'[<>:"/\\|?*]', "_", prompt)[:150]


def _iw_stem(prompt):
    """Infinite-World naming: first 30 chars, spaces → underscores."""
    return prompt[:30].replace(" ", "_")


def _find_iw_video(videos_path, prompt):
    """
    Find the last cumulative video for a given prompt, ignoring the 4-digit task prefix.
    Pattern: ????_{prompt[:30]}_chunk{N:03d}_cumulative.mp4
    Returns the path of the highest chunk, else None.
    """
    stem   = _iw_stem(prompt)
    needle = f"_{stem}_chunk"
    suffix = "_cumulative.mp4"
    found  = []
    try:
        for fname in os.listdir(videos_path):
            if needle in fname and fname.endswith(suffix):
                found.append(os.path.join(videos_path, fname))
    except FileNotFoundError:
        return None
    return sorted(found)[-1] if found else None


def build_video_pairs(videos_path, dimensions, image_types, resolution, num_samples):
    """
    Return list of (image_path, video_path) matching the JSON filters.

    Looks for videos in two naming formats:
      1. VBench standard:      {prompt}-{index}.mp4
      2. Infinite-World output: {task_idx:04d}_{prompt[:30]}_chunk{N:03d}_cumulative.mp4
    """
    info_list = json.load(open(_INFO_JSON, "r", encoding="utf-8"))
    allowed_types = {t.strip() for t in image_types.split(",") if t.strip()} \
                    if image_types else None

    image_folder = os.path.join(_CROP_BASE, resolution)

    seen_prompts = set()
    pairs   = []   # (image_path, video_path)
    missing = []   # descriptions of what wasn't found

    for info in info_list:
        if not any(d in info.get("dimension", []) for d in dimensions):
            continue
        if allowed_types and info.get("image_type") not in allowed_types:
            continue

        image_name = info["image_name"]
        prompt     = info["prompt_en"]
        key        = (image_name, prompt)
        if key in seen_prompts:
            continue
        seen_prompts.add(key)

        image_path = os.path.join(image_folder, image_name)

        found_any = False

        # ── Try VBench standard naming first ──────────────────────────────
        for index in range(num_samples):
            vbench_path = os.path.join(videos_path, f"{_safe(prompt)}-{index}.mp4")
            if os.path.exists(vbench_path):
                pairs.append((image_path, vbench_path))
                found_any = True

        # ── Fall back to Infinite-World cumulative naming ─────────────────
        if not found_any:
            iw_path = _find_iw_video(videos_path, prompt)
            if iw_path:
                pairs.append((image_path, iw_path))
                found_any = True

        if not found_any:
            missing.append(f"{_safe(prompt)}-[0..{num_samples-1}].mp4 / "
                           f"????_{_iw_stem(prompt)}_chunk*_cumulative.mp4")

    return pairs, missing


def run_evaluation(videos_path, dimensions, output_path, resolution, image_types, num_samples):
    import torch
    from vbench2_beta_i2v import VBenchI2V

    pairs, missing = build_video_pairs(
        videos_path, dimensions, image_types, resolution, num_samples
    )

    print(f"\nFound : {len(pairs)} videos")
    if missing:
        print(f"Missing: {len(missing)} videos")
        for p in missing[:10]:
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if not pairs:
        print("No videos found — nothing to evaluate.")
        return

    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_vbench = VBenchI2V(device, _INFO_JSON, output_path)
    my_vbench.evaluate(
        videos_path=videos_path,
        name="results",
        dimension_list=dimensions,
        resolution=resolution,
    )
    print(f"\nEvaluation results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="VBench-I2V metric evaluation on pre-generated videos"
    )
    parser.add_argument("--videos_path", required=True,
                        help="Folder of videos named {prompt}-{index}.mp4")
    parser.add_argument("--dimensions", nargs="+",
                        default=[
                            "i2v_background", "i2v_subject", "camera_motion",
                            "background_consistency", "subject_consistency",
                            "motion_smoothness", "dynamic_degree", "temporal_flickering",
                            "aesthetic_quality", "imaging_quality",
                        ],
                        help="VBench dimensions to evaluate (default: all 10)")
    parser.add_argument("--resolution", default="1-1",
                        choices=["1-1", "8-5", "7-4", "16-9"])
    parser.add_argument("--image_types", type=str, default="",
                        help="Comma-separated image_type filter, e.g. background,scenery")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Videos per prompt expected (default: 5)")
    parser.add_argument("--output_path", default=os.path.join(_THIS_DIR, "evaluation_results"),
                        help="Where to save evaluation JSON results")
    parser.add_argument("--list_only", action="store_true",
                        help="Only list found/missing videos, skip metric computation")
    args = parser.parse_args()

    if args.list_only:
        pairs, missing = build_video_pairs(
            args.videos_path, args.dimensions, args.image_types,
            args.resolution, args.num_samples,
        )
        print(f"Found : {len(pairs)}")
        print(f"Missing: {len(missing)}")
        for img, vid in pairs:
            print(f"  {os.path.basename(vid)}")
    else:
        run_evaluation(
            videos_path=args.videos_path,
            dimensions=args.dimensions,
            output_path=args.output_path,
            resolution=args.resolution,
            image_types=args.image_types,
            num_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()

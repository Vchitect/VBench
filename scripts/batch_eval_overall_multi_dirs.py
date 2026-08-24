import argparse
import csv
import importlib
import json
import os
import re
from collections import defaultdict

import torch

from vbench.utils import init_submodules


DEFAULT_DIMS = ["overall_consistency"]

CLIP_HEADER_RE = re.compile(r"^\s*Clip\s*(\d+)\s*[:：]\s*$", re.IGNORECASE)
ROLE_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+)\s*$")


def parse_prompts(path):
    clips = {}
    current_clip = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            header = CLIP_HEADER_RE.match(line)
            if header:
                current_clip = f"Clip{int(header.group(1))}"
                clips.setdefault(current_clip, {})
                continue
            match = ROLE_LINE_RE.match(line)
            if match and current_clip:
                role = match.group(1)
                prompt = match.group(2)
                clips[current_clip][role] = prompt
            elif match and current_clip is None:
                current_clip = "Clip1"
                clips.setdefault(current_clip, {})
                role = match.group(1)
                prompt = match.group(2)
                clips[current_clip][role] = prompt
    return clips


def save_json(path, data):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def write_csv(path, rows, fieldnames):
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.4f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def build_full_info_from_prompts(prompts_path, videos_dirs, dimension_list, exclude_roles):
    clips = parse_prompts(prompts_path)
    full_info_list = []
    missing = []

    for videos_dir in videos_dirs:
        for clip_name, role_map in clips.items():
            for role, prompt in role_map.items():
                if role in exclude_roles:
                    continue
                video_path = os.path.join(videos_dir, f"{role}.mp4")
                if not os.path.isfile(video_path):
                    missing.append(f"{videos_dir}/{clip_name}/{role}")
                    continue
                full_info_list.append(
                    {
                        "prompt_en": prompt,
                        "dimension": dimension_list,
                        "video_list": [video_path],
                        "clip": clip_name,
                        "role": role,
                        "videos_dir": videos_dir,
                    }
                )
    if missing:
        print("Missing videos:", ", ".join(sorted(missing)))
    return full_info_list


def align_results_with_full_info(results_dict, full_info_list):
    scores_by_index = defaultdict(dict)
    for dimension, result in results_dict.items():
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            continue
        video_results = result[1]
        if len(video_results) != len(full_info_list):
            print(
                f"Warning: {dimension} results length {len(video_results)} does not match "
                f"full_info length {len(full_info_list)}."
            )
        limit = min(len(video_results), len(full_info_list))
        for idx in range(limit):
            item = video_results[idx]
            scores_by_index[idx][dimension] = item.get("video_results")
    return scores_by_index


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate overall_consistency for multiple flat video folders."
    )
    parser.add_argument(
        "--videos_dir",
        action="append",
        required=True,
        help="Video folder containing role.mp4 files. Can be repeated.",
    )
    parser.add_argument(
        "--prompts_txt",
        required=True,
        help="Prompt txt with Clip sections.",
    )
    parser.add_argument(
        "--exclude_roles",
        nargs="+",
        default=["daqiao", "malasong"],
        help="Roles to skip (default: daqiao malasong).",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=DEFAULT_DIMS,
        help="Dimensions to evaluate.",
    )
    parser.add_argument(
        "--output_json",
        default="evaluation_results/overall_multi_dirs.json",
        help="Path to save evaluation results JSON.",
    )
    parser.add_argument(
        "--output_csv",
        default="evaluation_results/overall_multi_dirs.csv",
        help="Path to save evaluation results CSV.",
    )
    parser.add_argument(
        "--full_info_json",
        default="evaluation_results/overall_multi_dirs_full.json",
        help="Path to save the generated full_info JSON.",
    )
    parser.add_argument(
        "--reuse_json",
        action="store_true",
        help="Reuse an existing output_json to generate CSV without rerunning models.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on: cuda or cpu.",
    )
    parser.add_argument(
        "--read_frame",
        action="store_true",
        help="Read frames instead of videos when supported.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Load checkpoints from local cache if available.",
    )
    parser.add_argument(
        "--imaging_quality_preprocessing_mode",
        default="longer",
        help="imaging_quality preprocessing: longer, shorter, shorter_centercrop, None.",
    )
    parser.add_argument(
        "--include_prompt",
        action="store_true",
        help="Include prompt_en column in the output CSV.",
    )
    args = parser.parse_args()

    if os.path.abspath(args.full_info_json) == os.path.abspath(args.output_json):
        raise SystemExit("full_info_json and output_json must be different files.")

    if args.reuse_json:
        if not os.path.isfile(args.output_json):
            raise SystemExit(f"output_json not found: {args.output_json}")
        if not os.path.isfile(args.full_info_json):
            raise SystemExit(f"full_info_json not found: {args.full_info_json}")
        with open(args.output_json, "r", encoding="utf-8") as f:
            results_dict = json.load(f)
        with open(args.full_info_json, "r", encoding="utf-8") as f:
            full_info_list = json.load(f)
    else:
        full_info_list = build_full_info_from_prompts(
            args.prompts_txt, args.videos_dir, args.dimensions, set(args.exclude_roles)
        )
        if not full_info_list:
            raise SystemExit("No evaluation entries found. Check prompts or videos.")
        save_json(args.full_info_json, full_info_list)

        device = torch.device(args.device)
        submodules_dict = init_submodules(args.dimensions, local=args.local, read_frame=args.read_frame)

        results_dict = {}
        for dimension in args.dimensions:
            dimension_module = importlib.import_module(f"vbench.{dimension}")
            evaluate_func = getattr(dimension_module, f"compute_{dimension}")
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(
                args.full_info_json,
                device,
                submodules_list,
                imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
            )
            results_dict[dimension] = results

        save_json(args.output_json, results_dict)

    scores_by_index = align_results_with_full_info(results_dict, full_info_list)

    rows = []
    for idx, info in enumerate(full_info_list):
        row = {
            "videos_dir": info.get("videos_dir", ""),
            "clip": info.get("clip", ""),
            "role": info.get("role", ""),
            "video_path": info.get("video_list", [""])[0],
        }
        if args.include_prompt:
            row["prompt_en"] = info.get("prompt_en", "")
        for dimension in args.dimensions:
            row[dimension] = scores_by_index.get(idx, {}).get(dimension, "")
        rows.append(row)

    fieldnames = ["videos_dir", "clip", "role", "video_path"]
    if args.include_prompt:
        fieldnames.append("prompt_en")
    fieldnames += args.dimensions
    write_csv(args.output_csv, rows, fieldnames)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()


# Example:
# python scripts/batch_eval_overall_multi_dirs.py \
#   --videos_dir videos/10_edit_videos/long \
#   --videos_dir videos/concat_results_10_examples \
#   --videos_dir videos/long_results_10_examples \
#   --prompts_txt prompt_all_long.txt \
#   --dimensions overall_consistency \
#   --exclude_roles daqiao malasong \
#   --include_prompt \
#   --output_json evaluation_results/overall_multi_dirs.json \
#   --output_csv evaluation_results/overall_multi_dirs.csv \
#   --full_info_json evaluation_results/overall_multi_dirs_full.json




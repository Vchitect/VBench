import argparse
import csv
import importlib
import json
import os
import re
from pathlib import Path

import torch

from vbench.utils import get_prompt_from_filename, init_submodules


DEFAULT_DIMS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "temporal_style",
    "overall_consistency",
]


def find_videos(root_dir):
    video_paths = []
    for cur_root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                video_paths.append(os.path.join(cur_root, filename))
    return sorted(video_paths)


def build_full_info(video_paths, dimension_list):
    full_info_list = []
    for video_path in video_paths:
        full_info_list.append(
            {
                "prompt_en": get_prompt_from_filename(video_path),
                "dimension": dimension_list,
                "video_list": [video_path],
            }
        )
    return full_info_list


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


def _parse_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower() == "true":
        return 1.0
    if text.lower() == "false":
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


def append_mean_row(csv_path, label="MEAN"):
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not fieldnames:
        return

    rows = [row for row in rows if row.get("video_path") != label]

    sums = {name: 0.0 for name in fieldnames}
    counts = {name: 0 for name in fieldnames}
    for row in rows:
        for name in fieldnames:
            if name == "video_path":
                continue
            value = _parse_numeric(row.get(name))
            if value is None:
                continue
            sums[name] += value
            counts[name] += 1

    mean_row = {}
    for name in fieldnames:
        if name == "video_path":
            mean_row[name] = label
            continue
        if counts[name] == 0:
            mean_row[name] = ""
        else:
            mean_row[name] = f"{sums[name] / counts[name]:.4f}"

    rows.append(mean_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_video_paths_from_results(results_dict):
    video_paths = set()
    for result in results_dict.values():
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            continue
        video_results = result[1]
        for item in video_results:
            video_path = item.get("video_path")
            if video_path:
                video_paths.add(video_path)
    return sorted(video_paths)


def collect_video_paths_from_full_info(full_info_list):
    video_paths = set()
    for item in full_info_list:
        video_list = item.get("video_list")
        if not video_list:
            continue
        if isinstance(video_list, (list, tuple)):
            for path in video_list:
                if path:
                    video_paths.add(path)
        elif isinstance(video_list, str):
            video_paths.add(video_list)
    return sorted(video_paths)


def build_prompt_map_from_full_info(full_info_list):
    prompt_map = {}
    for item in full_info_list:
        prompt = item.get("prompt_en")
        video_list = item.get("video_list")
        if not prompt or not video_list:
            continue
        if isinstance(video_list, (list, tuple)):
            for path in video_list:
                if path:
                    prompt_map[path] = prompt
        elif isinstance(video_list, str):
            prompt_map[video_list] = prompt
    return prompt_map


CLIP_HEADER_RE = re.compile(r"^\s*Clip\s*(\d+)\s*[:：]\s*$", re.IGNORECASE)
ROLE_LINE_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*:\s*(.+)\s*$")
CLIP_DIR_RE = re.compile(r"clip\s*(\d+)", re.IGNORECASE)


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
                # No Clip headers: treat file as a single clip.
                current_clip = "Clip1"
                clips.setdefault(current_clip, {})
                role = match.group(1)
                prompt = match.group(2)
                clips[current_clip][role] = prompt
    return clips


def resolve_clip_dirs(clips, videos_dir, clip_names=None):
    if os.path.isdir(videos_dir):
        has_mp4 = any(
            filename.lower().endswith(".mp4")
            for filename in os.listdir(videos_dir)
            if os.path.isfile(os.path.join(videos_dir, filename))
        )
        if has_mp4:
            target_clips = clip_names or list(clips.keys())
            if len(target_clips) == 1:
                return {target_clips[0]: videos_dir}

    subdirs = []
    for entry in os.listdir(videos_dir):
        full = os.path.join(videos_dir, entry)
        if os.path.isdir(full):
            subdirs.append(entry)

    num_to_dirs = {}
    for subdir in subdirs:
        match = CLIP_DIR_RE.search(subdir)
        if not match:
            continue
        num = int(match.group(1))
        num_to_dirs.setdefault(num, []).append(subdir)

    clip_to_dir = {}
    target_clips = clip_names or clips.keys()
    for clip_name in target_clips:
        match = CLIP_HEADER_RE.match(f"{clip_name}:")
        if not match:
            raise SystemExit(f"Unexpected clip name: {clip_name}")
        num = int(match.group(1))
        candidates = num_to_dirs.get(num, [])
        if not candidates:
            raise SystemExit(f"No directory found for {clip_name} under {videos_dir}.")
        if len(candidates) > 1:
            raise SystemExit(
                f"Multiple directories match {clip_name}: {candidates}. "
                "Please keep only one matching folder."
            )
        clip_to_dir[clip_name] = os.path.join(videos_dir, candidates[0])
    return clip_to_dir


def index_videos_by_clip(clip_dirs):
    mapping = {}
    for clip_name, clip_dir in clip_dirs.items():
        for root, _, files in os.walk(clip_dir):
            for filename in files:
                if not filename.lower().endswith(".mp4"):
                    continue
                role = os.path.splitext(filename)[0]
                mapping.setdefault(clip_name, {})[role] = os.path.join(root, filename)
    return mapping


def build_full_info_from_prompts(prompts_path, videos_dir, dimension_list, clip_names=None):
    clips = parse_prompts(prompts_path)
    if clip_names:
        for clip_name in clip_names:
            if clip_name not in clips:
                raise SystemExit(f"Clip not found in prompts: {clip_name}")
    clip_dirs = resolve_clip_dirs(clips, videos_dir, clip_names)
    video_map = index_videos_by_clip(clip_dirs)

    full_info_list = []
    missing_videos = []
    missing_prompts = []
    target_clips = clip_names or sorted(clips.keys())
    for clip_name in target_clips:
        role_map = clips.get(clip_name, {})
        video_roles = set(video_map.get(clip_name, {}).keys())
        prompt_roles = set(role_map.keys())
        for role, prompt in role_map.items():
            video_path = video_map.get(clip_name, {}).get(role)
            if not video_path:
                missing_videos.append(f"{clip_name}/{role}")
                continue
            full_info_list.append(
                {
                    "prompt_en": prompt,
                    "dimension": dimension_list,
                    "video_list": [video_path],
                }
            )
        for role in sorted(video_roles - prompt_roles):
            missing_prompts.append(f"{clip_name}/{role}")
    if missing_videos or missing_prompts:
        full_info_list.append(
            {
                "dimension": [],
                "missing_videos": missing_videos,
                "missing_prompts": missing_prompts,
            }
        )
        if missing_videos:
            print("Missing videos:", ", ".join(sorted(missing_videos)))
        if missing_prompts:
            print("Missing prompts:", ", ".join(sorted(missing_prompts)))
    return full_info_list


def main():
    parser = argparse.ArgumentParser(description="Batch evaluate VBench custom_input videos.")
    parser.add_argument(
        "--videos_dir",
        default="videos",
        help="Root directory containing mp4 files (supports nested folders).",
    )
    parser.add_argument(
        "--output_json",
        default="evaluation_results/batch_eval_results.json",
        help="Path to save evaluation results JSON.",
    )
    parser.add_argument(
        "--output_csv",
        default="evaluation_results/batch_eval_results.csv",
        help="Path to save evaluation results CSV.",
    )
    parser.add_argument(
        "--full_info_json",
        default="evaluation_results/batch_full_info.json",
        help="Path to save the generated full_info JSON.",
    )
    parser.add_argument(
        "--use_full_info_json",
        action="store_true",
        help="Use an existing full_info_json instead of generating from filenames.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=DEFAULT_DIMS,
        help="Dimensions to evaluate.",
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
        "--prompts_txt",
        help="Prompt txt with Clip sections (use with --videos_dir to build full_info_json).",
    )
    parser.add_argument(
        "--clip",
        action="append",
        help="Clip name to include when using --prompts_txt (e.g., Clip1). Can be repeated.",
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
        help="Include prompt_en column in the output CSV for quick inspection.",
    )
    args = parser.parse_args()
    if os.path.abspath(args.full_info_json) == os.path.abspath(args.output_json):
        raise SystemExit("full_info_json and output_json must be different files.")

    if args.reuse_json:
        if not os.path.isfile(args.output_json):
            raise SystemExit(f"output_json not found: {args.output_json}")
        with open(args.output_json, "r", encoding="utf-8") as f:
            results_dict = json.load(f)
        video_paths = collect_video_paths_from_results(results_dict)
        if not video_paths:
            video_paths = find_videos(args.videos_dir)
    else:
        if args.use_full_info_json:
            if not os.path.isfile(args.full_info_json):
                raise SystemExit(f"full_info_json not found: {args.full_info_json}")
            with open(args.full_info_json, "r", encoding="utf-8") as f:
                full_info_list = json.load(f)
            video_paths = collect_video_paths_from_full_info(full_info_list)
            if not video_paths:
                raise SystemExit(f"No video paths found in: {args.full_info_json}")
            prompt_map = build_prompt_map_from_full_info(full_info_list)
        else:
            if args.prompts_txt:
                full_info_list = build_full_info_from_prompts(
                    args.prompts_txt, args.videos_dir, args.dimensions, args.clip
                )
                save_json(args.full_info_json, full_info_list)
                video_paths = collect_video_paths_from_full_info(full_info_list)
                prompt_map = build_prompt_map_from_full_info(full_info_list)
            else:
                video_paths = find_videos(args.videos_dir)
                if not video_paths:
                    raise SystemExit(f"No .mp4 files found under: {args.videos_dir}")

                full_info_list = build_full_info(video_paths, args.dimensions)
                save_json(args.full_info_json, full_info_list)
                prompt_map = {path: get_prompt_from_filename(path) for path in video_paths}
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

    score_by_video = {path: {} for path in video_paths}
    for dimension, result in results_dict.items():
        if not isinstance(result, (list, tuple)) or len(result) < 2:
            continue
        video_results = result[1]
        for item in video_results:
            video_path = item.get("video_path")
            score = item.get("video_results")
            if video_path in score_by_video:
                score_by_video[video_path][dimension] = score

    rows = []
    for video_path in video_paths:
        row = {"video_path": video_path}
        if args.include_prompt:
            row["prompt_en"] = prompt_map.get(video_path, "")
        for dimension in args.dimensions:
            row[dimension] = score_by_video[video_path].get(dimension, "")
        rows.append(row)

    fieldnames = ["video_path"]
    if args.include_prompt:
        fieldnames.append("prompt_en")
    fieldnames += args.dimensions
    write_csv(args.output_csv, rows, fieldnames)
    append_mean_row(args.output_csv)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()


# python scripts/batch_eval_custom_input.py \
#   --videos_dir videos/10_edit_videos/long \
#   --output_json evaluation_results/edit_contact_long.json \
#   --output_csv evaluation_results/edit_contact_long.csv


# python scripts/batch_eval_custom_input.py \
#   --reuse_json \
#   --output_json evaluation_results/long_results_10_examples.json \
#   --output_csv evaluation_results/long_results_10_examples.csv



# python scripts/batch_eval_custom_input.py \
#   --videos_dir videos \
#   --prompts_txt prompt_all_long.txt \
#   --dimensions overall_consistency \
#   --include_prompt \
#   --output_json evaluation_results/all_overall.json \
#   --output_csv evaluation_results/all_overall.csv \
#   --full_info_json evaluation_results/all_full_info.json

# 如果你只想先算 Clip1：
# python scripts/batch_eval_custom_input.py \
#   --videos_dir videos/clip10_results \
#   --prompts_txt prompt_all_long.txt \
#   --clip Clip10 \
#   --dimensions overall_consistency \
#   --include_prompt \
#   --output_json evaluation_results/clip10_overall.json \
#   --output_csv evaluation_results/clip10_overall.csv \
#   --full_info_json evaluation_results/clip10_full_info.json



# --prompts_txt 会自动生成 full_info_json（按 ClipX 目录匹配），然后直接跑评测。
# 不需要再单独执行 build_full_info_from_prompts.py。


# python scripts/batch_eval_custom_input.py \
#   --videos_dir videos \
#   --prompts_txt prompt_all_long.txt \
#   --dimensions overall_consistency \
#   --include_prompt \
#   --output_json evaluation_results/overall_10_clips.json \
#   --output_csv evaluation_results/overall_10_clips.csv \
#   --full_info_json evaluation_results/overall_10_clips_full.json

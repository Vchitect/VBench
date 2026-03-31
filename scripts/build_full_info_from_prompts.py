#!/usr/bin/env python3
import argparse
import json
import os
import re


CLIP_HEADER_RE = re.compile(r"^\s*Clip\s*(\d+)\s*[:：]\s*$", re.IGNORECASE)
CLIP_DIR_RE = re.compile(r"clip\s*(\d+)", re.IGNORECASE)
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


def index_videos_by_clip(videos_dir, clip_dirs):
    mapping = {}
    for clip_name, clip_dir in clip_dirs.items():
        for root, _, files in os.walk(clip_dir):
            for filename in files:
                if not filename.lower().endswith(".mp4"):
                    continue
                role = os.path.splitext(filename)[0]
                mapping.setdefault(clip_name, {})[role] = os.path.join(root, filename)
    return mapping


def build_full_info(clips, video_map, dimensions, clip_names=None):
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
                    "dimension": dimensions,
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
    parser = argparse.ArgumentParser(description="Build full_info JSON from prompt txt.")
    parser.add_argument("--prompts", required=True, help="Path to prompt_all_long.txt")
    parser.add_argument("--videos_dir", required=True, help="Root directory of mp4 videos.")
    parser.add_argument("--output", required=True, help="Output full_info JSON path.")
    parser.add_argument(
        "--clip",
        action="append",
        help="Clip name to include (e.g., Clip1). Can be repeated.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=["overall_consistency"],
        help="Dimensions to evaluate.",
    )
    args = parser.parse_args()

    clips = parse_prompts(args.prompts)
    if args.clip:
        for clip_name in args.clip:
            if clip_name not in clips:
                raise SystemExit(f"Clip not found in prompts: {clip_name}")
    clip_dirs = resolve_clip_dirs(clips, args.videos_dir, args.clip)
    video_map = index_videos_by_clip(args.videos_dir, clip_dirs)
    full_info_list = build_full_info(clips, video_map, args.dimensions, args.clip)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(full_info_list, f, ensure_ascii=True, indent=2)


if __name__ == "__main__":
    main()

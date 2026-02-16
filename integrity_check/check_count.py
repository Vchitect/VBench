import json
import os
import argparse
import sys

def check_num_videos(json_path, video_dir):
    """
    Checks if the total number of video files matches the expected count
    based on the input JSON filename.

    json file for i2v task is assumed to contain the string 'i2v'
    """
    # Logic preserved from original script:
    # distinct expected counts for i2v vs t2v based on filename
    if 'i2v' in os.path.basename(json_path).lower():
        expected_count = 5590
    else:
        expected_count = 4720

    print(f"--- Checking counts for: {os.path.basename(json_path)} ---")
    
    # Check if directory exists
    if not os.path.isdir(video_dir):
        print(f"! Error: Directory '{video_dir}' not found.")
        return False

    # Count files ending in .mp4
    files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]
    actual_count = len(files)

    # Compare and Report
    print(f"Expected videos: {expected_count}")
    print(f"Found videos:    {actual_count}")

    if actual_count == expected_count:
        print("OK! check_num_videos: SUCCESS. Count matches.")
        return True
    else:
        diff = expected_count - actual_count
        if diff > 0:
            print(f"! check_num_videos: FAIL. Missing {diff} videos.")
        else:
            print(f"! check_num_videos: FAIL. Found {abs(diff)} extra videos.")
        return False

def filter_missing_videos(json_path, video_dir, output_path):
    """
    Identifies specifically which entries from the JSON are missing
    corresponding video files in the directory.
    """
    if not os.path.exists(json_path):
        print(f"! Error: Input JSON '{json_path}' not found.")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"! Error: Failed to decode JSON from '{json_path}'.")
        return

    missing_entries = []
    print(f"\n--- Scanning {len(data)} entries against '{video_dir}' ---")

    for entry in data:
        # Adjust key if your json uses something other than 'image_name'
        # Fallback to 'prompt' or other keys if necessary, but using original logic:
        image_name = entry.get('image_name', '') 
        
        if not image_name:
            continue

        # Logic: Convert 'xxx.jpg' -> 'xxx' -> 'xxx-0.mp4'
        base_name = os.path.splitext(image_name)[0]
        expected_video_name = f"{base_name}-0.mp4"
        expected_video_path = os.path.join(video_dir, expected_video_name)
        
        # If video file is NOT found, add the whole entry to our list
        if not os.path.exists(expected_video_path):
            missing_entries.append(entry)

    # Save the missing entries to a new file
    if missing_entries:
        with open(output_path, 'w') as f:
            json.dump(missing_entries, f, indent=4)
        
        print(f"! Found {len(missing_entries)} missing videos.")
        print(f"Data for missing entries saved to: {output_path}")
    else:
        print("OK! All videos exist! No missing entries found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for missing sampled videos against a VBench JSON reference."
    )

    # Positional arguments
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the original JSON file (e.g., vbench2_i2v_full_info.json)"
    )
    parser.add_argument(
        "video_dir",
        type=str,
        help="Path to the folder containing sampled .mp4 files"
    )

    # Optional argument
    parser.add_argument(
        "--output",
        type=str,
        default="missing_entries.json",
        help="Path to save the resulting JSON of missing entries (default: missing_entries.json)"
    )

    args = parser.parse_args()

    # Run the checks
    check_num_videos(args.json_path, args.video_dir)
    filter_missing_videos(args.json_path, args.video_dir, args.output)
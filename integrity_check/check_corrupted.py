import subprocess
import os
import glob
import argparse
import sys
from tqdm import tqdm  # pip install tqdm

def check_integrity_ffmpeg(directory, extension="*.mp4"):
    # Normalize the extension path for glob
    search_pattern = os.path.join(directory, "**", extension)
    files = glob.glob(search_pattern, recursive=True)
    corrupt_files = []
    
    if not files:
        print(f"! No files found matching '{extension}' in '{directory}'")
        return []

    print(f"Scanning {len(files)} files using FFmpeg...")

    for video_path in tqdm(files):
        try:
            # Run ffmpeg, discard video output (-f null -), capture error logs
            # -v error: Only output fatal errors
            # -xerror: Stop on error
            cmd = [
                "ffmpeg", 
                "-v", "error", 
                "-i", video_path, 
                "-f", "null", 
                "-"
            ]
            
            # Run the command and capture output
            result = subprocess.run(
                cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.PIPE, 
                text=True
            )
            
            # If there is ANY output in stderr, the file has corruption
            if result.stderr:
                print(f"\n ! CORRUPT: {video_path}")
                # Print first line of error for context
                print(f"   Reason: {result.stderr.strip().splitlines()[0]}...") 
                corrupt_files.append(video_path)
                
        except Exception as e:
            print(f"! Script Error on {video_path}: {e}")

    return corrupt_files

if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(
        description="Recursively check integrity of video files using FFmpeg."
    )
    
    # Positional argument: The folder path
    parser.add_argument(
        "path", 
        type=str, 
        help="Path to the directory containing video files"
    )

    args = parser.parse_args()

    # Validate directory exists
    if not os.path.isdir(args.path):
        print(f"Error: The directory '{args.path}' does not exist.")
        sys.exit(1)

    # Run the check
    bad_ones = check_integrity_ffmpeg(args.path)
    
    print(f"\n--- Final Report ---")
    print(f"Found {len(bad_ones)} corrupt files.")
    
    if bad_ones:
        output_file = "corrupt_files_ffmpeg.txt"
        with open(output_file, "w") as f:
            f.write("\n".join(bad_ones))
        print(f"List saved to '{output_file}'")
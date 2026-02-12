import subprocess
import os
import glob
from tqdm import tqdm  # pip install tqdm (optional, for progress bar)

# This script checks if any of the sampled videos is corrupted
# Recommended step to run before starting evaluation


def check_integrity_ffmpeg(directory, extension="*.mp4"):
    files = glob.glob(os.path.join(directory, "**", extension), recursive=True)
    corrupt_files = []
    
    print(f"🔍 Strictly scanning {len(files)} files using FFmpeg...")

    # We use tqdm for a progress bar, if you don't have it, just iterate 'files'
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
                # Filter out minor warnings if necessary, but 'Invalid NAL' will show up here
                print(f"\n CORRUPT: {video_path}")
                print(f"   Reason: {result.stderr.strip().splitlines()[0]}...") # Print first line of error
                corrupt_files.append(video_path)
                
        except Exception as e:
            print(f"Script Error on {video_path}: {e}")

    return corrupt_files

if __name__ == "__main__":
    # Update this path to your video folder
    bad_ones = check_integrity_ffmpeg("/path/to/sampled/videos")
    
    print(f"\n--- Final Report ---")
    print(f"Found {len(bad_ones)} corrupt files.")
    
    if bad_ones:
        with open("corrupt_files_ffmpeg.txt", "w") as f:
            f.write("\n".join(bad_ones))
        print("List saved to 'corrupt_files_ffmpeg.txt'")
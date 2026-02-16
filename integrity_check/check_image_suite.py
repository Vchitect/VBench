import json
import os
import argparse
import sys

def filter_missing_image(json_path, image_dir, output_path):
    # 1. Validate inputs
    if not os.path.exists(json_path):
        print(f"! Error: Input JSON '{json_path}' not found.")
        sys.exit(1)

    if not os.path.isdir(image_dir):
        print(f"! Error: Image directory '{image_dir}' not found.")
        sys.exit(1)

    # 2. Load the original data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"! Error: Failed to decode JSON from '{json_path}'.")
        sys.exit(1)

    missing_entries = []
    print(f"Scanning {len(data)} entries against '{image_dir}'...")

    # 3. Check each entry
    for entry in data:
        # Adjust key if your json uses something other than 'image_name'
        image_name = entry.get('image_name', '') 
        
        if not image_name:
            continue

        # Construct full path
        expected_image_path = os.path.join(image_dir, image_name)
        
        # 4. If image file is NOT found, add the whole entry to our list
        if not os.path.exists(expected_image_path):
            # Optional: Print the specific missing file for debugging
            # print(f"Missing: {image_name}")
            missing_entries.append(entry)

    # 5. Save the missing entries to a new file
    if missing_entries:
        with open(output_path, 'w') as f:
            json.dump(missing_entries, f, indent=4)
        
        print(f"\n ! Found {len(missing_entries)} missing images.")
        print(f"Data for missing entries saved to: {output_path}")
    else:
        print("\nOK! All images exist! No missing entries found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check if the image suite is consistent with the VBench JSON."
    )

    # Positional arguments
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to the original JSON file (e.g., vbench2_i2v_full_info.json)"
    )
    parser.add_argument(
        "image_dir",
        type=str,
        help="Path to the directory containing the downloaded image suite"
    )

    # Optional argument
    parser.add_argument(
        "--output",
        type=str,
        default="inconsistent_image_suite.json",
        help="Path to save the missing entries JSON (default: inconsistent_image_suite.json)"
    )

    args = parser.parse_args()

    filter_missing_image(args.json_path, args.image_dir, args.output)
"""Post-process VBench eval_results.json into per-dimension + overall CSVs."""
import argparse
import csv
import glob
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_path", required=True)
    p.add_argument("--prefix", required=True, help="Filename prefix (e.g. folder name)")
    args = p.parse_args()

    jsons = sorted(glob.glob(os.path.join(args.output_path, "*_eval_results.json")))
    if not jsons:
        print(f"[write_csvs] No eval_results.json found in {args.output_path}")
        sys.exit(0)

    data = json.load(open(jsons[-1], encoding="utf-8"))
    os.makedirs(args.output_path, exist_ok=True)

    # Overall CSV
    overall_path = os.path.join(args.output_path, f"{args.prefix}_overall.csv")
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dimension", "overall_score"])
        for dim, (score, _) in data.items():
            w.writerow([dim, score])
    print(f"[write_csvs] {overall_path}")

    # Per-dimension CSVs
    for dim, (score, videos) in data.items():
        extra_keys = [k for k in (videos[0].keys() if videos else [])
                      if k not in ("video_path", "video_results")]
        dim_path = os.path.join(args.output_path, f"{args.prefix}_{dim}.csv")
        with open(dim_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_path", "video_results"] + extra_keys)
            for v in videos:
                w.writerow([v["video_path"], v["video_results"]] +
                           [v.get(k, "") for k in extra_keys])
        print(f"[write_csvs] {dim_path}")


if __name__ == "__main__":
    main()

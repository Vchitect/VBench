import argparse
import torch
import os, sys
from datetime import datetime

from competition_utils import transform_to_videos

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

from competitions import VBenchCompetition


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_path", 
        type=str, 
        required=False,
        help="folder that contains short videos or long videos"
    )
    parser.add_argument(
        "--frame_rate", 
        type=int, 
        required=False,
        help="frame rate of generated videos"
    )
    parser.add_argument(
        "--video_path", 
        type=str, 
        required=False,
        help="folder that contains the sampled videos"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./evaluate_results",
        help="output path that save evaluation results"
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        default="./short_prompt_list_test2.txt",
        help="Specify the path of the file that contains prompt lists"
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f"args: {args}")
    
    
    if not args.video_path:
        ### transform png frames to mp4 video
        assert args.submission_path is not None and args.frame_rate is not None, "You need to provide the submission_path\
            and the frame rate for generating the video."
        args.video_path = os.path.join(args.output_path, "evaluated_videos")
        transform_to_videos(args.submission_path, args.video_path, args.frame_rate)
    
    device = torch.device("cuda")
    myvbench = VBenchCompetition(device, None, args.output_path)
    
    print(f'start evaluation')
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    
    kwargs = {
        'imaging_quality_preprocessing_mode': 'longer'
    }
    

    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f.readlines()]
    
    myvbench.evaluate(
        videos_path = args.video_path,
        name = f'results_{current_time}',
        prompt_list=prompts,
        dimension_list = args.dimension,
        **kwargs
    )
    print("done")
    

if __name__ == "__main__":
    main()

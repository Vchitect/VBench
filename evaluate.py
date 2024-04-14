import torch
import os
from vbench import VBench
from datetime import datetime

import json
import argparse

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBench')
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/vbench/VBench_full_info.json',
        help="path to save the json file that contains the prompt and dimension information",
    )
    parser.add_argument(
        "--videos_path",
        type=str,
        required=True,
        help="folder that contains the sampled videos",
    )
    parser.add_argument(
        "--dimension",
        nargs='+',
        required=True,
        help="list of evaluation dimensions, usage: --dimension <dim_1> <dim_2>",
    )
    parser.add_argument(
        "--load_ckpt_from_local",
        type=bool,
        required=False,
        help="whether load checkpoints from local default paths (assuming you have downloaded the checkpoints locally",
    )
    parser.add_argument(
        "--read_frame",
        type=bool,
        required=False,
        help="whether directly read frames, or directly read videos",
    )
    parser.add_argument(
        "--custom_input",
        action="store_true",
        required=False,
        help="whether use custom input prompt or vbench prompt"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="""Specify the input prompt\n
        If not specified, filenames will be used as input prompts\n
        * Mutually exclusive to --prompt_file.\n
        ** This option must be used with --custom_input flag
        """
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=False,
        help="""Specify the path of the file that contains prompt lists\n
        If not specified, filenames will be used as input prompts\n
        * Mutually exclusive to --prompt.\n
        ** This option must be used with --custom_input flag
        """
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f'args: {args}')

    device = torch.device("cuda")
    my_VBench = VBench(device, args.full_json_dir, args.output_path)
    
    print(f'start evaluation')
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    prompt = []
    
    if (args.prompt_file is not None) and (args.prompt != ""):
        raise Exception("--prompt_file and --prompt cannot be used together")
    if (args.prompt_file is not None or args.prompt != "") and (not args.custom_input):
        raise Exception("must set --custom_input for using external prompt")

    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = json.load(f)
        assert type(prompt) == dict, "Invalid prompt file format. The correct format is {\"video_path\": prompt, ... }"
    elif args.prompt != "":
        prompt = [args.prompt]

    my_VBench.evaluate(
        videos_path = args.videos_path,
        name = f'results_{current_time}',
        prompt_list=prompt, # pass in [] to read prompt from filename
        dimension_list = args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        mode=args.custom_input,
    )
    print('done')


if __name__ == "__main__":
    main()

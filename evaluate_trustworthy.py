import torch
import os
from vbench2_beta_trustworthiness import VBenchTrustworthiness
from datetime import datetime

import argparse

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBench')
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_trustworthy_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/vbench2_beta_trustworthiness/vbench2_trustworthy.json',
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
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f'args: {args}')

    device = torch.device("cuda")
    my_VBench = VBenchTrustworthiness(device, args.full_json_dir, args.output_path)
    
    print(f'start evaluation')
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    my_VBench.evaluate(
        videos_path = args.videos_path,
        name = f'results_{current_time}',
        dimension_list = args.dimension,
        local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        custom_prompt=args.custom_input,
    )
    print('done')


if __name__ == "__main__":
    main()

import torch
import os
from vbench2_beta_i2v import VBenchI2V
from datetime import datetime

import argparse

def parse_args():

    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='VBenchI2V')
    parser.add_argument(
        "--output_path",
        type=str,
        default='./evaluation_i2v_results/',
        help="output path to save the evaluation results",
    )
    parser.add_argument(
        "--full_json_dir",
        type=str,
        default=f'{CUR_DIR}/vbench2_beta_i2v/vbench2_i2v_full_info.json',
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
        "--ratio",
        type=str,
        default=None,
        help="specify the target ratio",
    )
    parser.add_argument(
        "--imaging_quality_preprocessing_mode",
        type=str,
        required=False,
        default='longer',
        help="""This is for setting preprocessing in imaging_quality
        1. 'shorter': if the shorter side is more than 512, the image is resized so that the shorter side is 512.
        2. 'longer': if the longer side is more than 512, the image is resized so that the longer side is 512.
        3. 'shorter_centercrop': if the shorter side is more than 512, the image is resized so that the shorter side is 512. 
        Then the center 512 x 512 after resized is used for evaluation.
        4. 'None': no preprocessing
        """,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(f'args: {args}')
    
    kwargs = {
        'imaging_quality_preprocessing_mode': args.imaging_quality_preprocessing_mode
    }

    device = torch.device("cuda")
    my_VBench = VBenchI2V(device, args.full_json_dir, args.output_path)
    
    print(f'start evaluation')
    current_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

    my_VBench.evaluate(
        videos_path = args.videos_path,
        name = f'results_{current_time}',
        dimension_list = args.dimension,
        resolution = args.ratio,
        **kwargs
    )
    print('done')


if __name__ == "__main__":
    main()

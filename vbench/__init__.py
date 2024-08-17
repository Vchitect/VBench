import os

from .utils import get_prompt_from_filename, init_submodules, save_json, load_json
import importlib
from itertools import chain
from pathlib import Path

from .distributed import get_rank, print0


class VBench(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that VBench originally provides
        self.output_path = output_path              # output directory to save VBench results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        return ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]        

    def check_dimension_requires_extra_info(self, dimension_list):
        dim_custom_not_supported = set(dimension_list) & set([
            'object_class', 'multiple_objects', 'scene', 'appearance_style', 'color', 'spatial_relationship'
        ])

        assert len(dim_custom_not_supported) == 0, f"dimensions : {dim_custom_not_supported} not supported for custom input"


    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[], special_str='', verbose=False, mode='vbench_standard', **kwargs):
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions
        if mode=='custom_input':
            self.check_dimension_requires_extra_info(dimension_list)
            if os.path.isfile(videos_path):
                cur_full_info_list = [{"prompt_en": get_prompt_from_filename(videos_path), "dimension": dimension_list, "video_list": [videos_path]}]
                if len(prompt_list) == 1:
                    cur_full_info_list[0]["prompt_en"] = prompt_list[0]
            else:
                video_names = os.listdir(videos_path)

                cur_full_info_list = []

                for filename in video_names:
                    postfix = Path(os.path.join(videos_path, filename)).suffix
                    if postfix.lower() not in ['.mp4', '.gif', '.jpg', '.png']:
                        continue
                    cur_full_info_list.append({
                        "prompt_en": get_prompt_from_filename(filename), 
                        "dimension": dimension_list, 
                        "video_list": [os.path.join(videos_path, filename)]
                    })

                if len(prompt_list) > 0:
                    prompt_list = {os.path.join(videos_path, path): prompt_list[path] for path in prompt_list}
                    assert len(prompt_list) >= len(cur_full_info_list), """
                        Number of prompts should match with number of videos.\n
                        Got {len(prompt_list)=}, {len(cur_full_info_list)=}\n
                        To read the prompt from filename, delete --prompt_file and --prompt_list
                        """

                    all_video_path = [os.path.abspath(file) for file in list(chain.from_iterable(vid["video_list"] for vid in cur_full_info_list))]
                    backslash = "\n"
                    assert len(set(all_video_path) - set([os.path.abspath(path_key) for path_key in prompt_list])) == 0, f"""
                    The prompts for the following videos are not found in the prompt file: \n
                    {backslash.join(set(all_video_path) - set([os.path.abspath(path_key) for path_key in prompt_list]))}
                    """

                    video_map = {}
                    for prompt_key in prompt_list:
                        video_map[os.path.abspath(prompt_key)] = prompt_list[prompt_key]

                    for video_info in cur_full_info_list:
                        video_info["prompt_en"] = video_map[os.path.abspath(video_info["video_list"][0])]

        elif mode=='vbench_category':
            self.check_dimension_requires_extra_info(dimension_list)
            CUR_DIR = os.path.dirname(os.path.abspath(__file__))
            category_supported = [ Path(category).stem for category in os.listdir(f'prompts/prompts_per_category') ]# TODO: probably need refactoring again
            if 'category' not in kwargs:
                category = category_supported
            else:
                category = kwargs['category']

            assert category is not None, "Please specify the category to be evaluated with --category"
            assert category in category_supported, f'''
            The following category is not supported, {category}.
            '''

            video_names = os.listdir(videos_path)
            postfix = Path(video_names[0]).suffix

            with open(f'{CUR_DIR}/prompts_per_category/{category}.txt', 'r') as f:
                video_prompts = [line.strip() for line in f.readlines()]

            for prompt in video_prompts:
                video_list = []
                for filename in video_names:
                    if (not Path(filename).stem.startswith(prompt)):
                        continue
                    postfix = Path(os.path.join(videos_path, filename)).suffix
                    if postfix.lower() not in ['.mp4', '.gif', '.jpg', '.png']:
                        continue
                    video_list.append(os.path.join(videos_path, filename))

                cur_full_info_list.append({
                    "prompt_en": prompt, 
                    "dimension": dimension_list, 
                    "video_list": video_list 
                })

        else:
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)
            postfix = Path(video_names[0]).suffix
            for prompt_dict in full_info_list:
                # if the prompt belongs to any dimension we want to evaluate
                if set(dimension_list) & set(prompt_dict["dimension"]): 
                    prompt = prompt_dict['prompt_en']
                    prompt_dict['video_list'] = []
                    for i in range(5): # video index for the same prompt
                        intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'
                        if intended_video_name in video_names: # if the video exists
                            intended_video_path = os.path.join(videos_path, intended_video_name)
                            prompt_dict['video_list'].append(intended_video_path)
                            if verbose:
                                print0(f'Successfully found video: {intended_video_name}')
                        else:
                            print0(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
                    cur_full_info_list.append(prompt_dict)

        
        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print0(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path


    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='vbench_standard', **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'vbench.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print0(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        if get_rank() == 0:
            save_json(results_dict, output_name)
            print0(f'Evaluation results saved to {output_name}')

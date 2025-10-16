import os

from .utils import get_prompt_from_filename, init_submodules, save_json, load_json
import importlib
from itertools import chain
from pathlib import Path


class VBench2(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that VBench originally provides
        self.output_path = output_path              # output directory to save VBench results
        os.makedirs(self.output_path, exist_ok=True)

    def build_full_dimension_list(self, ):
        return ["Human_Anatomy", "Human_Identity", "Human_Clothes", "Diversity", "Composition", "Dynamic_Spatial_Relationship", 
                "Dynamic_Attribute", "Motion_Order_Understanding", "Human_Interaction", "Complex_Landscape", 'Complex_Plot', "Camera_Motion", 
                "Motion_Rationality", "Instance_Preservation", "Mechanics", "Thermotics", "Material", "Multi-View_Consistency"]        

    def check_dimension_requires_extra_info(self, dimension_list):
        dim_custom_not_supported = set(dimension_list) & set([
            'Composition', 'Dynamic_Attribute', 'Dynamic_Spatial_Relationship', 'Instance_Preservation', 'Complex_Plot', 'Complex_Landscape', 
            'Motion_Rationality', 'Motion_Order_Understanding', 'Mechanics', 'Thermotics', 'Material', "Camera_Motion", "Human_Interaction"
        ])

        assert len(dim_custom_not_supported) == 0, f"dimensions : {dim_custom_not_supported} not supported for custom input"

    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[], special_str='', verbose=False, mode='vbench_standard', **kwargs):
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions
        if mode=='custom_input':
            self.check_dimension_requires_extra_info(dimension_list)
            video_names = os.listdir(videos_path)
            assert len(video_names)>0, f"ERROR : The video files is empty"
            cur_full_info_list = []
            prompt_check_list = []
            for filename in video_names:
                postfix = Path(os.path.join(videos_path, filename)).suffix
                if postfix.lower() not in ['.mp4']:
                    continue
                if dimension_list[0]=='Diversity':
                    prompt_en = get_prompt_from_filename(filename)
                    if prompt_en in prompt_check_list:
                        continue
                    prompt_check_list.append(prompt_en)
                    item = {
                        "prompt_en": prompt_en, 
                        "dimension": dimension_list, 
                        "video_list": []
                    }
                    for ite in range(20):
                        item['video_list'].append(os.path.join(videos_path, f'{prompt_en}{special_str}-{str(ite)}{postfix}'))
                    cur_full_info_list.append(item)
                else:
                    cur_full_info_list.append({
                        "prompt_en": get_prompt_from_filename(filename), 
                        "dimension": dimension_list, 
                        "video_list": [os.path.join(videos_path, filename)]
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
                    if prompt_dict["dimension"][0]=='Diversity':
                        num=20
                    else:
                        num=3
                    for i in range(num): # video index for the same prompt
                        intended_video_name = f'{prompt[:180]}{special_str}-{str(i)}{postfix}'
                        if intended_video_name in video_names: # if the video exists
                            intended_video_path = os.path.join(videos_path, intended_video_name)
                            prompt_dict['video_list'].append(intended_video_path)
                            if verbose:
                                print(f'Successfully found video: {intended_video_name}')
                        else:
                            print(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
                            raise
                    cur_full_info_list.append(prompt_dict)

        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path


    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='vbench_standard', **kwargs):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)
        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        
        for dimension in dimension_list:
            try:
                if dimension=="Multi-View_Consistency":
                    dimension_change = "Multi_View_Consistency"
                else:
                    dimension_change = dimension
                dimension_module = importlib.import_module(f'vbench2.{dimension_change.lower()}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension_change.lower()}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')

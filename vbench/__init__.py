import os

from .utils import init_submodules, save_json, load_json
import importlib

class VBench(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that VBench originally provides
        self.output_path = output_path              # output directory to save VBench results
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=False)

    def build_full_dimension_list(self, ):
        return ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]        

    def build_full_info_json(self, videos_path, name, dimension_list, special_str=''):
        cur_full_info_list = load_json(self.full_info_dir)
        video_names = os.listdir(videos_path)
        postfix = '.'+ video_names[0].split('.')[-1]
        for prompt_dict in cur_full_info_list:
            prompt = prompt_dict['prompt_en']
            if f'{prompt}{special_str}-0{postfix}' in video_names:
                prompt_dict['video_list'] = [os.path.join(videos_path, f'{prompt}{special_str}-{str(i)}{postfix}') for i in range(5)]        
        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path

    def evaluate(self, videos_path, name, dimension_list=None, local=False, read_frame=False):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)
        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list)
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'vbench.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!')
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')

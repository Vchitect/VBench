import os
from .utils import init_submodules, save_json, load_json
from .aesthetic_quality import compute_aesthetic_quality
from .background_consistency import compute_background_consistency
from .subject_consistency import compute_subject_consistency
from .imaging_quality import compute_imaging_quality
from .object_class import compute_object_class
from .multiple_objects import compute_multiple_objects
from .color import compute_color
from .spatial_relationship import compute_spatial_relationship
from .scene import compute_scene
from .temporal_style import compute_temporal_style
from .overall_consistency import compute_overall_consistency
from .temporal_flickering import compute_temporal_flickering
from .motion_smoothness import compute_motion_smoothness
from .dynamic_degree import compute_dynamic_degree
from .human_action import compute_human_action
from .appearance_style import compute_appearance_style

class VBench(object):
    def __init__(self, device, full_info_dir, output_path):
        self.device = device                        # cuda or cpu
        self.full_info_dir = full_info_dir          # full json file that VBench originally provides
        self.output_path = output_path              # output directory to save VBench results
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=False)

    def build_full_dimension_list(self, ):
        return ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]        

    def build_full_info_json(self, videos_path, name, dimension_list):
        cur_full_info_list = load_json(self.full_info_dir)
        video_names = os.listdir(videos_path)
        postfix = '.'+ video_names[0].split('.')[-1]
        for prompt_dict in cur_full_info_list:
            prompt = prompt_dict['prompt_en']
            if prompt + '_0033-0'+postfix in video_names:
                prompt_dict['video_list'] = [os.path.join(videos_path, prompt+'_0033-'+str(i)+postfix) for i in range(5)]
        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        return cur_full_info_path

    def evaluate(self, videos_path, name, dimension_list=None, local=False):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local)
        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list)
        for dimension in dimension_list:
            try:
                evaluate_func = eval(f"compute_{dimension}")
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!')
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)

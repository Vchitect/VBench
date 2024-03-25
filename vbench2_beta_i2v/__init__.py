import os

from vbench2_beta_i2v.utils import init_submodules, save_json, load_json
from vbench import VBench
import importlib


class VBenchI2V(VBench):
    def build_full_dimension_list(self, ):
        return ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style", "i2v_subject", "i2v_background", "camera_motion"]     

    def evaluate(self, videos_path, name, dimension_list=None, local=False, read_frame=False, custom_prompt=False, resolution="1-1"):
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame, resolution=resolution)
        # print('BEFORE BUILDING')
        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, custom_prompt=custom_prompt)
        # print('AFTER BUILDING')
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'vbench2_beta_i2v.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete
            results = evaluate_func(cur_full_info_path, self.device, submodules_list)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')

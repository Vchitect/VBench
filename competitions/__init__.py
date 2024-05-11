import os
from vbench import VBench
from vbench.utils import init_submodules, save_json, get_prompt_from_filename
import importlib
from pathlib import Path
from itertools import chain


class VBenchCompetition(VBench):
    def __init__(self, device, full_info_dir, output_path):
        super().__init__(device, full_info_dir, output_path)
        self.dimension_map = {
            "temporal_quality": ["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree"],
            "frame_wise_quality": ["aesthetic_quality", "imaging_quality"],
            "text_alignment": ["overall_consistency", "clip_score"]
        }
    
    def build_full_dimension_list(self, ):
        return list(self.dimension_map.keys())       

    
    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[], **kwargs):
        cur_full_info_list=[] # to save the prompt and video path info for the current dimensions
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
            
            all_video_path = sorted(list(chain.from_iterable(vid["video_list"] for vid in cur_full_info_list)))
            assert len(all_video_path) == len(prompt_list), "the number of videos and prompts should be the same."
            
            video_map = dict(zip(all_video_path, prompt_list))

            for video_info in cur_full_info_list:
                video_info["prompt_en"] = video_map[video_info["video_list"][0]]

        
        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path
    
    
    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, **kwargs):
        results_dict = {}
        
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
            
        for dimension_key in dimension_list:
            dimension_l = self.dimension_map[dimension_key]
            submodules_dict = init_submodules(dimension_l, local=local, read_frame=read_frame)
            cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_l, prompt_list, **kwargs)
            
            dim_results = {}
            for dimension in dimension_l:
                try:
                    if dimension == "clip_score":
                        dimension_module = importlib.import_module(f'{dimension}')
                        submodules_list = []
                    else:
                        dimension_module = importlib.import_module(f'vbench.{dimension}')
                        submodules_list = submodules_dict[dimension]
                    evaluate_func = getattr(dimension_module, f'compute_{dimension}')
                except Exception as e:
                    raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')

                results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
                dim_results[dimension] = results

            if dimension_key == "temporal_quality":
                weighted_score = (1.0 * dim_results["subject_consistency"][0] + 1.0 * dim_results["background_consistency"][0] + 1.0 * dim_results["motion_smoothness"][0] + 0.5 * dim_results["dynamic_degree"][0]) / 3.5
            elif dimension_key == "frame_wise_quality":
                weighted_score = (1.0 * dim_results["aesthetic_quality"][0] + 1.0 * dim_results["imaging_quality"][0]) / 2.0
            elif dimension_key == "text_alignment":
                weighted_score = (1.0 * dim_results["overall_consistency"][0] + 1.0 * dim_results["clip_score"][0]) / 2.0
            
            results_dict[dimension_key] = [weighted_score, dim_results]
                
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)


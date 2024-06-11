import os
from vbench import VBench
from vbench.utils import init_submodules, save_json, get_prompt_from_filename, load_json
import importlib
from pathlib import Path
from itertools import chain

from vbench2_beta_long.utils import split_video_into_scenes, split_video_into_clips, load_clip_lengths, get_duration_from_json

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


    #### VBench Long
    def preprocess(self, videos_path, mode, threshold = 35.0, segment_length=16, duration=2, **kwargs):
        if "split_clip" in os.listdir(videos_path):
            print(f"Videos have been splitted into clips in {videos_path}/split_clip")
            return 

        split_scene_video_path = []
        if kwargs['use_semantic_splitting']:
            for video_file in os.listdir(videos_path):
                video_path = os.path.join(videos_path, video_file)
                if not video_path.endswith(('.mp4', '.avi', '.mov')):
                    continue
                
                video_name = os.path.splitext(video_file)[0]
                output_dir = os.path.join(videos_path, "split_scene", video_name)
                os.makedirs(output_dir, exist_ok=True)
                split_scene_flag = split_video_into_scenes(video_path, output_dir, threshold)
                if split_scene_flag:
                    split_scene_video_path.append(video_path)

        dimension_clip_length_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", kwargs['clip_length_config'])
        dimension_clip_length = load_clip_lengths(dimension_clip_length_config_path)

        base_output_dir = os.path.join(videos_path, "split_clip")
        os.makedirs(base_output_dir, exist_ok=True)


        for video_file in os.listdir(videos_path):
            video_path = os.path.join(videos_path, video_file)

            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue

            # duration = get_duration_from_json(video_path, full_info_list, dimension_clip_length)
            if mode == 'long_custom_input':
                duration = 2

            if video_path in split_scene_video_path:
                video_name = os.path.splitext(video_file)[0]
                video_scenes_path = os.path.join(os.path.dirname(video_path), "split_scene", video_name)
                for video_scene_path in os.listdir(video_scenes_path):
                    video_scene_path = os.path.join(video_scenes_path, video_scene_path)
                    split_video_into_clips(video_scene_path, base_output_dir, int(duration), fps=8)

            else:
                split_video_into_clips(video_path, base_output_dir, int(duration), fps=8)

        print(f"Splitting videos into clips in {base_output_dir}")


    def evaluate_long(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='long_custom_input', **kwargs):

        self.preprocess(videos_path, mode, **kwargs)

        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()

        for dimension_key in dimension_list:
            dimension_l = self.dimension_map[dimension_key]

            submodules_dict = init_submodules(dimension_l, local=local, read_frame=read_frame)

            cur_full_info_path = self.build_full_info_json_long(videos_path, name, dimension_l, prompt_list, **kwargs)
            
            dim_results = {}
            for dimension in dimension_l:
                try:
                    if dimension == "clip_score":
                        dimension_module = importlib.import_module(f'{dimension}')
                        submodules_list = []
                    else:
                        dimension_module = importlib.import_module(f'vbench2_beta_long.{dimension}')
                        submodules_list = submodules_dict[dimension]
                    evaluate_func = getattr(dimension_module, f'compute_long_{dimension}')
                except Exception as e:
                    raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
                
                print(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete

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
        print(f'Evaluation results saved to {output_name}')


    def build_full_info_json_long(self, videos_path, name, dimension_list, prompt_list=[], **kwargs):

        cur_full_info_dict = {} 

        splited_videos_path = os.path.join(videos_path, 'split_clip')
        
        for prompt_folder in os.listdir(splited_videos_path):
            prompt_folder_path = os.path.join(splited_videos_path, prompt_folder)
            if not os.path.isdir(prompt_folder_path):
                continue 

            base_prompt = prompt_folder.split('-Scene')[0]

            if base_prompt not in cur_full_info_dict:
                cur_full_info_dict[base_prompt] = {
                    "prompt_en": base_prompt,
                    "dimension": dimension_list,
                    "video_list": []
                }
            
            for video_file in os.listdir(prompt_folder_path):
                if video_file.endswith(('.mp4', '.avi', '.mov')):
                    
                    video_path = os.path.join(prompt_folder_path, video_file)
                    cur_full_info_dict[base_prompt]["video_list"].append(video_path)
        cur_full_info_list = list(cur_full_info_dict.values())

        if len(prompt_list) > 0:
            
            video_map = dict([(f"{k:04d}", v) for k, v in enumerate(prompt_list, 1)])
            
            for video_info in cur_full_info_list:
                video_info["prompt_en"] = video_map[video_info["prompt_en"].split("_")[0]]       

        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path
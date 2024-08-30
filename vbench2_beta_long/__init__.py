import os
import re
import importlib
from itertools import chain
from pathlib import Path
from vbench.utils import get_prompt_from_filename, init_submodules, save_json, load_json
from vbench2_beta_long.utils import split_video_into_scenes, split_video_into_clips, load_clip_lengths, get_duration_from_json
from vbench2_beta_long.temporal_flickering import filter_static_clips
from vbench import VBench




class VBenchLong(VBench):
    def build_full_dimension_list(self, ):
        return ["subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality", "object_class", "multiple_objects", "color", "spatial_relationship", "scene", "temporal_style", 'overall_consistency', "human_action", "temporal_flickering", "motion_smoothness", "dynamic_degree", "appearance_style"]

    def preprocess(self, videos_path, mode, threshold = 35.0, segment_length=16, duration=2, **kwargs):
        # static_filter_flag = (mode == 'long_vbench_standard' and (videos_path.split('/')[-1] == 'temporal_flickering' or 'temporal_flickering' in kwargs['preprocess_dimension_flag']))
        # static_filter_flag = kwargs['static_filter_flag']
        if "split_clip" in os.listdir(videos_path):
            # Get all folder names in the split_clip folder
            split_clip_path=os.path.join(videos_path,"split_clip")
            split_clip_folders_count = len([folder for folder in os.listdir(split_clip_path) if re.search(r'-\d+$', folder)])
            
            # Get the number of files in the videos_path folder that end with '.mp4'
            mp4_files_count = len([file for file in os.listdir(videos_path) if file.endswith('.mp4')])
            
            # Check if the number of folders matches the number of .mp4 files
            if split_clip_folders_count == mp4_files_count:
                print(f"Videos have been splitted into clips in {videos_path}/split_clip")
                return 

        # detect transistions
        split_scene_video_path = []
        if kwargs['use_semantic_splitting']:
            for video_file in os.listdir(videos_path):
                video_path = os.path.join(videos_path, video_file)
                if not video_path.endswith(('.mp4', '.avi', '.mov')):
                    continue
                
                # semantically consistent scenes splitting
                video_name = os.path.splitext(video_file)[0]
                output_dir = os.path.join(videos_path, "split_scene", video_name)
                os.makedirs(output_dir, exist_ok=True)
                split_scene_flag = split_video_into_scenes(video_path, output_dir, threshold)
                if split_scene_flag:
                    split_scene_video_path.append(video_path)

        full_info_list = load_json(self.full_info_dir)
        dimension_clip_length_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", kwargs['clip_length_config'])
        dimension_clip_length = load_clip_lengths(dimension_clip_length_config_path)

        # split video into clips
        base_output_dir = os.path.join(videos_path, "split_clip")
        os.makedirs(base_output_dir, exist_ok=True)

        for video_file in os.listdir(videos_path):
            video_path = os.path.join(videos_path, video_file)

            if not video_path.endswith(('.mp4', '.avi', '.mov')):
                continue

            duration = get_duration_from_json(video_path, full_info_list, dimension_clip_length)
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

        # finally, got floders under videos_path, which contain clips of each video
        print(f"Splitting videos into clips in {base_output_dir}")


    def evaluate(self, videos_path, name, prompt_list=[], dimension_list=None, local=False, read_frame=False, mode='vbench_standard', **kwargs):
        _dimensions = self.build_full_dimension_list()
        is_dimensional_structure = any(os.path.isdir(os.path.join(videos_path, dim)) for dim in _dimensions)
        kwargs['preprocess_dimension_flag'] = dimension_list
        if is_dimensional_structure:
            # 1. Under dimensions folders
            for dimension in _dimensions:
                dimension_path = os.path.join(videos_path, dimension)
                self.preprocess(dimension_path, mode, **kwargs)
        else:
            self.preprocess(videos_path, mode, **kwargs)

        # Now, long videos have been splitted into clips
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)
        # print('BEFORE BUILDING')
        # loop for build_full_info_json for clips

        cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)
        # print('AFTER BUILDING')
        for dimension in dimension_list:
            try:
                dimension_module = importlib.import_module(f'vbench2_beta_long.{dimension}')
                evaluate_func = getattr(dimension_module, f'compute_long_{dimension}')
            except Exception as e:
                raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}')
            submodules_list = submodules_dict[dimension]
            print(f'cur_full_info_path: {cur_full_info_path}') # TODO: to delete

            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
        output_name = os.path.join(self.output_path, name+'_eval_results.json')
        save_json(results_dict, output_name)
        print(f'Evaluation results saved to {output_name}')


    def build_full_info_json(self, videos_path, name, dimension_list, prompt_list=[], special_str='', verbose=False, mode='vbench_standard', **kwargs):
        cur_full_info_list=[]

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

        elif mode=='long_vbench_standard':
            # if kwargs['static_filter_flag'] and 'temporal_flickering' in dimension_list:
            #     videos_path = os.path.join(videos_path, 'temporal_filtered_cilps', 'filtered_videos')
            full_info_list = load_json(self.full_info_dir)
            video_names = os.listdir(videos_path)
            postfix = Path(video_names[0]).suffix
            video_clip_folder_names = [name.replace(postfix, '') for name in video_names]
            for prompt_dict in full_info_list:
                # if the prompt belongs to any dimension we want to evaluate
                if set(dimension_list) & set(prompt_dict["dimension"]):
                    prompt = prompt_dict['prompt_en']
                    prompt_dict['video_list'] = []
                    for i in range(kwargs['num_of_samples_per_prompt']): # video index for the same prompt
                        intended_video_name = f'{prompt}{special_str}-{str(i)}{postfix}'
                        intended_video_name_floder = f'{prompt}{special_str}-{str(i)}'
                        intended_video_clips_name_floder = os.path.join(videos_path, "split_clip", intended_video_name_floder)

                        if not os.path.exists(intended_video_clips_name_floder):
                            print(f'WARNING!!! This required video clips are not found! Missing benchmark videos can lead to unfair evaluation result. The missing video clips folder is: {intended_video_clips_name_floder}')
                            continue
                        for video_clip_name in os.listdir(intended_video_clips_name_floder):
                            if video_clip_name.split('_')[0] in video_clip_folder_names:
                                intended_video_path = os.path.join(intended_video_clips_name_floder, video_clip_name)
                                prompt_dict['video_list'].append(intended_video_path)
                            if verbose:
                                print(f'Successfully found video clips in : {intended_video_name_floder}')

                    cur_full_info_list.append(prompt_dict)
        elif mode=='long_custom_input':
            cur_full_info_dict = {} # to save the prompt and video path info for the current dimensions

            # get splitted video paths
            splited_videos_path = os.path.join(videos_path, 'split_clip')

            
            for prompt_folder in os.listdir(splited_videos_path):
                prompt_folder_path = os.path.join(splited_videos_path, prompt_folder)
                if not os.path.isdir(prompt_folder_path):
                    continue  # Skip if it's not a directory
                

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
                                print(f'Successfully found video: {intended_video_name}')
                        else:
                            print(f'WARNING!!! This required video is not found! Missing benchmark videos can lead to unfair evaluation result. The missing video is: {intended_video_name}')
                    cur_full_info_list.append(prompt_dict)


        cur_full_info_path = os.path.join(self.output_path, name+'_full_info.json')
        save_json(cur_full_info_list, cur_full_info_path)
        print(f'Evaluation meta data saved to {cur_full_info_path}')
        return cur_full_info_path

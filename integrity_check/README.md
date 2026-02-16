# :bookmark_tabs: Integrity Check

We provide useful scripts that users may find useful to facilitate the process of sampling videos and perfoming evaluation.

1. `check_corrupted.py` :  

   - Usage:

     - ```bash
       cd integrity_check
       python check_corrupted.py path/to/sampled_videos_folder
       ```

   - Descriptions:

     - A utility script that scans a directory of video files using FFmpeg to identify and report any corrupted or unreadable files.
     - May want to perform this check before starting the evaluation

   

2. `check_count.py`

   - Usage:

     - ```bash
       cd integrity_check
       python check_count.py path/to/vbench_full_info.json path/to/sampled_videos_folder
       ```

   - Descriptions:

     - A utility script to check if all the videos needed for VBench evaluation exist in the directory of your sampled videos
     - May want to perform this check before starting the evaluation
     - Assume user uses the default `VBench_full_info.json` for T2V and  `vbench2_i2v_full_info.json` for I2V

     

3. `check_image_suite.py` (for I2V)

   - Usage:

     - ```
       cd integrity_check
       python check_image_suite.py path/to/vbench2_i2v_full_info.json path/to/image_prompt_folder
       ```

   - Descriptions:

     - This script check if all the image prompt needed when sampling videos for VBench-I2V evaluation exit
     - May want to perform this check before starting the sampling (for I2V)

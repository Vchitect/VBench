# Competitions

More details will be announced at the official competition site.

## Short Videos

Sample videos from `competitions/short_prompt_list.txt`, and the sampled videos will be automatically evaluated in terms of 3 aspects:
| Evaluation Aspects | Automatic Evaluation | Human Evaluation |
| ----- | -----| -----|
|`temporal_quality`| - VBench dimensions: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, | Temporal consistency between frames, motion quality |
|`frame_wise_quality`| - VBench dimensions: `aesthetic_quality`, `imaging_quality`, | The quality of individual video frames |
|`text_alignment`| - VBench dimension: `overall_consistency`. - CLIP Score | Alignment between generated videos and text prompts |


TODO
- [ ] Add sampling and formatting requirements
- [ ] Add video (FPS, resolution, duration) requirements
- [ ] Add evaluation pipeline for all dimensions

## Long Videos

Sample videos from `competitions/long_prompt_list.json`

TODO
- [ ] Add sampling and formatting requirements
- [ ] Add video (FPS, resolution, duration) requirements
- [ ] Add evaluation pipeline for long videos

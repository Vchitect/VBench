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


We provide 20 stories for long video generation. Take one story for example:
```
"summary": "A panda wakes up in the morning and goes to school.",
"storyline": [
    "The panda stirs awake, nestled in a mound of blankets.",
    "The panda brushes its teeth diligently, a minty freshness lingering in the air.",
    "Sunlight bathes the breakfast table as the panda enjoys bamboo shoots and bread.",
    "Laden with its backpack, the panda heads out the door eagerly.",
    "Snowflakes dance around the panda as it walks to school.",
    "Inside the classroom, the panda listens attentively to the teacher."
]
```
- The `summary` is an overall description of the story.
- The `storyline` is a detailed and step-by-step description of the story.

There are several ways to generate a long video, depending on the capability of your model.
1. Take the `summary` as input text prompt, and generate a long video directly. This is useful when your text encoder or video generation has limited capacity in understanding long input text prompts.
2. Take the `storyline` list as input in a step-by-step manner, sample each item in the list separately, and concatenate your sampled videos into a long video. This might produce better results compared to directly using `summary`. This is useful when your model has limited capacity in handling long prompts, and your model is targeted for short video generation instead of long video generation.
3. Take the `storyline` list as input in one go, concatenate the list of sentences in the storyline to produce a long sentence, and feed this long sentence into your model to generate a long video in one go. This is useful when you model has good long prompt understanding capabilities, and is able to directly generate long videos.


TODO
- [ ] Add sampling and formatting requirements
- [ ] Add video (FPS, resolution, duration) requirements
- [ ] Add evaluation pipeline for long videos
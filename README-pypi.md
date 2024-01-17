# for project description in pypi
## :hammer: Installation
#### Install with pip
   ```
   pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
   pip install git+https://github.com/Vchitect/VBench.git
   ```

#### Install with git clone
    git clone https://github.com/Vchitect/VBench.git
    pip install -r VBench/requirements.txt
    pip install VBench
    
If there is an error during [detectron2](https://github.com/facebookresearch/detectron2) installation, see [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Usage
##### command line 
```bash
    vbench evaluate --videos_path $VIDEO_PATH --dimension $DIMENSION
```
For example:
```bash
    vbench evaluate --videos_path "sampled_videos/lavie/human_action" --dimension "human_action"
```
##### python
```python
    from vbench import VBench
    my_VBench = VBench(device, <path/to/VBench_full_info.json>, <path/to/save/dir>)
    my_VBench.evaluate(
        videos_path = <video_path>,
        name = <name>,
        dimension_list = [<dimension>, <dimension>, ...],
    )
```
For example: 
```python
    from vbench import VBench
    my_VBench = VBench(device, "VBench_full_info.json", "evaluation_results")
    my_VBench.evaluate(
        videos_path = "sampled_videos/lavie/human_action",
        name = "lavie_human_action",
        dimension_list = ["human_action"],
    )
```


## :gem: Pre-Trained Models
[Optional] Please download the pre-trained weights according to the guidance in the `model_path.txt` file for each model in the `pretrained` folder to `~/.cache/vbench`.

## :bookmark_tabs: Prompt Suite

We provide prompt lists are at `prompts/`. 

Check out [details of prompt suites](https://github.com/Vchitect/VBench/tree/master/prompts), and instructions for [**how to sample videos for evaluation**](https://github.com/Vchitect/VBench/tree/master/prompts).

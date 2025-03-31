# Sora Generates Videos with Stunning Geometrical Consistency
Generates Videos  Geometrical Consistency metric toolbox (GVGC metric)
##
The recently developed Sora model [1] has exhibited remarkable capabilities in video generation, sparking intense discussions regarding its ability to simulate real-world phenomena. Despite its growing popularity, there is a lack of established metrics to quantitatively evaluate its fidelity to real-world physics. In this paper, we introduce a new benchmark that assesses the quality of the generated videos based on their adherence to real-world physics principles. We employ a method that transforms the generated videos to 3D models, leveraging the premise that the accuracy of 3D reconstruction is heavily contingent on the video quality. From the perspective of 3D reconstruction, we use the fidelity of the geometric constraints satisfied by the constructed 3D models as a proxy to gauge the extent to which the generated videos conform to the rules of real-world physics.
## Homepase
https://sora-geometrical-consistency.github.io/
## Fast forward
<div align="left">
 <img src="img/teaser.png" width="80%">
</div>

<div align="left">
 <img src="img/fidelitymetric.png" width="80%">
</div>

<div align="left">
 <img src="img/stereomatching.png" width="80%">
</div>

<div align="left">
 <img src="img/matching.png" width="80%">
</div>


<div align="left">
 <img src="img/recon.png" width="80%">
</div>


## Data
### data download link:
https://drive.google.com/file/d/1E_7DR_DIvvWtDXn5KXUwXfBIA_3MhBMG/view?usp=drive_link
place the data file as fllows:
```python
root
|
---xxx.py
---XXX.py
---data
   |
   ---sora
   |
   ---gen2
   |
   ---pika
```
## Code
### code usage
```python
python Eval_all.py
```
## Software
### software whl download link:
https://drive.google.com/file/d/1scoJ-mLoZ_3ZkrALQfdwwnw-0Q30NB5e/view?usp=drive_link
### install:
```python
pip install GVGC-0.0.1-py3-none-any.whl
```

### software usage:
```python
#!/usr/bin/env python3
from AutoExtraFrame import AutoExtraFrame
from PatchAutoEvaluate import PatchAutoEvaluate
from PatchDenseMatch import PatchDenseMatch
from PatchDrawMatch import PatchDrawMatch

video_list = ["sora", "pika", "gen2"]
AutoExtraFrame(video_list)
PatchAutoEvaluate(video_list)
PatchDenseMatch(video_list)
PatchDrawMatch(video_list)
```
## Result
```python
```python
root
|
---xxx.py
---XXX.py
---data
   |
   ---sora
   |
   ---------brief_result
   ---------full_result
   ---------image_result
   |
   ---gen2
   |
   ---------brief_result
   ---------full_result
   ---------image_result
   |
   ---pika
   |
   ---------brief_result
   ---------full_result
   ---------image_result
```
```
## Media report
用 Sora 影片建 3D！ https://www.youtube.com/watch?v=X6n5ZCc7yy0 


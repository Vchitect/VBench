# FAQ

**Q: How can I join VBench Leaderboard?**<br>
A: There are 3 options to join the leaderboard:<br>
Option | Sampling Party | Evaluation Party |              Comments                         |
| :---: | :---: |  :---: |        :--------------    | 
| 1️⃣ | VBench Team | VBench Team | VBench Team handles everything — ideal for both open-source and closed-source models (if API access is provided). We periodically allocate resources to sample newly released models and perform evaluations. You can request us to perform sampling and evaluation, but the progress depends on our available resources. |
| 2️⃣ | Your Team | VBench Team | Submit your video samples via this [Google Form](https://forms.gle/Dy26sRobB6vouQZC7). If you prefer to evaluate on your own, refer to Option 3. |
| 3️⃣ | Your Team | Your Team | Already performed the full VBench evaluation? Submit your `eval_results.zip` files to the [VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)'s `[T2V]Submit here!` form. The evaluation results will be automatically updated to the leaderboard. Make sure to also fill in model details in the `[T2V]Submit here!` form. Submissions lacking essential information may be removed periodically. |

🔍 Transparency Note: Each leaderboard entry clearly states who performed the sampling and evaluation.



**Q: For option 2️⃣ and 3️⃣ - What if my model has been updated and now performs better? Can I make a new submission to the VBench Leaderboard?**<br>
A: Yes, you can submit an updated version of your model. Please clearly specify what changes have been made and how this version differs from your previous submission. Updates will only be evaluated if the modifications are explicitly described.

**Q: For option 2️⃣ - After sharing a cloud storage link to our sampled videos, how long will it take to receive the evaluation results?**<br>
A: Once we receive both your cloud storage link and the corresponding model information, evaluation typically takes 1–2 weeks, depending on the current queue.

**Q: For option 1️⃣ and 2️⃣ - Is the evaluation fully automated?**<br>
A: Yes. The evaluation strictly follows the open-source procedure available in our official code repository. It is fully reproducible, with no manual intervention or subjective judgment involved.

**Q: For option 2️⃣ - If the evaluation results are unsatisfactory, can we resubmit new videos for the same model?**<br>
A: Evaluation results are final and tied to the specific model version submitted. However, you may submit a newer checkpoint or updated version of the model for a fresh evaluation. In such cases, please include a brief explanation of the updates. Previous results will remain on the leaderboard and will not be removed.

**Q: For option 2️⃣ - how should I organize the sampled videos for submission?**<br>
A: You can place all sampled videos in a single folder. The filenames should follow the format shown in [this example list](https://github.com/Vchitect/VBench/blob/master/sampled_videos/sampled_videos.txt).


**Q: What’s the difference between VBench and VBench-Long?**<br>
A: Use [**VBench**](https://github.com/Vchitect/VBench?tab=readme-ov-file#usage) for evaluating videos shorter than 5.0 seconds (< 5.0s).
Use [**VBench-Long**](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_long) for evaluating videos 5.0 seconds or longer (≥ 5.0s).
Each benchmark is optimized for its respective video length to ensure fair and consistent evaluation.

**Q: Does VBench support running on Huawei Ascend NPU?**<br>
A: Yes. [Ascend Extension for PyTorch](https://github.com/Ascend/pytorch) develops `torch_npu` to adapt Ascend NPU to PyTorch so that developers who use the PyTorch can obtain powerful compute capabilities of Ascend AI Processors. Please install `torch` and `torch_npu` before installing vbench if you are using Ascend NPU devices. For more details, please refer to the official [installation guide](https://github.com/Ascend/pytorch#installation).

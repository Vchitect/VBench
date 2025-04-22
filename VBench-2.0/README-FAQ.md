# FAQ

**Q: How should I sample the videos using VBench-2.0 prompt suite?**<br>
A: We offer two ways: sampling based on per-dimension prompt or directly sampling by the full text list. The detailed instruction in [How to sample](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts#evaluate-all-dimensions)

**Q: If the name of the prompt exceeds the saving limit, what should I do?**<br>
A: In VBench-2.0, some prompt will exceed the saving limit, we will save the first 180 characters. The detailed instruction in [How to sample](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts#evaluate-all-dimensions)

**Q: How to name the videos if using augmented prompts or chinese prompts (whatever by myself or VBench-2.0)?**<br>
A: Note that the naming must follow the instruction of [How to sample](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts#evaluate-all-dimensions). We also provide our video name list and you can check the correctness [sample_videos.txt](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/sampled_videos/sampled_videos.txt) 

**Q: How can I join VBench-2.0 Leaderboard?**<br>
A: There are 3 options to join the leaderboard:<br>
Option | Sampling Party | Evaluation Party |              Comments                         |
| :---: | :---: |  :---: |        :--------------    | 
| 1️⃣ | VBench Team | VBench Team | VBench Team handles everything — ideal for both open-source and closed-source models (if API access is provided). We periodically allocate resources to sample newly released models and perform evaluations. You can request us to perform sampling and evaluation, but the progress depends on our available resources. |
| 2️⃣ | Your Team | VBench Team | Submit your video samples via this [Google Form](https://forms.gle/rjH6hmAHpZhRGdkv5). If you prefer to evaluate on your own, refer to Option 3. |
| 3️⃣ | Your Team | Your Team | Already performed the full VBench-2.0 evaluation? Submit your `eval_results.zip` files to the [VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)'s `VBench-2.0 Submit here!` form. The evaluation results will be automatically updated to the leaderboard. Make sure to also fill in model details in the `VBench-2.0 Submit here!` form. Submissions lacking essential information may be removed periodically. |

🔍 Transparency Note: Each leaderboard entry clearly states who performed the sampling and evaluation.



**Q: For option 2️⃣ and 3️⃣ - What if my model has been updated and now performs better? Can I make a new submission to the VBench-2.0 Leaderboard?**<br>
A: Yes, you can submit an updated version of your model. Please clearly specify what changes have been made and how this version differs from your previous submission. Updates will only be evaluated if the modifications are explicitly described.

**Q: For option 2️⃣ - After sharing a cloud storage link to our sampled videos, how long will it take to receive the evaluation results?**<br>
A: Once we receive both your cloud storage link and the corresponding model information, evaluation typically takes 1–2 weeks, depending on the current queue.

**Q: For option 1️⃣ and 2️⃣ - Is the evaluation fully automated?**<br>
A: Yes. The evaluation strictly follows the open-source procedure available in our official code repository. It is fully reproducible, with no manual intervention or subjective judgment involved.

**Q: For option 2️⃣ - If the evaluation results are unsatisfactory, can we resubmit new videos for the same model?**<br>
A: Evaluation results are final and tied to the specific model version submitted. However, you may submit a newer checkpoint or updated version of the model for a fresh evaluation. In such cases, please include a brief explanation of the updates. Previous results will remain on the leaderboard and will not be removed.

**Q: For option 2️⃣ - how should I organize the sampled videos for submission?**<br>
A: You can place all sampled videos in a single folder. The filenames should follow the format shown in [this example list](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/sampled_videos/sampled_videos.txt).

**Q: There seem to be some unreasonable parts in the prompt?**<br>
A1: `A person is doing xxx, suddenly they start to do xxx.` The `they` here is a singular, gender-neutral pronoun, which aims to avoid specifying the gender.

A2: `Aerial view, aerial view. Aerial view, aerial view. One blue balls and one red balls are on the wooden table and collide horizontally, bird's-eye view.` in `Instance Preservation`. The repetition of `aerial view` here is intentional to ensure that some models can successfully generate a top-down perspective.
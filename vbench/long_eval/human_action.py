
from vbench.core.human_action import compute_human_action
from vbench.long_eval.utils import reorganize_clips_results


def compute_long_human_action(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_human_action(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

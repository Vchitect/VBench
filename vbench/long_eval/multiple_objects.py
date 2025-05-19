from vbench.core.multiple_objects import compute_multiple_objects
from vbench.long_eval.utils import reorganize_clips_results


def compute_long_multiple_objects(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_multiple_objects(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

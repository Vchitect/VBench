
from vbench.object_class import compute_object_class
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_object_class(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_object_class(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

from vbench.dynamic_degree import compute_dynamic_degree
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_dynamic_degree(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_dynamic_degree(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)
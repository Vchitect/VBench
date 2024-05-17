
from vbench.color import compute_color
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_color(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_color(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)
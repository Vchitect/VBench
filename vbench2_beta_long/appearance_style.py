
from vbench.appearance_style import compute_appearance_style
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_appearance_style(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_appearance_style(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)
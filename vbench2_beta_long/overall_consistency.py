from vbench.overall_consistency import compute_overall_consistency
from vbench2_beta_long.utils import reorganize_clips_results

def compute_long_overall_consistency(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_overall_consistency(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

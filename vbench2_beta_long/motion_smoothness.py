from vbench.motion_smoothness import compute_motion_smoothness
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_motion_smoothness(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_motion_smoothness(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)
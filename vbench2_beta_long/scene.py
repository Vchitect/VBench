
from vbench.scene import compute_scene
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_scene(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_scene(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

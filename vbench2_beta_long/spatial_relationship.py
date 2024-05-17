
from vbench.spatial_relationship import compute_spatial_relationship
from vbench2_beta_long.utils import reorganize_clips_results


def compute_long_spatial_relationship(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_spatial_relationship(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

from vbench.core.spatial_relationship import compute_spatial_relationship
from vbench.long_eval.utils import reorganize_clips_results


def compute_long_spatial_relationship(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_spatial_relationship(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

from vbench.core.imaging_quality import compute_imaging_quality
from vbench.long_eval.utils import reorganize_clips_results


def compute_long_imaging_quality(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_imaging_quality(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results, dimension="imaging_quality")

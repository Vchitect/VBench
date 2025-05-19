
from vbench.core.temporal_style import compute_temporal_style
from vbench.long_eval.utils import reorganize_clips_results


def compute_long_temporal_style(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_temporal_style(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

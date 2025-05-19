import os


from collections import defaultdict

from vbench.core.aesthetic_quality import compute_aesthetic_quality
from vbench.utils import save_json, load_json
from vbench.long_eval.utils import reorganize_clips_results


def compute_long_aesthetic_quality(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_aesthetic_quality(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)

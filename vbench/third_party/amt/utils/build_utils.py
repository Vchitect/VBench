import importlib
import os
import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CUR_DIR, "../"))


def base_build_fn(module, cls, params):
    return getattr(importlib.import_module(
                    module, package=None), cls)(**params)


def build_from_cfg(config):
    module, cls = config['name'].rsplit(".", 1)
    params = config.get('params', {})
    return base_build_fn(module, cls, params)

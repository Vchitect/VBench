#  hack_registry.py
import logging

from mmengine.registry import Registry
from mmengine.logging import print_log
from typing import Type, Optional, Union, List


def _register_module(self,
                     module: Type,
                     module_name: Optional[Union[str, List[str]]] = None,
                     force: bool = False) -> None:
    """Register a module.

    Args:
        module (type): Module to be registered. Typically a class or a
            function, but generally all ``Callable`` are acceptable.
        module_name (str or list of str, optional): The module name to be
            registered. If not specified, the class name will be used.
            Defaults to None.
        force (bool): Whether to override an existing class with the same
            name. Defaults to False.
    """
    if not callable(module):
        raise TypeError(f'module must be Callable, but got {type(module)}')

    if module_name is None:
        module_name = module.__name__
    if isinstance(module_name, str):
        module_name = [module_name]
    for name in module_name:
        if not force and name in self._module_dict:
            existed_module = self.module_dict[name]
            # raise KeyError(f'{name} is already registered in {self.name} '
            #                f'at {existed_module.__module__}')
            print_log(
                f'{name} is already registered in {self.name} '
                f'at {existed_module.__module__}. Registration ignored.',
                logger='current',
                level=logging.INFO
            )
        self._module_dict[name] = module


Registry._register_module = _register_module
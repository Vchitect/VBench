"""CPU-only regression tests for the optional, single-NPU entry point.

Hardware and heavyweight import boundaries are stubbed; no NPU computation is simulated.
Run with: python -m unittest discover -s tests -v
"""

import argparse
import ast
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import shlex
import subprocess
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def launcher(cuda=False, npu=True, package=True, cuda_version="11.8"):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    torch.npu = types.SimpleNamespace(is_available=lambda: npu, set_device=mock.Mock())
    torch.device = mock.Mock(side_effect=lambda value: ("device", value))
    torch.version = types.SimpleNamespace(cuda=cuda_version)
    vbench = types.ModuleType("vbench")
    vbench.VBench = mock.Mock()
    distributed = types.ModuleType("vbench.distributed")
    distributed.dist_init = mock.Mock()
    distributed.print0 = mock.Mock()
    modules = {"torch": torch, "vbench": vbench, "vbench.distributed": distributed,
               "torch_npu": types.ModuleType("torch_npu") if package else None}
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
        yield load_module("vbench/launch/evaluate.py", "ascend_launch_test"), torch, distributed


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.cli = load_module("vbench/cli/evaluate.py", "ascend_cli_test")
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        self.cli.register_subparsers(subparsers)
        self.parser = parser
        self.evaluate_parser = subparsers.choices["evaluate"]

    def npu_args(self, *extra):
        self.assertIn("--device", self.evaluate_parser.format_help())
        return self.parser.parse_args(["evaluate", "--videos_path", "videos with spaces",
                                       "--dimension", "human_action", "--device", "npu:0", *extra])

    def test_cuda_is_still_the_default(self):
        args = self.parser.parse_args(["evaluate", "--videos_path", "videos",
                                       "--dimension", "human_action"])
        self.assertEqual(getattr(args, "device", None), "cuda")

    def test_npu_device_and_video_path_are_forwarded(self):
        args = self.npu_args()
        with mock.patch.object(self.cli.subprocess, "run") as run:
            self.cli.evaluate(args)
        command = shlex.split(run.call_args.args[0])
        self.assertEqual(command[command.index("--device") + 1], "npu:0")
        self.assertIn("--videos_path=videos with spaces", command)
        self.assertEqual(command[command.index("--nproc_per_node") + 1], "1")

    def test_multiple_npus_fail_before_spawning_a_child(self):
        args = self.npu_args("--ngpus", "2")
        with mock.patch.object(self.cli.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "single NPU"):
                self.cli.evaluate(args)
        run.assert_not_called()

    def test_child_failure_reaches_the_calling_process(self):
        args = self.parser.parse_args(["evaluate", "--videos_path", "videos",
                                       "--dimension", "human_action"])
        command = shlex.join([sys.executable, "-c", "raise SystemExit(7)"])
        with mock.patch.object(self.cli, "stringify_cmd", return_value=command):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                self.cli.evaluate(args)
        self.assertEqual(error.exception.returncode, 7)


class TestDeviceSelection(unittest.TestCase):
    def initialize(self, module, device="npu:0", dimensions=None):
        self.assertTrue(hasattr(module, "initialize_device"), "Add an explicit device initializer")
        return module.initialize_device(device, dimensions or ["human_action"])

    def test_cuda_keeps_existing_distributed_initialization(self):
        with launcher(cuda=True, package=False) as (module, torch, distributed):
            self.assertEqual(self.initialize(module, "cuda"), ("device", "cuda"))
            distributed.dist_init.assert_called_once_with()
            torch.npu.set_device.assert_not_called()

    def test_npu_selects_device_without_nccl_initialization(self):
        with launcher() as (module, torch, distributed):
            self.assertEqual(self.initialize(module), ("device", "npu:0"))
            torch.npu.set_device.assert_called_once_with("npu:0")
            distributed.dist_init.assert_not_called()

    def test_unavailable_npu_fails_without_cuda_fallback(self):
        for hardware, error in (({"package": False}, ImportError), ({"npu": False}, RuntimeError)):
            with self.subTest(hardware=hardware), launcher(**hardware) as (module, torch, distributed):
                self.assertTrue(hasattr(module, "initialize_device"))
                with self.assertRaises(error):
                    module.initialize_device("npu:0", ["human_action"])
                distributed.dist_init.assert_not_called()
                torch.npu.set_device.assert_not_called()

    def test_distributed_npu_launch_is_rejected(self):
        with launcher() as (module, _, distributed), mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            self.assertTrue(hasattr(module, "initialize_device"))
            with self.assertRaisesRegex(RuntimeError, "single NPU"):
                module.initialize_device("npu:0", ["human_action"])
            distributed.dist_init.assert_not_called()

    def test_other_dimensions_are_not_silently_advertised_as_npu_supported(self):
        with launcher() as (module, _, distributed):
            self.assertTrue(hasattr(module, "initialize_device"))
            with self.assertRaisesRegex(ValueError, "human_action"):
                module.initialize_device("npu:0", ["human_action", "scene"])
            distributed.dist_init.assert_not_called()

    def test_main_passes_the_selected_device_to_the_existing_public_api(self):
        with launcher() as (module, _, distributed):
            self.assertTrue(hasattr(module, "initialize_device"))
            with mock.patch.object(sys, "argv", ["evaluate.py", "--videos_path", "videos",
                                                  "--dimension", "human_action", "--device", "npu:0"]):
                module.main()
            self.assertEqual(module.VBench.call_args.args[0], ("device", "npu:0"))
            module.VBench.return_value.evaluate.assert_called_once()
            distributed.dist_init.assert_not_called()


class TestInstallation(unittest.TestCase):
    def check_installation(self, **hardware):
        tree = ast.parse((ROOT / "setup.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "check_torch_version")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"), namespace)
        with launcher(**hardware), redirect_stdout(io.StringIO()):
            namespace["check_torch_version"]()

    def test_npu_only_environment_can_install_without_cuda(self):
        try:
            self.check_installation(cuda=False, npu=True, cuda_version=None)
        except RuntimeError as error:
            self.fail(f"A supported NPU-only environment must pass the installation check: {error}")

    def test_existing_supported_cuda_installation_still_works_without_torch_npu(self):
        self.check_installation(cuda=True, package=False, cuda_version="11.8")
        self.check_installation(cuda=True, package=False, cuda_version="12.1")

    def test_existing_cuda_version_check_is_not_removed(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported CUDA version"):
            self.check_installation(cuda=True, cuda_version="12.8")


if __name__ == "__main__":
    unittest.main()

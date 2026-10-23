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


class Device:
    def __init__(self, value):
        self.type, _, index = str(value).partition(":")
        self.index = int(index) if index else None

    def __str__(self):
        return self.type if self.index is None else f"{self.type}:{self.index}"


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
    torch.device = Device
    torch.version = types.SimpleNamespace(cuda=cuda_version)
    vbench = types.ModuleType("vbench")
    vbench.VBench = mock.Mock()
    distributed = types.ModuleType("vbench.distributed")
    distributed.dist_init = mock.Mock()
    distributed.get_world_size = mock.Mock(return_value=1)
    distributed.print0 = mock.Mock()
    modules = {"torch": torch, "vbench": vbench, "vbench.distributed": distributed,
               "torch_npu": types.ModuleType("torch_npu") if package else None}
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
        assert (ROOT / "vbench/device.py").is_file(), "The entry points need a shared device helper"
        sys.modules["vbench.device"] = load_module("vbench/device.py", "vbench.device")
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

    def test_multiple_dimensions_are_forwarded_individually(self):
        self.assertEqual(next(action for action in self.evaluate_parser._actions
                              if action.dest == "dimension").nargs, "+")
        args = self.parser.parse_args(["evaluate", "--videos_path", "videos", "--device", "npu:0",
                                       "--dimension", "scene", "human_action"])
        with mock.patch.object(self.cli.subprocess, "run") as run:
            self.cli.evaluate(args)
        command = shlex.split(run.call_args.args[0])
        offset = command.index("--dimension")
        self.assertEqual(command[offset + 1:offset + 3], ["scene", "human_action"])

    def test_child_failure_reaches_the_calling_process(self):
        args = self.parser.parse_args(["evaluate", "--videos_path", "videos",
                                       "--dimension", "human_action"])
        command = shlex.join([sys.executable, "-c", "raise SystemExit(7)"])
        with mock.patch.object(self.cli, "stringify_cmd", return_value=command):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                self.cli.evaluate(args)
        self.assertEqual(error.exception.returncode, 7)

    def test_legacy_quoted_dimensions_still_reach_the_launcher(self):
        args = self.parser.parse_args(["evaluate", "--videos_path", "videos",
                                       "--dimension", "scene human_action"])
        with mock.patch.object(self.cli.subprocess, "run") as run:
            self.cli.evaluate(args)
        command = shlex.split(run.call_args.args[0])
        offset = command.index("--dimension")
        self.assertEqual(command[offset + 1:offset + 3], ["scene", "human_action"])


class TestDeviceSelection(unittest.TestCase):
    def initialize(self, module, device="npu:0"):
        self.assertTrue(hasattr(module, "initialize_device"), "Add an explicit device initializer")
        return module.initialize_device(device)

    def test_cuda_keeps_existing_distributed_initialization(self):
        with launcher(cuda=True, package=False) as (module, torch, distributed):
            self.assertEqual(str(self.initialize(module, "cuda")), "cuda")
            distributed.dist_init.assert_called_once_with()
            torch.npu.set_device.assert_not_called()

    def test_npu_selects_device_without_nccl_initialization(self):
        with launcher() as (module, torch, distributed):
            self.assertEqual(str(self.initialize(module)), "npu:0")
            self.assertEqual(str(torch.npu.set_device.call_args.args[0]), "npu:0")
            distributed.dist_init.assert_not_called()

    def test_unavailable_npu_fails_without_cuda_fallback(self):
        for hardware, error in (({"package": False}, ImportError), ({"npu": False}, RuntimeError)):
            with self.subTest(hardware=hardware), launcher(**hardware) as (module, torch, distributed):
                self.assertTrue(hasattr(module, "initialize_device"))
                with self.assertRaises(error):
                    module.initialize_device("npu:0")
                distributed.dist_init.assert_not_called()
                torch.npu.set_device.assert_not_called()

    def test_distributed_npu_launch_is_rejected(self):
        with launcher() as (module, _, distributed), mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            self.assertTrue(hasattr(module, "initialize_device"))
            with self.assertRaisesRegex(RuntimeError, "single NPU"):
                module.initialize_device("npu:0")
            distributed.dist_init.assert_not_called()

    def test_main_passes_the_selected_device_to_the_existing_public_api(self):
        with launcher() as (module, _, distributed):
            for entry in (module, load_module("evaluate.py", "root_evaluate_test")):
                for dimension in ("human_action", "scene", "object_class"):
                    with self.subTest(entry=entry.__name__, dimension=dimension), mock.patch.object(
                            sys, "argv", ["evaluate.py", "--videos_path", "videos",
                                          "--dimension", dimension, "--device", "npu:0"]):
                        entry.VBench.reset_mock()
                        entry.main()
                        self.assertEqual(str(entry.VBench.call_args.args[0]), "npu:0")
                        self.assertEqual(entry.VBench.return_value.evaluate.call_args.kwargs["dimension_list"],
                                         [dimension])
            distributed.dist_init.assert_not_called()

    def test_api_resolves_npu_without_starting_a_process_group(self):
        import tempfile

        with launcher() as (_, torch, distributed), tempfile.TemporaryDirectory() as directory:
            tree = ast.parse((ROOT / "vbench/__init__.py").read_text())
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VBench")
            constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
            namespace = {"os": os, "resolve_device": sys.modules["vbench.device"].resolve_device}
            exec(compile(ast.Module(body=[constructor], type_ignores=[]), "vbench/__init__.py", "exec"), namespace)
            instance = types.SimpleNamespace()
            namespace["__init__"](instance, "npu:0", "info.json", directory)
            self.assertIsInstance(instance.device, Device)
            self.assertEqual(str(instance.device), "npu:0")
            torch.npu.set_device.assert_called_once()
            distributed.dist_init.assert_not_called()

    def test_api_keeps_cpu_and_rejects_an_existing_multi_process_group(self):
        with launcher() as (_, torch, distributed):
            resolve = sys.modules["vbench.device"].resolve_device
            self.assertEqual(str(resolve("cpu")), "cpu")
            torch.npu.set_device.assert_not_called()
            distributed.get_world_size.return_value = 2
            with self.assertRaisesRegex(RuntimeError, "single NPU"):
                resolve("npu:0")


class TestEvaluatorDeviceHandling(unittest.TestCase):
    def function(self, path, name, namespace):
        tree = ast.parse((ROOT / path).read_text())
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), namespace)
        return namespace[name]

    def test_grit_config_receives_the_actual_device(self):
        get_parser = self.function("vbench/third_party/grit_src/image_dense_captions.py", "get_parser",
                                   {"CACHE_DIR": "cache", "CUR_DIR": "grit"})
        for name in ("cpu", "cuda", "npu:0"):
            with self.subTest(device=name):
                opts = get_parser(Device(name), "model.pth")["opts"]
                self.assertIn("MODEL.DEVICE", opts)
                self.assertEqual(opts[opts.index("MODEL.DEVICE") + 1], name)

    def test_motion_memory_uses_the_selected_accelerator(self):
        torch = types.SimpleNamespace(device=Device, tensor=mock.Mock(),
                                      cuda=types.SimpleNamespace(get_device_properties=mock.Mock()),
                                      npu=types.SimpleNamespace(get_device_properties=mock.Mock()))
        torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3
        torch.npu.get_device_properties.return_value.total_memory = 32 * 1024**3
        initialize = self.function("vbench/motion_smoothness.py", "initialization",
                                   {"torch": torch, "FrameProcess": mock.Mock()})
        with redirect_stdout(io.StringIO()):
            for name, memory in (("npu:0", 32 * 1024**3), ("cuda", 8 * 1024**3), ("cpu", 1)):
                instance = types.SimpleNamespace(device=Device(name))
                initialize(instance)
                self.assertEqual(instance.vram_avail, memory)

    def test_npu_detector_errors_are_not_converted_to_zero_scores(self):
        from contextlib import nullcontext

        detect = self.function("vbench/object_class.py", "get_dect_from_grit",
                               {"torch": types.SimpleNamespace(device=Device, no_grad=nullcontext)})
        model = types.SimpleNamespace(device=Device("npu:0"),
                                      run_caption_tensor=mock.Mock(side_effect=RuntimeError("unsupported op")))
        with self.assertRaisesRegex(RuntimeError, "unsupported op"):
            detect(model, [object()])
        model.device = Device("cuda")
        self.assertEqual(detect(model, [object()]), [set()])

    def test_npu_detector_can_legitimately_return_no_objects(self):
        from contextlib import nullcontext

        detect = self.function("vbench/object_class.py", "get_dect_from_grit",
                               {"torch": types.SimpleNamespace(device=Device, no_grad=nullcontext)})
        model = types.SimpleNamespace(device=Device("npu:0"),
                                      run_caption_tensor=lambda frame: ([], None))
        try:
            result = detect(model, [object()])
        except IndexError:
            self.fail("An empty detection result must remain a valid zero score")
        self.assertEqual(result, [set()])


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

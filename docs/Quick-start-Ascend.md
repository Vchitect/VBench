# Quick start: human_action on Ascend NPU

Run the original VBench `human_action` example on one Ascend NPU, using either
the installed CLI or the public Python API. The UMT model, 16-frame sampling,
preprocessing, scoring and JSON result format are unchanged.

This guide uses five original LaVie videos for **A person is strumming guitar**.
It demonstrates the evaluation flow; it does not reproduce a full benchmark or
leaderboard score. Other dimensions, multiple NPUs, VBench++ and VBench-2.0 are
outside the scope of this support.

## Prerequisites

- An Ascend NPU with its driver installed; `npu-smi info` must work.
- Linux aarch64, Python 3.12 and a working CANN environment. In a container,
  configure the NPU devices and driver mounts before running the example.
- Compatible `torch`, `torchvision` and `torch_npu` already installed. Follow
  the [Ascend PyTorch installation instructions](https://gitcode.com/Ascend/pytorch)
  to select a combination compatible with your CANN version. Do not install a
  CUDA wheel into the NPU environment.
- Access to Python package indexes, GitHub, the official video dataset and the
  [official UMT checkpoint](https://huggingface.co/OpenGVLab/VBench_Used_Models).
  The checkpoint is about 600 MB; allow at least 10 GB of writable disk space.

Reference environment:

| Component | Version |
| --- | --- |
| NPU | Ascend 910B4, one device |
| OS / Python / CANN | Ubuntu 22.04 aarch64 / 3.12 / 9.1.0 |
| torch / torch_npu | 2.9.0 CPU build / 2.9.0.post2 |
| torchvision | 0.24.0 CPU build |
| timm / NumPy | 0.9.12 / 1.26.4 |
| Decord | 0.6.0, CPU decoder |

CPU Decord and torchvision handle input preprocessing; UMT inference runs on
the NPU. `torch_npu` is optional for CUDA users and is not installed automatically.

## Check the environment

Activate your prepared Python environment, then load CANN and select one NPU:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
python - <<'PY'
import torch
import torch_npu

assert torch.npu.is_available(), "Ascend NPU is not available"
assert torch.npu.device_count() == 1, "Select one visible NPU for this example"
print("NPU:", torch.npu.get_device_name(0))
PY
```

Choose a different physical device in `ASCEND_RT_VISIBLE_DEVICES` if needed.
`npu:0` below refers to the first device visible to this process.

## Install VBench from source

Clone the source revision containing this support and work from its root:

```bash
git clone https://github.com/Vchitect/VBench.git
cd VBench
python -m pip install setuptools wheel "numpy==1.26.4"
```

On Linux aarch64, Decord 0.6.0 needs a CPU source build. Install the system
dependencies as root (or use `sudo` for the two `apt-get` commands):

```bash
apt-get update
apt-get install -y build-essential cmake pkg-config git wget ffmpeg \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev
git clone --recursive --depth 1 --branch v0.6.0 https://github.com/dmlc/decord.git decord-src
cmake -S decord-src -B decord-src/build -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build decord-src/build --parallel 4
python -m pip install --no-build-isolation --no-deps ./decord-src/python
```

Install VBench normally, using the repository's existing `requirements.txt`.
Keep the prepared NPU PyTorch stack and the reference preprocessing versions:

```bash
python -m pip install --no-build-isolation -e . \
    "timm==0.9.12" "numpy==1.26.4" "opencv-python==4.11.0.86"
python -c "import vbench.human_action; print('human_action import: OK')"
```

No separate dependency profile is required. The supported NPU evaluation path
in this guide remains limited to single-device `human_action`.

## Prepare the five official videos

Request access to the [official sampled-video dataset](https://huggingface.co/datasets/Vchitect/VBench_sampled_video)
with your own Hugging Face account. After access is granted, download the five
unmodified files, retaining their names and directory structure:

```bash
python -m pip install huggingface_hub
hf auth login
for index in 0 1 2 3 4; do
    hf download Vchitect/VBench_sampled_video \
        "lavie/human_action/A person is strumming guitar-${index}.mp4" \
        --repo-type dataset \
        --revision 8c89bb0218a12323d04821d11ec03c780d0dfb9c \
        --local-dir sampled_videos
done
```

The videos should now be in `sampled_videos/lavie/human_action/`. Use a matching
subset of the original prompt metadata so this small example does not expect
all videos from the full benchmark:

```bash
python - <<'PY'
import json
from pathlib import Path

prompt = "A person is strumming guitar"
info = json.loads(Path("vbench/VBench_full_info.json").read_text())
subset = [item for item in info
          if item["prompt_en"] == prompt and "human_action" in item["dimension"]]
assert len(subset) == 1
Path("human_action_full_info.json").write_text(json.dumps(subset, indent=2))
PY
```

VBench downloads the official UMT checkpoint on the first evaluation if it is
not already available. No alternative weights, generated test videos or modified
scoring code are used.

## Evaluate with the CLI

Use the installed `vbench evaluate` entry point for NPU evaluation, not the
legacy CUDA-only `python evaluate.py` entry point:

```bash
vbench evaluate \
    --device npu:0 \
    --ngpus 1 \
    --dimension human_action \
    --videos_path sampled_videos/lavie/human_action \
    --full_json_dir human_action_full_info.json \
    --output_path evaluation_results/npu-cli
```

An explicit NPU request fails if `torch_npu` or the device is unavailable; it
does not fall back to CPU or CUDA. The CLI rejects multi-NPU launches and other
dimensions for this NPU path. Existing CUDA commands keep their default device
and distributed initialization.

## Evaluate with the Python API

The public API already accepts a PyTorch device. Import `torch_npu` before
creating the NPU device, then use the same input files as above:

```python
import torch
import torch_npu
from vbench import VBench

benchmark = VBench(
    torch.device("npu:0"),
    "human_action_full_info.json",
    "evaluation_results/npu-api",
)
benchmark.evaluate(
    videos_path="sampled_videos/lavie/human_action",
    name="lavie_human_action",
    dimension_list=["human_action"],
)
```

CLI results are written to `evaluation_results/npu-cli/*_eval_results.json`.
The API example writes `evaluation_results/npu-api/lavie_human_action_eval_results.json`.
Each result contains the aggregate `human_action` score and per-video records.
A zero score is valid; completing the evaluation does not imply the generated
videos perform the requested action correctly.

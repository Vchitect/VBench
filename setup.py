#!/usr/bin/env python

from setuptools import find_packages, setup
import os

def check_torch_version():
    use_cuda = False

    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("\033[91mCUDA is not available. Please install a CUDA 11- or 12.1-enabled PyTorch version.\033[0m")

        cuda_version = torch.version.cuda
        print(cuda_version)
        if cuda_version not in ["11.6", "11.7", "11.8", "12.1"]:
            raise RuntimeError(f"\033[91mUnsupported CUDA version: {cuda_version}. Please install PyTorch with 11.6<=CUDA<=12.1.\033[0m")
        
        use_cuda = True
    except:
        print("\033[93mPlease install torch and torchvision compiled with cuda 11.8 before installing vbench\033[0m")
        print("\033[93mFor CUDA 11.8, run:\033[0m")
        print("\033[93m    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\033[0m")
        print("\033[93mFor CUDA 12.1, run:\033[0m")
        print("\033[93m    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\033[0m")
        print("\033[93mFor CUDA 11 with PyTorch < 2.0 run:\033[0m")
        print("\033[93m    pip install torch==1.13.1 torchvision==0.14.1\033[0m")
        print("\033[93mTry to use Ascend NPU.\033[0m")

    if use_cuda:
        return
            
    try:
        import torch
        import torch_npu
        
        if not torch.npu.is_available():
            raise RuntimeError("\033[91mAscend NPU is not available.\033[0m")
        
        print("\033[92mUsing Ascend NPU to accelerate the process.\033[0m")
    except:
        print("\033[93mPlease install torch torch_npu and torchvision before installing vbench if you are using Ascend NPU devices.\033[0m")
        print("\033[93mFor more details, please refer to the official installation guide: https://github.com/Ascend/pytorch#installation\033[0m")


def fetch_readme():
    with open('README-pypi.md', encoding='utf-8') as f:
        text = f.read()
    return text

def fetch_requirements():
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirements.txt')
    with open(filename, 'r') as f:
        envs = [line.rstrip('\n') for line in f.readlines() if '@' not in line]
    return envs

install_requires = fetch_requirements()
check_torch_version()
setup(name='vbench',
      version='0.1.5',
      description='Video generation benchmark',
      long_description=fetch_readme(),
      long_description_content_type='text/markdown',
      project_urls={
          'Source': 'https://github.com/Vchitect/VBench',
      },
      entry_points={
          'console_scripts': ['vbench=vbench.cli.vbench:main']
      },
      install_requires=install_requires,
      packages=find_packages(),
      include_package_data=True,
      license='Apache Software License 2.0',
)

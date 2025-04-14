#!/usr/bin/env python

from setuptools import find_packages, setup
import os

def check_torch_version():
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("\033[91mCUDA is not available. Please install a CUDA 11.8-enabled PyTorch version.\033[0m")

        cuda_version = torch.version.cuda
        print(cuda_version)
        if cuda_version not in ["11.8"]:
            raise RuntimeError(f"\033[91mUnsupported CUDA version: {cuda_version}. Please install PyTorch with CUDA 11.8.\033[0m")
    except:
        print("\033[93mPlease install torch and torchvision compiled with cuda 11.8 before installing vbench2\033[0m")
        print("\033[93mFor CUDA 11.8, run:\033[0m")
        print("\033[93m    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118\033[0m")
        raise

def fetch_readme():
    with open('README-pypi.md', encoding='utf-8') as f:
        text = f.read()
    return text

def fetch_requirements():
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'requirement.txt')
    with open(filename, 'r') as f:
        envs = [line.rstrip('\n') for line in f.readlines() if '@' not in line]
    return envs

install_requires = fetch_requirements()
check_torch_version()
setup(name='vbench2',
      version='0.1.1',
      description='Video generation benchmark',
      long_description=fetch_readme(),
      long_description_content_type='text/markdown',
      project_urls={
          'Source': 'https://github.com/Vchitect/VBench/tree/master/VBench-2.0',
      },
      entry_points={
          'console_scripts': ['vbench2=vbench2.cli.vbench2:main']
      },
      install_requires=install_requires,
      packages=find_packages(),
      include_package_data=True,
      license='Apache Software License 2.0',
)

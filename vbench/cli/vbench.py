import argparse
import importlib
from vbench import VBench

def main():
    parser = argparse.ArgumentParser(prog="vbench")
    subparsers = parser.add_subparsers(title='vbench subcommands')

    vbench_cmd = ['evaluate', 'static_filter']
    for cmd in vbench_cmd:
        module = importlib.import_module(f'vbench.cli.{cmd}')
        module.register_subparsers(subparsers)
    args = parser.parse_args()
    args.func(args)


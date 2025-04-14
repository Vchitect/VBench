import argparse
import importlib
import subprocess

vbench2_cmd = ['evaluate']

def main():
    parser = argparse.ArgumentParser(prog="vbench2", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(title='vbench2 subcommands')

    for cmd in vbench2_cmd:
        module = importlib.import_module(f'vbench2.cli.{cmd}')
        module.register_subparsers(subparsers)
    parser.set_defaults(func=help)
    args = parser.parse_args()
    args.func(args)

def help(args):
    subprocess.run(['vbench2', '-h'], check=True)

#!/bin/python3
from time import strftime
from omegaconf import OmegaConf
from argparse import ArgumentParser
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
import gin
from condor.condor_parameters import CondorParameters
from condor import gin_helpers
import subprocess

parser = ArgumentParser()
parser.add_argument('config', type=str)
parser.add_argument('name', type=str, default='', nargs='?')
args = parser.parse_args()
config = args.config

config_path = Path(config)

# Support both gin (*.gin) and OmegaConf YAML (*.yml / *.yaml) configs.
if config_path.suffix in {'.yml', '.yaml'}:
    # OmegaConf-style YAML config
    cfg = OmegaConf.load(config)
    condor_cfg = cfg.get('CondorParameters')
    if condor_cfg is None:
        raise ValueError(f"YAML config '{config}' is missing a 'CondorParameters' section.")
    condor_dict = OmegaConf.to_container(condor_cfg, resolve=True)
    condor_parameters = CondorParameters(**condor_dict)
else:
    # Default: gin-config file
    gin.parse_config_files_and_bindings(
        config_files=[config],
        bindings=None,
        skip_unknown=True,
        print_includes_and_imports=True,
    )
    condor_parameters = CondorParameters()

cmd = condor_parameters.cmd
cmd = cmd.replace('__CONFIG__', config)
environment = dict(os.environ)
environment.update(condor_parameters.environment)
print('CMD:',  cmd)
completed_process = subprocess.run(cmd, shell=True, check=False, executable='/bin/bash', env=environment)
sys.exit(completed_process.returncode)
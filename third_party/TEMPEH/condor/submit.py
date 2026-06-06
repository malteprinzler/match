#!/bin/python3
import time
from omegaconf import OmegaConf
from argparse import ArgumentParser
import htcondor
import os
from datetime import datetime
from time import strftime
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

OmegaConf.register_new_resolver("eval", eval)


def submit_job(config, config_path):
    name = config.condor.jobname
    submit_conf = config.condor.submit
    submit_conf["arguments"] = submit_conf["arguments"].replace("__CONFIG__", str(config_path))
    submit_conf["priority"] = args.bid - 1000
    OmegaConf.resolve(submit_conf)
    njobs = submit_conf.pop("njobs", 1)

    # creating experiment directory
    code_backup_root = config.condor.get("code_backup_root", None)
    if code_backup_root is None:
        exp_path = os.getcwd()
    else:
        exp_dirname = datetime.now().strftime("%y_%m_%d-%H_%M_%S-") + name
        exp_path = os.path.join(code_backup_root, exp_dirname)
        os.makedirs(exp_path)
        os.system(f"cp -r . {exp_path}")
        os.chdir(exp_path)

    if "log_root" in submit_conf:
        Path(submit_conf.log_root).parent.mkdir(exist_ok=True, parents=True)

    time.sleep(1.1)  # needed more than 1. so that exp_dirnames of multiconfigs have different directory name

    # Submitting job
    job = htcondor.Submit(submit_conf)
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job, count=njobs)
    print(job)
    print(f"Submitted {njobs} Job(s) with ID {submit_result.cluster()} {name}.\n\n")


parser = ArgumentParser()
parser.add_argument("bid", type=int)
parser.add_argument("config", type=str)
args = parser.parse_args()
conf = OmegaConf.load(args.config)

with open(args.config, "r") as f:
    if f.readline().strip().startswith("# MULTICONFIG"):
        is_multiconfig = True
    else:
        is_multiconfig = False

if is_multiconfig:
    global_variables = conf.MULTICONFIG_CONFIG.get('GLOBAL_VARIABLES', {})
    conf.GLOBAL=global_variables
    multiconfig_save_path = conf.MULTICONFIG_CONFIG.get('MULTICONFIG_SAVE_PATH', None)
    if multiconfig_save_path is not None:
        Path(multiconfig_save_path).parent.mkdir(parents=True, exist_ok=True)
        os.system(f'cp {args.config} {multiconfig_save_path}')

    parent_cfg = conf.PARENT_CONFIG
    child_cfgs = conf.CHILD_CONFIGS
    n_processes = len(child_cfgs)
    time_str = strftime("%Y%m%d_%H%M%S")
    tmp_cfg_stem = Path(conf.MULTICONFIG_CONFIG.TMP_CONFIG_STEM)
    for i, child_cfg in enumerate(child_cfgs):
        child_cfg = OmegaConf.merge(parent_cfg, child_cfg)
        child_cfg.GLOBAL = global_variables
        OmegaConf.resolve(child_cfg)
        child_cfg.pop('GLOBAL', None)
        config_path = tmp_cfg_stem.parent / f"{time_str}-{tmp_cfg_stem.name}-{i:03d}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(child_cfg, config_path)
        submit_job(child_cfg, config_path)
else:
    submit_job(conf, args.config)



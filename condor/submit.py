#!/bin/python3
import time
from argparse import ArgumentParser
import htcondor
import os
from datetime import datetime
from time import strftime
from pathlib import Path
import sys
from omegaconf import OmegaConf
sys.path.append(str(Path(__file__).resolve().parents[1]))
import gin
from condor.condor_parameters import CondorParameters
from condor import gin_helpers
import dataclasses
import shutil


def submit_job():
    parser = ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('bid', type=int)
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

    experiment_dir = os.path.join(condor_parameters.base_dir, condor_parameters.name)
    log_root = os.path.join(experiment_dir, '$(ClusterId)')
    Path(log_root).parent.mkdir(exist_ok=True, parents=True)

    # setting up string to define custom environment variables
    custom_environment_strs = list()
    for k, v in condor_parameters.environment.items():
        custom_environment_strs.append(f'{k}={v}')
    custom_environment_str = ' '.join(custom_environment_strs)


    submit_conf = dataclasses.asdict(condor_parameters)
    submit_conf['executable'] = 'condor/run.py'
    submit_conf['arguments'] = config + ' ' + condor_parameters.name
    submit_conf["priority"] = args.bid - 1000
    submit_conf["error"] = log_root + '_$(Process).out'
    submit_conf["output"] = log_root + '_$(Process).out'
    submit_conf["log"] = log_root + '_$(Process).log'
    submit_conf["exit"] = log_root + '_$(Process).exit'
    submit_conf['environment'] = f"\"CONDOR_ClusterId=$(ClusterId) CONDOR_Process=$(Process) CONDOR_WORLD_SIZE={condor_parameters.njobs} CONDOR_LOGFILE=$(log) CONDOR_OUTFILE=$(output) CONDOR_EXITFILE=$(exit) {custom_environment_str} \""
    submit_conf['on_exit_hold'] = '(ExitCode =?= 1) || (ExitCode =?= 3)'
    submit_conf['on_exit_hold_reason'] = 'ifThenElse(ExitCode =?= 3, "Planned Restart", ifThenElse( JobRunCount <= $(retries), "Failed, will resume", "Failed, but no more retries left"))'
    submit_conf['on_exit_hold_subcode'] = 'ifThenElse(JobRunCount <= $(retries) || (ExitCode =?= 3), 1, 2)'
    submit_conf['periodic_release'] = '( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 1) )'

    # explicitly include values that cannot be used by condor submit
    for k in ['backup_excludepatterns', 'backup_root', 'base_dir', 'cmd', 'name', 'njobs']:
        if k in submit_conf:
            submit_conf.pop(k)

    # creating experiment directory
    if condor_parameters.backup_root is None:
        code_path = os.getcwd()
    else:
        exp_dirname = datetime.now().strftime("%y_%m_%d-%H_%M_%S-") + condor_parameters.name
        code_path = os.path.join(condor_parameters.backup_root, exp_dirname)
        Path(code_path).parent.mkdir(exist_ok=True, parents=True)
        ignore_patterns = ['.git', '__pycache__'] + condor_parameters.backup_excludepatterns
        shutil.copytree(
            '.',
            code_path,
            ignore=shutil.ignore_patterns(*ignore_patterns),
            symlinks=True
        )
        # # saving symlink to code directory
        # if os.path.exists(f'{experiment_dir}/code'):
        #     os.system(f'rm {experiment_dir}/code')
        # os.system(f'ln -s {code_path} {experiment_dir}/code')
        os.chdir(code_path)


    time.sleep(1.1)  # needed more than 1. so that exp_dirnames of multiconfigs have different directory name

    # Submitting job
    job = htcondor.Submit(submit_conf)
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job, count=condor_parameters.njobs)
    print(job)
    print(f"Submitted {condor_parameters.njobs} Job(s) with ID {submit_result.cluster()} {condor_parameters.name}.\n\n")

if __name__ == '__main__':
    submit_job()

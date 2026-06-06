#!/bin/python3
from time import strftime
from omegaconf import OmegaConf
from argparse import ArgumentParser
import os
from pathlib import Path
from multiprocessing import Process
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

"""



> MULTICONFIG CONFIG FILE EXAMPLE:

# MULTICONFIG
MULTICONFIG_CONFIG:
  MULTICONFIG_SAVE_PATH: ${GLOBAL.outroot}/config.yaml
  TMP_CONFIG_STEM: /home/mprinzler/tmp/configs/2dprior_gha_from_camsweep-
  RUN_PARALLEL: True
  DISJOINT_GPUS: True
  GLOBAL_VARIABLES:
    inroot: /is/cluster/mprinzler/projects/DynJoker/Gaussian-Head-Avatar/data/DYNAMIC_DISTILLATION/003
    outrood: /is/cluster/mprinzler/projects/DynJoker/Gaussian-Head-Avatar/data/DYNAMIC_DISTILLATION/003


CHILD_CONFIGS:
  - input_dir: ${GLOBAL.inroot}/train/EVAL
    output_dir: ${GLOBAL.outroot}/train/EVAL
    image_pred_mode: gt

  - input_dir: ${GLOBAL.inroot}/train/CAM_SWEEP
    output_dir: ${GLOBAL.outroot}/train/CAM_SWEEP
    image_pred_mode: prior

PARENT_CONFIG:
  # general
  model_ckpt: assets/joker/pretrained/bfm_ft_NersembleCelebvtext_230000.bin
  input_dir: null
  output_dir: null
  image_pred_mode: null
  #fixing_jobs: [7]
  #fixing_jobs_ntotal: 20
  load_model: ${.model_ckpt}
  val_dataset:
    _target: joker.prior.data.dataset.CamSweepDataset
    _kwargs:
      root: ${...input_dir}
      nsamples: -1

  inference_kwargs:
    num_inference_steps: 100

  condor:
    jobname: infer_2dprior_gha_from_camsweep
    submit:
      executable: "condor/joker.sh"
      arguments: "joker/prior/data/preprocess/gha_dataset/predict_gha_dataset_from_joker_camsweep_dataset.py __CONFIG__"
      description: "${..jobname}"
      request_gpus: 1
      request_cpus: 4
      request_memory: 64000
      request_disk: "50G"
      requirements: '(TARGET.CUDADeviceName=="NVIDIA A100-SXM4-40GB") || (TARGET.CUDADeviceName=="NVIDIA A100-SXM4-80GB")'
      log_root: "${...output_dir}/$(ClusterId).$(Process)"
      error: "$(log_root).out"
      output: "$(log_root).out"
      log: "$(log_root).log"
      environment: "\"CONDOR_ClusterId=$(ClusterId) CONDOR_Process=$(Process) CONDOR_WorldSize=${.njobs} CONDOR_LOGFILE=$(log) CONDOR_OUTFILE=$(out) NUMEXPR_MAX_THREADS=128\""
      njobs: 1

  #  code_backup:
  #    root: /is/cluster/mprinzler/.fastcomposer_codebackups


"""


OmegaConf.register_new_resolver("eval", eval)

parser = ArgumentParser()
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

        # generate execution command
        cmd_prepend = ""
        if conf.MULTICONFIG_CONFIG.get('DISJOINT_GPUS', False):
            cmd_prepend = cmd_prepend + f"CUDA_VISIBLE_DEVICES={i}, "

        executable = child_cfg.condor.submit.executable
        arguments = child_cfg.condor.submit.arguments.replace("__CONFIG__", str(config_path))

        # all processes except the last one are running in background
        if i != n_processes - 1 and conf.MULTICONFIG_CONFIG.get('RUN_PARALLEL', False):
            p = Process(target=os.system, args=(f"{cmd_prepend} {executable} {arguments}",))
            p.start()
        else:
            os.system(f"{cmd_prepend} {executable} {arguments}")

else:
    executable = conf.condor.submit.executable
    arguments = conf.condor.submit.arguments.replace("__CONFIG__", args.config)
    os.system(f"{executable} {arguments}")

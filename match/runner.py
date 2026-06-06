import matplotlib
matplotlib.use("Agg")  # no-Tk backend, offscreen rendering

import warnings
import torch

warnings.filterwarnings("ignore")  # ignore all warnings
import psutil
import pudb
import os
import types
import gc
import tensorflow as tf
import tqdm
import torch
import accelerate
import lpips
from absl import flags
from match.options import Options
from match import data, models
from match.utils.log_experiment_util import TensorBoardLogger
from match.utils import general_util, vis_util, log_console_util, data_util
import gin
import numpy as np
import copy
import pprint
import time
import mediapy as media
import pynvml
import einops
import collections
import datetime
from match.utils import file_util
import sys
from pyvirtualdisplay import Display
from typing import List

class MatchRunner:
    def __init__(self, options: Options):
        self.global_update_step = 0
        self.validation_counter = 0
        self.train_loader = None
        self.val_loaders = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.pretrained_model_iter = None
        self.tracker = None
        self.lpips_loss = None

        self.opt = options
        self.process_options()

        self.logger = log_console_util.getLogger(__name__)
        log_console_util.basicConfig(level=self.opt.log_level)

        self.local_tmp_dir = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), os.path.basename(self.opt.base_folder)
        )
        self.exp_dir = file_util.Path(self.opt.base_folder)
        self.ckpt_dir = self.exp_dir / "checkpoints"
        self.mldash_dir = self.exp_dir / "mldash"

        self.model = models.MatchModel(self.opt)
        self.template_vertices = self.model.template_vertices.cpu().numpy().astype(np.float32)
        self.template_faces = self.model.template_triangles.cpu().numpy().astype(np.int32)
        self.template_faceuvcoords = self.model.template_triangle_uvs.cpu().numpy().astype(np.float32)
        self.gs_renderer = self.model.gs_renderer
        self.vertex_mask = self.model.vertex_mask

        # starting virtual display for headless cpu-based rendering
        self.display = Display()
        self.display.start()  # opening virtual display for headless cpu-based rendering

        tf.config.set_visible_devices([], 'GPU')  # disabling GPU usage for tensorflow
        self.init_accelerator(self.exp_dir)

        # Enable TF32 for faster training on Ampere GPUs
        if self.opt.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.load_checkpoint()

    def process_options(self):
        if self.opt.output_res is None:
            self.opt.output_res = self.opt.input_res

    def save_config_files(self, config_files: list[str]):
        for i in range(len(config_files)):
            try:
                file_util.copy(config_files[i], self.exp_dir / f"config_{i}.gin", True)
            except Exception as e:
                self.logger.info(
                    f"Failed to copy gin config from [{config_files[i]}] to"
                    f" [{self.exp_dir / f'config_{i}.gin'}]\n"
                )
                pass

    def init_accelerator(self, exp_dir: file_util.Path):
        opt = self.opt
        ddp_kwargs = accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=opt.find_unused_parameters
        )
        pg_kwargs = accelerate.InitProcessGroupKwargs(
            timeout=datetime.timedelta(minutes=opt.nccl_timeout)
        )
        dataloader_config = accelerate.DataLoaderConfiguration(dispatch_batches=False)
        accelerator = accelerate.Accelerator(
            project_dir=str(exp_dir),
            gradient_accumulation_steps=opt.gradient_accumulation_steps,
            dataloader_config=dataloader_config,
            mixed_precision=opt.mixed_precision,
            deepspeed_plugin=None,
            kwargs_handlers=[ddp_kwargs, pg_kwargs],
        )
        self.accelerator = accelerator
        return accelerator
    
    @property
    def do_log(self):
        return self.accelerator.is_main_process and (
                self.global_update_step % self.opt.log_freq == 0 
                or self.global_update_step == 1)

    @property
    def train_batch_size_per_gpu(self):
      return self.train_loader.batch_size
    
    @property
    def weight_dtype(self):  
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        return weight_dtype

    @property
    def device(self):
        return self.accelerator.device

    def forward_image(self, batch, return_masked_gaussians: bool = False):
        return self.model(batch, func_name="forward_image", return_masked_gaussians=return_masked_gaussians)

    def render_gaussians(self, masked_gaussian_parameters, C2W, fxfycxcy, height, width, bg_color):
        return self.accelerator.unwrap_model(self.model).gs_renderer.render(masked_gaussian_parameters, C2W, fxfycxcy, height, width, bg_color)

    def init_train_loader(self):
        if self.opt.overfit:
            self.logger.warning('OVERFITTING ENABLED!')
            p = copy.deepcopy(self.opt.val_loader_params[0])
            p['kwargs']['training'] = getattr(p['kwargs'], 'training', False)
            train_loader_cls = data.get_loader_cls(p['cls'])
            train_loader = train_loader_cls(**p['kwargs'])
            train_loader = data_util.RepeatFewBatchDataLoader(train_loader, self.opt.overfit_n_batches)
        else:
            train_loader_cls = data.get_loader_cls(self.opt.train_loader_params['cls'])
            self.opt.train_loader_params['kwargs']['training'] = getattr(self.opt.train_loader_params['kwargs'], 'training', True)
            train_loader = train_loader_cls(**self.opt.train_loader_params['kwargs'])
        
        self.logger.info(
            f"Loaded [{self.train_batch_size_per_gpu if self.opt.overfit else len(train_loader.dataset)}] training samples"
        )
        self.train_loader = train_loader
        return train_loader
      

    def init_val_loaders(self):
        opt = self.opt
        # val loaders
        if opt.overfit:
            val_loaders = [self.train_loader]
        else:
            val_loaders = []
            for p in opt.val_loader_params:
                p = copy.deepcopy(p)
                val_loader_cls = data.get_loader_cls(p['cls'])
                p['kwargs']['training'] = getattr(p['kwargs'], 'training', False)

                val_loaders.append(
                    val_loader_cls(**p['kwargs'])
                )

        dataset_names_val = [p['name'] for p in opt.val_loader_params]
        self.val_loaders = val_loaders
        self.dataset_names_val = dataset_names_val

    def split_val_sets_across_processes(self, world_size: int = 1, process_idx: int = 0):
        '''
        Adjusts val dataset parameters to split val datasets across multiple processes.
        '''
        assert self.val_loaders is None, "Must not be called after self.init_val_loaders()"
        for p in self.opt.val_loader_params:
            p['kwargs']['dataset_kwargs']['world_size'] = world_size
            p['kwargs']['dataset_kwargs']['process_idx'] = process_idx


    def init_loss_weights(self):
        '''
        Converts all loss weights in opt to lists if they are not already. Each item in the list corresponds to the loss weight value for one dataset. Useful in the mixed dataset scenario where e.g. the geometry-related losses should only be calculcated on one of the datasets.
        
        Must not be called before self.init_train_loader()
        '''
        n_datasets = getattr(self.train_loader, 'n_datasets', 1)
        for k in [
            "xyz",
            "mesh_vert",
            "scale",
            "opacity",
            "render",
            "l1",
            "ssim",
            "lpips",
        ]:
            weight_name = k + "_weight"
            weight_val = getattr(self.opt, weight_name)
            if not isinstance(weight_val, list):
                setattr(self.opt, weight_name, [weight_val] * n_datasets)
        return self.opt
    
    def init_lpips(self):      
        if self.accelerator.is_main_process:
            _ = lpips.LPIPS(net="vgg")
            del _
        self.accelerator.wait_for_everyone()  # wait for pretrained backbone weights to be downloaded
        lpips_loss = lpips.LPIPS(net="vgg").to(self.accelerator.device)
        lpips_loss = lpips_loss.requires_grad_(False)
        self.lpips_loss = lpips_loss.eval()
          
        # For DeepSpeed bug: model inputs could be `torch.nn.Module` (e.g., `lpips_loss`)
        def is_floating_point(self):
            return True
        self.lpips_loss.is_floating_point = types.MethodType(is_floating_point, self.lpips_loss)

    def init_optimizer(self):
        params_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = models.get_optimizer(params=params_to_optimize, **self.opt.optimizer)
        self.optimizer = optimizer
        self.params_to_optimize = list(params_to_optimize)
        return optimizer

    def init_scheduler(self):
       
        # fixing lr scheduler settings: accelerate automatically performs
        # accelerator.num_processes steps for the lr scheduler at every optimization
        # step (see https://huggingface.co/docs/accelerate/concept_guides/performance#learning-rates).
        # For more control, we counteract this explicitly
        lr_kwargs_corrected = dict()
        for k, v in self.opt.lr_scheduler.items():
            if "steps" in k:
                v = v * self.accelerator.num_processes
            lr_kwargs_corrected[k] = v
        self.lr_scheduler = models.get_lr_scheduler(
            optimizer=self.optimizer, **lr_kwargs_corrected
        )
        return self.lr_scheduler
    
    def accelerate_prepare(self):
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        train_loader = self.train_loader
        val_loaders = self.val_loaders

        model, optimizer, lr_scheduler, train_loader, *val_loaders  = self.accelerator.prepare(
            model, optimizer, lr_scheduler, train_loader, *val_loaders
        )

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loaders = val_loaders

    def train(self, training: bool = True):
        self.model.train(training)
    
    def eval(self):
        self.train(training=False)

    def get_unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)
    
    def get_compiled_model(self):
        compiled_model = self.get_unwrapped_model()
        compiled_model.compile()
        return compiled_model


    def load_checkpoint(self, ckpt_path: str = None):
        '''loads model weights from checkpoint'''
        if ckpt_path is None:
            if self.opt.load_pretrained_model is None:
                ckpt_path = None
            else:
                ckpt_path = general_util.get_ckpt_path(
                    os.path.join(self.opt.load_pretrained_model, "checkpoints"),
                    self.opt.load_pretrained_model_ckpt,
                )
        if ckpt_path is None:
            self.logger.info("No pretrained model specified.")
            ckpt_iter = None
        else:      
            self.logger.info(f"Loading pretrained model from {ckpt_path}.")
            ckpt_iter = ckpt_path.split("/")[-1] if ckpt_path is not None else None
            local_ckpt_path = file_util.Path(f"{self.local_tmp_dir}/pretrained_model")
            with file_util.AccelerateRemoteDirWrapper(
                str(ckpt_path),
                str(local_ckpt_path),
                self.accelerator,
                should_writeback=False,
            ) as f:
                fname = f.GetFilename()
                self.model = general_util.load_ckpt(
                    fname,
                    self.model,
                    self.accelerator,
                    strict=self.opt.load_pretrained_model_strict,
                )
        self.pretrained_model_iter = ckpt_iter
        return self.model, ckpt_iter

    def resume_checkpoint(self):
        '''recovers entire training state from checkpoint'''
        opt = self.opt
        ckpt_dir = self.ckpt_dir
        logger = self.logger

        # (Optional) Load pretrained model or checkpoint
        ckpt_path = general_util.get_ckpt_path(str(ckpt_dir), opt.resume_from_iter)
        if ckpt_path is None:  # try to load pretrained network
            # Load a pretrained model
            if opt.load_pretrained_model is not None:
                self.logger.info(
                    f"Load Match checkpoint from [{opt.load_pretrained_model}] iteration"
                    f" [{opt.load_pretrained_model_ckpt:06d}]\n"
                )
                ckpt_path = general_util.get_ckpt_path(
                    os.path.join(opt.load_pretrained_model, "checkpoints"),
                    opt.load_pretrained_model_ckpt,
                )
        if ckpt_path is None:
            logger.info("Training from scratch\n")
        else:
            logger.info(f"Loading checkpoint from [{ckpt_path}]\n")
            local_ckpt_path = f"{self.local_tmp_dir}/resume_ckpt"
            with file_util.AccelerateRemoteDirWrapper(
                str(ckpt_path),
                local_ckpt_path,
                self.accelerator,
                should_writeback=False,
            ) as f:
                self.accelerator.load_state(f.GetFilename(), strict=False)
            removed_states = self._drop_incompatible_optimizer_state()
            if removed_states > 0:
                logger.warning(
                    "Dropped [%d] optimizer state entries with incompatible tensor shapes after"
                    " resume checkpoint load. This usually means model parameter shapes changed"
                    " (e.g. hidden size updates), so those parameters will continue with fresh"
                    " optimizer moments.",
                    removed_states,
                )
            opt.resume_from_iter = ckpt_path.split("/")[-1]
            self.global_update_step = int(opt.resume_from_iter)

    def _drop_incompatible_optimizer_state(self) -> int:
        """Drops optimizer state tensors whose shape no longer matches the parameter shape."""
        optimizer = self.optimizer
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer

        if not hasattr(optimizer, "state") or not hasattr(optimizer, "param_groups"):
            return 0

        # Map params to readable names for mismatch logging.
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        param_name_by_id = {id(p): n for n, p in unwrapped_model.named_parameters()}

        dropped = 0
        state_tensor_keys = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
        for group in optimizer.param_groups:
            state_is_incompatible = False
            for param in group["params"]:
                state = optimizer.state.get(param)
                if not state:
                    continue
                has_incompatible_shape = any(
                    key in state
                    and torch.is_tensor(state[key])
                    and state[key].shape != param.shape
                    for key in state_tensor_keys
                )
                for key in state_tensor_keys:
                    if key in state and torch.is_tensor(state[key]) and state[key].shape != param.shape:
                        param_name = param_name_by_id.get(id(param), "<unknown_param>")
                        self.logger.warning(
                            "INCOMPATIBLE optimizer state for param [%s], key [%s]:"
                            " checkpoint_shape=%s, param_shape=%s",
                            param_name,
                            key,
                            tuple(state[key].shape),
                            tuple(param.shape),
                        )
                if has_incompatible_shape:
                    state_is_incompatible = True
            if state_is_incompatible:  # if any param in the group is incompatible, drop the entire group
                for param in group["params"]:
                    del optimizer.state[param]
                    dropped += 1
        return dropped

    def save_experiment_info(self):
        if self.accelerator.is_main_process:
            # Save all experimental parameters and model architecture of this run to a file (opt and configs)
            self.logger.info(
                f"Saving experiment parameters and model architecture to [{self.exp_dir}]\n"
            )
            general_util.save_experiment_params(self.opt, str(self.exp_dir))
            general_util.save_model_architecture(self.accelerator.unwrap_model(self.model), str(self.exp_dir))
            self.logger.info(
                "Finished saving experiment parameters and model architecture to"
                f" [{self.exp_dir}]\n"
            )
    def init_experiment_logger(self):
        # experiment logger
        self.logger.info(f"Creating TensorBoardLogger and log to ")
        tracker = TensorBoardLogger(
            accelerator=self.accelerator,
            log_path=str(self.exp_dir),
        )
        self.tracker = tracker
    
    def init_nvml_handles(self):
        nvml_handles = []
        for i in range(torch.cuda.device_count()):
            nvml_handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            self.logger.info(f"nvml_handle: {i}: {nvml_handles[i]}")
        self.nvml_handles = nvml_handles
        return nvml_handles

    def prepare_training(self, save_config_files: List[str] = []):
        opt = self.opt
        logger = self.logger
        pp = pprint.PrettyPrinter(indent=4)
        logger.info(f"Training Settings:\n{pp.pformat(vars(opt))}\n")
        pynvml.nvmlInit()        

        # Create an experiment directory using the `tag`

        file_util.makedirs(self.ckpt_dir, exist_ok=True)
        file_util.makedirs(self.mldash_dir, exist_ok=True)

        self.save_config_files(save_config_files)

        # Set the random seed
        if opt.seed >= 0:
            accelerate.utils.set_seed(opt.seed)
            logger.info(f"You have chosen to seed([{opt.seed}])\n")
        
        self.accelerator.wait_for_everyone()  # other processes wait for the main process
        self.init_nvml_handles()
        self.init_train_loader()
        self.init_val_loaders()
        self.init_loss_weights()
        self.init_lpips()
        self.init_optimizer()
        self.init_scheduler()
        self.accelerate_prepare()
        self.resume_checkpoint()
        self.save_experiment_info()
        self.init_experiment_logger()         

    def log_train_step(self, logs, progress_bar):
        # Checks if the accelerator has performed an optimization step behind the scenes
        if self.accelerator.sync_gradients:
            # Gather the losses across all processes for logging (if we use distributed training)
            for k, v in logs.items():
                if isinstance(v, torch.Tensor):
                    v = self.accelerator.gather(v.detach()).mean().item()
                logs[k] = v
            logs["training/lr"] = self.lr_scheduler.get_last_lr()[0]
        
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)

            # Log the training progress
            if self.do_log:
                self.logger.info(
                    f"[{self.global_update_step:06d} / {self.opt.max_train_steps:06d}] "
                    + f"loss: {logs['training/loss']:.4f}, step_duration:"
                    f" {logs['training/step_duration']:.2e}, lr:"
                    f" {logs['training/lr']:.2e}"
                )
                self.tracker.log(
                    logs,
                    step=self.global_update_step,
                )

    def run_training(self):
        opt = self.opt
        accelerator = self.accelerator
        progress_bar = tqdm.tqdm(
            range(opt.max_train_steps),
            initial=self.global_update_step,
            desc="Training",
            ncols=125,
            disable=not accelerator.is_main_process,
        )
        if opt.initial_eval:
            self.run_evaluation()

        while True:
            step_start_time = time.time()
            for batch in self.train_loader:
                if self.global_update_step == opt.max_train_steps:
                    self.logger.info("Training finished!\n")
                    progress_bar.close()
                    return
                logs = self.train_step(batch)
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                step_start_time = step_end_time
                logs["training/step_duration"] = step_duration
                self.log_train_step(logs, progress_bar)

                # save checkpoint
                if self.global_update_step % opt.save_freq == 0:  # 1. every `save_freq` steps
                    self.save_checkpoint()

                # Evaluate on the validation set
                if (
                    (
                        self.global_update_step % opt.early_eval_freq == 0
                        and self.global_update_step < opt.early_eval
                    )  # 1. more frequently at the beginning
                    or self.global_update_step % opt.eval_freq
                    == 0  # 2. every `eval_freq` steps
                ):
                    self.run_evaluation()
        
    def evaluate_on_valset(self, val_set_name, val_loader, out_dir):
        logger = self.logger
        global_update_step = self.global_update_step
        accelerator = self.accelerator
        opt = self.opt

        all_val_metrics, val_steps, log_dict = {}, 0, {}
        val_progress_bar = tqdm.tqdm(
                range(opt.max_val_steps),
                desc=f"Validation_{val_set_name}",
                ncols=125,
                disable=not accelerator.is_main_process,
        )
        vis_img_collector = collections.defaultdict(list)
        val_sample_counter = 0
            
        per_vertex_eucldist = np.zeros((0, len(self.template_vertices)), dtype=np.float32)
        for val_batch in iter(val_loader):
            val_batch = general_util.batch_to_device(val_batch, accelerator.device)
            visualize = val_sample_counter < opt.n_vissamples
            b, v, c, h, w = val_batch["image"].shape

            val_outputs = self.model(
                val_batch,
                self.lpips_loss,
                step=global_update_step,
                dtype=self.weight_dtype,
                render_uv=visualize,
                render_img=visualize,
                render_gauss=visualize,
                log_all=True,
            )

            val_logs = dict([
                (k, v)
                for k, v in val_outputs.items()
                if not k.startswith("images")
            ])
            val_logs = accelerator.reduce(val_logs, reduction="mean")
            for k, v in val_logs.items():
                all_val_metrics.setdefault(k, []).append(v)

            # mesh vertex prediction
            verts_pred = self.model(
                val_batch, func_name="forward_mesh", dtype=self.weight_dtype
            )

            verts_gt = val_batch["verts"]

            verts_pred_filtered = einops.rearrange(
                einops.rearrange(verts_pred, "B V C -> V B C")[self.vertex_mask],
                "V B C -> B V C",
            )

            verts_gt_filtered = einops.rearrange(
                einops.rearrange(verts_gt, "B V C -> V B C")[self.vertex_mask],
                "V B C -> B V C",
            )

            mesh_mse = torch.mean(
                torch.square((verts_pred_filtered - verts_gt_filtered) * 1000)
            )
            mesh_mse = accelerator.reduce(mesh_mse, reduction="mean")

            mesh_mae = torch.mean(torch.abs(verts_pred_filtered - verts_gt_filtered) * 1000)
            mesh_mae = accelerator.reduce(mesh_mae, reduction="mean")

            per_vertex_eucldist = np.concatenate([per_vertex_eucldist, 
                                                    torch.linalg.norm(
                                                    (verts_pred_filtered - verts_gt_filtered) * 1000,
                                                    ord=2,
                                                    dim=-1,
                                                ).detach().cpu().numpy().astype(np.float32)], axis=0)

            all_val_metrics.setdefault("mesh_mse", []).append(mesh_mse)
            all_val_metrics.setdefault("mesh_mae", []).append(mesh_mae)

            val_progress_bar.update(1)
            val_steps += 1

            # visualize
            if visualize and accelerator.is_main_process:
                for k, v in val_outputs.items():
                    if k.startswith("images"):
                        vis_img_collector[
                            k.replace("images", f"images_{val_set_name}")
                        ].append(v.detach().cpu())

                # visualizing vertex predictions
                for i in range(len(verts_pred)):
                    vis_vert_img = vis_util.vis_vert_prediction(
                        verts_pred[i].cpu().numpy(),
                        verts_gt[i].cpu().numpy(),
                        self.template_faces,
                        einops.rearrange(
                            val_batch["image"], "B V C H W -> B V H W C"
                        )
                        .cpu()
                        .numpy()[i, : opt.num_input_views],
                        figscale=4.0,
                        rot90=opt.rot90
                    )
                    vis_img_collector[f"images_{val_set_name}/vert_vis"].append(
                        einops.rearrange(
                            torch.from_numpy(vis_vert_img), "H W C -> 1 1 C H W"
                        )
                    )
            val_sample_counter += b
            accelerator.wait_for_everyone()

            if (
                opt.max_val_steps is not None
                and val_steps == opt.max_val_steps
            ):
                break

        val_progress_bar.close()

        all_val_metrics_mean = dict([
            (k, torch.stack(v).mean().cpu())
            for k, v in all_val_metrics.items()
        ])

        logger.info(
            f"Eval {val_set_name} [{global_update_step:06d} /"
            f" {opt.max_train_steps:06d}] "
            + "".join([
                f"{k}: {v.item():.4f}, "
                for k, v in all_val_metrics_mean.items()
            ])
            + f"\n"
        )

        log_dict.update(
            dict([
                (f"validation_{val_set_name}/" + k, v)
                for k, v in all_val_metrics_mean.items()
            ])
        )

        # visualizing uv vertex mse
        per_vertex_eucldist = np.mean(per_vertex_eucldist, axis=0)
        uv_vertex_eucldist_vis = vis_util.vis_uv_scores(vertex_scores=per_vertex_eucldist, faces=self.template_faces, faces_uvcoords=self.template_faceuvcoords, vmin=0, vmax=50, splat_size=0.7)
        uv_vertex_eucldist_vis = einops.rearrange(torch.from_numpy(uv_vertex_eucldist_vis.astype(np.float32)/255), 'h w c -> 1 1 c h w')
        vis_img_collector[f"images_{val_set_name}/uv_eucl_dist"] = [uv_vertex_eucldist_vis]

        # adding validation visualizations to log_dict
        for k, v in vis_img_collector.items():
            v = torch.cat(v, dim=0)[: opt.n_vissamples]
            if k.endswith("vert_vis"):
                for i in range(len(v)):
                    log_dict[k + f"_{i}"] = v[i : i + 1]
            else:
                log_dict[k] = v

        accelerator.wait_for_everyone()

        # storing all images
        for k, v in log_dict.items():
            if k.startswith("images"):
                outpath_ = out_dir / f"{k}.jpg"
                outpath_.parent.mkdir(parents=True, exist_ok=True)
                media.write_image(
                    outpath_,
                    einops.rearrange(
                        v.detach().cpu().numpy(),
                        "B V C H W -> (B H) (V W) C",
                    ),
                )

        return log_dict

    @torch.no_grad()
    def run_evaluation(self):
        logger = self.logger
        global_update_step = self.global_update_step
        accelerator = self.accelerator
        exp_dir = self.exp_dir
        opt = self.opt

        logger.info(f"Evaluating at step [{global_update_step:06d}]\n")
        torch.cuda.empty_cache()
        gc.collect()
        evaluation_folder = exp_dir / f"evaluation/{global_update_step:07d}"
        tmp_evaluation_folder = file_util.Path(
            f"{self.local_tmp_dir}/evaluation/{global_update_step:07d}"
        )
        if accelerator.is_main_process:
            file_util.makedirs(tmp_evaluation_folder, exist_ok=True)
            file_util.makedirs(evaluation_folder.parent, exist_ok=True)
        
        self.model.eval()
        val_set_names = (
                self.dataset_names_val
                if self.dataset_names_val is not None
                else list(map(str, range(len(self.val_loaders))))
            )
        log_dict = {}
        for val_set_name, val_loader in zip(val_set_names, self.val_loaders):
           log_dict.update(
                self.evaluate_on_valset(val_set_name, val_loader, tmp_evaluation_folder)
            )
            
        if accelerator.is_main_process:
            # log validation metrics
            self.tracker.log(
                log_dict,
                step=global_update_step,
            )           

            # copy tmp evaluation folder to remote evaluation folder 
            file_util.copy(
                tmp_evaluation_folder,
                evaluation_folder,
                overwrite=True,
            )
            file_util.delete_recursively(tmp_evaluation_folder)

        accelerator.wait_for_everyone()
        logger.info(f"Finished evaluation at step [{global_update_step:06d}]\n")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(10)
        self.validation_counter +=1
        if (opt.restart_after_eval_frequency is not None) and (self.validation_counter % opt.restart_after_eval_frequency ==0 ):
            logger.info(f'Planned restarting of job after {self.validation_counter} validations')
            self.graceful_exit(3)

    def save_checkpoint(self, out_dir: file_util.Path = None):   
        # Atomically save checkpoint
        
        local_save_dir = file_util.Path(
              f"{self.local_tmp_dir}/checkpoints/{self.global_update_step:06d}"
          )
        remote_save_dir = file_util.Path(out_dir) if out_dir is not None else self.ckpt_dir / f"{self.global_update_step:06d}"
        remote_save_dir_tmp = file_util.Path(str(remote_save_dir) + "_tmp")
        if self.accelerator.is_main_process:
            if remote_save_dir_tmp.exists():
                file_util.delete_recursively(remote_save_dir_tmp)
            file_util.makedirs(remote_save_dir_tmp)
        with file_util.AccelerateRemoteDirWrapper(
            str(remote_save_dir_tmp),
            tmp_filename=str(local_save_dir),
            accelerator=self.accelerator,
            should_writeback=True,
        ) as f:
            self.accelerator.save_state(f.GetFilename())
        if self.accelerator.is_main_process:
            if remote_save_dir.exists():
                file_util.delete_recursively(remote_save_dir)
            file_util.rename(remote_save_dir_tmp, remote_save_dir)
        self.accelerator.wait_for_everyone()
        gc.collect()        

    def graceful_exit(self, exit_code: int):
        """
        Gracefully shut down a distributed training job and exit with a given code.
        Works with torch.distributed and 🤗 Accelerate.

        Args:
            exit_code (int): the code to exit with (default: 0).
        """
        self.logger.info("Gracefully exiting the training job...\n")
        self.display.stop()

        if self.tracker is not None:
            self.tracker.finish()

        # Write exit code to file
        if 'CONDOR_EXITFILE' in os.environ and self.accelerator.is_main_process:
            exit_path = file_util.Path(os.environ['CONDOR_EXITFILE'])
            exit_path.parent.mkdir(exist_ok=True, parents=True)
            with open(exit_path, 'w') as f:
                f.write(str(exit_code))
        self.accelerator.wait_for_everyone()

        try:
            # If running under accelerate, end gracefully
            self.accelerator.end_training()
        except Exception:
            pass
        sys.exit(0)
            
    def train_step(self, batch):
        model = self.model
        accelerator = self.accelerator
        opt = self.opt

        model.train()

        batch = general_util.batch_to_device(batch, accelerator.device)
        logs = dict()

        for i in range(torch.cuda.device_count()):
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handles[i]).gpu
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handles[i]).used
            logs[f"Resources/GPU/{i}_util"] = gpu_util
            logs[f"Resources/GPU/{i}_mem(GB)"] = gpu_memory / (1024**3)
        mem = psutil.virtual_memory()
        logs["Resources/mem_total(GB)"] = mem.total / (1024**3)
        logs["Resources/mem_avail(GB)"] = mem.available / (1024**3)

        with accelerator.accumulate(model):
            outputs = model(
                batch,
                self.lpips_loss,
                step=self.global_update_step + 1,
                dtype=self.weight_dtype,
            )  # `step` starts from 1
            logs.update(
                dict([
                    ("training/" + k, v)
                    for k, v in outputs.items()
                    if not k.startswith("images")
                ])
            )
            loss = outputs["loss"]

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(self.params_to_optimize, opt.max_grad_norm)

            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
        if accelerator.sync_gradients:
            self.global_update_step += 1
        
        
        return logs
      

            

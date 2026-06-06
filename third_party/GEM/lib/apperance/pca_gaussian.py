# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2025 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: wojciech.zielonka@tuebingen.mpg.de, wojciech.zielonka@tu-darmstadt.de

import copy
import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion
import pudb
from lib.F3DMM.masks.masking import Masking

from pathlib import Path
from utils.geometry import AttrDict
import einops


def axis_angle_to_quaternion(axis_angles: th.Tensor) -> th.Tensor:
    rot_mats = axis_angle_to_matrix(axis_angles)
    quaternions = matrix_to_quaternion(rot_mats)
    return quaternions


def _make_orthogonal(A):
    """Assume that A is a tall matrix.

    Compute the Q factor s.t. A = QR (A may be complex) and diag(R) is real and non-negative.
    """
    X, tau = th.geqrf(A)
    Q = th.linalg.householder_product(X, tau)
    # The diagonal of X is the diagonal of R (which is always real) so we normalise by its signs
    Q *= X.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
    return Q


def _is_orthogonal(Q, eps=None):
    n, k = Q.size(-2), Q.size(-1)
    Id = th.eye(k, dtype=Q.dtype, device=Q.device)
    # A reasonable eps, but not too large
    eps = 10. * n * th.finfo(Q.dtype).eps
    return th.allclose(Q.mH @ Q, Id, atol=eps)


class PCApperance(nn.Module):
    CHANNEL_DIMENSIONS = dict(geometry=3, opacity=1, scales=3, rotation=4, colors=3)

    def __init__(self, config) -> None:
        super().__init__()
        self.masking = Masking()
        self.dynamic_color = config.train.get('dynamic_colors', False)
        self.pca_all = config.train.get('pca_all', False)
        self.optimizable_means = config.train.get('optimizable_means', False)

        pca = th.load(self.get_pca_path(config), weights_only=False)
        gt = th.load(self.get_coeffs_path(config), weights_only=False)
        rootRT = th.load(self.get_rootRT_path(config), weights_only=False) if Path(self.get_rootRT_path(config)).exists() else None
        self.n_components = pca["config"] 

        self.components = nn.ParameterDict()
        self.masks = sorted(list(pca["geometry" if 'geometry' in pca else 'all'].keys()))
        self.mods = sorted(list(pca.keys()))
        if not self.dynamic_color:
            self.mods.remove("colors")
        self.mods.remove("tex_map")
        self.mods.remove("config")
        self.means = nn.ParameterDict() if self.optimizable_means else {}
        self.scales = {}
        self.n = sum(self.n_components.values())
        self.disable_dynamic_mouth_colors = config.train.get('disable_dynamic_mouth_colors', False)
        self.disable_dynamic_mouth_opacities = config.train.get('disable_dynamic_mouth_opacities', False)
        self.disable_dynamic_mouth_scales = config.train.get('disable_dynamic_mouth_scales', False)


        logger.info(f"Loaded PCApperance with {self.n_components} components with {self.masks}")

        static = ["config"]

        self.channels = {}
        self.gt_coeffs = {}
        self.rootRTs = rootRT

        for mod, masks in pca.items():
            if mod in static:
                continue
            for k, v in masks.items():
                key = self.get_key(mod, k)

                if mod == 'tex_map':
                    self.register_buffer(f"tex_map_{k}", pca["tex_map"][k].int())
                    continue
                elif mod == 'colors' and not self.dynamic_color:
                    self.register_buffer(f"colors_{k}", pca["colors"][k].float())
                    continue

                var = th.from_numpy(v["variance"]).float()
                std = th.sqrt(var)
                self.gt_coeffs[key] = (th.from_numpy(gt[mod][k][:, : self.n_components[mod]]).float() / std[: self.n_components[mod]]).cuda()
                self.components[key] = th.from_numpy(v["components"]).float()[: self.n_components[mod], :]
                self.scales[key] = std[: self.n_components[mod], None].float().cuda()
                mean = th.from_numpy(v["mean"]).float()
                if not self.optimizable_means:
                    mean = mean.cuda()
                self.means[key] =  mean
                self.channels[key] = v["channels"]

    def basis_parameters(self):
        return self.components.parameters()
    
    def means_parameters(self):
        if self.optimizable_means:
            return self.means.parameters()
        else:
            return []
    
    def means_parameters_flat(self):
        '''
        
        Returns:
            flattened tensor of mean parameters
        '''
        means_flat = [p.flatten() for p in self.means_parameters()]
        means_flat = th.cat(means_flat)
        return means_flat
    
    def basis_parameters_flat(self):
        '''
        
        Returns:
            flattened tensor of mean parameters
        '''
        components_flat = [p.flatten() for p in self.basis_parameters()]
        components_flat = th.cat(components_flat)
        return components_flat

    @staticmethod
    def get_pca_path(config):
        return f"{config.train.pca_dir}/GAUSSIAN_PCA.ptk"

    @staticmethod
    def get_coeffs_path(config):
        return f"{config.train.pca_dir}/GAUSSIAN_PCA_coeffs.ptk"

    @staticmethod
    def get_rootRT_path(config):
        return f"{config.train.pca_dir}/GAUSSIAN_PCA_rootRT.ptk"

    def get_mean(self, mod):
        mean = []
        mod_ = 'all' if self.pca_all else mod
        for mask in self.masks:
            key = self.get_key(mod_, mask)
            C = self.channels[key]
            mean.append(self.means[key].reshape(-1, C))
        mean = th.cat(mean).contiguous()
        if self.pca_all:
            mean = self.unzip_all({'all':mean})[mod]
        return mean

    def get_key(self, mod, mask):
        return f"{mod}_{mask}"

    def to_key_parts(self, key):
        # For backward compatibility, could have been "+" instead of "_"
        mod = key.split("_")[0]
        mask = "_".join(key.split("_")[1:])
        return mod, mask

    def resize(self, n):
        for k, v in self.components.items():
            self.gt_coeffs[k] = self.gt_coeffs[k][:, :n]
            self.components[k] = self.components[k][:n, :]
            self.scales[k] = self.scales[k][:n, :]

    def keys(self):
        return self.mods

    def get(self, idx):
        results = {}
        for k in self.gt_coeffs.keys():
            results[k] = self.gt_coeffs[k][idx]

        return results
    
    def get_rootRT(self, idx):
        return self.rootRTs[idx] if self.rootRTs is not None else None

    def make_orthagonal(self):
        for k in self.components.keys():
            A = self.components[k].T
            if not _is_orthogonal(A):
                logger.info(f"Orthagonalizing basis of {k}")
                self.components[k] = _make_orthogonal(A).T

    def to_coeffs(self, values, is_batch=False):
        coeffs = {}
        for mask in self.masks:
            n = 0
            for mod in self.mods:
                key = self.get_key(mod, mask)
                i = self.n_components[mod]
                if is_batch:
                    coeffs[key] = values[mask][:, n:n+i]
                else:
                    coeffs[key] = values[mask][n:n+i]
                n += i

        return coeffs
    
    def unzip_all(self, results):
        if 'all' in results: 
            unzipped_results = dict([(k, v) for k, v in results.items() if k != 'all'])
            all = results['all']
            counter = 0
            for k in sorted(self.CHANNEL_DIMENSIONS.keys()):
                if (not self.dynamic_color) and k == 'colors':
                    continue
                unzipped_results[k] = all[..., counter:counter + self.CHANNEL_DIMENSIONS[k]]
                counter += self.CHANNEL_DIMENSIONS[k]
            results = unzipped_results
        return results
    
    def get_tex_map(self):
        tex_maps = []
        for m in self.masks:            
            tex_maps.append(getattr(self, f"tex_map_{m}"))
        return th.cat(tex_maps)


    def inverse_transform(self, coeffs, masking=None):
        results = {m: [] for m in self.mods}
        results["tex_map"] = []
        
        is_batch = len(next(iter(coeffs.values())).shape) == 2
        if not is_batch:
            coeffs = dict([(k, v.unsqueeze(0)) for k, v in coeffs.items()])
        B = next(iter(coeffs.values())).shape[0]

        for k in coeffs.keys():
            C = self.channels[k]
            components = self.components[k].clone()
            if self.disable_dynamic_mouth_colors:
                assert k == 'all_full'
                pca_color_components = self.unzip_all(dict(all=components.view(self.n, -1, self.channels[k])))['colors']  # (ncomponents, ngaussians, 3)
                pca_color_components[:,self.masking.mouth()[self.tex_map_full]] = 0  # inplace operation

            if self.disable_dynamic_mouth_opacities:
                assert k == 'all_full'
                pca_color_components = self.unzip_all(dict(all=components.view(self.n, -1, self.channels[k])))['opacity']  # (ncomponents, ngaussians, 3)
                pca_color_components[:,self.masking.mouth()[self.tex_map_full]] = 0  # inplace operation

            if self.disable_dynamic_mouth_scales:
                assert k == 'all_full'
                pca_color_components = self.unzip_all(dict(all=components.view(self.n, -1, self.channels[k])))['scales']  # (ncomponents, ngaussians, 3)
                pca_color_components[:,self.masking.mouth()[self.tex_map_full]] = 0  # inplace operation

            
            values = (th.matmul(coeffs[k], self.scales[k] * components) + self.means[k]).reshape(B, -1, C)
            mod, name_mask = self.to_key_parts(k)
            if masking != None: # TODO: UNTESTED!
                if name_mask in masking["keys"]:
                    values = einops.repeat(self.means[k], '(n c) -> b n c', b=B, c=C)
                if "mask" in masking and len(self.masks) == 1:
                    values = einops.rearrange(values, 'b n c -> n b c')  
                    values[masking["mask"]] = einops.repeat(self.means[k].reshape(-1, C)[masking["mask"]], 'n c -> n b c', b=B)
                    values = einops.rearrange(values, 'n b c -> b n c')

            results[mod].append(values)

        for m in self.masks:            
            results["tex_map"].append(getattr(self, f"tex_map_{m}"))
            if not self.dynamic_color:
                if not 'colors' in results:
                    results['colors'] = list()
                results["colors"].append(getattr(self, f"colors_{m}"))

        results = {k: th.cat(v, dim=-1).contiguous() for k, v in results.items()}
        results = self.unzip_all(results)
        results['apperance'] = results.pop('colors')

        # results["rotation"] = axis_angle_to_quaternion(results["rotation"])
        if not is_batch:
            results = dict([(k, v.squeeze(0) if k != 'tex_map' else v) for k, v in results.items()])

        return AttrDict(results)

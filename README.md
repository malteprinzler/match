# MATCH 
Official repository for the paper

**Feed-forward Gaussian Registration for Head Avatar Creation and Editing**, ***CVPR 2026***.

<a href="https://malteprinzler.github.io/" target="_blank">Malte Prinzler</a><sup>1,2</sup>, <a href="https://www.paulogotardo.com/" target="_blank">Paulo Gotardo</a><sup>2</sup>, <a href="https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html/" target="_blank">Siyu Tang</a><sup>1</sup>, <a href="https://sites.google.com/site/bolkartt/" target="_blank">Timo Bolkart</a><sup>2</sup>

<sup>1</sup>ETH Zürich, <sup>2</sup>  
Work partially done while Malte was an intern at Google.

<a href='http://arxiv.org/abs/2603.15811'><img src='https://img.shields.io/badge/arXiv-2603.15811-red'></a> <a href='https://malteprinzler.github.io/projects/match/'><img src='https://img.shields.io/badge/project page-MATCH-Green'></a> <a href='#citation'><img src='https://img.shields.io/badge/cite-blue'></a>

https://github.com/user-attachments/assets/a95b154e-5a41-4315-802c-5b2a3d2555ac

TL;DR:  Given calibrated multi-view images, MATCH infers static Gaussian splat textures in 0.5 seconds. The resulting Gaussians are in dense semantic correspondence across subjects and expressions. This allows for various applications such as editing, expression transfer, and fast avatar optimization. 

## ⚡️ Quick start guide
### 🛠️ 1. Create conda environment and install requirements

```bash
# 1. Clone repo
git clone https://github.com/malteprinzler/match/
cd match

# 2. Create conda environment for MATCH:
conda create -f environment.yaml
conda activate match

# 3. Set correct conda environment variables
conda env config vars set CUDA_HOME=<path-to-your-local-cuda12.6-installation>
conda deactivate ; conda activate match

# 4. Install requirements that need more love
pip install --no-build-isolation 'git+https://github.com/mattloper/chumpy@51d5afd92a8ded3637553be8cef41f328a1c863a'
pip install --no-build-isolation 'git+https://github.com/malteprinzler/gsplat_2dgs_sm90.git'
pip install --no-build-isolation 'git+https://github.com/malteprinzler/dqtorch.git'
pip install --no-build-isolation 'git+https://github.com/rahul-goel/fused-ssim/@031f321343102d542efc64f8161f11406d78e9d0'
GCC=gcc-9 CC=gcc-9 CXX=g++-9 pip install --no-build-isolation 'third_party/GEM/submodules/diff-gaussian-rasterization'
pip install --no-build-isolation 'third_party/GEM/submodules/simple-knn'
pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47'
pip install git+https://github.com/MarilynKeller/mesh
pip install opendr==0.78 --no-build-isolation
cd third_party/TEMPEH/modules/liegroups && python setup.py install && cd -
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.1_cu126.html
pip install third_party/pyrender
cd third_party/GEM/styleunet/stylegan2_ops/; python setup.py install ; cd -
pip install 'git+https://github.com/MarilynKeller/mesh@e8b04be3aaa1d262a3b7a05ef9e4085b4836b049'
pip install 'mediapipe>=0.10.30' 'protobuf>=5.28,<6'
pip install numpy==2.2.6


# 5. register match as package
pip install -e match


```

### 📥 2. Download assets
Download the assets from https://drive.google.com/drive/folders/1Lq5gvGe6MHHlfSSafKmrieztskAc3-bB?usp=sharing and unzip them under `./assets/`


### 🎬 3. Download example data from Ava256
run `condor/run.py configs/data/download_ava256_quickstart.gin`
this will download a small subset of the Ava-256 dataset to `data/ava-256`

### 🌐 4. Infer coarse mesh
run `condor/run.py configs/tempeh/save_predictions_quickstart.yml`
this will save the coarse mesh and uv renderings to `data/tempeh_predictions/ava-256/quickstart`

### 🚀 5. Inferring MATCH
run `condor/run.py configs/match/save_camera_sweeps_quickstart.gin`
this will infer MATCH and save the results to `experiments/match/quickstart`


## Driving a pretrained MATCH-based GEM avatar with a monocular video
- download the example data from https://drive.google.com/drive/folders/1lbT4Rph91dKs2St0VpJWlNKSqNOthFDe?usp=sharing and place it under `./example_data`
- run `condor/run.py configs/GEM/cross_eval_APP152.yml`. This will create the reenacted videos under `experiments/GEM/avatars/UNION_FINAL_APP152/FINAL_regressor_pcacomponents150/val_predictions`


## Creating a MATCH-based GEM avatar from multi-view videos
We demonstrate how a set of registered Gaussian splat textures of a person predicted by MATCH can be leveraged to create lightweight animatable head avatars. 

- Download the example data by running `condor/run.py configs/data/download_ava256_GEM.gin`
- Infer the coarse mesh initialization on the data: `condor/run.py configs/tempeh/save_predictions_GEM.yml`
- If you haven't done so yet, create a separate `sapiens_lite` conda environment by running `source scripts/sapiens/install_environment.sh`.
- activate the `sapiens_lite` environment
- create the sapiens segmentation masks by running `scripts/sapiens/seg_ava256.sh data/ava-256 data/sapiens_segmentations/ava-256/framestride_1 1 "APP152 PGO261"`
- MATCH prediction: To predict textures of Gaussian splats for a particular subject that can then be converted into an animatable avatar, please run `condor/run.py configs/match/save_GEM_dataset.gin`. By default, this will store the results under `experiments/match/GEM_datasets`
- Then run `condor/run.py configs/GEM/combine_GEM_datasets.gin`. This creates meta-data to combine the predictions on different sequences into one dataset which is stored under `match/GEM_datasets/<subject_id>/UNION_GEM_<subject_id>`.

- Fit FLAME against registrations: `condor/run.py configs/GEM/fit_flame_APP152.yml`

- Compute PCA over the registered Gaussians: `condor/run.py configs/GEM/pca_gauss_APP152.yml`

- Predict expression features for monocular driving for both target and driving subjects: 
    `condor/run.py configs/GEM/predict_face_features_APP152.yml`
    `condor/run.py configs/GEM/predict_face_features_PGO261.yml`

- Launch coefficient regressor training: `condor/run.py configs/GEM/train_avatar_APP152.yml`



## Training TEMPEH
- Download the entire ava-256 dataset using `condor/run.py configs/data/download_ava256_full.gin`
- Start tempeh training by running `condor/run.py configs/tempeh/train_tempeh.yaml`

## Training MATCH
Coming soon...
<!-- - Make sure you downloaded the entire ava-256 dataset as described [here](#training-tempeh).
- Download the NeRSemble dataset following the instructions from the [official repository](https://github.com/tobias-kirschstein/nersemble-data). For all further steps, we assume that the nersemble dataset root is `data/nersemble/`
-  -->



<!-- 
### Data preprocessing
#### Sapiens segmentation: 
We require sapiens segmentation masks for training. 
- Please follow the instructions <a href='https://github.com/facebookresearch/sapiens/blob/main/lite/README.md#-installation'>here</a> to set up the `sapiens_lite` conda environment.
- Download the pretrained segmentation weights for the bfloat16 version of the `sapiens-seg-1b` model from <a href='https://huggingface.co/facebook/sapiens'>here</a> and place it under `assets/sapiens/checkpoints/bfloat16/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_bfloat16.pt2`
- edit `configs/sapiens/segment_ava256.gin` according to your needs
- run `condor/run.py configs/sapiens/segment_ava256.gin` which will start the semantic segmentation.  -->

## License
MATCH is distributed under the MIT license. However, it builds on the third party software [GEM](https://github.com/Zielon/GEM/blob/main/LICENSE), [pyrender](https://github.com/mmatl/pyrender/blob/master/LICENSE), [sapiens](https://github.com/facebookresearch/sapiens/blob/main/LICENSE), and [TEMPEH](https://github.com/TimoBolkart/TEMPEH/blob/main/LICENSE) for which different licenses may apply. If you use their code, please make sure to obey the respective licenses. 

## 📖 Citation

```tex
@inproceedings{prinzler2026match,
  title={Feed-forward Gaussian Registration for Head Avatar Creation and Editing},
  author={Prinzler, Malte and Gotardo, Paulo and Tang, Siyu and Bolkart, Timo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```



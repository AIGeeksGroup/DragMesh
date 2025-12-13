# <img src="./assets/dragmesh_logo.png" alt="DragMesh logo" width="60"/> DragMesh: Interactive 3D Generation Made Easy

Official repository for the paper 
> **DragMesh: Interactive 3D Generation Made Easy**. 
>  
> [Tianshan Zhang](https://neptune-t.github.io/aca-web-html/)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*†, [Hao Tang](https://ha0tang.github.io/)<sup>#</sup>
> 
> \*Equal contribution. †Project lead. <sup>#</sup>Corresponding author. 
> 
> ### [Paper](https://www.arxiv.org/abs/2512.06424) | [Website](https://aigeeksgroup.github.io/DragMesh) | [Models](https://huggingface.co/AIGeeksGroup/DragMesh)

> [!NOTE]  
>  GAPartNet (link above) is the canonical dataset source for all articulated assets used in DragMesh.  



https://github.com/user-attachments/assets/428b0d36-50ab-4b46-ab17-679ad22c826b



## 🧾 Citation
If you find DragMesh helpful, please cite:
```bibtex
@article{zhang2025dragmesh,
  title={DragMesh: Interactive 3D Generation Made Easy},
  author={Zhang, Tianshan and Zhang, Zeyu and Tang, Hao},
  journal={arXiv preprint arXiv:2512.06424},
  year={2025}
}
```

---

## ✨ Intro

While generative models have excelled at creating static 3D content, the pursuit of systems that understand how objects move and respond to interactions remains a fundamental challenge. Current methods for articulated motion lie at a crossroads: they are either physically consistent but too slow for real-time use, or generative but violate basic kinematic constraints. We present DragMesh, a robust framework for real-time interactive 3D articulation built around a lightweight motion generation core. Our core contribution is a novel decoupled kinematic reasoning and motion generation framework. First, we infer the latent joint parameters by decoupling semantic intent reasoning (which determines the joint type) from geometric regression (which determines the axis and origin using our Kinematics Prediction Network (KPP-Net)). Second, to leverage the compact, continuous, and singularity-free properties of dual quaternions for representing rigid body motion, we develop a novel Dual Quaternion VAE (DQ-VAE). This DQ-VAE receives these predicted priors, along with the original user drag, to generate a complete, plausible motion trajectory. To ensure strict adherence to kinematics, we inject the joint priors at every layer of the DQ-VAE's non-autoregressive Transformer decoder using FiLM (Feature-wise Linear Modulation) conditioning. This persistent, multi-scale guidance is complemented by a numerically-stable cross-product loss to guarantee axis alignment. This decoupled design allows DragMesh to achieve real-time performance and enables plausible, generative articulation on novel objects without retraining, offering a practical step toward generative 3D intelligence.


## 📰 News


## ✅ TODO
- [x] Upload the DragMesh paper and project page.
- [x] Release the training and inference code.
- [x] Provide GAPartNet processing pipeline and LMDB builder.
- [x] Share checkpoints on Hugging Face.
- [ ] Create an interactive presentation.
- [ ] Publish a Hugging Face Space for browser-based manipulation.

## ⚡ Quick Start
### 🧩 Environment Setup
We ship a full Conda specification in `environment.yml` (environment name: `dragmesh`). It targets Python 3.10, CUDA 12.1, and PyTorch 2.4.1. Create or update via:
```bash
conda env create -f environment.yml
conda activate dragmesh
# or update an existing env
conda env update -f environment.yml --prune
```

> The spec already installs trimesh, pyrender, pygltflib, viser, Objaverse, SAPIEN, pytorch3d, and tiny-cuda-nn. If you prefer a minimal setup, install those packages manually before running the scripts.

### 🛠️ Native Extensions
Chamfer distance kernels are required for the VAE loss. Clone and build the upstream project:
```bash
git clone https://github.com/ThibaultGROUEIX/ChamferDistancePytorch.git
cd ChamferDistancePytorch
python setup.py install
cd ..
```

## 📦 Data Preparation (GAPartNet)
1. Visit https://pku-epic.github.io/GAPartNet/ and download the articulated assets for the categories listed in `config/category_split_v2.json`.  
2. Arrange files so that each object folder contains `mobility_annotation_gapartnet.urdf`, `meta.json`, and textured meshes (`*.obj`). Example:
   ```
   data/gapartnet/<object_id>/
     |- mobility_annotation_gapartnet.urdf
     |- meta.json
     |- textured_objs/*.obj
   ```
3. Convert to LMDB for fast training IO:
   ```bash
   python utils/build_lmdb.py \
     --dataset_root data/gapartnet \
     --output_prefix data/dragmesh \
     --config config/category_split_v2.json \
     --num_frames 16 \
     --num_points 4096
   # Produces data/dragmesh_train.lmdb and data/dragmesh_val.lmdb
   ```
4. Use `utils/balanced_dataset_utils.get_motion_type_weights` with `WeightedRandomSampler` if you need balanced revolute/prismatic sampling.

## 🧠 Training
### Dual Quaternion VAE
```bash
python scripts/train_vae_v2.py \
  --lmdb_train_path data/dragmesh_train.lmdb \
  --lmdb_val_path data/dragmesh_val.lmdb \
  --data_split_json_path config/category_split_v2.json \
  --output_dir outputs/vae \
  --num_epochs 300 \
  --batch_size 16 \
  --latent_dim 256 \
  --num_frames 16 \
  --mesh_recon_weight 10.0 \
  --cd_weight 30.0 \
  --kl_weight 0.001 \
  --kl_anneal_epochs 80 \
  --use_tensorboard --use_wandb
```

### Kinematics Prediction Pipeline (KPP-Net)
```bash
python scripts/train_predictor.py \
  --lmdb_train_path data/dragmesh_train.lmdb \
  --lmdb_val_path data/dragmesh_val.lmdb \
  --data_split_json_path config/category_split_v2.json \
  --output_dir outputs/kpp \
  --batch_size 32 \
  --num_epochs 200 \
  --encoder_type attention \
  --head_type decoupled \
  --predict_type True
```

Both scripts log to TensorBoard and optionally Weights & Biases. Check `modules/loss.py` and `modules/predictor_loss.py` for objective details.

## 🧪 Inference
### Batch Sweep (dataset mode)
```bash
python inference_animation.py \
  --dataset_root data/gapartnet \
  --checkpoint best_model.pth \
  --sample_id 40261 \
  --output_dir results_deterministic \
  --num_samples 5 \
  --num_frames 16
```
Outputs MP4, GIF, and animated GLB per object. If you plan to process a large dataset using dual-quaternion ground truth (no manual drags), prefer this script because running only KPP predictions frame-by-frame may introduce cumulative drift that eventually breaks physical alignment.

### Custom Mesh Manipulation (manual input)
```bash
python inference_pipeline.py \
  --mesh_file assets/cabinet.obj \
  --mask_file assets/cabinet_vertex_labels.npy \
  --mask_format vertex \
  --drag_point 0.12,0.48,0.05 \   # example: x,y,z point on the movable part
  --drag_vector 0.0,0.0,0.2 \     # example: direction+magnitude of the drag
  --manual_joint_type revolute \
  --kpp_checkpoint best_model_kpp.pth \
  --vae_checkpoint best_model.pth \
  --output_dir outputs/cabinet_demo \
  --num_samples 3
```
Supply drag points/vectors directly through the CLI (no viewer UI). Use `--manual_joint_type revolute` or `--manual_joint_type prismatic` to force a specific motion family when needed. If you omit the manual override, the pipeline first trusts KPP-Net and, when `--llm_endpoint` + `--llm_api_key` are provided, backs off to the LLM-based classifier described in `inference_pipeline.py`. Outputs share the same MP4/GIF/GLB format as the batch pipeline.

## 👀 Visualization
- GIF/MP4 exports rely on `pyrender` and `imageio`. For headless servers, set `PYOPENGL_PLATFORM=osmesa`.  
- `inference_animation.py` also exports animated GLB files for direct use in GLTF viewers.  
- For additional visualization tooling (e.g., rerun or Blender scripts), see `inference_animation.py` and `inference_pipeline.py`.

## 👩‍💻 Case Study
| Scenario | Description |
| --- | --- |
| Drawer opening | Translational motion predicted entirely from drag cues. |
| Microwave door | Revolute joint inference with FiLM conditioned motion generation. |
| Bucket handle | High curvature rotations showing the benefit of dual quaternions. |

## 🎬 Demo Gallery

**Translational drags**
| | | |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/18def78b-8e90-45c1-8f87-11a07af8aab2" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/3bc592cb-b9c7-4416-9784-8641b6100767" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/f3364136-2e41-459b-82b6-b3a6787c9390" controls width="260"></video> |
| <video src="https://github.com/user-attachments/assets/ee4d749e-00fa-4de8-80c0-4c2a30e4f308" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/f082760d-9f20-4819-b7fe-6f45ae501103" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/7efac575-19e4-4265-8f77-e7cff4fdad0f" controls width="260"></video> |

**Rotational drags**
| | | |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/883c5a47-4c7a-4cfd-9df4-ff9b278e1e2b" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/d4608ba9-6055-4a5f-9f63-e8c427c64301" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/4263d858-c470-4e2d-b32b-b3e2570916ab" controls width="260"></video> |
| <video src="https://github.com/user-attachments/assets/25ed4dc5-40f7-46c8-809f-d921788187ac" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/d7c8682b-8d05-42f8-80e8-6cfee6ed09f9" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/d976a523-c07d-44d0-85ea-9b4da1b57b94" controls width="260"></video> |

**Self-spin / free-spin**
| | | |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/91d2e288-8f87-40be-8332-018a13ab6e8e" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/b8af10dd-d6cf-4360-bd63-3a322be3a505" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/0a4bcdce-3c31-4d29-82ac-fc693680dce2" controls width="260"></video> |
| <video src="https://github.com/user-attachments/assets/7332d57e-4a47-45d6-b1a0-7d353188533e" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/0d06f7f2-2e42-4358-8975-00ea32a9d70e" controls width="260"></video> | <video src="https://github.com/user-attachments/assets/07acbc6d-7a95-4ed2-9495-6eef86dc7b73" controls width="260"></video> |

## 🗂️ Repository Tour
| Path | Content |
| --- | --- |
| `modules/model_v2.py` | Dual Quaternion VAE (encoder, decoder, FiLM Transformer). |
| `modules/predictor.py` | KPP-Net architecture. |
| `modules/data_loader_v2.py` | GAPartNet parsing and dual quaternion labels. |
| `utils/balanced_dataset_utils.py` | LMDB dataset builder and balanced sampling utilities. |
| `scripts/train_vae_v2.py`, `scripts/train_predictor.py` | Training entry points. |
| `inference_animation*.py`, `inference_pipeline.py` | Inference pipelines (batch and interactive). |
| `ChamferDistancePytorch/` | CUDA kernels for Chamfer distance and auxiliary metrics. |

### 🌳 Project Tree (annotated)
```
DragMesh/
├── assets/                      # Logos, teaser figures, future demo media
│   ├── dragmesh_logo.png
│   └── teaser.png
checkpoints/                
│   ├── dqvae.pth             
│   └── kpp.pth
├── ChamferDistancePytorch/      # CUDA/C++ Chamfer distance implementation (build with setup.py)
├── config/
│   └── category_split_v2.json   # GAPartNet in-domain split definition
├── modules/
│   ├── model_v2.py              # Dual Quaternion VAE architecture
│   ├── predictor.py             # KPP-Net for kinematic reasoning
│   ├── loss.py                  # VAE objectives (Chamfer, dual quaternions, constraints)
│   ├── predictor_loss.py        # Loss terms for KPP-Net
│   └── data_loader_v2.py        # GAPartNet loader + dual quaternion ground truth builder
├── scripts/
│   ├── train_vae_v2.py          # Training loop for the VAE motion prior
│   └── train_predictor.py       # Training loop for KPP-Net
├── utils/
│   ├── balanced_dataset_utils.py # LMDB dataset class + balanced sampling helper
│   ├── dataset_utils.py          # Category-aware dataset wrappers
│   └── build_lmdb.py             # CLI to build LMDBs from GAPartNet folders
├── partnet/
│   └── Hunyuan3D-Part/           # External resources (P3-SAM, XPart docs)
├── results_deterministic/        # Placeholder for inference outputs (MP4/GIF/GLB)
├── inference_animation.py        # Batch evaluation + GLB export
├── inference_animation_kpp.py    # Dataset-driven animation tests (legacy interface)
├── inference_pipeline.py         # Interactive mesh manipulation pipeline
├── environment.yml               # Conda environment (name: dragmesh)
├── README.md                     
```

## 🙏 Acknowledgement
We thank the GAPartNet team for the articulated dataset, and upstream projects such as ChamferDistancePytorch, Objaverse, SAPIEN, and PyTorch3D for their open-source contributions.

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=AIGeeksGroup/DragMesh&type=Date)](https://www.star-history.com/#AIGeeksGroup/DragMesh&Date)

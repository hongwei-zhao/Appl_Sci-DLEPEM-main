# Dynamic LoRA-Experts and Prototype-Ensemble Matching (DLEPEM) for Class-Incremental Learning

[![Framework](https://img.shields.io/badge/PyTorch-Methodology-red.svg)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official PyTorch implementation for **DLEPEM**: *Dynamic LoRA-Experts and Prototype-Ensemble Matching for Class-Incremental Learning*.

DLEPEM provides an innovative parameter-efficient framework that tackles the **Stability-Plasticity Intertwinement** in class-incremental learning (CIL). By introducing MoE-style dynamic allocation of LoRA-Experts with an ensemble prototyping match, it isolates representation disruptions across subsequent tasks, successfully sidestepping catastrophic forgetting and achieving state-of-the-art results on both standard CIL and few-shot CIL (FSCIL) paradigms.

<div align=center><img src="https://markdownimg-hw.oss-cn-beijing.aliyuncs.com/20251009102328.png" style="zoom: 50%;" /></div>

---

## 🔥 Key Contributions

1. **Dynamic LoRA-Expert**: Dynamically instantiates a task-specific low-rank (LoRA) expert for each new task, freezing prior ones. It operates on *Vision Transformers (ViT)* as `MLP-Experts` or `QV-Experts`, naturally eliminating inter-task interference and enforcing strict stability without suppressing plasticity.
2. **Prototype-Ensemble Matching**: Simultaneously leverages the frozen backbone's robust generalized pre-trained representations and the dynamic router's domain-specific characteristics to perform an ensemble match, guaranteeing accurate Module-Identity Inference (MII).
3. **Rehearsal-Free**: Eradicates the need for any storage of old data samples, averting critical privacy concerns or huge deployment resource demands.

---

## 🌟 Dependencies

1. `torch >= 2.0.1`
2. `torchvision >= 0.15.2`
3. `timm == 0.6.12`
4. `tqdm`, `numpy`, `scipy`, `easydict`

*Install packages via pip:*
```bash
pip install torch==2.0.1 torchvision==0.15.2 timm==0.6.12 tqdm numpy scipy easydict
```

---

## 🔑 Running Experiments

The execution flow relies on JSON configuration files for managing all hyperparameters and routing mechanics dynamically.

### 1. Configuration Check
Edit corresponding configuration files in the `scripts/` folder (e.g., `scripts/dlepem_cub_B0_Inc20.json`) prior to starting exactly as needed. 

Key hyperparameters configurable via `scripts/*.json`:
- **`init_cls`**: Classes allocated for the base initialization phase. 

- **`increment`**: The class incremental step payload per subsequent task phase.

- **`backbone_type`**: Designates ViT variants configured with DLEPEM. (*e.g., `vit_base_patch16_224_in21k_dlepem`*)

- **`lora_positions`**: Controls Expert type (`["q","v"]` forms the highly capable **QV-Expert**, whereas `["mlp"]` forms the **MLP-Expert**).

- **`router_train_method`**: Distinguishes the router execution policy. DLEPEM implements the highly capable `prototype_ensemble` router algorithm alongside dual knowledge distillation (`Plasticity Feature Distillation` and `Stability Feature Distillation`).

- **`alpha`**: The distillation coefficient balancing the Plasticity Feature Distillation (PFD) and Stability Feature Distillation (SFD) losses (default: `0.04`).

- **`router_epochs`** & **`router_lr`**: Iterations and learning rate dedicated to training the Router Expert stage independently from the main task.

- **`rank`**: The bottleneck dimensionality $r$ for the Low-Rank Adaptation (LoRA) matrices.

  > **Note regarding Multi-GPU support:** While the codebase references `DistributedDataParallel`, full DDP support is currently experimental. Executing on single GPUs (via specifying one device ID in the `"device"` field of the config) is recommended for stable reproducibility.

### 2. Executing

Run the model from the project root utilizing the target `[JSON Configuration]`:

```bash
# Example Run with CUB200 configurations for DLEPEM-QV
python main.py --config=./scripts/dlepem_cub_B0_Inc20.json

# Example Run with CIFAR100 configurations 
python main.py --config=./scripts/dlepem_cifar_B0_Inc10.json
```

---

## 🔎 Datasets

We support 5 established metrics out of the box. Implementations map to standard incremental preprocessing settings.
- **CIFAR100**: will be automatically downloaded by the code.
- **CUB200**:  Google Drive: [link](https://drive.google.com/file/d/1XbUpnWpJPnItt5zQ6sHJnsjPncnNLvWb/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EVV4pT9VJ9pBrVs2x0lcwd0BlVQCtSrdbLVfhuajMry-lA?e=L6Wjsc)
- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)
- **OmniBenchmark**: Google Drive: [link](https://drive.google.com/file/d/1AbCP3zBMtv_TDXJypOCnOgX8hJmvJm3u/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EcoUATKl24JFo3jBMnTV2WcBwkuyBH0TmCAy6Lml1gOHJA?e=eCNcoA)
- **VTAB**: Google Drive: [link](https://drive.google.com/file/d/1xUiwlnx4k0oDhYi26KL5KwrCAya-mvJ_/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EQyTP1nOIH5PrfhXtpPgKQ8BlEFW2Erda1t7Kdi3Al-ePw?e=Yt4RnV)

When utilizing custom or downloaded datasets, specify the absolute folder trajectory mapping manually internally:
```python
# In `utils/data.py`
def download_data(self):
    train_dir = '[YOUR_DATA_PATH]/train/'
    test_dir = '[YOUR_DATA_PATH]/val/'
```

---

## 👨‍🏫 Acknowledgments

This codebase was developed with inspiration and structures adapted from:
- [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)

We express our immense gratitude to developers across the Class Incremental Learning paradigms for sharing foundational architectures.

## ACCUP: Test-Time Adaptation for Time Series

**Augmented Contrastive Clustering with Uncertainty-Aware Prototyping (PyTorch)**

> Implementation and practical notes for test-time adaptation (TTA) on time-series classification tasks (e.g., EEG/FD/HAR).

---

## Table of Contents

* [Overview](#overview)
* [Key Highlights](#key-highlights)
* [Repository Layout](#repository-layout)
* [Requirements](#requirements)
* [Setup](#setup)
* [Data Preparation](#data-preparation)
* [Quick Start](#quick-start)

  * [A. Pretrain the Source Model](#a-pretrain-the-source-model)
  * [B. Test-Time Adaptation with ACCUP](#b-test-time-adaptation-with-accup)
* [Important Hyperparameters](#important-hyperparameters)
* [Tips & FAQ](#tips--faq)
* [Citation](#citation)
* [License](#license)
* [Contributing](#contributing)

---

## Overview

**ACCUP** adapts a pre-trained source model at **test time** without target labels. The method:

1. Generates **augmented views** (e.g., magnitude warp) and **ensembles** features/logits for stability.
2. Maintains a **memory of support samples** and selects **low-entropy** examples per class to form **uncertainty-aware prototypes**.
3. For each test sample, compares **prototype-based prediction entropy** vs **augmented-ensemble entropy** and **picks the lower-entropy** one as the pseudo label source.
4. Applies a **supervised contrastive loss (SupCon)** across multi-views (raw/aug/average) to encourage **intra-class compactness** and **inter-class separation**.
5. **Updates only** BN parameters and a small subset of convolution layers for **stable online TTA**.

---

## Key Highlights

* **Prototype reliability** via per-class **Top-K low-entropy supports**.
* **Entropy-wise selection** between prototype prediction and augmented-ensemble prediction.
* **Multi-view SupCon** across raw/aug/averaged representations.
* **Low intrusion**: no target labels; minimal parameters updated.

---

## Repository Layout

```
.
├─ algorithms/
│  ├─ accup.py                 # ACCUP main algorithm
│  └─ base_tta_algorithm.py    # Base class for TTA + softmax-entropy utility
├─ configs/
│  ├─ data_model_configs.py    # Dataset/model shapes and config (EEG/FD/HAR)
│  └─ tta_hparams_new.py       # Hyperparameters (training + ACCUP per dataset)
├─ dataloader/
│  ├─ augmentations.py         # Magnitude warp & other (optional) augmentations
│  ├─ dataloader.py            # Plain dataloader (raw view)
│  └─ demo_dataloader.py       # Augmented dataloader: returns (raw, aug, aug2)
├─ loss/
│  └─ sup_contrast_loss.py     # Supervised contrastive loss (SupCon)
├─ models/
│  ├─ da_models.py             # CNN backbone + linear classifier
│  └─ loss.py                  # Label smoothing CE, conditional entropy, etc.
├─ pre_train_model/
│  └─ pre_train_model.py       # PreTrainModel wrapper (feature_extractor + classifier)
├─ notebooks/
│  └─ ACCUP_codecells.ipynb    # Annotated notebook version
└─ README.md / requirements.txt / LICENSE / ...
```

> Names may vary slightly in your repo; adjust imports accordingly.

---

## Requirements

* Python ≥ 3.9
* PyTorch ≥ 1.12 (match your CUDA)
* scikit-learn (metrics)
* SciPy (spline for magnitude warp)
* torchvision (handy utilities)

Install:

```bash
pip install -r requirements.txt
# If you don't have a requirements file:
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn
```

---

## Data Preparation

Expected structure (example):

```
data/<DATASET>/<SCENARIO>/
  ├─ train_<domain_id>.pt
  └─ test_<domain_id>.pt
```

Each `.pt` should contain at least:

* `"samples"`: Tensor/ndarray, shape normalized to `(N, C, L)` internally.
* `"labels"`: Tensor/ndarray (optional for TTA, used for evaluation).

**Normalization**:

* `Load_Dataset` computes per-channel mean/std and standardizes the raw view.
* The augmented loader (`demo_dataloader.py`) **standardizes raw and aug views separately** to avoid distribution leakage.

---

## Quick Start

### A. Pretrain the Source Model

Supervised training on source domains with **Label Smoothing Cross-Entropy**:

```python
from configs.data_model_configs import EEG as DatasetCfg   # e.g., EEG
from configs.tta_hparams_new import EEG as HpCfg
from dataloader.dataloader import data_generator
from models.da_models import get_backbone_class
from pre_train_model.pre_train_model import PreTrainModel
from your_module.pretrain import pre_train_model  # or wherever your function lives

dataset_cfg = DatasetCfg()
hp_all = HpCfg()
train_hp = hp_all.train_params
alg_hp   = hp_all.alg_hparams['ACCUP']  # contains pre_learning_rate, etc.

backbone = get_backbone_class('CNN')
src_dl = data_generator(
    data_path="data/EEG/0_11", domain_id="0",
    dataset_configs=dataset_cfg, hparams=train_hp, dtype="train"
)

# Minimal metric helper (replace with your own)
class Meter:
    def __init__(self): self.sum=0; self.cnt=0; self.avg=0
    def update(self, v, n): self.sum+=v*n; self.cnt+=n; self.avg=self.sum/self.cnt
avg_meter = {'Src_cls_loss': Meter()}

import logging
logger = logging.getLogger("pretrain"); logger.setLevel(logging.DEBUG)

device = "cuda:0"
src_state_dict, pretrained_model = pre_train_model(
    backbone=backbone, configs=dataset_cfg, hparams={**train_hp, **alg_hp},
    src_dataloader=src_dl, avg_meter=avg_meter, logger=logger, device=device
)
```

Returns:

* `src_state_dict`: convenient checkpoint (`state_dict`) of the source-only model.
* `pretrained_model`: full `nn.Module` with `feature_extractor` + `classifier`.

### B. Test-Time Adaptation with ACCUP

Use the augmented dataloader and the ACCUP algorithm:

```python
from dataloader.demo_dataloader import data_generator_demo
from algorithms.accup import ACCUP
import torch.optim as optim
import torch

device = "cuda:0"

trg_dl = data_generator_demo(
    data_path="data/EEG/0_11", domain_id="11",  # example: adapt from domain 0 -> 11
    dataset_configs=dataset_cfg, hparams=train_hp, dtype="test"
)

# Initialize ACCUP with the pretrained model.
tta = ACCUP(
    configs=dataset_cfg,
    hparams={**train_hp, **alg_hp},
    model=pretrained_model,
    optimizer=lambda params: optim.Adam(params, lr=alg_hp['learning_rate'])
).to(device)

# Online TTA: infer + adapt on-the-fly
for (raw, aug, _), _, _ in trg_dl:
    raw = raw.float().to(device)
    aug = aug.float().to(device)
    preds = tta.forward((raw, aug))  # internally: ensemble -> prototype -> entropy pick -> SupCon step
    # TODO: record/evaluate preds as needed
```

What happens per batch:

* Extract features for raw & augmentation, **ensemble** them.
* Update **memory supports**, select per-class **Top-K low-entropy** supports to form prototypes.
* Compare prototype-based vs ensemble-based entropy; **choose lower** to create pseudo labels.
* Run **SupCon** across (raw, aug, averaged) views; **update only** BN and a few conv layers.

---

## Important Hyperparameters

Defined per dataset in `configs/tta_hparams_new.py`:

* `filter_K`: max low-entropy supports per class (`-1` = keep all).
* `tau`: temperature for prototype-similarity logits (larger = sharper).
* `temperature`: SupCon temperature (smaller = stronger separation).
* `pre_learning_rate` / `learning_rate`: source training / TTA learning rates.
* `steps`: TTA update steps per `forward` (usually `1`).

---

## Tips & FAQ

* **Batch size & BN**: very small batches can destabilize BN during TTA; prefer smaller LR or update only BN parameters.
* **Support memory growth**: by default we keep accumulating; consider a sliding window if drift is a concern.
* **Multiple augmentations**: current loop uses one augmentation for ensembling; you can add `aug2` for multi-view averaging/voting.
* **Loss mixing**: ACCUP uses SupCon as the main driver; you can optionally add conditional entropy minimization if your experiments call for it.
* **Notebook diffs**: strip outputs before committing (e.g., `nbstripout`) to keep clean PRs.
* **Large files**: track `.pt/.pth` with Git LFS or ignore them.

---

## Citation

If this repo helps your research, please cite the paper and this implementation:

```
@article{ACCUP2024,
  title   = {Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test-Time Adaptation},
  author  = {Authors},
  year    = {2024}
}

@misc{ACCUPCode,
  title        = {ACCUP PyTorch Implementation},
  howpublished = {\url{https://github.com/<your-username>/<repo-name>}}
}
```

---

## License

Research/replication use by default; see `LICENSE` for details (MIT or Apache-2.0 recommended).

---

## Contributing

PRs welcome! Ideas:

1. New backbones (TCN/ResNet/LSTM).
2. More augmentations and ensembling strategies.
3. Metrics & visualization (t-SNE/UMAP).
4. Additional datasets/config templates.

---

If you want, I can tailor the **paths/imports** in the Quick Start to match your exact repo structure — just share your tree (e.g., `tree -L 2`).

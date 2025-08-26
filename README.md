# SincPD (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V6HkhIrxScbJlf4KdTDLV_V4evYaQ8UN?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Status](https://img.shields.io/badge/status-active-green)

![Model](asset/Main_model_plot-1.png)

## 🔎 Overview
**SincPD** is an explainable deep learning model for **Parkinson’s Disease (PD) diagnosis** and **severity estimation** using **gait cycle analysis**.  
It leverages **SincNet filters** as adaptive bandpass filters to extract key frequency features from vertical Ground Reaction Force (vGRF) signals, providing **better interpretability** compared to conventional black-box neural networks.

- Paper: [arXiv preprint](https://arxiv.org/abs/2502.17463)
- Authors: Armin Salimi-Badr, Mahan Veisi, Sadra Berangi

---

## 🧩 Key Contributions
- **Adaptive Sinc Filters** for interpretable feature extraction from gait signals.  
- **Explainable AI** via clustering-based pruning (K-Means + silhouette).  
- **PD Diagnosis & Severity Estimation** with state-of-the-art accuracy.  
- **Reproducibility** ensured through Jupyter notebooks and modular Python package.  

---

## 🏗 Repository Structure
```text
SincPD/
├─ pyproject.toml             # Packaging (PEP 621)
├─ requirements.txt           # Dependencies
├─ Makefile                   # Common tasks (train, test, lint, ...)
├─ README.md
├─ data/                      # (empty) place PhysioNet gait dataset here
│   └─ README.md
├─ notebooks/                 # Reproducible experiments
│   ├─ 00_quickstart.ipynb
│   └─ SincNet_PD_Severity.ipynb
├─ scripts/                   # Helper scripts (demo, shell)
├─ src/sincpd/                # Source code (installable package)
│   ├─ models/                # SincConv + CNN head
│   ├─ data/                  # Datasets & transforms
│   ├─ training/              # Training loop & optimizer
│   ├─ pruning/               # KMeans-based pruning
│   ├─ eval/                  # Metrics & evaluation
│   ├─ utils/                 # Seed, IO, plotting
│   ├─ configs/               # YAML configs
│   └─ cli.py                 # Command-line interface
└─ tests/                     # Pytest smoke-tests
```

---

## 📊 Performance (from paper)
- **98.77% accuracy** for PD vs. healthy classification  
- **97.22% accuracy** for PD severity prediction

---

## 📥 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/MahanVeisi8/SincPD.git
cd SincPD
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

---

## 📂 Dataset
We use the [PhysioNet Gait Database](https://physionet.org/content/gaitpdb/1.0.0/).  
Please download and place it under `data/physionet_gait/`.

---

## 🚀 Usage

### Quickstart (demo with synthetic data)
```bash
make demo
```
Trains a tiny model on a dummy dataset and saves a checkpoint to `runs/demo/`.

### Training on real data
```bash
sincpd train --config src/sincpd/configs/default.yaml
```

### Pruning filters
```bash
sincpd prune --ckpt runs/diag_default/model.pt
```

### Evaluate a checkpoint
```bash
sincpd eval --ckpt runs/diag_default/model.pt --split val
```

---

## 📓 Reproducibility
- All experiments are available as Jupyter notebooks (`notebooks/`).  
- Notebooks import from the `src/` package to ensure consistency.  
- Seeds are fixed via `sincpd.utils.seed` for reproducible results.  

---

## 📈 Roadmap
- [ ] Release pretrained models.  
- [ ] Add support for severity regression tasks.  
- [ ] Extend to other biosignal domains (EEG, protein sequences).  

---

## 👥 Contributors
- **Mahan Veisi**
- **Sadra Berangi**
- **Armin Salimi-Badr**

---

## 📜 Citation
```bibtex
@misc{salimibadr2025sincpdexplainablemethodbased,
  title={SincPD: An Explainable Method based on Sinc Filters to Diagnose Parkinson's Disease Severity by Gait Cycle Analysis}, 
  author={Armin Salimi-Badr and Mahan Veisi and Sadra Berangi},
  year={2025},
  eprint={2502.17463},
  archivePrefix={arXiv},
  primaryClass={eess.SP},
  url={https://arxiv.org/abs/2502.17463}
}
```

---

## ⚖️ License
MIT (see [LICENSE](./LICENSE))

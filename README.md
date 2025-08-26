# SincPD (PyTorch) — **Working README**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V6HkhIrxScbJlf4KdTDLV_V4evYaQ8UN?usp=sharing)

SincPD is an **explainable deep learning model** for **Parkinson’s Disease (PD) diagnosis** and **severity estimation** from **gait vGRF signals**.  
The repository provides **modular PyTorch code**, a **command-line interface (CLI)**, **reproducible notebooks**, YAML configs, and lightweight tests.

---

## 0) Requirements
- Python 3.10 or 3.11  
- pip (latest version recommended)  
- (Optional) Git and Make  

---

## 1) Clone and create virtual environment

### Linux / macOS
```bash
git clone https://github.com/MahanVeisi8/SincPD.git
cd SincPD
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -e .[dev]
```

### Windows (PowerShell)
```powershell
git clone https://github.com/MahanVeisi8/SincPD.git
cd SincPD
py -3 -m venv .venv
.\.venv\Scripts\activate
py -m pip install -U pip
py -m pip install -e .[dev]
```

---

## 2) Quick Demo (no real data required)

### Linux / macOS
```bash
python3 scripts/demo_quickstart.py
# Output: runs/demo/model.pt
```

### Windows
```powershell
py scripts\demo_quickstart.py
# Output: runs\demo\model.pt
```

This demo uses a dummy dataset to verify the installation.

---

## 3) Training / Evaluation / Pruning via CLI

### Linux / macOS
```bash
python3 -m sincpd.cli train --config src/sincpd/configs/default.yaml
python3 -m sincpd.cli eval --ckpt runs/diag_default/model.pt --split val
python3 -m sincpd.cli prune --ckpt runs/diag_default/model.pt
```

### Windows
```powershell
py -m sincpd.cli train --config src\sincpd\configs\default.yaml
py -m sincpd.cli eval --ckpt runs\diag_default\model.pt --split val
py -m sincpd.cli prune --ckpt runs\diag_default\model.pt
```

> Tip: Using `python -m sincpd.cli` ensures the CLI always works even if `sincpd` is not directly on PATH.

---

## 4) Repository Structure
```
SincPD/
  notebooks/              # Jupyter notebooks
  src/sincpd/             # Package source code
    models/               # SincConv + CNN head
    data/                 # datasets.py, transforms.py
    training/             # trainer.py, optim.py
    pruning/              # kmeans_prune.py
    eval/                 # metrics.py
    utils/                # seed.py, io.py, plotting.py
    configs/              # default.yaml, severity.yaml
    cli.py                # CLI entry point
  tests/                  # Pytest smoke tests
  scripts/                # demo_quickstart.py
  runs/                   # outputs (created after training)
```

---

## 5) Real Dataset
- Dataset: [PhysioNet Gait Database](https://physionet.org/content/gaitpdb/1.0.0/)  
- Place under: `data/physionet_gait/`  
- If the format differs, adapt `src/sincpd/data/datasets.py`.  

---

## 6) Testing & Code Quality
```bash
pytest -q
pre-commit run --all-files
```

---

## 7) Troubleshooting
- **`sincpd: command not found`** → use `python -m sincpd.cli ...` instead.  
- **`python: command not found`** → use `python3` (Linux/macOS) or `py` (Windows).  
- **PyTorch installation fails?**
  - CPU-only:
    ```bash
    python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```
    or on Windows:
    ```powershell
    py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

---

## 8) Citation
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

## 9) License
MIT (see [LICENSE](./LICENSE))

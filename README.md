# SincNet-PD-Diagnosis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1V6HkhIrxScbJlf4KdTDLV_V4evYaQ8UN?usp=sharing)
![Python](https://img.shields.io/badge/Python-3.8-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Status](https://img.shields.io/badge/status-active-green)

## Overview
This repository contains the implementation of an **explainable deep learning model** for diagnosing **Parkinson’s Disease (PD)** using **gait cycle analysis**. The proposed method is based on **SincNet filters**, which act as adaptive bandpass filters to extract key frequency features from gait signals. This approach ensures better interpretability of model decisions compared to traditional deep learning techniques.

## Paper Reference
**Title:** An Explainable Method based on Adaptive Sinc Filters to Diagnose Parkinson’s Disease Severity by Gait Cycle Analysis  
**Authors:** Armin Salimi-Badr, Mahan Veisi, Sadra Berangi  
**Status:** Under review at AIHC Journal  

## Key Contributions
- **Adaptive Sinc Filters:** Used for feature extraction from vertical Ground Reaction Force (vGRF) signals.
- **Explainable AI:** Filters are pruned using **K-means clustering** and **silhouette scores** to identify significant frequency bands.
- **PD Diagnosis & Severity Estimation:** Model predicts both **PD presence** and **severity levels** with high accuracy.
- **Preprocessing & Optimization:** vGRF signals are processed to remove noise, and redundant filters are pruned for efficiency.

## Dataset
The study utilizes publicly available gait datasets from **PhysioNet**:
- [PhysioNet Gait Dataset](https://physionet.org/content/gaitpdb/1.0.0/)

## Model Architecture
- **Preprocessing:** Standardization of gait cycle signals.
- **Feature Extraction:** SincNet layers extract important frequency bands.
- **Pruning & Optimization:** Clustering-based pruning to enhance interpretability.
- **Classification:** Fully connected layers for binary and multi-class PD classification.

## Performance
The model achieves:
- **98.77% accuracy** for PD classification.
- **97.22% accuracy** for severity prediction.

## Repository Structure (To be Updated)
```plaintext
├── data/            # Dataset (to be added)
├── models/          # Model implementations
├── notebooks/       # Jupyter notebooks for experiments
├── results/         # Evaluation metrics & visualizations
├── README.md        # Project documentation
```

## Installation & Usage
```bash
git clone https://github.com/YOUR_ORG/SincNet-PD-Diagnosis.git
cd SincNet-PD-Diagnosis
# Install dependencies (to be added later)
```

## Future Work
- Adding **pretrained models**.
- Implementing **generative modeling** for synthetic gait data.
- Expanding to **protein sequence analysis**.

## Contributors
- **Mahan Veisi**
- **Sadra Berangi**
- **Armin Salimi-Badr** 

## License
TBD (MIT/Apache 2.0 or based on journal requirements)

## Citation
If you use this work, please cite the corresponding paper (to be updated upon publication).

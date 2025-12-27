# Unifying VLM-Guided Flow Matching and Spectral Anomaly Detection for Interpretable Veterinary Diagnosis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Canine_Pneumothorax-blue.svg)](Data/)

> **Official implementation** of the paper: *"Unifying VLM-Guided Flow Matching and Spectral Anomaly Detection for Interpretable Veterinary Diagnosis"*.

## 📖 Abstract

Automatic diagnosis of canine pneumothorax is challenged by data scarcity and the need for trustworthy models. This paper introduces a novel diagnostic paradigm that reframes the task as a synergistic process of **signal localization** and **spectral detection**.

1.  **Localization:** We employ a Vision-Language Model (VLM) to guide an iterative **Flow Matching** process, progressively refining segmentation masks to achieve superior boundary accuracy.
2.  **Detection:** We apply **Random Matrix Theory (RMT)** to analyze features from the suspected lesion. Healthy tissue is modeled as predictable random noise, while pneumothorax is identified by detecting statistically significant outlier eigenvalues (non-random pathological signals).

This synergy of generative segmentation and first-principles statistical analysis yields a highly accurate and interpretable diagnostic system.

![Method Overview](assets/fig1.pdf)


## ✨ Key Features

* **Iterative Flow Matching:** Uses VLM guidance to refine segmentation masks step-by-step.
* **Spectral Anomaly Detection:** Leveraging Random Matrix Theory (RMT) for interpretable classification without reliance on black-box confidence scores.
* **New Dataset:** Introduces a pixel-level annotated dataset for canine pneumothorax.

## 📂 Dataset

We provide the **Canine Pneumothorax Dataset** with pixel-level annotations.

* **Download:** [Insert Link to Google Drive / HuggingFace / OneDrive]
* **Structure:**
    ```bash
    data/
    ├── images/
    │   ├── train/
    │   └── test/
    ├── masks/
    │   ├── train/
    │   └── test/
    └── annotations.json
    ```

## 🛠️ Installation

### Prerequisites
* Linux or macOS
* Python 3.8+
* PyTorch 1.13+
* CUDA 11.0+ (Recommended)

### Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/project-name.git](https://github.com/yourusername/project-name.git)
cd project-name

# Create a virtual environment
conda create -n pneumo_diagnosis python=3.9
conda activate pneumo_diagnosis

# Install dependencies
pip install -r requirements.txt

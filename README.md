# Chest X-Ray Classification using CNN

## link for database: https://drive.google.com/file/d/1CPZM7OmIJUaAJhAP460Epcfu2xonW4uA/view?usp=sharing

## Overview
This project aims to build a deep learning model capable of classifying chest X-ray images into three categories:  
**Normal**, **Pneumonia**, and **Tuberculosis**.  
The goal is to assist medical professionals in early diagnosis through automated image analysis.

---
## Team
Nurlan Marat, Aiym Tleuberdinova, Amina Rysbekova
---

## Project Workflow
### 1. Data Preparation
- Dataset organized into three splits: **train**, **validation**, and **test**.  
- Directory structure:
datasets/
├── train/
│ ├── Normal/
│ ├── Pneumonia/
│ └── Tuberculosis/
├── val/
│ ├── Normal/
│ ├── Pneumonia/
│ └── Tuberculosis/
└── test/
├── Normal/
├── Pneumonia/
└── Tuberculosis/
- A configuration file `data.yaml` contains dataset information and class names.
- Data augmentation (rotation, color jitter, cropping, flipping) was used to improve model generalization.

---

## 🧠 Model
- Implemented using **PyTorch** and **Torchvision**.
- CNN and transfer learning approaches (e.g., ResNet18) were tested.
- Fine-tuning was applied with weighted loss to handle class imbalance.

---

## Training Setup
- Optimizer: **Adam**
- Learning rate: 0.0003 → 0.0001 (for fine-tuning)
- Scheduler: **ReduceLROnPlateau**
- Loss: **Weighted CrossEntropyLoss**
- Batch size: 64 → 128 (tested for stability)
- Early stopping and checkpoint saving included.

---

## 📊 Results

| Class         | Precision | Recall | F1-score |
|----------------|------------|---------|----------|
| Normal         | 0.64 | 0.81 | 0.71 |   0.73
| Pneumonia      | 0.79 | 0.96 | 0.87 |   0.85       
| Tuberculosis   | 0.95 | 0.62 | 0.75 |   0.75 
| **Accuracy**   |     |     | **0.76** |

Best performance achieved after fine-tuning with class weights and reduced learning rate.  
Model showed high recall for Pneumonia and strong precision for Tuberculosis.

---

## Training Performance
- Early stopping used to avoid overfitting.
- Learning rate automatically reduced when validation loss plateaued.
- Checkpoints saved every 2 epochs.
- Final accuracy: **76%** (balanced performance across all classes).

---

## 📂 Project Structure
finalAML/
│
├── __pycache__/                # Python cache files
├── best_model/                 # Best model weights (.pth)
│   └── best23.pth
├── checkpoints/                # Intermediate checkpoints
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_20.pth
│   └── ...
├── datasets/                   # Dataset root directory
│   ├── train/
│   │   ├── Normal/
│   │   ├── Pneumonia/
│   │   └── Tuberculosis/
│   ├── val/
│   │   ├── Normal/
│   │   ├── Pneumonia/
│   │   └── Tuberculosis/
│   └── test/
│       ├── Normal/
│       ├── Pneumonia/
│       └── Tuberculosis/
│   └── data.yaml               # Dataset metadata
│
├── main.ipynb                  # Main Jupyter notebook (training + evaluation)
├── streamlit_deployed.py       # Streamlit app for model demo
├── README.md                   # Project description (this file)
└── metrics_report.pdf           # Results and visualizations



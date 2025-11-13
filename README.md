<h1>Overview</h1>
<p>This project aims to build a deep learning model capable of classifying chest X-ray images into three categories:
Normal, Pneumonia, and Tuberculosis.
The goal is to assist medical professionals in early diagnosis through automated image analysis.</p>

<h2>1.Project Structure</h2>
finalAML/
│
├── best_model/
│   └── best23.pth
├── checkpoints/
│   ├── checkpoint_epoch_10.pth
│   ├── checkpoint_epoch_20.pth
│   └── ...
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── main.py / notebook.ipynb
├── README.md
└── metrics_report.pdf

<h2>2. Data Preparation</h2>

Dataset organized into three splits: train, validation, and test.

Directory structure:
datasets/
├── train/
│   ├── Normal/
│   ├── Pneumonia/
│   └── Tuberculosis/
├── val/
│   ├── Normal/
│   ├── Pneumonia/
│   └── Tuberculosis/
└── test/
    ├── Normal/
    ├── Pneumonia/
    └── Tuberculosis/

<h2>3.Model</h2>

Implemented using PyTorch and Torchvision.
Cusom CNN and earning approaches were tested.
Fine-tuning was applied with weighted loss to handle class imbalance.

<h2>4.Technologies</h2>
Python 3.12
PyTorch
Torchvision
NumPy, Pandas
Matplotlib, Seaborn
scikit-learn

<h2>5.Team</h2>
Nurlan Marat, 
Aiym Tleuberdinova, 
Amina Rysbekova





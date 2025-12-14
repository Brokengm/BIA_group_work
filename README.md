# 👁️ GlaucoScan


## Overview

This project is a deep learning-based application designed to assist in the preliminary analysis of **Fundus Camera Images** to detect the signs of **Glaucoma**.

## GlaucoScan's Three-stage Pipeline

![image](https://github.com/Brokengm/BIA_group_work/blob/main/Pipeline.png)


## Features

* **Targeted Diagnostic Focus:** Combines robust preprocessing and UNet segmentation to ensure the classifier focuses exclusively on the critical optic disc region, maximizing diagnostic accuracy.
* **High-Performance Backbone:** Employs a fine-tuned EfficientNet-B3 model to achieve reliable and efficient feature extraction for binary glaucoma classification.
* **Fundus Image–Optimized Processing:** The pipeline is custom-tailored to neutralize the visual noise (dark backgrounds, contrast variation) inherent in diverse clinical fundus images.
* **End-to-End Automated Workflow:** Provides a fully integrated pipeline from image input to prediction output, minimizing manual intervention.
* **User-Oriented Graphical Interface:** Offers an intuitive GUI that allows non-technical users to perform glaucoma analysis with minimal effort.

## 🛠️ Installation and Setup  
This software offers two primary methods for usage: running the analysis via **Python scripts** or utilizing the comprehensive **standalone desktop application** built with PyQt5.

### 🐍 Source Code Installation (Python Environment)

### 1  Prerequisites

* Python 3.8+
* NVIDIA GPU (Recommended for faster inference, otherwise runs on CPU)

### 2  Environment Setup

#### 2.1 Clone the repository

```bash
git clone https://github.com/Brokengm/BIA_group_work.git
cd final_version
```

#### 2.2 Create and activate the environment

```bash
# Using conda
conda create -n glaucoma python=3.9
conda activate glaucoma
```

> 💡 Pro Tip: We highly recommend using a **Conda environment** for dependency management. Conda simplifies package isolation and installation, especially for scientific and deep learning packages like PyTorch.
>
> ✨ Further Reading: [Conda Documentation](https://docs.conda.io/en/latest/index.html) | [Managing Conda Environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)


#### 2.3 Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Use appropriate CUDA version
pip install -r requirements.txt 
# The required packages include: PyQt5, numpy, Pillow, timm, scikit-image
```

#### 2.4 Prepare files
For the application to run, you must download the these files and place them directly into the same directory:

* `384_unet.pth` UNet Segmentation Model
* `efficientnet.pth` EfficientNet Classification Model
* `app_gui.py` Main application, GUI (PyQt5) logic, and prediction pipeline entry point.
* `model.py` Contains UNet and CombinedModel (EfficientNet) class definitions and model loading logic.
* `preprocess.py` Contains all image processing functions (cropping, CLAHE, segmentation, etc.).
* `welcome.jpeg` The picture of main page.

### 3 Execute the script

Ensuring you have the 6 files prepared, run the main application script under the same directory.

```bash
python app_gui.py
```

### 🍎 Standalone Software Download (MacOS)

Download [GlaucoScan.zip](https://github.com/Brokengm/BIA_group_work/blob/main/GlaucoScan.zip), and decompress it to run the software immediately.

## Analysis Steps

1.  **Welcome Screen:** Click **"Let's get started"**.
2.  **Load Image:** Click the **"Add image"** button to select a fundus image (`.jpg`, `.png`, etc.). The image will appear in the left panel.
3.  **Analyze:** Click **"Run analysis"**. The system will perform:
    * Preprocessing (border removal, contrast enhancement).
    * Optic Disc Segmentation (using UNet).
    * Optic Disc Cropping (centering on the segmented region).
    * Glaucoma Classification (using EfficientNet-B3).
4.  **View Results:** The classification result, including the predicted class (`Glaucoma` or `Non-Glaucoma`) and the precise probability scores, will be displayed in the right panel.

*A visual walkthrough of these analysis steps is available in the **[Introduction Video](https://github.com/Brokengm/BIA_group_work/blob/main/introduction_video.mp4)**.*

## ⚠️ Disclaimer


**This software is intended for course work and educational demonstration only. It must NOT be used for real medical diagnosis or self-assessment of glaucoma.** Always consult a qualified medical professional for health concerns.





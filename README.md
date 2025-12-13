# 👁️ GlaucoScan


## Overview

This project is a deep learning-based desktop application designed to assist in the preliminary inspection of **Fundus Camera Images** for signs of **Glaucoma**.

It utilizes a three-stage computer vision pipeline:

1.  **Preprocessing:** This included background removal to eliminate black borders, contrast enhancement using gamma correction and other image augmentation.
2.  **UNet Segmentation:** Accurately localizes and segments the optic disc and cup region.
3.  **EfficientNet-B3 Classification:** Classifies the cropped region to distinguish between **Non-Glaucoma** and potential **Glaucoma** cases.

The system is built using **PyTorch** and deployed with a user-friendly **PyQt5 GUI**, allowing researchers and medical professionals to perform local, real-time image analysis.

## 🚀 Features

* **Two-Stage Preprocessing:** Implements robust preprocessing and UNet segmentation to ensure the classification model focuses only on the critical optic disc region, maximizing diagnostic accuracy.
* **High-Performance Backbone:** Utilizes a fine-tuned **EfficientNet-B3** network for fast and reliable feature extraction and classification.
* **Fundus Image Optimized:** The segmentation and preprocessing steps are specifically tailored for varying qualities and sizes of fundus camera images, effectively removing dark backgrounds and centering on the region of interest.
* **Confidence Metrics:** Displays precise probability scores for both the "Glaucoma" and "Non-Glaucoma" categories.
* **Clean Desktop UI:** A fully featured, self-contained desktop interface built with PyQt5, optimizing ease of use for local image analysis.

## 🛠️ Installation and Setup

### Prerequisites

* Python 3.8+
* NVIDIA GPU (Recommended for faster inference, otherwise runs on CPU)

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Brokengm/BIA_group_work.git
    cd final_version
    ```

2.  **Create and activate the environment (recommended):**
    ```bash
    # Using conda
    conda create -n glaucoma python=3.9
    conda activate glaucoma
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Use appropriate CUDA version
    pip install -r requirements.txt 
    # The required packages include: PyQt5, numpy, Pillow, timm, scikit-image
    ```

4.  **Files you need:**
    For the application to run, you must download the these files and place them directly into the same directory:

    * `384_unet.pth` UNet Segmentation Model
    * `efficientnet.pth` EfficientNet Classification Model
    * `app_gui.py` Main application, GUI (PyQt5) logic, and prediction pipeline entry point.
    * `model.py` Contains UNet and CombinedModel (EfficientNet) class definitions and model loading logic.
    * `preprocess.py` Contains all image processing functions (cropping, CLAHE, segmentation, etc.).
    * `welcome.jpeg` The picture of main page.

## 💡 Usage

### Running the Application

1.  Ensure you have completed the **Installation and Setup** steps.
2.  Run the main application script:

    ```bash
    python app_gui.py
    ```

### Analysis Steps

1.  **Welcome Screen:** Click **"Let's get started"**.
2.  **Load Image:** Click the **"Add image"** button to select a fundus image (`.jpg`, `.png`, etc.). The image will appear in the left panel.
3.  **Analyze:** Click **"Run analysis"**. The system will perform:
    * Preprocessing (border removal, contrast enhancement).
    * Optic Disc Segmentation (using UNet).
    * Optic Disc Cropping (centering on the segmented region).
    * Glaucoma Classification (using EfficientNet-B3).
4.  **View Results:** The classification result, including the predicted class (`Glaucoma` or `Non-Glaucoma`) and the precise probability scores, will be displayed in the right panel.

## ⚠️ Disclaimer


**This software is intended for course work and educational demonstration only. It must NOT be used for real medical diagnosis or self-assessment of glaucoma.** Always consult a qualified medical professional for health concerns.





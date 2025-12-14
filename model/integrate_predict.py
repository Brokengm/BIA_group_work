# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import jaccard_score, precision_score, recall_score, f1_score

from skimage import exposure, util
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing, square
import scipy.io as sio

"""### Use U-net to crop image

"""

# code tested in notebook
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        logits = self.conv(dec1)
        return self.sigmoid(logits)

def get_bounding_box(binary_mask):
    binary = binary_mask > 0
    if not np.any(binary):
        height, width = binary_mask.shape
        return 0, 0, width, height

    labeled_image = label(binary)
    regions = regionprops(labeled_image)
    largest_region = max(regions, key=lambda r: r.area)
    min_row, min_col, max_row, max_col = largest_region.bbox
    return min_col, min_row, max_col - min_col, max_row - min_row


def gamma_correct(img, gamma=0.4):
    img_float = util.img_as_float(img)
    corrected = exposure.adjust_gamma(img_float, gamma=gamma)
    return util.img_as_ubyte(corrected)


def enhance_contrast_clahe(image, clip_limit=0.02, kernel_size=8):
    img_float = util.img_as_float(image)
    enhanced = exposure.equalize_adapthist(
        img_float,
        kernel_size=kernel_size,
        clip_limit=clip_limit
    )
    return util.img_as_ubyte(enhanced)


def segment_optic_disc_from_mat(image_path, mat_dir):
    image_stem = Path(image_path).stem
    mat_path = os.path.join(mat_dir, f"{image_stem}.mat")
    mat_data = sio.loadmat(mat_path)
    mask = mat_data['mask']
    optic_disc_mask = (mask > 0).astype(np.uint8) * 255
    return optic_disc_mask


def segment_optic_disc_by_clustering(gray_image):
    pixel_values = gray_image.reshape(-1, 1).astype(np.float32)
    kmeans = KMeans(n_clusters=6, n_init=3, random_state=0).fit(pixel_values)
    cluster_labels = kmeans.labels_.reshape(gray_image.shape)
    cluster_brightness_means = []
    for cluster_id in range(6):
        pixels_in_cluster = pixel_values[cluster_labels.ravel() == cluster_id]
        mean_brightness = pixels_in_cluster.mean() if len(pixels_in_cluster) > 0 else -np.inf
        cluster_brightness_means.append(mean_brightness)
    brightest_cluster_ids = np.argsort(cluster_brightness_means)[-2:]
    optic_disc_mask = np.isin(cluster_labels, brightest_cluster_ids)
    return (optic_disc_mask * 255).astype(np.uint8)


def segment_optic_disc_with_unet(image_path, unet_model):
    pil_image = Image.open(image_path).convert('RGB')
    original_img = np.array(pil_image)

    img_tensor = util.img_as_float(original_img)
    img_tensor = resize(img_tensor, (384, 384), anti_aliasing=True)
    img_tensor = torch.FloatTensor(img_tensor).permute(2, 0, 1).unsqueeze(0)
    model_device = next(unet_model.parameters()).device
    img_tensor = img_tensor.to(model_device)

    with torch.no_grad():
        output = unet_model(img_tensor)
        mask_pred = output.squeeze().cpu().numpy()
        mask_pred = (mask_pred > 0.5).astype(np.uint8)

    mask_pred = resize(mask_pred, original_img.shape[:2], anti_aliasing=False, order=0)
    mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255

    return mask_pred


def crop_fundus_region(original_image, threshold=10):
    if original_image.ndim == 2:
        original_image = np.stack([original_image] * 3, axis=-1)

    gray_image = util.img_as_ubyte(rgb2gray(original_image))

    if np.max(gray_image) <= threshold:
        foreground_mask = np.ones_like(gray_image, dtype=np.uint8) * 255
    else:
        foreground_mask = (gray_image > threshold).astype(np.uint8) * 255

    min_col, min_row, width, height = get_bounding_box(foreground_mask)

    margin = min(100, width // 4, height // 4)
    col_start = max(0, min_col + margin)
    row_start = max(0, min_row + margin)
    col_end = min(original_image.shape[1], min_col + width - margin)
    row_end = min(original_image.shape[0], min_row + height - margin)

    if col_end <= col_start or row_end <= row_start:
        col_start, row_start, col_end, row_end = 0, 0, original_image.shape[1], original_image.shape[0]

    cropped_rgb = original_image[row_start:row_end, col_start:col_end]
    cropped_gray = gray_image[row_start:row_end, col_start:col_end]
    return cropped_rgb, cropped_gray


def refine_mask_with_morphology(binary_mask):
    binary = binary_mask > 0
    opened = binary_opening(binary, footprint=square(5))
    closed = binary_closing(opened, footprint=square(5))
    return (closed * 255).astype(np.uint8)


def crop_optic_disc_region(rgb_image, optic_disc_mask):
    min_col, min_row, width, height = get_bounding_box(optic_disc_mask)

    scale_x = rgb_image.shape[1] / optic_disc_mask.shape[1]
    scale_y = rgb_image.shape[0] / optic_disc_mask.shape[0]
    min_col = int(min_col * scale_x)
    width = int(width * scale_x)
    min_row = int(min_row * scale_y)
    height = int(height * scale_y)

    center_x = min_col + width // 2
    center_y = min_row + height // 2

    half_crop_size = 250 
    col_start = max(0, center_x - half_crop_size)
    row_start = max(0, center_y - half_crop_size)
    col_end = min(rgb_image.shape[1], center_x + half_crop_size)
    row_end = min(rgb_image.shape[0], center_y + half_crop_size)

    if col_end <= col_start or row_end <= row_start:
        col_start, row_start = 0, 0
        col_end = min(rgb_image.shape[1], 500)
        row_end = min(rgb_image.shape[0], 500)

    optic_disc_crop = rgb_image[row_start:row_end, col_start:col_end]
    final_optic_disc_crop = util.img_as_ubyte(
        resize(optic_disc_crop, (384, 384), anti_aliasing=True)
    )
    return final_optic_disc_crop



def preprocess_fundus_image(image_path, mat_dir=None, method='unet', unet_model=None, debug=False):
    pil_image = Image.open(image_path).convert("RGB")
    original_image = np.array(pil_image)

    cropped_rgb, cropped_gray = crop_fundus_region(original_image, threshold=10)

    enhanced_gray = gamma_correct(cropped_gray, gamma=0.4)
    enhanced_gray = enhance_contrast_clahe(enhanced_gray)

    if method == 'clustering':
        optic_disc_mask = segment_optic_disc_by_clustering(enhanced_gray)
    elif method == 'unet':
        if unet_model is None:
            raise ValueError("method='unet', must provide model")
        optic_disc_mask = segment_optic_disc_with_unet(image_path, unet_model)
    elif method == 'mat':
        if mat_dir is None:
            raise ValueError("method='mat', must provide mat file path")
        optic_disc_mask = segment_optic_disc_from_mat(image_path, mat_dir)
    else:
        raise ValueError(f"unknown mehtod: {method}")

    refined_mask = refine_mask_with_morphology(optic_disc_mask)

    enhanced_rgb = enhance_contrast_clahe(cropped_rgb)
    final_crop = crop_optic_disc_region(enhanced_rgb, refined_mask)

    if debug:
        return {
            "original_image": original_image,
            "cropped_fundus": cropped_rgb,
            "optic_disc_mask": refined_mask,
            "final_optic_disc_crop": final_crop
        }
    else:
        return final_crop

def load_model(model_class, model_save_path, in_channels=3, out_channels=1, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"use devide: {device}")

    model = model_class(in_channels=in_channels, out_channels=out_channels).to(device)
    model.eval()

    try:
        with torch.serialization.safe_globals([np.dtype, np.core.multiarray.scalar]):
            checkpoint = torch.load(model_save_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"weights_only=True failed: {e}")
        print("try weights_only=False...")
        checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"model loaded")
    return model

def batch_preprocess_images(
    input_directory,
    output_directory,
    method='unet',
    unet_model=None,
    image_extension="*.jpg",
    save_quality=95
):
    os.makedirs(output_directory, exist_ok=True)
    print(f"output directory exists: {output_directory}")

    image_paths = sorted(glob(os.path.join(input_directory, image_extension)))
    if not image_paths:
        print(f"No any {image_extension} file in {input_directory}")
        return


    for idx, image_path in enumerate(image_paths, 1):
        filename = Path(image_path).name
        print(f" ({idx}/{len(image_paths)}) now: {filename}")

        final_crop = preprocess_fundus_image(
            image_path=image_path,
            method=method,
            unet_model=unet_model if method == 'unet' else None)

        output_filename = Path(image_path).parts[-1]
        output_path = os.path.join(output_directory, output_filename)

        img_to_save = Image.fromarray(final_crop)
        img_to_save.save(output_path, format='JPEG', quality=save_quality)

        print(f" saved in : {output_path}")

def preprocess_single_image(
    image_path: str,
    method: str = 'unet',
    unet_model=None,
    output_directory: str = None,
    save_quality: int = 95
) -> np.ndarray:

    final_crop = preprocess_fundus_image(
        image_path=image_path,
        method=method,
        unet_model=unet_model if method == 'unet' else None
    )


    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

        output_filename = Path(image_path).parts[-1]
        output_path = os.path.join(output_directory, output_filename)

        img_to_save = Image.fromarray(final_crop)

        try:
            img_to_save.save(output_path, format='JPEG', quality=save_quality)
            print(f"Saved as a JPG file (Quality={save_quality}) to: {output_path}")
        except Exception as e:
            print(f"Error when saving: {e}")

# Efficientnet Model: Prediction

import torch
from torch import nn
import timm

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)

        num_features = self.cnn_model.num_features

        self.cnn_projection = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 30)
        )

        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(30, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, *x):
        image = x[0]
        x1 = self.cnn_model(image)
        x1 = self.cnn_projection(x1)
        x = self.final_layers(x1)
        return x

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm


# model weights
EFFICIENTNET_WEIGHTS_PATH = '/mnt/best_fold_4.pth'
CLASS_NAMES = ['Non-Glaucoma', 'Glaucoma']

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_instance = None

def get_loaded_model():
    """
    Lazy Load EfficientNet
    """
    global model_instance
    if model_instance is None:
        print(f"First time loading EfficientNet...")
        
        model = CombinedModel().to(DEVICE)

        try:
            model.load_state_dict(torch.load(EFFICIENTNET_WEIGHTS_PATH, map_location=DEVICE))
            model.eval()
            model_instance = model
            print("Successfully loaded.")
        except Exception as e:
            print(f"Model weights loading failed. Please check the path and class definition: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    return model_instance

class InferenceDataset(Dataset):
    """The Dataset used for batch inference only reads the image paths."""
    def __init__(self, file_paths: list, transform):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        return image, Path(img_path).name

def predict_multiple_images(file_paths: list, batch_size: int = 16) -> pd.DataFrame:
    """
    Perform batch predictions on multiple U-Net cropped images. 
    
    Args:
        file_paths (list): The complete paths of the images to be predicted.
        batch_size (int): Batch size. 
        
    Returns:
        pd.DataFrame: A table containing the file names, predicted categories, and probabilities.
    """
    if not file_paths:
        return pd.DataFrame()

    model = get_loaded_model()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    dataset = InferenceDataset(file_paths, transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # do prediction
    all_results = []

    with torch.no_grad():
        for images, img_names in tqdm(data_loader, desc="Batch Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predicted_indices = np.argmax(probabilities, axis=1)

            # collect results
            for i, name in enumerate(img_names):
                pred_index = predicted_indices[i]

                result = {
                    'filename': name,
                    'predicted_class': CLASS_NAMES[pred_index]
                }
                for j, class_name in enumerate(CLASS_NAMES):
                    result[f'prob_{class_name}'] = probabilities[i, j]

                all_results.append(result)

    return pd.DataFrame(all_results)

def predict_single_image(image_path: str) -> dict:
    """
    Make predictions on the cropped images of a single U-Net. 

    Args:
        image_path (str): The complete path of the image to be predicted. 
    Returns:
        dict: A dictionary containing the file name, predicted category and probability.
    """
    if not os.path.exists(image_path):
        return {'error': f"file not found: {image_path}"}

    model = get_loaded_model()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE) # (1, C, H, W)

    except Exception as e:
        return {'filename': Path(image_path).name, 'error': f"Image loading or preprocessing failed: {e}"}

    # conduct prediction
    with torch.no_grad():
        outputs = model(image_tensor)

    # result analyzation
    probabilities = torch.softmax(outputs, dim=1).cpu().squeeze().numpy()
    predicted_index = np.argmax(probabilities)

    result = {
        'filename': Path(image_path).name,
        'predicted_class': CLASS_NAMES[predicted_index],
    }

    for i, class_name in enumerate(CLASS_NAMES):
        result[f'prob_{class_name}'] = float(probabilities[i])

    return result

unet_model = load_model(
    model_class=UNet,
    model_save_path="/mnt/384_unet.pth",
    in_channels=3,
    out_channels=1
)

# batch_preprocess_images(
#     input_directory="/kaggle/input/glaucoma-detection/ORIGA/ORIGA/Images/",
#     output_directory="/kaggle/working/cropped_images",
#     method='unet',
#     unet_model=unet_model,
#     image_extension="*.jpg",
#     save_quality=95
# )

preprocess_single_image(
    image_path="/mnt/001.jpg",
    method='unet',
    unet_model=unet_model,
    output_directory= '/mnt/output',
    save_quality=95 
)

# example path for a single image
SINGLE_IMAGE_PATH_EXAMPLE = '/mnt/33.jpg'
# example path for a batch
MULTIPLE_IMAGE_PATHS_EXAMPLE = [
    'path/to/your/Unet_Cropped_Dataset/sample_001.jpg',
    'path/to/your/Unet_Cropped_Dataset/sample_002.jpg',
    'path/to/your/Unet_Cropped_Dataset/sample_003.jpg',
]

# single image
print("\n Single image prediction")
single_result = predict_single_image(SINGLE_IMAGE_PATH_EXAMPLE)
print(single_result)

# batch_results_df = predict_multiple_images(MULTIPLE_IMAGE_PATHS_EXAMPLE)
# print(batch_results_df)

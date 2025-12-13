# preprocess.py

import os
import numpy as np
from pathlib import Path

from PIL import Image

import torch
from torchvision import transforms

from skimage import exposure, util
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing, square

# Import model dependencies (model.py should be in the same directory)
from model import UNet, get_unet_model, DEVICE

# =======================================================
#Image processing functions
# =======================================================
def get_bounding_box(binary_mask):
    """Finds the bounding box of the largest connected component in a binary mask."""
    binary = binary_mask > 0
    if not np.any(binary):
        # Return full image size if mask is empty
        height, width = binary_mask.shape
        return 0, 0, width, height

    labeled_image = label(binary)
    regions = regionprops(labeled_image)
    # Find the largest object (most likely the disc)
    largest_region = max(regions, key=lambda r: r.area) 
    min_row, min_col, max_row, max_col = largest_region.bbox
    return min_col, min_row, max_col - min_col, max_row - min_row


def gamma_correct(img, gamma=0.4):
    """Apply standard gamma correction to adjust brightness."""
    img_float = util.img_as_float(img)
    corrected = exposure.adjust_gamma(img_float, gamma=gamma)
    return util.img_as_ubyte(corrected)


def enhance_contrast_clahe(image, clip_limit=0.02, kernel_size=8):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement."""
    img_float = util.img_as_float(image)
    enhanced = exposure.equalize_adapthist(
        img_float,
        kernel_size=kernel_size,
        clip_limit=clip_limit
    )
    return util.img_as_ubyte(enhanced)


def segment_optic_disc_with_unet(image_path, unet_model):
    """Run UNet inference to generate the optic disc segmentation mask."""
    pil_image = Image.open(image_path).convert('RGB')
    original_img = np.array(pil_image)

    # Pre-process raw image for UNet input (resize to 384x384, normalize, to tensor)
    img_tensor = util.img_as_float(original_img)
    img_tensor = resize(img_tensor, (384, 384), anti_aliasing=True)
    img_tensor = torch.FloatTensor(img_tensor).permute(2, 0, 1).unsqueeze(0)

    model_device = next(unet_model.parameters()).device
    img_tensor = img_tensor.to(model_device)

    with torch.no_grad():
        output = unet_model(img_tensor)
        mask_pred = output.squeeze().cpu().numpy()
        mask_pred = (mask_pred > 0.5).astype(np.uint8) # Binarize at 0.5 threshold

    # Resize mask back to original image size for accurate cropping
    mask_pred = resize(mask_pred, original_img.shape[:2], anti_aliasing=False, order=0)
    mask_pred = (mask_pred > 0.5).astype(np.uint8) * 255

    return mask_pred


def crop_fundus_region(original_image, threshold=10):
    """Crops the image to the main fundus region, removing large black borders."""
    if original_image.ndim == 2:
        original_image = np.stack([original_image] * 3, axis=-1)

    gray_image = util.img_as_ubyte(rgb2gray(original_image))

    # Create foreground mask based on intensity threshold
    if np.max(gray_image) <= threshold:
        foreground_mask = np.ones_like(gray_image, dtype=np.uint8) * 255
    else:
        foreground_mask = (gray_image > threshold).astype(np.uint8) * 255

    min_col, min_row, width, height = get_bounding_box(foreground_mask)

    # Apply a small margin/padding logic to the crop
    margin = min(100, width // 4, height // 4)
    col_start = max(0, min_col + margin)
    row_start = max(0, min_row + margin)
    col_end = min(original_image.shape[1], min_col + width - margin)
    row_end = min(original_image.shape[0], min_row + height - margin)

    # Fallback to full image crop if calculated region is invalid
    if col_end <= col_start or row_end <= row_start:
        col_start, row_start, col_end, row_end = 0, 0, original_image.shape[1], original_image.shape[0]

    cropped_rgb = original_image[row_start:row_end, col_start:col_end]
    return cropped_rgb


def refine_mask_with_morphology(binary_mask):
    """Applies binary opening and closing to smooth the segmentation mask."""
    binary = binary_mask > 0
    opened = binary_opening(binary, footprint=square(5))
    closed = binary_closing(opened, footprint=square(5))
    return (closed * 255).astype(np.uint8)


def crop_optic_disc_region(rgb_image, optic_disc_mask):
    """Crops a fixed-size region (500x500 scaled to 384x384) centered on the optic disc."""
    # Find bounding box of the disc in the mask
    min_col, min_row, width, height = get_bounding_box(optic_disc_mask)

    # Scale coordinates back to the original image dimensions
    scale_x = rgb_image.shape[1] / optic_disc_mask.shape[1]
    scale_y = rgb_image.shape[0] / optic_disc_mask.shape[0]
    min_col = int(min_col * scale_x)
    width = int(width * scale_x)
    min_row = int(min_row * scale_y)
    height = int(height * scale_y)

    center_x = min_col + width // 2
    center_y = min_row + height // 2

    # Define crop window centered at the disc
    half_crop_size = 250  # Targets a 500x500 region
    col_start = max(0, center_x - half_crop_size)
    row_start = max(0, center_y - half_crop_size)
    col_end = min(rgb_image.shape[1], center_x + half_crop_size)
    row_end = min(rgb_image.shape[0], center_y + half_crop_size)

    # Fallback/safety check
    if col_end <= col_start or row_end <= row_start:
        col_start, row_start = 0, 0
        col_end = min(rgb_image.shape[1], 500)
        row_end = min(rgb_image.shape[0], 500)

    optic_disc_crop = rgb_image[row_start:row_end, col_start:col_end]
    
    # Final resize to the classifier input size (384x384)
    final_optic_disc_crop = util.img_as_ubyte(
        resize(optic_disc_crop, (384, 384), anti_aliasing=True)
    )
    return final_optic_disc_crop


def preprocess_fundus_image(image_path, unet_model, debug=False):
    """
    End-to-end preprocessing pipeline: 
    1. Removes black borders.
    2. Runs UNet segmentation.
    3. Crops a 384x384 region centered on the segmented optic disc.
    """
    pil_image = Image.open(image_path).convert("RGB")
    original_image = np.array(pil_image)

    # 1. Crop out large black surrounding regions
    cropped_rgb = crop_fundus_region(original_image, threshold=10)

    # 2. Image enhancements (gamma and CLAHE applied for segmentation robustness)
    enhanced_gray = gamma_correct(rgb2gray(cropped_rgb), gamma=0.4)
    enhanced_gray = enhance_contrast_clahe(enhanced_gray)

    # 3. Segmentation
    optic_disc_mask = segment_optic_disc_with_unet(image_path, unet_model)
    refined_mask = refine_mask_with_morphology(optic_disc_mask)

    # 4. Final crop using the refined mask
    enhanced_rgb = enhance_contrast_clahe(cropped_rgb)
    final_crop = crop_optic_disc_region(enhanced_rgb, refined_mask)

    if debug:
        # Useful for debugging the intermediate steps
        return {
            "original_image": original_image,
            "cropped_fundus": cropped_rgb,
            "optic_disc_mask": refined_mask,
            "final_optic_disc_crop": final_crop
        }
    else:
        return final_crop
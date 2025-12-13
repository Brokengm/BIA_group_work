def seg_kmeans(input_path, output_path, threshold=10):
    import cv2
    import numpy as np
    from pathlib import Path
    from sklearn.cluster import KMeans
    from tqdm.notebook import tqdm
    import os

    def gamma_correct(img, gamma=0.4):
        img = img.astype(np.uint8)
        lookUpTable = np.empty((1,256), np.uint8)
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        out = cv2.LUT(img, lookUpTable)
        return out

    def clahe(image, cl=2.0, tgs=8):
        clahe_ = cv2.createCLAHE(clipLimit=cl, tileGridSize=(tgs, tgs))
        cl1 = clahe_.apply(image)
        return cl1

    def clahe_rgb(img, cl=2.0, tgs=8):
        g, b, r = cv2.split(img)
        g, b, r = clahe(g), clahe(b), clahe(r)
        return cv2.merge([g, b, r])

    def apply_gaussian(img):
        img = cv2.GaussianBlur(img, (5,5), 0)
        return img

    def get_bounding_box(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return [x, y, w, h]

    def remove_black_padding(img, threshold):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        x, y, w, h = get_bounding_box(th)
        
        c_size = 300
        final_size = 224
        images = []

        for i, image in enumerate([clahe(gray), gamma_correct(gray, 3), clahe_rgb(img), img]):
            crop = image[y+c_size:y+h-c_size, x+c_size:x+w-c_size]
            crop_resized = cv2.resize(crop, (final_size, final_size))
            if i == 2:
                images.append(crop)
            images.append(crop_resized)
        
        return images

    def ensure_cluster_groups(data_2d, labels, clusters=4):
        mean_intensities = [data_2d[labels == i].mean() for i in range(clusters)]
        label_map = {i: label for i, label in sorted(enumerate(mean_intensities), key=lambda x: x[1])}
        label_map = {k: i for i, k in enumerate(label_map.keys())}
        mapped_labels = np.vectorize(label_map.get)(labels)
        return mapped_labels

    def cluster_image(img):
        data_2d = img.reshape(-1, 1)
        kmeans = KMeans(n_clusters=6, n_init=3, random_state=0).fit(data_2d)
        labels = kmeans.labels_
        labels = ensure_cluster_groups(data_2d, labels, 6)
        labels = labels.reshape(img.shape)
        mask = np.isin(labels, [4, 5])
        mask = (mask * 255).astype(np.uint8)
        return mask

    def apply_morphology(binary_mask):
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations = 1)
        final = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 2)
        return final

    def crop_cup_disc(img, mask):
        x, y, w, h = get_bounding_box(mask)
        width_ratio = img.shape[1]/mask.shape[1]
        height_ratio = img.shape[0]/mask.shape[0]
        x, w = int(x * width_ratio), int(w * width_ratio)
        y, h = int(y * height_ratio), int(h * height_ratio)
        
        cx = x + w//2
        cy = y + h//2
        
        c_size = 300
        x_start = (cx-c_size) if (cx > c_size) else 0
        y_start = (cy-c_size) if (cy > c_size) else 0
        
        crop = img[y_start:cy+c_size, x_start:cx+c_size]
        crop = cv2.resize(crop, (224, 224))

        return crop
    
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    out = remove_black_padding(img, threshold)
    if out is None:
        raise ValueError("Failed to remove black padding / no region detected")

    _, corrected_gray, rgb_clahe_org, _, _ = out

    segmented = cluster_image(corrected_gray)
    segmented = apply_morphology(segmented)

    cropped = crop_cup_disc(rgb_clahe_org, segmented)
    if cropped is None:
        raise ValueError("Failed to crop optic disc")

    cv2.imwrite(output_path, cropped)
    print(f"Saved segmented optic disc → {output_path}")

    return cropped

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optic Disc Segmentation")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, required=True, help="Where to save the segmented output")

    args = parser.parse_args()

    seg_kmeans(args.input, args.output)
    

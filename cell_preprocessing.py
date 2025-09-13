#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import skimage
from skimage.measure import regionprops_table, label
from skimage.exposure import equalize_adapthist, match_histograms
from skimage.filters import threshold_otsu
from skimage.color import rgb2hsv, hsv2rgb


def create_circular_mask(h, w, center=None, radius=None):
    """
    Crea una maschera circolare booleana.
    """
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def preprocess_image(image_array: np.ndarray, enhance_contrast: bool = False, isolate_and_crop: bool = False,
                     do_tiling: bool = False, scale_factor: int = 1, tile_size: int = 640,
                     do_histogram_matching: bool = False, reference_image: np.ndarray = None,
                     ) -> tuple:
    """
    Pipeline di preprocessing su un'immagine.
    """
    processed_image = image_array.copy()

    # 1. Rimuovi canale alpha se presente
    if processed_image.shape[-1] == 4:
        processed_image = processed_image[:, :, 0:3]
    
    # 2. Isolamento e ritaglio del campione
    if isolate_and_crop:
        try:
            red_exposed = processed_image[:, :, 0]
            thresh = threshold_otsu(red_exposed)
            binary_red_exposed = red_exposed < thresh
            label_bg = label(binary_red_exposed, background=True)
            props = regionprops_table(label_bg, properties=('centroid', 'axis_major_length', 'bbox', 'area', 'eccentricity'))
            df = pd.DataFrame(props).loc[lambda d: d.area > 10000].sort_values('eccentricity')

            center = (df['centroid-1'].iat[0], df['centroid-0'].iat[0])
            radius = df['axis_major_length'].iat[0] / 2
            mask = create_circular_mask(processed_image.shape[0], processed_image.shape[1], center=center, radius=radius)
            processed_image = processed_image * np.stack((mask, mask, mask), axis=-1)

            xmin, ymin, xmax, ymax = [df[f'bbox-{i}'].iat[0] for i in range(4)]
            processed_image = processed_image[xmin:xmax, ymin:ymax]
        except (IndexError, ValueError):
            print("Attenzione: impossibile isolare e ritagliare il campione. Passaggio saltato.")

    # 3. Miglioramento contrasto
    if enhance_contrast:
        hsv_image = rgb2hsv(processed_image)
        v_channel = hsv_image[:, :, 2]
        v_channel_enhanced = equalize_adapthist(v_channel, clip_limit=0.01)
        hsv_image[:, :, 2] = v_channel_enhanced
        processed_image = (hsv2rgb(hsv_image) * 255).astype(np.uint8)

    original_processed_image = processed_image.copy()
    
    # 4. Scaling
    if scale_factor > 1:
        h, w = processed_image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        processed_image = cv2.resize(processed_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    images_to_process = [processed_image]
    coords = [(0, 0)]

    # 5. Tiling
    if do_tiling and scale_factor > 1:
        tiles_with_coords = tile_image_by_grid(processed_image, grid_size=scale_factor)
        images_to_process = [t[0] for t in tiles_with_coords]
        coords = [t[1] for t in tiles_with_coords]

    # 6. Histogram Matching
    if do_histogram_matching:
        if reference_image is not None:
            matched_images = []
            target_h, target_w = images_to_process[0].shape[:2]
            try:
                # Ridimensiona la reference
                scaled_ref = cv2.resize(reference_image, (target_w, target_h), interpolation=cv2.INTER_AREA)

                # Assicura 3 canali
                if scaled_ref.ndim == 2:
                    scaled_ref = cv2.cvtColor(scaled_ref, cv2.COLOR_GRAY2BGR)

                for img in images_to_process:
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    matched = match_histograms(img, scaled_ref, channel_axis=-1)
                    matched_images.append(matched.astype(np.uint8))

                images_to_process = matched_images
            except Exception as e:
                print(f"[AVVISO] Histogram Matching fallito: {e}")
        else:
            print("Attenzione: Histogram matching richiesto ma nessuna immagine di riferimento fornita. Passaggio saltato.")
    
    return images_to_process, coords, original_processed_image, scale_factor


def tile_image_by_grid(image: np.ndarray, grid_size: int) -> list:
    """
    Suddivide un'immagine in una griglia di N x N tasselli.
    """
    h, w = image.shape[:2]
    tile_h, tile_w = h // grid_size, w // grid_size
    tiles = []
    for i in range(grid_size):
        for j in range(grid_size):
            y, x = i * tile_h, j * tile_w
            tile = image[y:y + tile_h, x:x + tile_w]
            tiles.append((tile, (x, y)))
    return tiles

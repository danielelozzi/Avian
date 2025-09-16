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
                     do_histogram_matching: bool = False, reference_image: np.ndarray = None,
                     input_magnification: float = 200.0, training_magnification: float = 200.0,
                     tile_size: int = 640, overlap: int = 100
                     ) -> dict:
    """Pipeline di preprocessing su un'immagine.

    Restituisce un dizionario contenente le immagini processate e i parametri
    necessari per aggiornare eventuali annotazioni (es. bounding box, maschere).

    Esempio di utilizzo per scalare bounding box dopo la chiamata:
    """
    original_shape = image_array.shape[:2]
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
    
    images_to_process = [processed_image]
    
    # 4. Histogram Matching
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
    
    processed_image = images_to_process[0]

    # 5. Scaling e Tiling
    scale_factor = training_magnification / input_magnification
    processed_tiles, tile_coords = scale_and_tile_image(
        processed_image, scale_factor, tile_size, overlap
    )

    return {
        "processed_tiles": processed_tiles,
        "tile_coords": tile_coords,
        "original_processed_image": original_processed_image,
        "scale_factor": scale_factor,
        "original_shape": original_shape
    }


def scale_and_tile_image(image: np.ndarray, scale_factor: float, tile_size: int, overlap: int) -> (list, list):
    """
    Ridimensiona un'immagine e la suddivide in tasselli sovrapposti.
    """
    if scale_factor != 1.0:
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LANCZOS4
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    else:
        scaled_image = image

    h, w = scaled_image.shape[:2]
    step = tile_size - overlap
    tiles = []
    coords = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            
            tile = scaled_image[y1:y2, x1:x2]
            
            # Se il tassello è più piccolo della dimensione minima, lo si padda
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded_tile[0:tile.shape[0], 0:tile.shape[1]] = tile
                tile = padded_tile

            tiles.append(tile)
            coords.append((x1, y1))

    return tiles, coords

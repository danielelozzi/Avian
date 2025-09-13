#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import skimage
from skimage.measure import regionprops_table, label, find_contours
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu, sobel
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt


def create_circular_mask(h, w, center=None, radius=None):
    """
    Crea una maschera circolare booleana.

    Args:
        h (int): Altezza dell'immagine/maschera.
        w (int): Larghezza dell'immagine/maschera.
        center (tuple, optional): Coordinate (x, y) del centro del cerchio. 
                                  Se None, viene usato il centro dell'immagine.
        radius (int, optional): Raggio del cerchio. Se None, viene usato il raggio
                                più grande possibile che non esca dai bordi.

    Returns:
        ndarray: Un array booleano dove True rappresenta l'interno del cerchio.
    """
    if center is None:  # usa il centro dell'immagine
        center = (int(w / 2), int(h / 2))
    if radius is None:  # usa la distanza minima tra il centro e i bordi
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


def preprocess_image(image_array: np.ndarray, enhance_contrast: bool = False, isolate_and_crop: bool = False,
                     do_tiling: bool = False, tile_size: int = 640, overlap: int = 128,
                     do_scaling: bool = False, input_magnification: int = 100, train_magnification: int = 100,
                     do_histogram_matching: bool = False, reference_image: np.ndarray = None,
                     ) -> tuple:
    """
    Esegue una pipeline di preprocessing su un'immagine seguendo un ordine specifico.

    Args:
        image_array (np.ndarray): L'immagine di input come array NumPy.
        isolate_and_crop (bool): Se True, isola l'area circolare del campione e la ritaglia.
        enhance_contrast (bool): Se True, migliora il contrasto dell'immagine.
        do_tiling (bool): Se True, suddivide l'immagine in tasselli.
        tile_size (int): Dimensione dei tasselli.
        overlap (int): Sovrapposizione dei tasselli.
        do_scaling (bool): Se True, esegue lo scaling dell'immagine/tasselli.
        input_magnification (int): Ingrandimento dell'immagine di input.
        train_magnification (int): Ingrandimento usato per l'addestramento del modello.
        do_histogram_matching (bool): Se True, esegue l'histogram matching.
        reference_image (np.ndarray, optional): Immagine di riferimento per l'histogram matching.

    Returns:
        tuple: Una tupla contenente (lista di immagini, lista di coordinate, immagine originale processata).
               Se il tiling non è attivo, la lista di immagini e coordinate conterrà un solo elemento.
    """
    processed_image = image_array.copy()

    # 1. Esclusione del canale alpha se presente
    if processed_image.shape[-1] == 4:
        processed_image = processed_image[:, :, 0:3]
    
    # 2. Isolamento e ritaglio del campione (opzionale)
    if isolate_and_crop:
        try:
            # Identificazione dell'area del vetrino usando threshold_otsu
            red_exposed = processed_image[:, :, 0]
            thresh = threshold_otsu(red_exposed)
            binary_red_exposed = red_exposed < thresh
            label_bg = label(binary_red_exposed, background=True)
            props = regionprops_table(label_bg, properties=('centroid', 'axis_major_length', 'bbox', 'area', 'eccentricity'))
            df = pd.DataFrame(props).loc[lambda d: d.area > 10000].sort_values('eccentricity')

            # Creazione di una maschera circolare e applicazione
            center = (df['centroid-1'].iat[0], df['centroid-0'].iat[0])
            radius = df['axis_major_length'].iat[0] / 2
            mask = create_circular_mask(processed_image.shape[0], processed_image.shape[1], center=center, radius=radius)
            processed_image = processed_image * np.stack((mask, mask, mask), axis=-1)

            # Ritaglio dell'immagine sull'area di interesse
            xmin, ymin, xmax, ymax = [df[f'bbox-{i}'].iat[0] for i in range(4)]
            processed_image = processed_image[xmin:xmax, ymin:ymax]
        except (IndexError, ValueError) as e:
            # Se il rilevamento del campione fallisce, restituisce l'immagine così com'è
            print("Attenzione: impossibile isolare e ritagliare il campione. Il passaggio è stato saltato.")

    # 3. Miglioramento del contrasto (opzionale)
    if enhance_contrast:
        # Conversione in spazio colore HSV per separare la luminosità (V) dal colore (H, S)
        hsv_image = rgb2hsv(processed_image)
        
        # Applica CLAHE (Contrast Limited Adaptive Histogram Equalization) al canale V (Value)
        v_channel = hsv_image[:, :, 2]
        v_channel_enhanced = equalize_adapthist(v_channel, clip_limit=0.01)
        hsv_image[:, :, 2] = v_channel_enhanced
        
        # Riconverti in RGB e assicurati che sia nel formato corretto (uint8)
        processed_image = (hsv2rgb(hsv_image) * 255).astype(np.uint8)

    original_processed_image = processed_image.copy()
    images_to_process = [processed_image]
    coords = [(0, 0)]

    # 4. Tiling (opzionale)
    if do_tiling:
        tiles_with_coords = tile_image(processed_image, tile_size=tile_size, overlap=overlap)
        images_to_process = [t[0] for t in tiles_with_coords]
        coords = [t[1] for t in tiles_with_coords]

    # 5. Scaling (opzionale)
    if do_scaling and input_magnification != train_magnification:
        scale_factor = train_magnification / input_magnification
        scaled_images = []
        for img in images_to_process:
            if scale_factor != 1.0:
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                # Usa INTER_AREA per rimpicciolire, INTER_LANCZOS4 per ingrandire
                interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LANCZOS4
                scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
                scaled_images.append(scaled_img)
            else:
                scaled_images.append(img)
        images_to_process = scaled_images

    # 6. Histogram Matching (opzionale)
    if do_histogram_matching:
        if reference_image is not None:
            matched_images = []
            for img in images_to_process:
                matched_images.append(manual_histogram_matching(img, reference_image))
            images_to_process = matched_images
        else:
            print("Attenzione: Histogram matching richiesto ma nessuna immagine di riferimento fornita. Passaggio saltato.")

    return images_to_process, coords, original_processed_image


def manual_histogram_matching(source_image: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
    """
    Allinea il colore e il contrasto dell'immagine sorgente a quelli dell'immagine di riferimento
    utilizzando l'histogram matching su ogni canale di colore.

    Args:
        source_image (np.ndarray): L'immagine da modificare (in formato BGR).
        reference_image (np.ndarray): L'immagine di riferimento (in formato BGR).

    Returns:
        np.ndarray: L'immagine sorgente con colori e contrasto allineati.
    """
    matched_image = np.zeros_like(source_image)
    for i in range(3):  # Itera sui canali B, G, R
        hist_src, _ = np.histogram(source_image[:, :, i].ravel(), 256, [0, 256])
        hist_ref, _ = np.histogram(reference_image[:, :, i].ravel(), 256, [0, 256])

        cdf_src = hist_src.cumsum()
        cdf_ref = hist_ref.cumsum()

        cdf_src_normalized = cdf_src * hist_ref.sum() / cdf_src.sum()

        lookup_table = np.zeros(256, dtype='uint8')
        g = 0
        for j in range(256):
            while g < 256 and cdf_ref[g] < cdf_src_normalized[j]:
                g += 1
            lookup_table[j] = g

        matched_image[:, :, i] = cv2.LUT(source_image[:, :, i], lookup_table)

    return matched_image


def tile_image(image: np.ndarray, tile_size: int, overlap: int) -> list:
    """
    Suddivide un'immagine in tasselli sovrapposti.

    Args:
        image (np.ndarray): L'immagine da suddividere.
        tile_size (int): La dimensione (larghezza e altezza) di ogni tassello.
        overlap (int): Il numero di pixel di sovrapposizione tra i tasselli.

    Returns:
        list: Una lista di tuple, dove ogni tupla contiene un tassello e le sue coordinate (x, y) originali.
    """
    h, w, _ = image.shape
    stride = tile_size - overlap
    tiles = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            tile = image[y:y_end, x:x_end]
            if tile.shape[0] > overlap and tile.shape[1] > overlap: # Salta tasselli troppo piccoli
                tiles.append((tile, (x, y)))
    return tiles
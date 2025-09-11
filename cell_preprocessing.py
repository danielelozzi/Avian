#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import skimage
from skimage.measure import regionprops_table, label
from skimage.exposure import rescale_intensity, equalize_adapthist
from skimage.filters import threshold_otsu
from PIL import Image
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


def preprocess_image(image_array: np.ndarray, enhance_contrast: bool = True, isolate_and_crop: bool = True) -> np.ndarray:
    """
    Preprocessa un'immagine di un vetrino per isolare e ritagliare l'area di interesse.

    Args:
        image_array (np.ndarray): L'immagine di input come array NumPy.
        enhance_contrast (bool): Se True, migliora il contrasto dell'immagine.
        isolate_and_crop (bool): Se True, isola l'area circolare del campione e la ritaglia.

    Returns:
        np.ndarray: L'immagine preprocessata e ritagliata come array NumPy.
    """
    processed_image = image_array.copy()

    # 1. Esclusione del canale alpha se presente
    if processed_image.shape[-1] == 4:
        processed_image = processed_image[:, :, 0:3]

    # 2. Miglioramento del contrasto (opzionale)
    if enhance_contrast:
        # Usa multi-otsu a 3 classi per definire i limiti.
        # È un compromesso efficiente che si adatta all'immagine.
        red_channel = processed_image[:, :, 0]
        thresholds = skimage.filters.threshold_multiotsu(red_channel, classes=3)
        p_low, p_high = thresholds[0], thresholds[1]
        processed_image = rescale_intensity(processed_image, in_range=(p_low, p_high))

    # 3. Isolamento e ritaglio del campione (opzionale)
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
        except (IndexError, ValueError):
            # Se il rilevamento del campione fallisce, restituisce l'immagine così com'è
            print("Attenzione: impossibile isolare e ritagliare il campione. Il passaggio è stato saltato.")

    return processed_image
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import skimage
from skimage.measure import regionprops_table, label
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_multiotsu


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


def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    """
    Preprocessa un'immagine di un vetrino per isolare e ritagliare l'area di interesse.

    Args:
        image_array (np.ndarray): L'immagine di input come array NumPy.

    Returns:
        np.ndarray: L'immagine preprocessata e ritagliata come array NumPy.
    """
    # 1. Esclusione del canale alpha se presente
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, 0:3]

    # 2. Miglioramento del contrasto basato sul canale rosso
    red_channel = image_array[:, :, 0]
    multi_threshold_0 = threshold_multiotsu(red_channel, classes=6)
    cell_exposed = rescale_intensity(image_array, in_range=(multi_threshold_0[0], multi_threshold_0[-1]))

    # 3. Identificazione dell'area del vetrino (background)
    red_exposed = cell_exposed[:, :, 0]
    multi_threshold_1 = threshold_multiotsu(red_exposed, classes=6)
    binary_red_exposed = red_exposed < multi_threshold_1[0]
    label_bg = label(binary_red_exposed, background=True)
    props = regionprops_table(label_bg, properties=('centroid', 'axis_major_length', 'bbox', 'area', 'eccentricity'))
    df = pd.DataFrame(props).loc[lambda d: d.area > 10000].sort_values('eccentricity')

    # 4. Creazione di una maschera circolare e applicazione
    center = (df['centroid-1'].iat[0], df['centroid-0'].iat[0])
    radius = df['axis_major_length'].iat[0] / 2
    mask = create_circular_mask(image_array.shape[0], image_array.shape[1], center=center, radius=radius)
    cell_exposed = cell_exposed * np.stack((mask, mask, mask), axis=-1)

    # 5. Ritaglio dell'immagine sull'area di interesse
    xmin, ymin, xmax, ymax = [df[f'bbox-{i}'].iat[0] for i in range(4)]
    return cell_exposed[xmin:xmax, ymin:ymax]
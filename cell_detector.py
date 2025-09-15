import cv2
import glob
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes, Masks
import os
import numpy as np


def detect_cells_yolo(source, model):
    """
    Esegue l'inferenza di object detection con YOLO.
    La libreria ridimensiona e adatta automaticamente l'immagine sorgente
    per farla corrispondere alle dimensioni di addestramento del modello.
    """
    model_yolo = YOLO(model)
    results = model_yolo.predict(source, save=True, conf=0.5, iou=0.7)
    return results


def merge_tile_results(
    tile_results: list,
    original_shape: tuple,
    scale_factor: float = 1.0,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Results:
    """
    Unisce i risultati dai tasselli, rimappa le coordinate e applica NMS.

    Args:
        tile_results (list): Lista di tuple (risultato_yolo, (offset_x, offset_y)).
        original_shape (tuple): La forma (h, w) dell'immagine originale.
        scale_factor (float): Il fattore di scala applicato prima del tiling.
        conf_threshold (float): Soglia di confidenza per NMS.
        iou_threshold (float): Soglia IoU per NMS.

    Returns:
        Results: Un oggetto Results di Ultralytics con i rilevamenti uniti.
    """
    all_boxes = []
    all_scores = []
    all_classes = []
    all_segments = []

    if not tile_results:
        # Se non ci sono risultati, restituisci un oggetto Results vuoto ma con le dimensioni corrette
        return Results(orig_img=np.zeros((*original_shape, 3), dtype=np.uint8), path="", names={}, boxes=None, masks=None)

    template_result = next(
        (res for res, _ in tile_results if res is not None and res.boxes is not None), None
    )
    if template_result is None:
        # Se nessun tassello ha prodotto risultati, restituisci un oggetto vuoto
        return Results(orig_img=np.zeros((*original_shape, 3), dtype=np.uint8), path="", names={}, boxes=None, masks=None)

    # --- CORREZIONE CHIAVE ---
    # Creiamo un nuovo oggetto Results da zero con le dimensioni dell'immagine originale,
    # invece di usare un template da un tassello (che ha dimensioni errate, es. 640x640).
    final_result = Results(orig_img=np.zeros((*original_shape, 3), dtype=np.uint8), path=template_result.path, names=template_result.names)

    for result, (x_offset, y_offset) in tile_results:
        if result is None or result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset
        all_boxes.extend(boxes)

        all_scores.extend(result.boxes.conf.cpu().numpy())
        all_classes.extend(result.boxes.cls.cpu().numpy())

        if result.masks is not None:
            masks_np = result.masks.data.cpu().numpy()  # [N, H, W]
            for mask in masks_np:
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if contours:
                    seg = max(contours, key=cv2.contourArea).squeeze(1)
                    seg[:, 0] += x_offset
                    seg[:, 1] += y_offset
                    all_segments.append(seg)
                else:
                    all_segments.append(np.array([]))
        elif result.boxes is not None:
            all_segments.extend([np.array([])] * len(result.boxes))

    if not all_boxes:
        return template_result.new()

    indices = cv2.dnn.NMSBoxes(np.array(all_boxes, dtype=np.float32), np.array(all_scores, dtype=np.float32), conf_threshold, iou_threshold)

    if len(indices) == 0:
        return template_result.new()

    indices = indices.flatten()

    final_boxes_xyxy = np.array(all_boxes)[indices]
    final_scores = np.array(all_scores)[indices]
    final_classes = np.array(all_classes)[indices]
    final_segments = [all_segments[i] for i in indices]

    # 🔧 FIX: conversione in float per evitare errore di casting
    if scale_factor != 1.0:
        inv_scale_factor = 1.0 / scale_factor
        final_boxes_xyxy = final_boxes_xyxy.astype(np.float32) * inv_scale_factor
        new_segments = []
        for seg in final_segments:
            if seg.size > 0:
                seg = seg.astype(np.float32) * inv_scale_factor
            new_segments.append(seg)
        final_segments = new_segments

    final_result.boxes = Boxes(
        np.hstack(
            (final_boxes_xyxy, final_scores[:, np.newaxis], final_classes[:, np.newaxis])
        ),
        orig_shape=original_shape,
    )

    if any(seg.size > 0 for seg in final_segments):
        # Ricostruisce le maschere bitmap dai poligoni finali
        # Le dimensioni della maschera devono corrispondere a quelle dell'immagine
        final_mask_tensor = []
        h, w = final_result.orig_shape[:2]

        for seg in final_segments:
            if seg.size > 0:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [seg.astype(np.int32)], 1)
                final_mask_tensor.append(mask)
            else:
                final_mask_tensor.append(np.zeros((h, w), dtype=np.uint8))

        # --- CORREZIONE FINALE ---
        # L'oggetto Masks ha bisogno sia delle maschere binarie (per il calcolo)
        # sia dei segmenti poligonali (per il disegno). Dobbiamo fornirgli entrambi.
        # I segmenti devono essere normalizzati rispetto alle dimensioni dell'immagine.
        if final_mask_tensor:
            # Creando l'oggetto Masks dalle maschere binarie, la libreria calcolerà automaticamente i poligoni (attributo .xy)
            final_result.masks = Masks(masks=np.array(final_mask_tensor), orig_shape=final_result.orig_shape)

    return final_result


def main():
    model_path = os.path.join(os.getcwd(), "yolov8nseg_avian_100epoche.pt")
    image_path = os.path.join(os.getcwd(), "test_img/IMG_3064.jpg")

    if not os.path.exists(image_path):
        print(f"File non trovato: {image_path}")
        return

    results = detect_cells_yolo(source=image_path, model=model_path)
    print("Inferenza completata. I risultati sono salvati nella directory 'runs/segment'.")


if __name__ == "__main__":
    main()

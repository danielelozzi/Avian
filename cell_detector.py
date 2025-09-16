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
    scale_factor: float,
    conf_threshold: float,
    iou_threshold: float = 0.45
) -> Results:
    """
    Unisce i risultati dai tasselli, rimappa le coordinate e applica NMS.

    Args:
        tile_results (list): Lista di tuple (risultato_yolo, (offset_x, offset_y)).
        original_shape (tuple): La forma (h, w) dell'immagine originale.
        scale_factor (float): Il fattore di scala applicato all'immagine prima del tiling.
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
        return Results(orig_img=np.zeros((*original_shape, 3), dtype=np.uint8), path="", names={}, boxes=None, masks=None)

    # Trova un risultato valido per ottenere i nomi delle classi
    template_result = next((res for res, _ in tile_results if res and res.boxes), None)
    if template_result is None:
        return Results(orig_img=np.zeros((*original_shape, 3), dtype=np.uint8), path="", names={}, boxes=None, masks=None)

    # Crea un oggetto Results vuoto con le proprietà corrette
    final_result = Results(
        orig_img=np.zeros((*original_shape, 3), dtype=np.uint8),
        path="",
        names=template_result.names,
        boxes=None, # Verranno aggiunti dopo
        masks=None  # Verranno aggiunti dopo
    )

    for result, (x_offset, y_offset) in tile_results:
        if not result or not result.boxes:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset
        all_boxes.extend(boxes)

        all_scores.extend(result.boxes.conf.cpu().numpy())
        all_classes.extend(result.boxes.cls.cpu().numpy())

        if result.masks is not None:
            masks_np = result.masks.data.cpu().numpy()
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
        else:
            all_segments.extend([np.array([])] * len(result.boxes))

    if not all_boxes:
        return final_result

    indices = cv2.dnn.NMSBoxes(np.array(all_boxes, dtype=np.float32), np.array(all_scores, dtype=np.float32), conf_threshold, iou_threshold)

    if not hasattr(indices, 'flatten'):
        return final_result # Restituisce un risultato vuoto ma con le dimensioni corrette

    indices = indices.flatten()

    final_boxes_xyxy = np.array(all_boxes)[indices]
    final_scores = np.array(all_scores)[indices]
    final_classes = np.array(all_classes)[indices]
    final_segments = [all_segments[i] for i in indices]

    if scale_factor != 1.0:
        final_boxes_xyxy /= scale_factor
        new_segments = []
        for seg in final_segments:
            if seg.size > 0:
                seg = seg.astype(np.float32) / scale_factor
            new_segments.append(seg)
        final_segments = new_segments

    final_result.boxes = Boxes(
        np.hstack(
            (final_boxes_xyxy, final_scores[:, None], final_classes[:, None])
        ),
        orig_shape=original_shape,
    )

    if any(seg.size > 0 for seg in final_segments):
        final_masks_data = []
        for seg in final_segments:
            mask = np.zeros(original_shape, dtype=np.uint8)
            if seg.size > 0:
                cv2.fillPoly(mask, [seg.astype(np.int32)], 1)
            final_masks_data.append(mask)
        
        final_result.masks = Masks(
            masks=np.array(final_masks_data), orig_shape=original_shape
        )

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

import cv2
import glob
from ultralytics import YOLO
from ultralytics.engine.results import Results
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

def merge_tile_results(tile_results: list, original_shape: tuple, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Results:
    """
    Unisce i risultati dai tasselli, rimappa le coordinate e applica NMS.

    Args:
        tile_results (list): Lista di tuple (risultato_yolo, (offset_x, offset_y)).
        original_shape (tuple): La forma (h, w) dell'immagine originale.
        conf_threshold (float): Soglia di confidenza per NMS.
        iou_threshold (float): Soglia IoU per NMS.

    Returns:
        Results: Un oggetto Results di Ultralytics con i rilevamenti uniti.
    """
    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []
    
    if not tile_results:
        return None

    # Prendi un risultato valido come template
    template_result = next((res for res, _ in tile_results if res is not None), None)
    if template_result is None:
        # Se nessun risultato è valido, non possiamo creare un oggetto Results.
        # In questo caso, è meglio restituire un oggetto Results vuoto ma valido.
        # Creiamo un template fittizio per questo scopo.
        from ultralytics.engine.results import Boxes, Masks
        return Results(orig_img=np.zeros((*original_shape, 3), dtype=np.uint8), path="", names={}, boxes=Boxes(np.array([]), original_shape), masks=Masks(np.array([]), original_shape))

    for result, (x_offset, y_offset) in tile_results:
        if result is None or result.boxes is None:
            continue

        # Rimappa Bounding Boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset
        
        all_boxes.extend(boxes)
        all_scores.extend(result.boxes.conf.cpu().numpy())
        all_classes.extend(result.boxes.cls.cpu().numpy())

        # Rimappa Maschere
        if result.masks is not None:
            for mask_data in result.masks.data.cpu().numpy():
                # Ridimensiona la maschera alla dimensione del tile
                tile_h, tile_w = result.orig_shape
                mask_resized = cv2.resize(mask_data, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                
                # Crea una maschera vuota grande quanto l'immagine originale
                full_mask = np.zeros(original_shape[:2], dtype=np.uint8)
                
                # Posiziona la maschera ridimensionata
                h, w = mask_resized.shape
                y_end, x_end = y_offset + h, x_offset + w
                full_mask[y_offset:y_end, x_offset:x_end] = mask_resized
                all_masks.append(full_mask)

    if not all_boxes:
        return template_result.new()

    # Applica Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, conf_threshold, iou_threshold)
    
    # Crea un nuovo oggetto Results con i dati filtrati
    final_boxes = np.array(all_boxes)[indices]
    final_scores = np.array(all_scores)[indices]
    final_classes = np.array(all_classes)[indices]
    final_masks = np.array(all_masks)[indices] if all_masks else None

    # Se final_masks ha una sola dimensione (un solo risultato), aggiungi una dimensione per renderlo [1, H, W]
    if final_masks is not None and final_masks.ndim == 2:
        final_masks = np.expand_dims(final_masks, axis=0)


    # Combina i dati dei box in formato [x1, y1, x2, y2, conf, cls]
    combined_data = np.hstack((final_boxes, final_scores[:, np.newaxis], final_classes[:, np.newaxis]))

    # Crea l'oggetto Results finale
    final_result = template_result.new()
    final_result.orig_shape = original_shape # Assicura che orig_shape sia sempre impostato
    final_result.boxes = type(template_result.boxes)(combined_data, orig_shape=original_shape)
    if final_masks is not None and template_result.masks is not None:
        final_result.masks = type(template_result.masks)(final_masks, orig_shape=original_shape)
        # Forza il calcolo dei contorni poligonali (segmenti) accedendo alla proprietà .xy
        _ = final_result.masks.xy

    return final_result

def main():
    model_path = os.path.join(os.getcwd(), 'yolov8nseg_avian.pt')
    # Modifica il percorso del file sorgente per usare IMG_3064.jpg
    image_path = os.path.join(os.getcwd(), 'test_img/IMG_3064.jpg')
    
    if not os.path.exists(image_path):
        print(f"File non trovato: {image_path}")
        return

    results = detect_cells_yolo(source=image_path, model=model_path)
    print("Inferenza completata. I risultati sono salvati nella directory 'runs/segment'.")

if __name__ == "__main__":
    main()
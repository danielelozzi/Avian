#!/usr/bin/env python
# coding: utf-8

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont
import datetime

# --- CONFIGURAZIONE DEL MODELLO ---
# Queste costanti dovrebbero corrispondere a come il modello è stato addestrato.
MODEL_INPUT_SIZE = (512, 384)  # Dimensione corretta (larghezza, altezza) attesa dal modello

# Mappa gli ID di output del modello ai nomi delle classi e ai colori
CLASS_MAP = {
    0: {"id": 1, "name": "eritrociti", "color": "red"},
    1: {"id": 2, "name": "trombociti", "color": "cyan"},
    2: {"id": 3, "name": "linfociti", "color": "yellow"},
    # Aggiungi altre classi se necessario
}

CATEGORIES = [{"id": v["id"], "name": v["name"], "supercategory": "cell"} for k, v in CLASS_MAP.items()]


def non_max_suppression(boxes, scores, iou_threshold):
    """Applica la Soppressione dei Non-Massimi (NMS) per pulire i box sovrapposti."""
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
        
    return keep


class CellDetector:
    """
    Classe che gestisce il caricamento e l'inferenza di un modello ONNX
    per la rilevazione di cellule.
    """
    def __init__(self, model_path='efficientNet_B0.onnx', provider='CPUExecutionProvider'):
        print(f"Caricamento del modello da: {model_path}")
        print(f"Utilizzo del provider di esecuzione: {provider}")
        self.session = ort.InferenceSession(model_path, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        print("Modello caricato con successo.")

    def detect(self, image_array: np.ndarray, conf_threshold=0.5, iou_threshold=0.4):
        """
        Esegue la rilevazione di oggetti sull'immagine fornita.
        """
        original_h, original_w = image_array.shape[:2]

        # 1. Preprocessing dell'immagine per il modello
        image_pil = Image.fromarray(image_array).resize(MODEL_INPUT_SIZE)
        input_tensor = np.array(image_pil, dtype=np.float32)
        input_tensor = input_tensor / 255.0  # Normalizzazione
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Aggiungi batch dimension

        # 2. Esegui inferenza
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # --- IPOTESI SULL'OUTPUT DEL MODELLO ---
        # Assumo che l'output sia una lista di array [num_detections, 6]
        # dove 6 sta per [x1, y1, x2, y2, confidence, class_id]
        # Se il tuo modello ha un output diverso, questa parte va adattata.
        detections = outputs[0]

        # 3. Post-processing        
        if detections is None or len(detections) == 0:
            return np.array(image_array), self._create_empty_coco(original_w, original_h)

        boxes, scores, class_ids = [], [], []
        for det in detections:
            # Aggiunto controllo per la dimensione del rilevamento per evitare IndexError
            if len(det) < 6:
                continue

            confidence = det[4]
            if confidence >= conf_threshold:
                boxes.append(det[:4])
                scores.append(confidence)
                class_ids.append(int(det[5]))

        if not boxes:
            return np.array(image_array), self._create_empty_coco(original_w, original_h)

        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Scala i box alle dimensioni del modello (es. 512x512)
        boxes[:, [0, 2]] *= MODEL_INPUT_SIZE[0]
        boxes[:, [1, 3]] *= MODEL_INPUT_SIZE[1]

        # Applica NMS
        keep_indices = non_max_suppression(boxes, scores, iou_threshold)
        
        final_boxes = boxes[keep_indices]
        final_scores = scores[keep_indices]
        final_class_ids = np.array(class_ids)[keep_indices]

        # Scala i box finali alle dimensioni originali dell'immagine
        final_boxes[:, [0, 2]] *= (original_w / MODEL_INPUT_SIZE[0])
        final_boxes[:, [1, 3]] *= (original_h / MODEL_INPUT_SIZE[1])

        # 4. Genera l'output (immagine con box e dati COCO)
        return self._format_output(image_array, final_boxes, final_scores, final_class_ids)

    def _format_output(self, image_array, boxes, scores, class_ids):
        pil_image = Image.fromarray(image_array)
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        coco_output = self._create_empty_coco(image_array.shape[1], image_array.shape[0])
        cell_counts = {cat["name"]: 0 for cat in CATEGORIES}

        for i, box in enumerate(boxes):
            class_id = class_ids[i]
            if class_id not in CLASS_MAP:
                continue

            info = CLASS_MAP[class_id]
            class_name, color = info["name"], info["color"]
            cell_counts[class_name] += 1

            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - 15), f"{class_name} {scores[i]:.2f}", fill=color, font=font)

            # Aggiungi annotazione COCO
            bbox_coco = [x1, y1, x2 - x1, y2 - y1]
            coco_annotation = {
                "id": i + 1, "image_id": 1, "category_id": info["id"],
                "bbox": bbox_coco, "area": bbox_coco[2] * bbox_coco[3],
                "segmentation": [], "iscrowd": 0, "score": float(scores[i])
            }
            coco_output["annotations"].append(coco_annotation)

        coco_output["summary_counts"] = cell_counts
        return np.array(pil_image), coco_output

    def _create_empty_coco(self, width, height):
        return {
            "info": {"description": "Avian Cell Annotations (ONNX)", "version": "1.0", "date_created": datetime.date.today().isoformat()},
            "licenses": [{"id": 1, "name": "N/A", "url": ""}],
            "images": [{"id": 1, "width": width, "height": height, "file_name": "image.png", "license": 1}],
            "annotations": [],
            "categories": CATEGORIES
        }
import cv2
import glob
from ultralytics import YOLO
import os

def detect_cells_yolo(source, model):
    """
    Esegue l'inferenza di object detection con YOLO.
    La libreria ridimensiona e adatta automaticamente l'immagine sorgente
    per farla corrispondere alle dimensioni di addestramento del modello.
    """
    model_yolo = YOLO(model)
    results = model_yolo.predict(source, save=True, conf=0.5, iou=0.7)
    return results

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
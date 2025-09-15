from ultralytics import YOLO
import torch
import os

def train_model():
    """
    Esegue il fine-tuning di un modello di segmentazione YOLOv8n sul dataset delle cellule ematiche aviarie.

    Il processo si articola nei seguenti passaggi:
    1. Caricamento di un modello YOLOv8 per la segmentazione (`yolov8n-seg.pt`), pre-addestrato sul dataset COCO.
    2. Rilevamento e selezione automatica del dispositivo di calcolo più performante disponibile
       (CUDA per GPU NVIDIA, MPS per Apple Silicon, altrimenti CPU).
    3. Avvio del processo di addestramento (`model.train()`) utilizzando i parametri specificati.
       Viene applicata una vasta gamma di tecniche di data augmentation per migliorare la robustezza
       del modello e la sua capacità di generalizzare, specialmente in presenza di classi sbilanciate.
       Le augmentations includono:
       - Trasformazioni geometriche: rotazioni, traslazioni, scaling, shear, cambi di prospettiva.
       - Flipping: specchiamento orizzontale e verticale.
       - Tecniche avanzate:
         - `mosaic`: combina quattro immagini in una per esporre il modello a contesti e scale diverse.
         - `mixup`: mescola due immagini e le loro etichette per regolarizzare il modello.
         - `copy_paste`: copia oggetti da un'immagine e li incolla su un'altra, aumentando la frequenza
           delle classi meno comuni (particolarmente utile per la segmentazione).
    4. Al termine dell'addestramento, il modello con le migliori performance sul set di validazione
       (salvato come `best.pt` nella cartella dei risultati) viene esportato in formato PyTorch (`.pt`)
       con un nome standard (`yolov8nseg_avian.pt`) per un facile utilizzo nell'applicazione principale.
    """
    # Carica un modello pre-addestrato per la segmentazione
    model = YOLO('yolov8n-seg.pt')
    
    # Seleziona il dispositivo: 'cuda' per NVIDIA, 'mps' per Apple Silicon, altrimenti 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Training using device: {device}")

    # Esegui il fine-tuning del modello
    # I parametri come epochs, imgsz, batch, etc., possono essere regolati.
    # Aggiungiamo parametri di data augmentation per migliorare la robustezza del modello
    # e aiutare con eventuali classi sbilanciate.
    results = model.train(
        data='avian_blood_cells.yaml',
        device=device,
        epochs=100,
        imgsz=640,
        batch=8,
        name='yolov8n_avian_blood_seg_augmented', # Nome della cartella per i risultati
        # Parametri di Data Augmentation
        degrees=25.0,  # Rotazione immagine (+/- gradi)
        translate=0.1, # Traslazione immagine (+/- frazione)
        scale=0.5,     # Ridimensionamento immagine (+/- gain)
        shear=2.0,     # Shear dell'immagine (+/- gradi)
        perspective=0.0005, # Prospettiva dell'immagine
        flipud=0.5,    # Probabilità di flip verticale
        fliplr=0.5,    # Probabilità di flip orizzontale
        mosaic=1.0,    # Augmentation a mosaico (probabilità)
        mixup=0.1,     # Augmentation mixup (probabilità)
        copy_paste=0.1 # Augmentation copy-paste per la segmentazione (probabilità)
    )
    
    # Esporta il modello nel formato .pt
    # Il modello migliore viene salvato automaticamente come 'best.pt' nella cartella dei risultati.
    # Qui lo esportiamo con un nome specifico.
    model.export(format='pt', name='yolov8nseg_avian.pt')
    print(f"\nModello salvato come 'yolov8nseg_avian.pt' nella directory corrente.")

if __name__ == '__main__':
    train_model()
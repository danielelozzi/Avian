from ultralytics import YOLO
import torch

def train_model():
    """
    Esegue il fine-tuning di un modello YOLOv8n-seg.
    """
    # Carica un modello pre-addestrato per la segmentazione
    model = YOLO('yolov8n-seg.pt')
    
    # Seleziona il dispositivo: 'cuda' per NVIDIA, 'mps' per Apple Silicon, altrimenti 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Training using device: {device}")

    # Esegui il fine-tuning del modello
    # I parametri come epochs, imgsz, batch, etc., possono essere regolati.
    results = model.train(
        data='avian_blood_cells.yaml',
        device=device,
        epochs=50,
        imgsz=640,
        batch=8,
        name='yolov8n_avian_blood_seg' # Nome della cartella per i risultati
    )

if __name__ == '__main__':
    train_model()
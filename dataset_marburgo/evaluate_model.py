from ultralytics import YOLO
import torch

def evaluate_model():
    """
    Valuta un modello di segmentazione YOLOv8 addestrato su un set di dati.

    Questa funzione carica il modello fine-tunato e lo esegue sul set di dati
    specificato nel file .yaml (usando lo split 'test' o 'val').
    Stampa a schermo le metriche di performance per la segmentazione,
    tra cui Precision, Recall e mAP (basata su IoU).
    """
    # --- PARAMETRI DI CONFIGURAZIONE ---
    MODEL_PATH = '../yolov8nseg_avian.pt'  # Percorso al modello addestrato
    DATA_CONFIG_PATH = 'avian_blood_cells.yaml' # File di configurazione del dataset

    # Seleziona il dispositivo: 'cuda' per NVIDIA, 'mps' per Apple Silicon, altrimenti 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Evaluation using device: {device}")

    try:
        # Carica il modello addestrato
        model = YOLO(MODEL_PATH)

        # Esegui la validazione/valutazione sul dataset
        # La funzione `val()` calcola automaticamente tutte le metriche richieste
        # usando il set di dati specificato in `DATA_CONFIG_PATH`.
        # Per impostazione predefinita, usa lo split 'val', ma possiamo forzare 'test'.
        metrics = model.val(data=DATA_CONFIG_PATH, split='test', device=device)

        print("\n--- Riepilogo Metriche di Segmentazione ---")
        # Le metriche principali sono già state stampate a schermo dalla funzione val().
        # Qui accediamo ai valori specifici per la maschera (segmentazione).
        print(f"Precision (Mask): {metrics.mask.p.mean():.4f}")
        print(f"Recall (Mask): {metrics.mask.r.mean():.4f}")
        print(f"mAP50 (Mask): {metrics.mask.map50:.4f}  (Intersection over Union > 50%)")
        print(f"mAP50-95 (Mask): {metrics.mask.map:.4f} (Intersection over Union da 50% a 95%)")

    except Exception as e:
        print(f"Si è verificato un errore durante la valutazione: {e}")

if __name__ == '__main__':
    evaluate_model()
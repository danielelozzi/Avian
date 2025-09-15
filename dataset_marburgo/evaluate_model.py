from ultralytics import YOLO
import torch
import pandas as pd

def evaluate_model():
    """
    Valuta un modello di segmentazione YOLOv8 addestrato su un set di dati.
    e salva la matrice di confusione come immagine.

    Questa funzione carica il modello fine-tunato e lo esegue sul set di dati
    specificato nel file .yaml (usando lo split 'test' o 'val').
    Stampa a schermo le metriche di performance per la segmentazione,
    tra cui Precision, Recall e mAP (basata su IoU), sia in forma aggregata
    che dettagliata per ogni singola classe, utilizzando tabelle per una
    migliore leggibilità.
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

        # --- MATRICE DI CONFUSIONE ---
        # La funzione val() calcola la matrice di confusione, noi la plottiamo e salviamo.
        # Il file verrà salvato nella stessa cartella dei risultati della validazione.
        confusion_matrix = metrics.confusion_matrix
        confusion_matrix.plot(save_dir=metrics.save_dir, names=model.names)
        print(f"\nMatrice di confusione salvata in: {metrics.save_dir}/confusion_matrix.png")

        # --- METRICHE DETTAGLIATE PER CLASSE ---
        class_names = model.names

        # 1. Metriche di Segmentazione (Mask) per classe
        seg_metrics_data = {
            'Classe': [],
            'Precision': [],
            'Recall': [],
            'mAP50': [],
            'mAP50-95': []
        }
        for i, name in class_names.items():
            seg_metrics_data['Classe'].append(name)
            seg_metrics_data['Precision'].append(metrics.mask.p[i])
            seg_metrics_data['Recall'].append(metrics.mask.r[i])
            seg_metrics_data['mAP50'].append(metrics.mask.ap50[i])
            seg_metrics_data['mAP50-95'].append(metrics.mask.ap[i, 0]) # mAP50-95 è la prima colonna di ap

        seg_df = pd.DataFrame(seg_metrics_data)
        print("\n\n--- Metriche di Segmentazione (Mask) per Classe ---")
        print(seg_df.to_string(index=False, float_format="%.4f"))

        # 2. Metriche di Classificazione (Box) per classe (simile a classification_report)
        box_metrics_data = {
            'Classe': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'mAP50': []
        }
        for i, name in class_names.items():
            box_metrics_data['Classe'].append(name)
            box_metrics_data['Precision'].append(metrics.box.p[i])
            box_metrics_data['Recall'].append(metrics.box.r[i])
            box_metrics_data['F1-Score'].append(metrics.box.f1[i])
            box_metrics_data['mAP50'].append(metrics.box.ap50[i])

        box_df = pd.DataFrame(box_metrics_data)
        print("\n\n--- Metriche di Classificazione (Box) per Classe ---")
        print(box_df.to_string(index=False, float_format="%.4f"))

    except Exception as e:
        print(f"Si è verificato un errore durante la valutazione: {e}")

if __name__ == '__main__':
    evaluate_model()
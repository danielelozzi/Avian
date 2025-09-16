import argparse
from ultralytics import YOLO
import sys

def inspect_model_classes(model_path):
    """
    Carica un modello YOLO e stampa le informazioni sulle sue classi.

    Args:
        model_path (str): Il percorso al file del modello (.pt).
    """
    try:
        # Carica il modello specificato
        print(f"Caricamento del modello da: {model_path}...")
        model = YOLO(model_path)
        
        # Ottieni le informazioni sulle classi. model.names è un dizionario {indice: nome_classe}
        class_names = model.names
        num_classes = len(class_names)
        
        if num_classes > 0:
            print(f"\n✅ Il modello è stato addestrato su {num_classes} classi.")
            print("\n--- Elenco delle Classi ---")
            for index, name in sorted(class_names.items()):
                print(f"  - Indice {index}: {name}")
            print("--------------------------")
        else:
            print("⚠️ Non è stato possibile trovare informazioni sulle classi nel modello.")

    except FileNotFoundError:
        print(f"❌ ERRORE: Il file del modello non è stato trovato al percorso: {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Si è verificato un errore imprevisto: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # Configura il parser per accettare argomenti da riga di comando
    parser = argparse.ArgumentParser(
        description="Ispeziona le classi di un modello YOLOv8 addestrato (.pt).",
        epilog="Esempio di utilizzo: python inspect_model.py yolov8n_cat_pose_cls.pt"
    )
    
    # Aggiungi l'argomento per il percorso del modello
    parser.add_argument(
        'model_path', 
        type=str, 
        help="Percorso al file del modello YOLO (.pt) da ispezionare."
    )
    
    args = parser.parse_args()
    
    # Esegui la funzione di ispezione
    inspect_model_classes(args.model_path)
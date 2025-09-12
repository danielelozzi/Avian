import json
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict

# --- PARAMETRI DI CONFIGURAZIONE ---
# Modifica questa frazione per usare una percentuale diversa del dataset (es. 0.2 per il 20%, 1.0 per il 100%)
DATASET_FRACTION = 1.0

# Percentuali per la suddivisione del dataset campionato
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1 # Deve essere 1.0 - TRAIN_SIZE - VAL_SIZE

def convert_coco_to_yolo(images_info, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, split_name):
    """
    Converte le annotazioni COCO in formato YOLO per la segmentazione.

    Args:
        images_info (list): Lista di dizionari, ognuno per un'immagine.
        annotations_by_image_id (dict): Dizionario di annotazioni raggruppate per image_id.
        cat_id_to_yolo_id (dict): Mappa da category_id a indice YOLO.
        output_dir (Path): Directory di output per il dataset YOLO.
        image_dirs (list): Lista delle cartelle originali delle immagini (es. ['train', 'val']).
        split_name (str): Nome dello split (es. 'train', 'val').
    """
    images_path = output_dir / 'images' / split_name
    labels_path = output_dir / 'labels' / split_name
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {split_name} set...")
    for img_info in tqdm(images_info):
        img_w, img_h = img_info['width'], img_info['height']
        img_id = img_info['id']
        anns = annotations_by_image_id.get(img_id, [])
        
        # Copia l'immagine nella cartella di destinazione
        # Cerca l'immagine in tutte le possibili directory di origine
        src_img_path = None
        for img_dir in image_dirs:
            potential_path = img_dir / img_info['file_name']
            if potential_path.exists():
                src_img_path = potential_path
                break
        
        if not src_img_path:
            print(f"Warning: Image {img_info['file_name']} not found. Skipping.")
            continue
            
        dst_img_path = images_path / img_info['file_name']
        shutil.copy(src_img_path, dst_img_path)

        # Crea il file di etichetta YOLO
        label_file_path = labels_path / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_file_path, 'w') as f:
            for ann in anns:
                yolo_cat_id = cat_id_to_yolo_id[ann['category_id']]
                
                # Gestisce sia RLE che poligoni
                segmentation = ann['segmentation']
                if isinstance(segmentation, dict) and 'counts' in segmentation:
                    # Formato RLE, non gestito in questo script base.
                    # YOLOv8 può gestire RLE ma la conversione è più complessa.
                    # Per ora, saltiamo queste annotazioni se presenti.
                    # print(f"Skipping RLE annotation for image {img_id}")
                    continue
                elif isinstance(segmentation, list):
                    # Formato poligono
                    # Assumiamo che il primo poligono sia quello principale
                    seg = segmentation[0]
                    normalized_seg = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / img_w
                        y = seg[i+1] / img_h
                        normalized_seg.extend([x, y])
                    
                    seg_str = " ".join(map(str, normalized_seg))
                    f.write(f"{yolo_cat_id} {seg_str}\n")

def load_and_merge_coco(base_dir):
    """Carica e unisce i file train.json e val.json."""
    all_images = []
    all_annotations = []
    all_categories = []
    
    # Mappa per evitare duplicati di immagini e annotazioni
    image_ids = set()
    ann_ids = set()

    for json_file in ['train.json', 'val.json']:
        coco_path = base_dir / json_file
        if not coco_path.exists():
            print(f"Warning: {json_file} not found. Skipping.")
            continue
        
        with open(coco_path, 'r') as f:
            data = json.load(f)
        
        if not all_categories:
            all_categories = data['categories']

        for img in data['images']:
            if img['id'] not in image_ids:
                all_images.append(img)
                image_ids.add(img['id'])

        for ann in data['annotations']:
            if ann['id'] not in ann_ids:
                all_annotations.append(ann)
                ann_ids.add(ann['id'])

    return all_images, all_annotations, all_categories

def main():
    # Controllo coerenza delle percentuali
    if not np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1.0):
        raise ValueError("Le somme delle percentuali di TRAIN, VAL, e TEST devono dare 1.0")

    base_dir = Path('./dataset_segmentation')
    output_dir = Path('./avian_yolo_dataset')
    
    # 1. Pulisci la directory di output se esiste
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"Cleaned output directory: {output_dir}")

    # 2. Carica e unisci i dataset
    print("Loading and merging COCO datasets...")
    all_images, all_annotations, categories = load_and_merge_coco(base_dir)
    print(f"Total images found: {len(all_images)}")
    print(f"Total annotations found: {len(all_annotations)}")

    # 3. Campiona una frazione del dataset totale
    if DATASET_FRACTION < 1.0:
        num_to_sample = int(len(all_images) * DATASET_FRACTION)
        print(f"Sampling {num_to_sample} images ({DATASET_FRACTION*100:.1f}% of total)...")
        sampled_images = random.sample(all_images, num_to_sample)
    else:
        sampled_images = all_images
        print("Using the full dataset.")

    # 4. Suddivisione stratificata in train, val, test
    print("Splitting dataset into train, validation, and test sets...")
    # Crea un array di etichette per la stratificazione (es. la classe più rara per immagine)
    image_labels = defaultdict(lambda: -1)
    for ann in all_annotations:
        # Usiamo la category_id per la stratificazione
        if image_labels[ann['image_id']] == -1 or ann['category_id'] < image_labels[ann['image_id']]:
            image_labels[ann['image_id']] = ann['category_id']
    
    labels_for_split = [image_labels[img['id']] for img in sampled_images]

    # Prima divisione: train vs (val + test)
    train_val_size = VAL_SIZE + TEST_SIZE
    train_images, val_test_images, _, _ = train_test_split(
        sampled_images, labels_for_split, train_size=TRAIN_SIZE, random_state=42, stratify=labels_for_split
    )
    
    # Ricrea le etichette per la seconda divisione
    labels_for_second_split = [image_labels[img['id']] for img in val_test_images]
    
    # Seconda divisione: val vs test
    val_images, test_images, _, _ = train_test_split(
        val_test_images, labels_for_second_split, test_size=(TEST_SIZE / train_val_size), random_state=42, stratify=labels_for_second_split
    )
    
    print(f"Train set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")
    print(f"Test set size: {len(test_images)}")

    # 5. Prepara i dati per la conversione
    annotations_by_image_id = defaultdict(list)
    for ann in all_annotations:
        annotations_by_image_id[ann['image_id']].append(ann)
        
    cat_ids = sorted([cat['id'] for cat in categories])
    cat_id_to_yolo_id = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    
    image_dirs = [base_dir / 'train', base_dir / 'val']

    # 6. Converti e salva i set
    convert_coco_to_yolo(train_images, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, 'train')
    convert_coco_to_yolo(val_images, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, 'val')
    convert_coco_to_yolo(test_images, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, 'test')

    print("\nDataset preparation complete!")

if __name__ == '__main__':
    main()
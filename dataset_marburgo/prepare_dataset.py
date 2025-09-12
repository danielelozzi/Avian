import json
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import numpy as np
from collections import defaultdict
from pycocotools import mask as mask_util # <-- NUOVA IMPORTAZIONE
import cv2 # <-- NUOVA IMPORTAZIONE

# --- PARAMETRI DI CONFIGURAZIONE ---
DATASET_FRACTION = 0.2
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1

def rle_to_polygons(rle, height, width):
    """
    Converte una maschera RLE in una lista di poligoni.
    """
    binary_mask = mask_util.decode(rle)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if contour.size >= 6: # Un poligono valido ha almeno 3 punti (6 coordinate)
            polygons.append(contour.flatten().tolist())
    return polygons

def convert_coco_to_yolo(images_info, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, split_name):
    """
    Converte le annotazioni COCO in formato YOLO per la segmentazione.
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

        label_file_path = labels_path / f"{Path(img_info['file_name']).stem}.txt"
        with open(label_file_path, 'w') as f:
            for ann in anns:
                yolo_cat_id = cat_id_to_yolo_id[ann['category_id']]
                
                segmentation = ann['segmentation']
                polygons = []

                if isinstance(segmentation, dict) and 'counts' in segmentation:
                    # --- NUOVA GESTIONE RLE ---
                    rle = {'counts': segmentation['counts'], 'size': [img_h, img_w]}
                    polygons = rle_to_polygons(rle, img_h, img_w)
                elif isinstance(segmentation, list):
                    # Gestione poligoni esistente
                    polygons = segmentation

                # Scrivi i poligoni nel file di etichetta
                for seg in polygons:
                    if not seg: continue
                    normalized_seg = []
                    for i in range(0, len(seg), 2):
                        x = seg[i] / img_w
                        y = seg[i+1] / img_h
                        normalized_seg.extend([x, y])
                    
                    seg_str = " ".join(map(str, normalized_seg))
                    f.write(f"{yolo_cat_id} {seg_str}\n")

# ... (il resto del file da load_and_merge_coco in poi rimane invariato) ...
def load_and_merge_coco(base_dir):
    """Carica e unisce i file train.json e val.json."""
    all_images = []
    all_annotations = []
    all_categories = []
    
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
    if not np.isclose(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 1.0):
        raise ValueError("Le somme delle percentuali di TRAIN, VAL, e TEST devono dare 1.0")

    base_dir = Path('./dataset_segmentation')
    output_dir = Path('../avian_yolo_dataset')
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    print(f"Cleaned output directory: {output_dir}")

    print("Loading and merging COCO datasets...")
    all_images, all_annotations, categories = load_and_merge_coco(base_dir)
    print(f"Total images found: {len(all_images)}")
    print(f"Total annotations found: {len(all_annotations)}")

    if DATASET_FRACTION < 1.0:
        num_to_sample = int(len(all_images) * DATASET_FRACTION)
        print(f"Sampling {num_to_sample} images ({DATASET_FRACTION*100:.1f}% of total)...")
        sampled_images = random.sample(all_images, num_to_sample)
    else:
        sampled_images = all_images
        print("Using the full dataset.")

    print("Splitting dataset into train, validation, and test sets...")
    image_labels = defaultdict(lambda: -1)
    for ann in all_annotations:
        if image_labels[ann['image_id']] == -1 or ann['category_id'] < image_labels[ann['image_id']]:
            image_labels[ann['image_id']] = ann['category_id']
    
    labels_for_split = [image_labels[img['id']] for img in sampled_images]

    train_val_size = VAL_SIZE + TEST_SIZE
    train_images, val_test_images, _, _ = train_test_split(
        sampled_images, labels_for_split, train_size=TRAIN_SIZE, random_state=42, stratify=labels_for_split
    )
    
    labels_for_second_split = [image_labels[img['id']] for img in val_test_images]
    
    val_images, test_images, _, _ = train_test_split(
        val_test_images, labels_for_second_split, test_size=(TEST_SIZE / train_val_size), random_state=42, stratify=labels_for_second_split
    )
    
    print(f"Train set size: {len(train_images)}")
    print(f"Validation set size: {len(val_images)}")
    print(f"Test set size: {len(test_images)}")

    annotations_by_image_id = defaultdict(list)
    for ann in all_annotations:
        annotations_by_image_id[ann['image_id']].append(ann)
        
    cat_ids = sorted([cat['id'] for cat in categories])
    cat_id_to_yolo_id = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    
    image_dirs = [base_dir / 'train', base_dir / 'val']

    convert_coco_to_yolo(train_images, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, 'train')
    convert_coco_to_yolo(val_images, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, 'val')
    convert_coco_to_yolo(test_images, annotations_by_image_id, cat_id_to_yolo_id, output_dir, image_dirs, 'test')

    print("\nDataset preparation complete!")

if __name__ == '__main__':
    main()
import yaml
from pathlib import Path
import json

def create_yolo_yaml():
    """
    Crea il file YAML di configurazione per il training con YOLO.
    """
    # The path should be relative to the 'dataset_marburgo' directory
    # where the script is run.
    dataset_path = Path('../avian_yolo_dataset')

    train_coco_path = Path('./dataset_segmentation/train.json')

    # Estrai i nomi delle categorie dal file COCO originale
    with open(train_coco_path, 'r') as f:
        coco_data = json.load(f)
    
    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]

    # Use paths relative to the project structure
    yaml_data = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(class_names)}
    }

    # Save the YAML file inside the dataset_marburgo directory
    with open('avian_blood_cells.yaml', 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    
    print("avian_blood_cells.yaml created successfully.")

if __name__ == '__main__':
    create_yolo_yaml()
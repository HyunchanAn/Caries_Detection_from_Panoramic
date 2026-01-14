import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(json_path, image_dir, output_dir, target_classes=None):
    """
    Converts COCO JSON annotations to YOLO format.
    
    Args:
        json_path (str): Path to the COCO JSON file.
        image_dir (str): Directory containing the images.
        output_dir (str): Directory where labels and images will be saved (YOLO structure).
        target_classes (list): List of class names to include. If None, all classes are included.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create directories
    labels_dir = os.path.join(output_dir, 'labels')
    images_output_dir = os.path.join(output_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # Map category IDs to YOLO class IDs (0, 1, 2...)
    # Check if we need to filter classes
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Create a mapping from COCO category ID to YOLO class ID
    # If target_classes is provided, only map those.
    cat_id_to_yolo_id = {}
    yolo_id = 0
    
    print("Categories found in JSON:")
    for cat_id, name in categories.items():
        print(f"ID: {cat_id}, Name: {name}")
        
    # Manual mapping strategy or auto
    # For DENTEX, we might want to map specific disease names to 'Caries'
    # Simplified: Map everything or specific logic
    
    # If target_classes is provided, we filter.
    # Example logic: if 'caries' in name.lower() -> class 0
    
    # Using a simple list based mapping for now
    sorted_cat_ids = sorted(categories.keys())
    for cat_id in sorted_cat_ids:
        cat_name = categories[cat_id]
        if target_classes:
            if cat_name in target_classes:
                cat_id_to_yolo_id[cat_id] = target_classes.index(cat_name)
        else:
            cat_id_to_yolo_id[cat_id] = yolo_id
            yolo_id += 1

    print("Category Mapping (COCO ID -> YOLO ID):", cat_id_to_yolo_id)

    # Process images
    images_info = {img['id']: img for img in data['images']}
    
    for img_id, img_info in tqdm(images_info.items(), desc="Converting"):
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']
        
        # Source image path
        src_img_path = os.path.join(image_dir, file_name)
        if not os.path.exists(src_img_path):
            # Try finding it recursively or check extension
            continue 

        # Create label file
        label_file = os.path.splitext(file_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # Find annotations for this image
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        if not annotations:
            continue
            
        yolo_lines = []
        for ann in annotations:
            cat_id = ann['category_id']
            if cat_id not in cat_id_to_yolo_id:
                continue
                
            yolo_cls = cat_id_to_yolo_id[cat_id]
            bbox = ann['bbox'] # [x_min, y_min, width, height]
            
            # Convert to YOLO center_x, center_y, w, h (normalized)
            x_min, y_min, w, h = bbox
            
            center_x = (x_min + w / 2) / img_w
            center_y = (y_min + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            
            # Clip to [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            yolo_lines.append(f"{yolo_cls} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")
        
        if yolo_lines:
            with open(label_path, 'w') as f_out:
                f_out.write('\n'.join(yolo_lines))
            
            # Copy image to YOLO directory
            dst_img_path = os.path.join(images_output_dir, file_name)
            shutil.copy2(src_img_path, dst_img_path)

if __name__ == "__main__":
    # Example Usage
    # Modify these paths according to your actual data location
    JSON_PATH = "../data/raw/train_quadrant_enumeration_disease.json" # Downloaded DENTEX JSON
    IMAGE_DIR = "../data/raw/xrays" # Downloaded Images
    OUTPUT_DIR = "../data/processed/train"
    
    # If None, converts all classes. 
    # If you want specific classes, list them: ['Caries', 'Deep Caries']
    # Note: DENTEX classes might be 'Caries', 'Periapical Lesion', etc. Check your JSON.
    TARGET_CLASSES = None 
    
    if os.path.exists(JSON_PATH) and os.path.exists(IMAGE_DIR):
        print("Starting conversion...")
        convert_coco_to_yolo(JSON_PATH, IMAGE_DIR, OUTPUT_DIR, TARGET_CLASSES)
        print("Conversion complete.")
    else:
        print("Data not found. Please check paths in src/data_converter.py")

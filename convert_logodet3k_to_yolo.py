"""
Convert LogoDet-3K dataset from Pascal VOC format to YOLO format
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import random

# Configuration
LOGODET3K_PATH = r"C:\Users\WIN11\.cache\kagglehub\datasets\lyly99\logodet3k\versions\1\LogoDet-3K"
OUTPUT_PATH = r"F:\apcs\ComputerVision\Thesis\logodet3k_yolo"

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Categories in LogoDet-3K
CATEGORIES = ['Necessities', 'Others', 'Sports', 'Transportation','Electronic']


def parse_xml_annotation(xml_path):
    """Parse Pascal VOC XML annotation file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    # Get all objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return img_width, img_height, objects


def convert_bbox_to_yolo(img_width, img_height, bbox):
    """Convert Pascal VOC bbox to YOLO format (normalized center coords + width/height)"""
    xmin, ymin, xmax, ymax = bbox
    
    # Calculate center coordinates and dimensions
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    width = xmax - xmin
    height = ymax - ymin
    
    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height


def collect_all_brands_and_images():
    """Collect all brand names and image paths from the dataset"""
    brand_to_images = defaultdict(list)
    all_brands = set()
    
    print("Collecting all brands and images...")
    
    for category in CATEGORIES:
        category_path = os.path.join(LOGODET3K_PATH, category)
        if not os.path.exists(category_path):
            print(f"Warning: Category {category} not found at {category_path}")
            continue
        
        # Each subdirectory is a brand
        for brand_name in os.listdir(category_path):
            brand_path = os.path.join(category_path, brand_name)
            if not os.path.isdir(brand_path):
                continue
            
            all_brands.add(brand_name)
            
            # Find all jpg files in the brand folder
            for file in os.listdir(brand_path):
                if file.endswith('.jpg'):
                    img_path = os.path.join(brand_path, file)
                    xml_path = img_path.replace('.jpg', '.xml')
                    
                    if os.path.exists(xml_path):
                        brand_to_images[brand_name].append({
                            'img_path': img_path,
                            'xml_path': xml_path,
                            'category': category
                        })
    
    print(f"Found {len(all_brands)} unique brands")
    print(f"Found {sum(len(imgs) for imgs in brand_to_images.values())} images")
    
    return brand_to_images, sorted(all_brands)


def create_yolo_structure(output_path):
    """Create YOLO dataset directory structure"""
    splits = ['train', 'val', 'test']
    
    # Create main directories
    for split in splits:
        images_dir = os.path.join(output_path, 'images', split)
        labels_dir = os.path.join(output_path, 'labels', split)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Created YOLO directory structure at {output_path}")


def split_dataset(brand_to_images):
    """Split dataset into train/val/test while keeping brand consistency"""
    train_data = []
    val_data = []
    test_data = []
    
    print("Splitting dataset...")
    
    for brand_name, images in brand_to_images.items():
        # Shuffle images for this brand
        random.shuffle(images)
        
        n_images = len(images)
        n_train = int(n_images * TRAIN_RATIO)
        n_val = int(n_images * VAL_RATIO)
        
        train_data.extend(images[:n_train])
        val_data.extend(images[n_train:n_train + n_val])
        test_data.extend(images[n_train + n_val:])
    
    print(f"Split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    return train_data, val_data, test_data


def process_and_copy_data(data_list, split_name, brand_to_id, output_path):
    """Process annotations and copy images to YOLO format"""
    images_dir = os.path.join(output_path, 'images', split_name)
    labels_dir = os.path.join(output_path, 'labels', split_name)
    
    print(f"Processing {split_name} split...")
    
    for idx, data in enumerate(data_list):
        img_path = data['img_path']
        xml_path = data['xml_path']
        
        # Parse XML annotation
        try:
            img_width, img_height, objects = parse_xml_annotation(xml_path)
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            continue
        
        # Generate unique filename
        img_filename = f"{split_name}_{idx:06d}.jpg"
        label_filename = f"{split_name}_{idx:06d}.txt"
        
        # Copy image
        dst_img_path = os.path.join(images_dir, img_filename)
        shutil.copy2(img_path, dst_img_path)
        
        # Create YOLO annotation file
        dst_label_path = os.path.join(labels_dir, label_filename)
        with open(dst_label_path, 'w') as f:
            for obj in objects:
                brand_name = obj['name']
                bbox = obj['bbox']
                
                # Get class ID
                if brand_name not in brand_to_id:
                    print(f"Warning: Brand {brand_name} not in brand list!")
                    continue
                
                class_id = brand_to_id[brand_name]
                
                # Convert bbox to YOLO format
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    img_width, img_height, bbox
                )
                
                # Write YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(data_list)} images")
    
    print(f"Completed {split_name} split: {len(data_list)} images")


def create_data_yaml(brands, output_path):
    """Create data.yaml configuration file for YOLO"""
    yaml_path = os.path.join(output_path, 'data.yaml')
    
    with open(yaml_path, 'w') as f:
        f.write(f"# LogoDet-3K dataset in YOLO format\n")
        f.write(f"path: {output_path}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n")
        f.write(f"\n")
        f.write(f"# Number of classes\n")
        f.write(f"nc: {len(brands)}\n")
        f.write(f"\n")
        f.write(f"# Class names\n")
        f.write(f"names:\n")
        for idx, brand in enumerate(brands):
            f.write(f"  {idx}: {brand}\n")
    
    print(f"Created data.yaml at {yaml_path}")


def main():
    print("=" * 60)
    print("LogoDet-3K to YOLO Converter")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check if source dataset exists
    if not os.path.exists(LOGODET3K_PATH):
        print(f"Error: Dataset not found at {LOGODET3K_PATH}")
        return
    
    # Collect all brands and images
    brand_to_images, all_brands = collect_all_brands_and_images()
    
    if not all_brands:
        print("Error: No brands found in dataset!")
        return
    
    # Create brand to ID mapping
    brand_to_id = {brand: idx for idx, brand in enumerate(all_brands)}
    
    print(f"\nBrand to ID mapping created ({len(brand_to_id)} classes)")
    print(f"Sample brands: {list(all_brands)[:5]}...")
    
    # Create YOLO directory structure
    create_yolo_structure(OUTPUT_PATH)
    
    # Split dataset
    train_data, val_data, test_data = split_dataset(brand_to_images)
    
    # Process and copy data for each split
    process_and_copy_data(train_data, 'train', brand_to_id, OUTPUT_PATH)
    process_and_copy_data(val_data, 'val', brand_to_id, OUTPUT_PATH)
    process_and_copy_data(test_data, 'test', brand_to_id, OUTPUT_PATH)
    
    # Create data.yaml
    create_data_yaml(all_brands, OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("Conversion completed successfully!")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_PATH}")
    print(f"Total classes: {len(all_brands)}")
    print(f"Train images: {len(train_data)}")
    print(f"Val images: {len(val_data)}")
    print(f"Test images: {len(test_data)}")
    print("\nYou can now use this dataset with YOLOv8:")
    print(f"  model.train(data='{os.path.join(OUTPUT_PATH, 'data.yaml')}')")


if __name__ == '__main__':
    main()

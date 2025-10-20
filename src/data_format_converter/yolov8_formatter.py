import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import shutil
from ultralytics import YOLO
from IPython.display import display, Image
import matplotlib.image as mpimg

# ================================
# 1. Define Paths and Class Mapping
# ================================
images_dir = 'dataset/images/jpegimages-trainval'  # Path to images
annotations_hbb_dir = 'dataset/annotations/horizontal bounding boxes'  # HBB annotations (XML)
annotations_obb_dir = 'dataset/annotations/oriented bounding boxes'  # OBB annotations (not processed here)
output_dir = 'dataset/YOLOv8'  # Output directory for YOLO formatted annotations

class_mapping = {
    'golffield': 0,
    'Expressway-toll-station': 1,
    'vehicle': 2,
    'trainstation': 3,
    'chimney': 4,
    'storagetank': 5,
    'ship': 6,
    'harbor': 7,
    'airplane': 8,
    'groundtrackfield': 9,
    'tenniscourt': 10,
    'dam': 11,
    'basketballcourt': 12,
    'Expressway-Service-area': 13,
    'stadium': 14,
    'airport': 15,
    'baseballfield': 16,
    'bridge': 17,
    'windmill': 18,
    'overpass': 19
}

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Inverse mapping (for later counting, mapping class id back to class name)
id_to_class = {v: k for k, v in class_mapping.items()}
# Function to convert bounding box coordinates to YOLO format
def convert_coordinates(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

# Function to process XML files
def process_xml_annotations(annotations_dir, output_dir, is_hbb=True):
    print(f"Processing {'HBB' if is_hbb else 'OBB'} annotations in {annotations_dir}...")
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    print(f"Found {len(xml_files)} XML files.")

    for annotation_file in xml_files:
        print(f"\nProcessing file: {annotation_file}")

        # Parse XML file
        xml_path = os.path.join(annotations_dir, annotation_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"Error parsing {annotation_file}: {e}")
            continue

        # Get image size
        size = root.find('size')
        if size is None:
            print(f"Warning: No size information in {annotation_file}")
            continue
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        print(f"Image size: {width}x{height}")

        # Get corresponding image file
        image_name = root.find('filename').text
        image_path = os.path.join(images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        print(f"Image found: {image_path}")

        # Write YOLO formatted annotations to output file
        output_annotation_path = os.path.join(output_dir, os.path.splitext(annotation_file)[0] + '.txt')
        with open(output_annotation_path, 'w') as f:
            objects = root.findall('object')
            if not objects:
                print(f"No objects found in {annotation_file}")
                continue
            print(f"Found {len(objects)} objects in {annotation_file}")

            for obj in objects:
                cls = obj.find('name').text
                if cls not in class_mapping:
                    print(f"Skipping unknown class: {cls}")
                    continue
                cls_id = class_mapping[cls]

                if is_hbb:
                    # Process HBB annotations
                    bbox = obj.find('bndbox')
                    if bbox is None:
                        print(f"Warning: No bounding box in object for class {cls}")
                        continue
                    xmin = float(bbox.find('xmin').text)
                    ymin = float(bbox.find('ymin').text)
                    xmax = float(bbox.find('xmax').text)
                    ymax = float(bbox.find('ymax').text)

                    # Convert bounding box coordinates to YOLO format
                    bbox_yolo = convert_coordinates((width, height), (xmin, xmax, ymin, ymax))
                else:
                    # Process OBB annotations (skip for now, as YOLO does not natively support OBB)
                    print(f"Skipping OBB annotation for class {cls}")
                    continue

                # Write YOLO formatted annotation to output file
                f.write(f"{cls_id} {' '.join([str(coord) for coord in bbox_yolo])}\n")
                print(f"Written object: class={cls}, bbox={bbox_yolo}")

# Process HBB annotations
process_xml_annotations(annotations_hbb_dir, output_dir, is_hbb=True)

# Process OBB annotations (optional, as YOLO does not natively support OBB)
# process_xml_annotations(annotations_obb_dir, output_dir, is_hbb=False)
# ================================
# 2. Convert XML Annotations to YOLO Format
# ================================


# Process only HBB annotations for YOLO conversion


print("\nConversion to YOLOv8 format completed.")
print("Class Mapping:")
print(class_mapping)

# ================================
# 3. Count Objects per Class from YOLO Annotations
# ================================
# Here, we iterate over the YOLO annotation (.txt) files in output_dir
class_counts = {class_name: 0 for class_name in class_mapping}

for annotation_file in os.listdir(output_dir):
    if not annotation_file.endswith('.txt'):
        continue
    annotation_path = os.path.join(output_dir, annotation_file)
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            class_name = id_to_class.get(cls_id, None)
            if class_name is not None:
                class_counts[class_name] += 1

# Sort classes based on number of objects (descending)
sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

# Plot number of instances per class
plt.figure(figsize=(12, 6))
sns.barplot(x=list(sorted_class_counts.keys()), y=list(sorted_class_counts.values()), palette="hls")
plt.title('Number of Objects per Class (Sorted)')
plt.xlabel('Class')
plt.ylabel('Number of Objects')
plt.xticks(rotation=90)
plt.show()

# ================================
# 4. Split Dataset into Train, Validation, and Test Sets
# ================================
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'
subfolders = ['images', 'labels']
split_ratios = [0.7, 0.15, 0.15]

# Create directories for each split with subfolders for images and labels
for directory in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(directory, subfolders[0]), exist_ok=True)
    os.makedirs(os.path.join(directory, subfolders[1]), exist_ok=True)

# Get list of image files and shuffle them
image_files = os.listdir(images_dir)
random.shuffle(image_files)

total_images = len(image_files)
num_train = int(split_ratios[0] * total_images)
num_val = int(split_ratios[1] * total_images)
num_test = total_images - num_train - num_val

# Copy images and corresponding YOLO annotations to the train, val, and test directories
for i, split in enumerate([train_dir, val_dir, test_dir]):
    # Calculate start and end indices for the current split
    start_index = sum([int(r * total_images) for r in split_ratios[:i]])
    if split == train_dir:
        num_images = num_train
    elif split == val_dir:
        num_images = num_val
    else:
        num_images = num_test
    end_index = start_index + num_images

    for image_file in image_files[start_index:end_index]:
        # Copy image file
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(split, subfolders[0], image_file))
        
        # Copy corresponding annotation file (YOLO .txt format)
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_src = os.path.join(output_dir, annotation_file)
        annotation_dst = os.path.join(split, subfolders[1], annotation_file)
        if os.path.exists(annotation_src):
            shutil.copy(annotation_src, annotation_dst)
        else:
            # If no annotation exists, create an empty file
            with open(annotation_dst, 'w') as f:
                f.write('')

print("Dataset split into train, val, and test folders with images and labels subfolders.")

# ================================
# 5. Count and Plot Class Distributions for Each Split
# ================================
def count_classes_in_folder(folder):
    counts = {class_name: 0 for class_name in class_mapping}
    labels_folder = os.path.join(folder, 'labels')
    if not os.path.exists(labels_folder):
        return counts
    for label_file in os.listdir(labels_folder):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_folder, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                class_name = id_to_class.get(cls_id, None)
                if class_name is not None:
                    counts[class_name] += 1
    return counts

train_class_counts = count_classes_in_folder(train_dir)
val_class_counts = count_classes_in_folder(val_dir)
test_class_counts = count_classes_in_folder(test_dir)

# Sort counts by class names (alphabetically)
sorted_train_class_counts = dict(sorted(train_class_counts.items()))
sorted_val_class_counts = dict(sorted(val_class_counts.items()))
sorted_test_class_counts = dict(sorted(test_class_counts.items()))

# Plot using subplots: Train, Validation, Test
plt.figure(figsize=(20, 8))

# Train subplot
plt.subplot(1, 3, 1)
plt.bar(list(sorted_train_class_counts.keys()), list(sorted_train_class_counts.values()), 
        color=sns.color_palette("hls", len(train_class_counts)))
plt.title('Train')
plt.ylabel('Number of Instances', fontweight='bold')
plt.xticks(rotation=90)
for key, value in sorted_train_class_counts.items():
    plt.text(key, value, str(value), ha='center', va='bottom')

# Validation subplot
plt.subplot(1, 3, 2)
plt.bar(list(sorted_val_class_counts.keys()), list(sorted_val_class_counts.values()), 
        color=sns.color_palette("hls", len(val_class_counts)))
plt.title('Validation')
plt.xlabel('Class', fontweight='bold')
plt.xticks(rotation=90)
for key, value in sorted_val_class_counts.items():
    plt.text(key, value, str(value), ha='center', va='bottom')

# Test subplot
plt.subplot(1, 3, 3)
plt.bar(list(sorted_test_class_counts.keys()), list(sorted_test_class_counts.values()), 
        color=sns.color_palette("hls", len(test_class_counts)))
plt.title('Test')
plt.xticks(rotation=90)
for key, value in sorted_test_class_counts.items():
    plt.text(key, value, str(value), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Combined horizontal bar plot for class distribution in each split
plt.figure(figsize=(10, 15))
class_names = sorted(train_class_counts.keys())
num_classes = len(class_names)
bar_height = 0.3
index = np.arange(num_classes)

plt.barh(index, [sorted_train_class_counts[cls] for cls in class_names], bar_height,
         label='Train', color='skyblue')
plt.barh(index + bar_height, [sorted_val_class_counts[cls] for cls in class_names], bar_height,
         label='Validation', color='lightgreen')
plt.barh(index + 2 * bar_height, [sorted_test_class_counts[cls] for cls in class_names], bar_height,
         label='Test', color='salmon')

plt.title('Class Distribution in Train, Validation, and Test Sets')
plt.xlabel('Number of Instances', fontweight='bold')
plt.ylabel('Class', fontweight='bold')
plt.yticks(index + bar_height, class_names)
plt.legend()
plt.tight_layout()
plt.show()

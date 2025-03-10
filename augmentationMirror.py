import os
import cv2
import numpy as np
import imgaug.augmenters as iaa
from tqdm import tqdm
import random

ROOT_DIR = os.getcwd()
OID_DIR = os.path.join(ROOT_DIR, 'OID')
OID_NORMALIZED_DIR = os.path.join(ROOT_DIR, 'OID_normalized')
ROBOFLOW_DIR = os.path.join(ROOT_DIR, 'Roboflow')
CLASSES_FILE = os.path.join(ROOT_DIR, 'classes.txt')
CLASSES_ROBOFLOW_FILE = os.path.join(ROOT_DIR, 'classes_roboflow.txt')

# Augmentadores: espejo horizontal + zoom aleatorio (1.0 a 1.2)
flip_augmenter_horizontal = iaa.Fliplr(1.0)  # 100% de probabilidad de aplicar espejo
flip_augmenter_vertical = iaa.Flipud(1.0) 

# Directorios base
base_dirs = ["train", "validation", "test"]

# Procesar cada conjunto de datos (train, validation, test)
for base_dir in base_dirs:
    base_path = os.path.join(OID_NORMALIZED_DIR, base_dir)
    if not os.path.exists(base_path):
        continue  # Saltar si el directorio no existe

    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        images_path = os.path.join(class_path, "images")
        labels_path = os.path.join(class_path, "labels")

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            continue  # Saltar si no existen ambas carpetas

        # Procesar cada imagen
        for img_name in tqdm(os.listdir(images_path), desc=f"Procesando {class_name} en {base_dir}"):
            if not img_name.endswith((".jpg", ".png")):
                continue  # Saltar archivos no imagen
            
            img_path = os.path.join(images_path, img_name)
            label_path = os.path.join(labels_path, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

            img = cv2.imread(img_path)
            if img is None or not os.path.exists(label_path):
                continue  # Saltar imágenes corruptas o sin etiquetas

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape

            # Leer etiquetas YOLO
            with open(label_path, "r") as f:
                lines = f.readlines()

            boxes = []
            class_labels = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                boxes.append([x_center, y_center, w, h])
                class_labels.append(class_id)

            # Aplicar espejo horizontal (solo espejo)
            flipped_img = flip_augmenter_horizontal(image=img)

            # Ajustar bounding boxes para el espejo
            flipped_boxes = [[1.0 - x, y, w, h] for x, y, w, h in boxes]

            # Guardar imagen con solo el espejo
            output_img_name_flip = f"flipH_{img_name}"
            output_img_path_flip = os.path.join(images_path, output_img_name_flip)
            cv2.imwrite(output_img_path_flip, cv2.cvtColor(flipped_img, cv2.COLOR_RGB2BGR))

            # Guardar etiquetas YOLO ajustadas para el espejo
            output_label_name_flip = output_img_name_flip.replace(".jpg", ".txt").replace(".png", ".txt")
            output_label_path_flip = os.path.join(labels_path, output_label_name_flip)
            with open(output_label_path_flip, "w") as f:
                for class_id, (x, y, w, h) in zip(class_labels, flipped_boxes):
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


            img_both = cv2.imread(output_img_path_flip)
            if img_both is None or not os.path.exists(output_label_path_flip):
                continue  # Saltar imágenes corruptas o sin etiquetas

            img_both = cv2.cvtColor(img_both, cv2.COLOR_BGR2RGB)
            height, width, _ = img_both.shape

            # Leer etiquetas YOLO
            with open(output_label_path_flip, "r") as f:
                lines = f.readlines()

            boxes = []
            class_labels = []
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                boxes.append([x_center, y_center, w, h])
                class_labels.append(class_id)

            # Aplicar espejo vertical (solo espejo)
            flipped_img = flip_augmenter_vertical(image=img_both)

            # Ajustar bounding boxes para el espejo
            flipped_boxes = [[x, 1.0 - y, w, h] for x, y, w, h in boxes]

            # Guardar imagen con solo el espejo
            output_img_name_flip = f"flipV_{output_img_name_flip}"
            output_img_path_flip = os.path.join(output_class_images, output_img_name_flip)
            cv2.imwrite(output_img_path_flip, cv2.cvtColor(flipped_img, cv2.COLOR_RGB2BGR))

            # Guardar etiquetas YOLO ajustadas para el espejo
            output_label_name_flip = output_img_name_flip.replace(".jpg", ".txt").replace(".png", ".txt")
            output_label_path_flip = os.path.join(output_class_labels, output_label_name_flip)
            with open(output_label_path_flip, "w") as f:
                for class_id, (x, y, w, h) in zip(class_labels, flipped_boxes):
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

            
            # Aplicar espejo vertical (solo espejo)
            flipped_img = flip_augmenter_vertical(image=img)

            # Ajustar bounding boxes para el espejo
            flipped_boxes = [[x, 1.0 - y, w, h] for x, y, w, h in boxes]

            # Guardar imagen con solo el espejo
            output_img_name_flip = f"flipV_{img_name}"
            output_img_path_flip = os.path.join(images_path, output_img_name_flip)
            cv2.imwrite(output_img_path_flip, cv2.cvtColor(flipped_img, cv2.COLOR_RGB2BGR))

            # Guardar etiquetas YOLO ajustadas para el espejo
            output_label_name_flip = output_img_name_flip.replace(".jpg", ".txt").replace(".png", ".txt")
            output_label_path_flip = os.path.join(labels_path, output_label_name_flip)
            with open(output_label_path_flip, "w") as f:
                for class_id, (x, y, w, h) in zip(class_labels, flipped_boxes):
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

print("✅ Data Augmentation completada y guardada en las carpetas correspondientes")

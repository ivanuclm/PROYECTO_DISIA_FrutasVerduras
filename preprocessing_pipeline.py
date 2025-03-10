import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import imgaug.augmenters as iaa
import random
from roboflow import Roboflow


# Rutas principales
ROOT_DIR = os.getcwd()
OID_DIR = os.path.join(ROOT_DIR, 'OID')
OID_NORMALIZED_DIR = os.path.join(ROOT_DIR, 'OID_normalized')
ROBOFLOW_DIR = os.path.join(ROOT_DIR, 'Roboflow')
CLASSES_FILE = os.path.join(ROOT_DIR, 'classes.txt')
CLASSES_ROBOFLOW_FILE = os.path.join(ROOT_DIR, 'classes_roboflow.txt')

print("🔹 Cargando clases desde classes.txt... ✅")

# Cargar clases y asignar índices
with open(CLASSES_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
class_index = {name: idx for idx, name in enumerate(class_names)}

# Paso 0: Descargamos productos adicionales de Roboflow
def download_roboflow_datasets():
    print("📥 Descargando datasets adicionales de Roboflow...")
    os.makedirs(ROBOFLOW_DIR, exist_ok=True)
    datasets = {
        "Avocado": ("latihancnn", "alpukat-pepaya", 1),
        "Bellpepper": ("swarajs-workspace", "capsicum-detection-5dzng", 1),
        "Cauliflower": ("swarajs-workspace", "cauli-flower-detection", 1),
        "Garlic": ("workplace-bz64d", "garlic-uerzz", 1),
        "Kiwi": ("pham-van-loc", "yolox3", 2),
        "Onion": ("yeoworm", "onion-dkyks", 2),
        "Plum": ("shj1864-naver-com", "plum-saygz", 1)
    }
    
    rf = Roboflow(api_key="jP9akRvjCWtBSSZB2sWk")
    
    for category, (workspace, project_name, version) in datasets.items():
        print(f"🔄 Descargando {category}...")
        project = rf.workspace(workspace).project(project_name)
        dataset = project.version(version).download("yolov7")
        extracted_path = os.path.join(ROOT_DIR, dataset.location)
        renamed_path = os.path.join(ROBOFLOW_DIR, category)
        shutil.move(extracted_path, renamed_path)
        print(f"✅ {category} descargado y renombrado correctamente.")
    
    print("📥 Todas las descargas de Roboflow completadas!")

# Paso 1: Limpieza de datos (Quitamos Grapefruit y Coconut porque se ha detectado que las imágenes no son válidas)
def clean_data(categories):
    categories_list = categories.split(", ")
    print("🧹 Limpiando las clases inválidas:", categories_list)

    for category in categories:
        if category in class_index:
            for subset in ["train", "test", "validation"]:
                subset_path = os.path.join(OID_DIR, subset, category)
                if os.path.exists(subset_path):
                    shutil.rmtree(subset_path)
    print("✅ Limpieza de datos completada!")

# Paso 2: Convertir anotaciones de OID a formato YOLOv7
def convert_annotations():
    print("📝 Convirtiendo anotaciones de OID a YOLOv7...")
    os.chdir(OID_DIR)
    DIRS = os.listdir(OID_DIR)
    
    for DIR in DIRS:
        if os.path.isdir(DIR):
            os.chdir(DIR)
            print(f"📂 Procesando conjunto: {DIR}")
            
            for category in os.listdir(os.getcwd()):
                if os.path.isdir(category) and category in class_index:
                    os.chdir(category)
                    print(f"🔹 Convirtiendo anotaciones para: {category}")
                    os.chdir("Label")
                    
                    for filename in tqdm(os.listdir(os.getcwd())):
                        if filename.endswith(".txt"):
                            annotations = []
                            with open(filename) as f:
                                lines = f.readlines()
                            
                            for line in lines:
                                labels = line.split()
                                coords = list(map(float, labels[1:]))
                                image_file = os.path.join("..", labels[0] + ".jpg")
                                image = cv2.imread(image_file)
                                
                                if image is not None:
                                    coords[2] -= coords[0]
                                    coords[3] -= coords[1]
                                    x_diff, y_diff = coords[2] / 2, coords[3] / 2
                                    coords[0] = (coords[0] + x_diff) / image.shape[1]
                                    coords[1] = (coords[1] + y_diff) / image.shape[0]
                                    coords[2] /= image.shape[1]
                                    coords[3] /= image.shape[0]
                                    
                                    newline = f"{class_index[category]} " + " ".join(map(str, coords)) + "\n"
                                    annotations.append(newline)
                                
                            with open(filename, "w") as outfile:
                                outfile.writelines(annotations)
                        
                    os.chdir(".."); os.chdir("..");
            os.chdir("..");
    os.chdir(ROOT_DIR)
    print("✅ Conversión de anotaciones completada!")

# Paso 3: Organizar dataset en OID_normalized
def organize_oid_dataset():
    print("📂 Organizamos los datos en OID_normalized...")
    for subset in ["train", "test", "validation"]:
        subset_source = os.path.join(OID_DIR, subset)
        subset_dest = os.path.join(OID_NORMALIZED_DIR, subset)
        os.makedirs(subset_dest, exist_ok=True)
        
        for category in os.listdir(subset_source):
            category_source = os.path.join(subset_source, category)
            category_dest = os.path.join(subset_dest, category)
            images_dest = os.path.join(category_dest, "images")
            labels_dest = os.path.join(category_dest, "labels")
            os.makedirs(images_dest, exist_ok=True)
            os.makedirs(labels_dest, exist_ok=True)
            
            for file in os.listdir(category_source):
                file_path = os.path.join(category_source, file)
                if file.endswith(('.jpg', '.png')):
                    shutil.copy(file_path, images_dest)
                elif file.endswith('.txt'):
                    shutil.copy(file_path, labels_dest)
    print("✅ Organización de datos completada!")

# Paso 4: Integrar datasets de Roboflow
def integrate_roboflow():
    print("🔄 Integrando datasets de Roboflow...")
    with open(CLASSES_ROBOFLOW_FILE, "r") as f:
        roboflow_classes = [line.strip() for line in f.readlines()]
    roboflow_index = {name: idx for idx, name in enumerate(roboflow_classes)}
    
    for category in os.listdir(ROBOFLOW_DIR):
        category_path = os.path.join(ROBOFLOW_DIR, category)
        if os.path.isdir(category_path) and category in roboflow_index:
            for subset in ["train", "test", "valid"]:
                source_subset = os.path.join(category_path, subset)
                target_subset = os.path.join(OID_NORMALIZED_DIR, subset, category)
                os.makedirs(target_subset, exist_ok=True)
                images_dest = os.path.join(target_subset, "images")
                labels_dest = os.path.join(target_subset, "labels")
                os.makedirs(images_dest, exist_ok=True)
                os.makedirs(labels_dest, exist_ok=True)
                
                for img_file in os.listdir(os.path.join(source_subset, "images")):
                    shutil.copy(os.path.join(source_subset, "images", img_file), images_dest)
                
                for label_file in os.listdir(os.path.join(source_subset, "labels")):
                    label_path = os.path.join(source_subset, "labels", label_file)
                    new_label_path = os.path.join(labels_dest, label_file)
                    
                    with open(label_path, "r") as f:
                        lines = f.readlines()
                    new_lines = [f"{roboflow_index[category]} " + " ".join(line.split()[1:]) + "\n" for line in lines]
                    with open(new_label_path, "w") as f:
                        f.writelines(new_lines)
    print("✅ Integración de Roboflow completada!")

# Paso 5: Data Augmentation con espejo horizontal y vertical
def augmentations_mirror():
    print("🔄 Aplicando Data Augmentation (Espejo Horizontal y Vertical) sobre las imágenes y sus etiquetas...")
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
   
# Función principal
def main():
    clean_categories = ["Grapefruit", "Coconut"]
    clean_data(clean_categories)
    convert_annotations()
    organize_oid_dataset()
    integrate_roboflow()
    augmentations_mirror()
    print("🎉 ¡Preprocesamiento de datos completado! Listo para entrenamiento en YOLOv7")

if __name__ == "__main__":
    main()
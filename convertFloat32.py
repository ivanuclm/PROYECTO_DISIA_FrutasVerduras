import os
import cv2
import numpy as np
from tqdm import tqdm

# Directorio base donde están las imágenes (ajusta según tu estructura)
BASE_DIR = "OID_normalized_pruebas"

def convert_images_to_float32():
    print("🔄 Convertiendo imágenes a float32...")

    for base_dir in ["train", "validation", "test"]:
        base_path = os.path.join(BASE_DIR, base_dir)
        if not os.path.exists(base_path):
            print(f"⚠️ {base_path} no existe. Saltando...")
            continue

        for class_name in os.listdir(base_path):
            class_path = os.path.join(base_path, class_name)
            images_path = os.path.join(class_path, "images")

            if not os.path.exists(images_path):
                print(f"⚠️ No se encontró carpeta de imágenes en {class_path}. Saltando...")
                continue

            for img_name in tqdm(os.listdir(images_path), desc=f"Procesando {class_name} en {base_dir}"):
                if not img_name.endswith((".jpg", ".png")):
                    continue  # Saltar archivos que no sean imágenes

                img_path = os.path.join(images_path, img_name)

                # Leer imagen
                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ No se pudo leer {img_name}. Saltando...")
                    continue  # Saltar imágenes corruptas

                # Convertir a float32 y normalizar a [0,1]
                img = img.astype(np.float32) / 255.0

                # Volver a guardar la imagen (OpenCV solo guarda uint8, así que la reescalamos)
                cv2.imwrite(img_path, (img * 255).astype(np.uint8))

    print("✅ Conversión a float32 completada!")

if __name__ == "__main__":
    convert_images_to_float32()

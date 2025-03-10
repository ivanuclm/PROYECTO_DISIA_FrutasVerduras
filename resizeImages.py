import os
import cv2
from tqdm import tqdm

# Directorio raíz de las imágenes OID
OID_DIR = "OID_pruebas/Dataset"
TARGET_SIZE = (640, 640)

def resize_images_and_labels():
    print("📏 Redimensionando imágenes a 640x640 y ajustando anotaciones...")
    
    for subset in ["train", "test", "validation"]:
        subset_path = os.path.join(OID_DIR, subset)
        
        if not os.path.exists(subset_path):
            print(f"⚠️ {subset} no encontrado, saltando...")
            continue

        for category in os.listdir(subset_path):
            category_path = os.path.join(subset_path, category)
            # images_path = os.path.join(category_path, "")
            images_path = category_path
            labels_path = os.path.join(category_path, "Label")

            if not os.path.exists(images_path) or not os.path.exists(labels_path):
                print(f"⚠️ No se encontró {category} en {subset}, saltando...")
                continue
            
            print(f"📂 Procesando {category} en {subset}...")

            for img_name in tqdm(os.listdir(images_path), desc=f"Redimensionando {category}"):
                img_path = os.path.join(images_path, img_name)
                label_path = os.path.join(labels_path, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

                if not os.path.exists(label_path):
                    if img_name != "Label":
                        print(f"⚠️ No hay etiqueta para {img_name}, saltando...")
                    continue

                # Leer imagen
                img = cv2.imread(img_path)
                if img is None:
                    print(f"❌ No se pudo leer {img_name}, saltando...")
                    continue

                # Obtener dimensiones originales
                h_orig, w_orig, _ = img.shape

                # Redimensionar imagen a 640x640
                img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                cv2.imwrite(img_path, img_resized)

                # Ajustar etiquetas YOLOv7
                with open(label_path, "r") as f:
                    lines = f.readlines()

                new_annotations = []
                for line in lines:
                    parts = line.strip().split()
                    class_id = parts[0]
                    x_center, y_center, w, h = map(float, parts[1:])

                    # Ajustar coordenadas
                    x_center = x_center * w_orig / 640
                    y_center = y_center * h_orig / 640
                    w = w * w_orig / 640
                    h = h * h_orig / 640

                    new_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

                # Guardar etiquetas corregidas
                with open(label_path, "w") as f:
                    f.writelines(new_annotations)

    print("✅ Redimensionado de imágenes y ajuste de etiquetas completado!")

if __name__ == "__main__":
    resize_images_and_labels()

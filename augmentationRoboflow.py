import os
import shutil

# Definir las carpetas base
roboflow_dir = "Roboflow"
oid_normalized_dir = "OID_normalized"

# Cargar la lista de clases y asignar índices
classes_file = "classes_roboflow.txt"
with open(classes_file, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

class_index = {name: idx for idx, name in enumerate(class_names)}

# Subconjuntos de datos
subsets = ["train", "test", "valid"]
subset_mapping = {"valid": "validation"}  # Renombrar 'valid' a 'validation'

# Procesar cada categoría en Roboflow
for category in os.listdir(roboflow_dir):
    category_path = os.path.join(roboflow_dir, category)

    if not os.path.isdir(category_path) or category not in class_index:
        print(f"⚠️ Saltando {category}: No está en classes_roboflow.txt o no es una carpeta válida.")
        continue

    class_id = class_index[category]  # Obtener el índice de la clase

    for subset in subsets:
        source_subset = os.path.join(category_path, subset)
        target_subset = os.path.join(oid_normalized_dir, subset_mapping.get(subset, subset), category)

        if not os.path.exists(source_subset):
            print(f"⚠️ No se encontró {source_subset}, saltando.")
            continue

        # Crear estructura de carpetas destino
        images_dest = os.path.join(target_subset, "images")
        labels_dest = os.path.join(target_subset, "labels")
        os.makedirs(images_dest, exist_ok=True)
        os.makedirs(labels_dest, exist_ok=True)

        # Mover imágenes
        for img_file in os.listdir(os.path.join(source_subset, "images")):
            shutil.copy(os.path.join(source_subset, "images", img_file), os.path.join(images_dest, img_file))

        # Procesar etiquetas
        for label_file in os.listdir(os.path.join(source_subset, "labels")):
            label_path = os.path.join(source_subset, "labels", label_file)
            new_label_path = os.path.join(labels_dest, label_file)

            with open(label_path, "r") as f:
                lines = f.readlines()

            # Reemplazar la primera columna con el índice correcto
            new_lines = [f"{class_id} " + " ".join(line.split()[1:]) + "\n" for line in lines]

            with open(new_label_path, "w") as f:
                f.writelines(new_lines)

        print(f"✅ {category} agregado en {target_subset}")

print("🚀 Fusión de datos completada.")

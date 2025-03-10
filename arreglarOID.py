import os
import shutil

# Directorios de origen y destino
source_dir = "OID/Dataset"
destination_dir = "OID_normalized"
subsets = ["train", "test", "validation"]

# Crear estructura de carpetas en el destino
for subset in subsets:
    subset_source = os.path.join(source_dir, subset)
    subset_dest = os.path.join(destination_dir, subset)
    os.makedirs(subset_dest, exist_ok=True)
    
    # Obtener todas las clases en el conjunto actual
    categories = [d for d in os.listdir(subset_source) if os.path.isdir(os.path.join(subset_source, d))]
    
    for category in categories:
        category_source = os.path.join(subset_source, category)
        category_dest = os.path.join(subset_dest, category)
        
        # Crear carpetas images y labels en el destino
        images_dest = os.path.join(category_dest, "images")
        labels_dest = os.path.join(category_dest, "labels")
        os.makedirs(images_dest, exist_ok=True)
        os.makedirs(labels_dest, exist_ok=True)
        
        # Mover imágenes y etiquetas normalizadas (que están sueltas)
        for file in os.listdir(category_source):
            file_path = os.path.join(category_source, file)
            
            if file.endswith(('.jpg', '.png', '.jpeg')):
                shutil.copy(file_path, os.path.join(images_dest, file))
            elif file.endswith('.txt') and "Labels" not in file_path:  # Evitar las etiquetas sin normalizar
                shutil.copy(file_path, os.path.join(labels_dest, file))

print("Organización de archivos completada en 'OID_normalized'.")
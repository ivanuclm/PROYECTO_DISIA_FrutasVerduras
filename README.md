# **Descarga y Preparación del Dataset de Imágenes para YOLOv7**

Este documento describe los pasos para descargar el dataset de imágenes de **Open Images Dataset** utilizando **OIDv4_ToolKit** y transformarlo al formato **YOLOv7** de forma preliminar, permitiendo modificaciones en caso de utilizar otras versiones en el futuro.

---

## **1️⃣ Instalación de YOLOv7**
El primer paso es clonar el repositorio oficial de **YOLOv7** y asegurarnos de que tenemos todas las dependencias necesarias instaladas.

```bash
mkdir PROYECTO_Grupo4
cd PROYECTO_Grupo4
git clone https://github.com/WongKinYiu/yolov7.git YOLOv7
cd YOLOv7
pip install -r requirements.txt
cd ..
```
Esto instalará **YOLOv7** y sus dependencias para futuras pruebas y entrenamiento.

---

## **2️⃣ Instalación de OIDv4_ToolKit para descargar Open Images Dataset**
Ahora descargamos e instalamos **OIDv4_ToolKit**, la herramienta que utilizaremos para descargar imágenes y etiquetas del **Open Images Dataset**.

```bash
git clone https://github.com/theAIGuysCode/OIDv4_ToolKit.git
cd OIDv4_ToolKit/
pip install -r requirements.txt
pip install urllib3==1.25.10  # Evita problemas de compatibilidad
```

---

## **3️⃣ Descarga del Dataset de Open Images**

Para descargar las imágenes y sus etiquetas, utilizamos el siguiente comando, asegurándonos de ejecutarlo en un **terminal** y **no en una libreta de Jupyter**, ya que utiliza la función `os.get_terminal_size()` que no funciona en las celdas de Jupyter.

```bash
python main.py downloader -y --classes "Apple" "Grape" "Pear" "Strawberry" "Tomato" "Lemon" "Banana" "Orange" "Peach" "Mango" "Pineapple" "Grapefruit" "Pomegranate" "Watermelon" "Cantaloupe" "Cucumber" "Radish" "Artichoke" "Potato" "Asparagus" "Pumpkin" "Zucchini" "Cabbage" "Carrot" "Broccoli" --type_csv train --limit 5000
```

📌 **Explicación del comando:**
- `downloader` → Llama a la función de descarga de imágenes.
- `-y` → Acepta automáticamente los términos de Open Images.
- `--classes` → Lista de clases de frutas y verduras a descargar.
- `--type_csv train` → Descarga imágenes del conjunto de entrenamiento.
- `--limit 5000` → Limita la descarga a **5000 imágenes por clase**.

---

## **4️⃣ Organización del Dataset**

Una vez completada la descarga, se crea una carpeta llamada **"OID"** dentro del directorio `OIDv4_ToolKit/`. Debemos **mover esta carpeta** al directorio raíz del proyecto **"PROYECTO_Grupo4"** para poder ejecutar el resto de los scripts de normalización y agregación de datos.

```bash
mv OIDv4_ToolKit/OID PROYECTO_Grupo4/OID
```

Esta carpeta **OID** contiene:
- **Imágenes descargadas**.
- **Etiquetas sin normalizar**.

El siguiente paso será ejecutar los scripts de normalización y fusión con datasets externos (ej. **Roboflow**) para generar el dataset final **OID_normalized**.

---

📌 **Próximos Pasos:**
✅ Normalización de etiquetas y conversión a formato YOLO.
✅ Integración con otras fuentes de datos para clases faltantes.
✅ Preparación del dataset final para entrenamiento de YOLOv7.

🚀 **El dataset final se guardará en `OID_normalized` con imágenes y anotaciones en formato YOLO listo para entrenamiento.**


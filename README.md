# **Descarga y Preparación del Dataset de Imágenes para YOLOv7**

🚨 **¡ATENCIÓN!** 🚨
Si no deseas ejecutar la descarga con el Toolkit y prefieres saltarte este paso, puedes descargar directamente el dataset desde **Google Drive**. Solo tienes que:
1. Descargar la carpeta `OID` desde [este enlace](https://drive.google.com/drive/u/2/folders/1AeTtp5Fn6BXYLyS_WBxq7iNSwIE4_Gwk).
2. Mover `OID` a la carpeta del repositorio clonado `PROYECTO_DISIA_FrutasVerduras`, al mismo nivel que `preprocessing_pipeline.py`.
3. Ir directamente al **paso 6** y ejecutar el pipeline de preprocesamiento con:
   ```bash
   python preprocessing_pipeline.py
   ```
Esto generará `OID_normalized/` sin necesidad de descargar las imágenes con OIDv4Toolkit. También puedes descargar el dataset ya preprocesado directamente desde el enlace anterior.

La carpeta `Roboflow` dentro del Drive contiene los datasets seleccionados de Roboflow para la agregación de productos ya descargados de manera manual. Sin embargo, el script `preprocessing_pipeline.py` **se encarga de descargar automáticamente** estos datasets al ejecutarse. Estos archivos ocupan aproximadamente **300 MB**, por lo que la descarga es rápida y sin complicaciones.

---


Este documento describe los pasos para descargar el dataset de imágenes de **Open Images Dataset**, convertirlo al formato **YOLOv7**, y realizar el preprocesamiento necesario para el entrenamiento del modelo.


---

## **1️⃣ Requisitos del Proyecto**

Antes de ejecutar el preprocesamiento, asegúrate de tener instaladas todas las dependencias. Se recomienda usar **Python 3.11.9**.

📌 **Si usas Python 3.12**, instala `setuptools` para evitar problemas con `numpy`.

### **📌 Instalación de dependencias**
Ejecuta el siguiente comando para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

Si encuentras errores con `numpy` en **Python 3.12**, instala `setuptools` manualmente:

```bash
pip install setuptools==58.2.0
```

El archivo `requirements.txt` incluye:
- `numpy==1.23.5` (compatible con `imgaug`)
- `imgaug` (para data augmentation)
- `opencv-python` (procesamiento de imágenes)
- `shutil, os, random` (manejo de archivos y directorios)
- `argparse, fileinput, tqdm` (herramientas para scripts y progreso de tareas)
- `setuptools==58.2.0` (comentado, necesario si usas Python 3.12)
- `urllib3==1.25.10` (evita problemas de compatibilidad con `OIDv4_Toolkit`)

---

## **2️⃣ Instalación de YOLOv7**
Antes de nada, debes contar y trabajar con este repositorio descargado (mediante git clone, o descargando directamente el .zip).
El primer paso es clonar el repositorio oficial de **YOLOv7** en nuestra copia local del proyecto y asegurarnos de que tenemos todas las dependencias necesarias instaladas.

```bash
git clone https://github.com/WongKinYiu/yolov7.git YOLOv7
cd YOLOv7
pip install -r requirements.txt
cd ..
```
Esto instalará **YOLOv7** y sus dependencias para futuras pruebas y entrenamiento.

---

## **3️⃣ Instalación de OIDv4_ToolKit para descargar Open Images Dataset**
Ahora descargamos e instalamos **OIDv4_ToolKit**, la herramienta que utilizaremos para descargar imágenes y etiquetas del **Open Images Dataset**.

```bash
git clone https://github.com/theAIGuysCode/OIDv4_ToolKit.git
cd OIDv4_ToolKit/
pip install -r requirements.txt
pip install urllib3==1.25.10  # Evita problemas de compatibilidad
```

---

## **4️⃣ Descarga del Dataset de Open Images**

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

## **5️⃣ Organización del Dataset**

Una vez completada la descarga, se crea una carpeta llamada **"OID"** dentro del directorio `OIDv4_ToolKit/`. Debemos **mover esta carpeta** al directorio raíz del proyecto **"PROYECTO_Grupo4"** para poder ejecutar el resto de los scripts de normalización y agregación de datos.

```bash
mv OIDv4_ToolKit/OID PROYECTO_Grupo4/OID
```

Esta carpeta **OID** contiene:
- **Imágenes descargadas**.
- **Etiquetas sin normalizar**.

El siguiente paso será ejecutar los scripts de normalización y fusión con datasets externos (ej. **Roboflow**) para generar el dataset final **Dataset_FrutasVerduras**.

---

## **6️⃣ Ejecución del Pipeline de Preprocesamiento**
Una vez tenemos los datos descargados, ejecutamos el **pipeline de preprocesamiento** para:
✅ Convertir anotaciones a formato YOLOv7.  
✅ Organizar el dataset en `OID_normalized`.  
✅ Integrar datasets adicionales desde `Roboflow`.  
✅ Aplicar **Data Augmentation** (espejo horizontal y vertical, multiplicando x4 el número de imágenes).  

Ejecuta:

```bash
python preprocessing_pipeline.py
```

📌 Esto generará un dataset final en `OID_normalized/` con todas las imágenes y anotaciones listas para YOLOv7.

### **🛠 Solución si `preprocessing_pipeline.py` no funciona**
Si por algún motivo el script no se ejecuta correctamente, sigue estos pasos manualmente:

1. **Redimensionar a 640x640**:
   ```bash
   python resizeImages.py
   ```
2. **Convertir anotaciones a formato YOLOv7**:
   ```bash
   python convert_annotations.py
   ```
3. **Organizar el dataset en `OID_normalized`**:
   ```bash
   python arreglarOID.py
   ```
4. **Integrar datasets adicionales desde Roboflow**:
   ```bash
   python augmentationRoboflow.py
   ```
5. **Aplicar Data Augmentation**:
   ```bash
   python augmentationsMirror.py
   ```

Estos pasos garantizan que el dataset esté listo en `OID_normalized/`.


---

📌 **Próximos Pasos:**
✅ Verificar que el dataset funciona con YOLOv7 u otras versiones.
✅ Comenzar el entrenamiento del modelo escogido.  
✅ Evaluar el rendimiento del modelo con validaciones.

🚀 **El dataset final estará en `OID_normalized/`, listo para entrenamiento en YOLO.**
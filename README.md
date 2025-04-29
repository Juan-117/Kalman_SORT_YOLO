# 🚀 Kalman_SORT_YOLO: Detección y Seguimiento de Objetos

Este proyecto implementa detección de objetos en video usando **YOLOv8** y seguimiento de los objetos detectados utilizando **DeepSORT** (Filtro de Kalman + Asociación de apariencias).

El enfoque está especialmente ajustado para seguir **vehículos (carros)** de forma precisa, ignorando otros objetos como personas o trenes.

---

## 📦 Estructura del Proyecto

Kalman_SORT_YOLO/
├── env/                     # Entorno virtual (local, no en GitHub)
├── scripts/
│   ├── test_yolo_detection.py   # Solo detección con YOLOv8
│   └── test_yolo_deepsort.py    # Detección + Tracking con DeepSORT
├── videos/
│   └── test.mp4              # Video de prueba
├── .gitignore
├── .flake8
├── .pre-commit-config.yaml


---

## 🛠 Tecnologías usadas

- [Python 3.10+](https://www.python.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [deep_sort_realtime](https://github.com/Lev0r/deep_sort_realtime)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

## 🚀 ¿Cómo correr el proyecto?

### 1. Clona el repositorio

```bash
git clone https://github.com/Juan-117/Kalman_SORT_YOLO.git
cd Kalman_SORT_YOLO

### 2. Crea y activa un entorno virtual

```bash
python3 -m venv env
source env/bin/activate  # En Linux/Mac

### 3. Instala dependencias
```bash
pip install -r requirements.txt

O instalar
```bash
pip install ultralytics
pip install deep_sort_realtime
pip install opencv-python

### 4. Correr detección básica
```bash
python scripts/test_yolo_detection.py

### 5. Correr detección + tracking
```bash
python scripts/test_yolo_deepsort.py




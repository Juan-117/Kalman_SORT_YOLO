import cv2
import torch
from ultralytics import YOLO

# Información de GPU
print("¿GPU disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de GPU:", torch.cuda.get_device_name(0))

# Ruta absoluta al video
video_path = "/home/juan/Kalman_SORT_YOLO/videos/test.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

# Crear ventana reutilizable
cv2.namedWindow("Detección YOLOv8", cv2.WINDOW_NORMAL)

# Cargar modelo YOLOv8 (nano por velocidad)
model = YOLO("yolov8n.pt")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Fin del video o error.")
        break

    # Detección
    results = model(frame)

    # Dibujar resultados
    annotated_frame = results[0].plot()

    # Mostrar en una sola ventana
    cv2.imshow("Detección YOLOv8", annotated_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

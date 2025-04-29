# scripts/test_yolo_deepsort.py

import os

import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# Verificación de GPU
print("¿GPU disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de GPU:", torch.cuda.get_device_name(0))

# Cargar modelo YOLOv8
model = YOLO("yolov8m.pt")  # Puedes usar yolov8n.pt, yolov8s.pt, etc.

# Inicializar DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

# Ruta absoluta al video
video_path = os.path.abspath(os.path.join("..", "videos", "test.mp4"))
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ No se pudo abrir el video.")
    exit()

# Crear ventana OpenCV
cv2.namedWindow("YOLOv8 + DeepSORT", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Fin del video o error de lectura.")
        break

    # Inference YOLO
    results = model(frame, verbose=False)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        cls = int(box.cls[0])

        # SOLO trackear "car" (class 2 en COCO)
        if cls == 2:
            bbox = [
                x1.item(),
                y1.item(),
                x2.item() - x1.item(),
                y2.item() - y1.item(),
            ]  # formato xywh
            detections.append((bbox, conf.item(), cls))

    # Actualizar el tracker
    if len(detections) > 0:
        tracks = tracker.update_tracks(detections, frame=frame)
    else:
        tracks = []

    # Dibujar cajas e IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = track.to_ltrb()  # left, top, right, bottom

        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"ID {track_id}",
            (int(l), int(t) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow("YOLOv8 + DeepSORT", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

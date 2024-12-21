import cv2
from ultralytics import YOLO
import os
import uuid

# Inicializar modelo YOLO
model = YOLO("yolov8n.pt")  # Asegúrate de tener este archivo en tu entorno

# Directorios locales
input_videos_folder = "./static/videos/"
output_videos_folder = "./static/videos_procesados/"
thumbnails_folder = "./static/thumbnails/"

# Crear carpetas si no existen
os.makedirs(input_videos_folder, exist_ok=True)
os.makedirs(output_videos_folder, exist_ok=True)
os.makedirs(thumbnails_folder, exist_ok=True)

def process_video(video_path):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    
    # Configuración para guardar el video procesado
    filename, _ = os.path.splitext(video_name)
    processed_video_name = f"{filename}_processed.mp4"
    processed_video_path = os.path.join(output_videos_folder, processed_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
    
    thumbnails = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detección con YOLO
        results = model.predict(frame, save=False, conf=0.5)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{model.names[int(box.cls)]} {float(box.conf):.2f}"
                # Dibujar detecciones en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Generar miniaturas cada 10 frames
                if len(thumbnails) < 10:  # Solo guardar hasta 10 miniaturas
                    thumbnail_name = f"{filename}_{uuid.uuid4().hex}.jpg"
                    thumbnail_path = os.path.join(thumbnails_folder, thumbnail_name)
                    thumbnail = frame[y1:y2, x1:x2]
                    cv2.imwrite(thumbnail_path, thumbnail)
                    thumbnails.append(thumbnail_path)
        
        # Escribir el frame procesado
        out.write(frame)
    
    cap.release()
    out.release()
    return processed_video_path, thumbnails

def process_all_videos():
    # Obtener todos los archivos de video en la carpeta de entrada
    video_files = [f for f in os.listdir(input_videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        video_path = os.path.join(input_videos_folder, video_file)
        print(f"Procesando: {video_path}")
        processed_path, thumbnails = process_video(video_path)
        print(f"Video procesado: {processed_path}")
        print(f"Miniaturas generadas: {thumbnails}")

if __name__ == "__main__":
    process_all_videos()

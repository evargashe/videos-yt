from flask import Flask, url_for, render_template, request, redirect, session, jsonify, send_from_directory
import os
import cv2
from ultralytics import YOLO
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = "key"

VIDEOS_FOLDER = os.path.join(app.root_path, 'static/videos')
PROCESSED_VIDEOS_FOLDER = os.path.join(app.root_path, 'static/videos_procesados')
THUMBNAILS_FOLDER = os.path.join(app.root_path, 'static/thumbnails')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Inicializar el modelo YOLO
model = YOLO("yolov8n.pt")

# Crear carpetas si no existen
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para procesar el video
def process_video(video_path):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    
    # Configuración para guardar el video procesado
    filename, _ = os.path.splitext(video_name)
    processed_video_name = f"{filename}_processed.mp4"
    processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, processed_video_name)
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
                    thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_name)
                    thumbnail = frame[y1:y2, x1:x2]
                    cv2.imwrite(thumbnail_path, thumbnail)
                    thumbnails.append(thumbnail_path)
        
        # Escribir el frame procesado
        out.write(frame)
    
    cap.release()
    out.release()
    return processed_video_path, thumbnails

# Ruta para la página principal
@app.route("/", endpoint="index")
def index():
    videos = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    return render_template("index.html", videos=videos)

# Ruta para subir videos
@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return "No se seleccionó ningún archivo", 400

    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(VIDEOS_FOLDER, filename))
        return redirect(url_for("index"))  # Redirigir a la página principal
    else:
        return "Archivo no permitido", 400

# Ruta para ver un video específico
@app.route("/video/<filename>", endpoint="video")
def video(filename):
    # Verificar que el archivo existe
    video_path = os.path.join(VIDEOS_FOLDER, filename)
    if os.path.exists(video_path):
        return render_template("video.html", video=filename)
    else:
        return "Video no encontrado", 404

# Ruta para procesar el video
# Ruta para procesar el video
@app.route("/process_video/<filename>", methods=["POST"])
def process_video_route(filename):
    # Obtener la ruta completa del video
    video_path = os.path.join(VIDEOS_FOLDER, filename)
    
    # Procesar el video
    processed_video_path, thumbnails = process_video(video_path)
    
    # Extraer solo el nombre de las miniaturas (sin la ruta completa)
    thumbnails = [os.path.basename(thumbnail) for thumbnail in thumbnails]
    
    # Obtener el nombre del archivo procesado
    processed_video_filename = processed_video_path.split(os.sep)[-1]  # Extraer solo el nombre del archivo
    
    # Redirigir a la página que muestra el video original y el procesado
    return render_template("video_processed.html", 
                           original_video=filename, 
                           processed_video=processed_video_filename, 
                           thumbnails=thumbnails)



# Ruta para mostrar los videos procesados y las miniaturas
@app.route("/videos")
def videos():
    video_files = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    return render_template("index.html", videos=video_files)

# Ruta para mostrar las miniaturas generadas
@app.route("/thumbnails/<filename>")
def thumbnails(filename):
    # Aquí puedes servir las miniaturas generadas
    return send_from_directory(THUMBNAILS_FOLDER, filename)

@app.errorhandler(404)
def page_not_found(error):
    return (render_template('404.html'), 404)

if __name__ == "__main__":
    app.run(debug=True)

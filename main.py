from flask import Flask, url_for, render_template, request, redirect, send_from_directory
import os
import cv2
from ultralytics import YOLO
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = "key"

# Carpetas para videos y resultados
VIDEOS_FOLDER = os.path.join(app.root_path, 'static/videos')
PROCESSED_VIDEOS_FOLDER = os.path.join(app.root_path, 'static/videos_procesados')
THUMBNAILS_FOLDER = os.path.join(app.root_path, 'static/thumbnails')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Inicializar el modelo YOLO
model = YOLO("yolov8n.pt")

# Crear carpetas si no existen
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)


import subprocess

def convert_to_browser_friendly(input_path, output_path):
    """Convierte un video a un formato compatible con navegadores usando FFmpeg."""
    command = [
        "ffmpeg", "-i", input_path,  # Archivo de entrada
        "-vcodec", "libx264",       # Códec de video
        "-acodec", "aac",           # Códec de audio
        "-strict", "experimental",  # Opción experimental para AAC
        output_path                 # Archivo de salida
    ]
    try:
        subprocess.run(command, check=True)  # Ejecutar el comando
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error al convertir el video: {e}")
        return None



def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_video(video_path):
    """Procesa el video utilizando YOLOv8 y genera un video procesado más miniaturas."""
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    # Configuración para guardar el video procesado
    filename, _ = os.path.splitext(video_name)
    processed_video_name = f"{filename}_processed_temp.avi"  # Guardar como archivo temporal
    processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, processed_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    thumbnails = []
    frame_count = 0

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

                # Generar miniaturas cada 50 frames
                if frame_count % 50 == 0 and len(thumbnails) < 10:
                    thumbnail_name = f"{filename}_{uuid.uuid4().hex}.jpg"
                    thumbnail_path = os.path.join(THUMBNAILS_FOLDER, thumbnail_name)
                    thumbnail = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
                    cv2.imwrite(thumbnail_path, thumbnail)
                    thumbnails.append(thumbnail_path)

        # Escribir el frame procesado
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Convertir el video procesado a un formato compatible
    final_processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, f"{filename}_processed.mp4")
    convert_to_browser_friendly(processed_video_path, final_processed_video_path)

    # Eliminar el archivo temporal
    os.remove(processed_video_path)

    return final_processed_video_path, thumbnails



@app.route("/", endpoint="index")
def index():    
    videos = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    print(videos)
    return render_template("index.html", videos=videos)


@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return "No se seleccionó ningún archivo", 400

    file = request.files['video']
    if file and allowed_file(file.filename):
        original_filename = file.filename
        original_path = os.path.join(VIDEOS_FOLDER, original_filename)

        # Guardar el archivo original
        file.save(original_path)

        # Generar un nuevo nombre para el video convertido
        converted_filename = f"{os.path.splitext(original_filename)[0]}_converted.mp4"
        converted_path = os.path.join(VIDEOS_FOLDER, converted_filename)

        # Convertir el video
        converted_path = convert_to_browser_friendly(original_path, converted_path)

        # Verificar si la conversión fue exitosa
        if converted_path:
            os.remove(original_path)  # Eliminar el video original si se convirtió correctamente
            return redirect(url_for("index"))
        else:
            return "Error al convertir el video", 500
    else:
        return "Archivo no permitido", 400


@app.route("/video/<filename>", endpoint="video")
def video(filename):
    video_path = os.path.join(VIDEOS_FOLDER, filename)
    if os.path.exists(video_path):
        return render_template("video.html", video=filename)
    else:
        return "Video no encontrado", 404


@app.route("/process_video/<filename>", methods=["POST"])
def process_video_route(filename):
    video_path = os.path.join(VIDEOS_FOLDER, filename)
    processed_video_path, thumbnails = process_video(video_path)

    processed_video_filename = os.path.basename(processed_video_path)
    print(f"Original: {video_path}, Procesado: {processed_video_path}")

    return render_template(
        "video_processed.html",
        original_video=filename,
        processed_video=processed_video_filename,
        thumbnails=[os.path.basename(thumbnail) for thumbnail in thumbnails]
    )




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

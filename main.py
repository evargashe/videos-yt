from flask import Flask, url_for, render_template, request, redirect, send_from_directory, jsonify
import os
import cv2
from ultralytics import YOLO
import uuid
import numpy as np
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = "key"

# Carpetas para videos y resultados
VIDEOS_FOLDER = os.path.join(app.root_path, 'static/videos')
PROCESSED_VIDEOS_FOLDER = os.path.join(app.root_path, 'static/videos_procesados')
THUMBNAILS_FOLDER = os.path.join(app.root_path, 'static/thumbnails')
HEATMAPS_FOLDER = os.path.join(app.root_path, 'static/heatmaps')
TIMESTAMPS_FOLDER = os.path.join(app.root_path, 'static/timestamps')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Inicializar el modelo YOLO
model = YOLO("yolov8n.pt")

# Crear carpetas si no existen
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(THUMBNAILS_FOLDER, exist_ok=True)
os.makedirs(HEATMAPS_FOLDER, exist_ok=True)
os.makedirs(TIMESTAMPS_FOLDER, exist_ok=True)

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
    """Procesa el video utilizando YOLOv8, genera un video procesado, miniaturas, mapa de calor y timestamps."""
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

    # Inicializar mapa de calor
    heatmap = np.zeros((height, width), dtype=np.float32)

    thumbnails = []
    timestamps = {}

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
                cls_name = model.names[int(box.cls)]

                # Dibujar detecciones en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Actualizar el mapa de calor
                heatmap[y1:y2, x1:x2] += 1

                # Generar miniaturas cada 50 frames
                # os.makedirs(os.path.join(THUMBNAILS_FOLDER, filename), exist_ok=True)
                # if frame_count % 50 == 0 and len(thumbnails) < 10:
                #     thumbnail_name = f"{filename}_{uuid.uuid4().hex}.jpg"
                #     thumbnail_path = os.path.join(THUMBNAILS_FOLDER, filename, thumbnail_name) 
                #     thumbnail = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
                #     cv2.imwrite(thumbnail_path, thumbnail)
                #     thumbnails.append(thumbnail_path)

                if frame_count % 50 == 0 and len(thumbnails) < 10:
                    os.makedirs(os.path.join(THUMBNAILS_FOLDER, filename), exist_ok=True)
                    thumbnail_name = f"{filename}_{uuid.uuid4().hex}.jpg"
                    thumbnail_path = os.path.join(THUMBNAILS_FOLDER, filename, thumbnail_name)
                    thumbnail = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else frame
                    if thumbnail.size > 0:  # Asegúrate de que el fragmento de imagen es válido
                        cv2.imwrite(thumbnail_path, thumbnail)
                        thumbnails.append(thumbnail_path)
                    else:
                        print(f"Miniatura inválida en el frame {frame_count}: {x1}, {y1}, {x2}, {y2}")

                # Guardar timestamps
                thumbnail_path2 = os.path.join('\\static\\thumbnails', filename, thumbnail_name)
                if cls_name not in timestamps:
                    timestamps[cls_name] = []
                timestamps[cls_name].append({
                    "time": frame_count / fps,
                    "thumbnail": thumbnail_path2 if len(thumbnails) <= 10 else None
                })

        # Escribir el frame procesado
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Normalizar y guardar el mapa de calor como imagen
    heatmap = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap = np.uint8(heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Guardar el mapa de calor
    heatmap_path = os.path.join(HEATMAPS_FOLDER, f"{filename}_heatmap.jpg")
    cv2.imwrite(heatmap_path, heatmap_colored)

    # Guardar los timestamps en un archivo JSON
    timestamps_path = os.path.join(TIMESTAMPS_FOLDER, f"{filename}_timestamps.json")
    with open(timestamps_path, "w") as f:
        json.dump(timestamps, f, indent=4)

    # Convertir el video procesado a un formato compatible
    final_processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, f"{filename}_processed.mp4")
    convert_to_browser_friendly(processed_video_path, final_processed_video_path)

    # Eliminar el archivo temporal
    os.remove(processed_video_path)

    return final_processed_video_path, thumbnails, heatmap_path, timestamps_path




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


@app.route("/process_video/<filename>", methods=["GET", "POST"])
def process_video_route(filename):
    
    #processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, f"{filename}_processed.mp4")
    #heatmap_path = os.path.join(HEATMAPS_FOLDER, f"{filename}_heatmap.jpg")
    #timestamps_path = os.path.join(TIMESTAMPS_FOLDER, f"{filename}_timestamps.json")

    # Verifica si ya se ha procesado el video antes (se puede verificar si existen los archivos de salida)
    base_filename, ext = os.path.splitext(filename)
    processed_video_path = os.path.join(PROCESSED_VIDEOS_FOLDER, f"{base_filename}_processed.mp4")
    heatmap_path = os.path.join(HEATMAPS_FOLDER, f"{base_filename}_heatmap.jpg")
    timestamps_path = os.path.join(TIMESTAMPS_FOLDER, f"{base_filename}_timestamps.json")
    thumbnails_path = os.path.join(THUMBNAILS_FOLDER, base_filename)
    #base_thumbnails_path = os.path.basename(thumbnails_path)
    #print(f"thumbnails_path: {thumbnails_path}")
    
    # Si el video no ha sido procesado previamente, procesarlo
    if not os.path.exists(processed_video_path) or not os.path.exists(timestamps_path):
        print(f"Archivos no encontrados: {processed_video_path}, {timestamps_path}")
        # Procesar el video
        video_path = os.path.join(VIDEOS_FOLDER, filename)
        processed_video_path, thumbnails, heatmap_path, timestamps_path = process_video(video_path)
    else:
        print(f"Archivos encontrados: {processed_video_path}, {timestamps_path}")


    # Cargar el archivo JSON de timestamps
    with open(timestamps_path, "r") as f:
        timestamps = json.load(f)

    # Obtener el query de la búsqueda (si existe)
    query = request.args.get('query', '').lower()

    # Filtrar resultados si hay una consulta
    if query:
        filtered_timestamps = {key: [entry for entry in value if query in key.lower()]
                               for key, value in timestamps.items()}
    else:
        filtered_timestamps = timestamps

    
    #for thumbnail in os.listdir(thumbnails_path):
    #    print(url_for('static', filename=f"thumbnails/"+ base_filename +'/'+ thumbnail))
    #    relative_path = os.path.join(subdirectory, thumbnail)

    thumbnails = [
        url_for('static', filename=f"thumbnails/{base_filename}/{thumbnail}")
        for thumbnail in os.listdir(thumbnails_path)
    ]



    return render_template( 
        "video_processed.html",
        original_video=filename,
        processed_video=os.path.basename(processed_video_path),
        #thumbnails=[os.path.basename(thumbnail) for thumbnail in thumbnails],
        thumbnails=thumbnails,
        heatmap=os.path.basename(heatmap_path),
        timestamps=os.path.basename(timestamps_path),
        filtered_timestamps=filtered_timestamps,  # Pasar los resultados filtrados
        query=query  # Pasar la consulta
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
    app.run(host="0.0.0.0", port=5000, debug=True)

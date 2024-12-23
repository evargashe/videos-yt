# Usa una imagen base de Python
FROM python:3.9-buster

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala dependencias del sistema necesarias para OpenCV, FFmpeg y otras
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copia los archivos del proyecto al contenedor
COPY . .

# Actualiza pip e instala las dependencias desde requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Exponer el puerto (si es necesario, por ejemplo, para Flask)
EXPOSE 5000

# Comando por defecto para ejecutar tu script principal
CMD ["python", "main.py"]

import cv2

cap = cv2.VideoCapture("./static/videos/videos-detection freeze.mp4")
if not cap.isOpened():
    print("No se puede abrir el video.")
else:
    print("El video se cargó correctamente.")
cap.release()

import cv2
import os
 
# Nombre del archivo de imagen en tu carpeta raíz
imagen_path = 'people.jpg'  # cambia el nombre si tu archivo es diferente
 
# Verificar que exista la imagen
if not os.path.exists(imagen_path):
    print(f"❌ No se encontró la imagen '{imagen_path}' en la carpeta actual.")
    exit()
 
# Cargar el clasificador de rostros de OpenCV
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
 
# Leer la imagen
imagen = cv2.imread(imagen_path)
if imagen is None:
    print("❌ No se pudo cargar la imagen. Revisa el formato o la ruta.")
    exit()
 
# Convertir a escala de grises (mejor rendimiento en detección)
gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
 
# Detectar rostros con parámetros ajustados para mejor precisión
rostros = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # más sensible a tamaños de rostro diferentes
    minNeighbors=4,    # menos estricto, detecta más
    minSize=(30, 30)   # tamaño mínimo de rostro
)
 
# Dibujar rectángulos alrededor de los rostros detectados
for (x, y, w, h) in rostros:
    cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
# Guardar la imagen procesada
resultado_path = 'resultado.jpg'
cv2.imwrite(resultado_path, imagen)
 
print(f"✅ Se detectaron {len(rostros)} rostro(s). Imagen guardada como '{resultado_path}'.")
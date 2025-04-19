
# Librerías estándar para procesamiento de imágenes y modelo
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2

# Tipado para respuestas
from typing import List

# FastAPI y módulos asociados
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Crear instancia de la aplicación FastAPI
app = FastAPI()

# Habilitar CORS para permitir peticiones desde el frontend (ej. Angular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔁 Puedes reemplazar con ["http://localhost:4200"] para mayor seguridad
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo previamente entrenado para reconocimiento de dígitos manuscritos
model = tf.keras.models.load_model("api/modelo_digitos_manuscritos.h5")

# Endpoint para procesar y predecir los dígitos manuscritos desde una imagen
@app.post("/predecir")
async def predecir_imagen(file: UploadFile = File(...)):
    # Leer el archivo enviado como binario
    contents = await file.read()

    # Convertir a imagen en escala de grises con PIL
    image = Image.open(io.BytesIO(contents)).convert("L")
    img = np.array(image)

    # Invertir colores si el fondo es claro (blanco), para mantener compatibilidad con MNIST (fondo negro)
    if np.mean(img) > 127:
        img = 255 - img

    # Binarizar la imagen para destacar el contorno del dígito
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Encontrar los contornos de los elementos en la imagen (dígitos)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar los contornos de izquierda a derecha según coordenada X
    contornos = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])

    resultados = []

    # Recorrer cada contorno detectado
    for contorno in contornos:
        x, y, w, h = cv2.boundingRect(contorno)

        # Filtrar contornos pequeños (ruido)
        if w > 10 and h > 10:
            # Obtener región de interés cuadrada centrada en el contorno
            lado = max(w, h)
            cx, cy = x + w // 2, y + h // 2
            x_inicio = max(cx - lado // 2, 0)
            y_inicio = max(cy - lado // 2, 0)
            x_final = x_inicio + lado
            y_final = y_inicio + lado

            # Recortar el dígito
            digito = img[y_inicio:y_final, x_inicio:x_final]

            # Redimensionar a 20x20 sin distorsión
            digito = cv2.resize(digito, (20, 20), interpolation=cv2.INTER_AREA)

            # Crear un lienzo 28x28 (formato MNIST) y centrar el dígito dentro
            canvas = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - 20) // 2
            y_offset = (28 - 20) // 2
            canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digito

            # Normalizar los píxeles entre 0 y 1
            canvas = canvas.astype(np.float32) / 255.0

            # Preparar la entrada en forma de tensor para el modelo: (1, 28, 28, 1)
            entrada = np.expand_dims(canvas, axis=(0, -1))

            # Realizar la predicción
            prediccion = model.predict(entrada, verbose=0)[0]
            clase = int(np.argmax(prediccion))

            # Agregar el dígito detectado a la lista de resultados
            resultados.append(clase)

    # Devolver la respuesta en formato JSON
    return JSONResponse(content={"digitos_detectados": resultados})

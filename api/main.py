
# # Librerías estándar para procesamiento de imágenes y modelo
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import io
# import cv2

# # Tipado para respuestas
# from typing import List

# # FastAPI y módulos asociados
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse

# # Crear instancia de la aplicación FastAPI
# app = FastAPI()

# # Habilitar CORS para permitir peticiones desde el frontend (ej. Angular)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 🔁 Puedes reemplazar con ["http://localhost:4200"] para mayor seguridad
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Cargar el modelo previamente entrenado para reconocimiento de dígitos manuscritos
# model = tf.keras.models.load_model("api/modelo_digitos_manuscritos.h5")

# # Endpoint para procesar y predecir los dígitos manuscritos desde una imagen
# @app.post("/predecir")
# async def predecir_imagen(file: UploadFile = File(...)):
#     # Leer el archivo enviado como binario
#     contents = await file.read()

#     # Convertir a imagen en escala de grises con PIL
#     image = Image.open(io.BytesIO(contents)).convert("L")
#     img = np.array(image)

#     # Invertir colores si el fondo es claro (blanco), para mantener compatibilidad con MNIST (fondo negro)
#     if np.mean(img) > 127:
#         img = 255 - img

#     # Binarizar la imagen para destacar el contorno del dígito
#     _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

#     # Encontrar los contornos de los elementos en la imagen (dígitos)
#     contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Ordenar los contornos de izquierda a derecha según coordenada X
#     contornos = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])

#     resultados = []

#     # Recorrer cada contorno detectado
#     for contorno in contornos:
#         x, y, w, h = cv2.boundingRect(contorno)

#         # Filtrar contornos pequeños (ruido)
#         if w > 10 and h > 10:
#             # Obtener región de interés cuadrada centrada en el contorno
#             lado = max(w, h)
#             cx, cy = x + w // 2, y + h // 2
#             x_inicio = max(cx - lado // 2, 0)
#             y_inicio = max(cy - lado // 2, 0)
#             x_final = x_inicio + lado
#             y_final = y_inicio + lado

#             # Recortar el dígito
#             digito = img[y_inicio:y_final, x_inicio:x_final]

#             # Redimensionar a 20x20 sin distorsión
#             digito = cv2.resize(digito, (20, 20), interpolation=cv2.INTER_AREA)

#             # Crear un lienzo 28x28 (formato MNIST) y centrar el dígito dentro
#             canvas = np.zeros((28, 28), dtype=np.uint8)
#             x_offset = (28 - 20) // 2
#             y_offset = (28 - 20) // 2
#             canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digito

#             # Normalizar los píxeles entre 0 y 1
#             canvas = canvas.astype(np.float32) / 255.0

#             # Preparar la entrada en forma de tensor para el modelo: (1, 28, 28, 1)
#             entrada = np.expand_dims(canvas, axis=(0, -1))

#             # Realizar la predicción
#             prediccion = model.predict(entrada, verbose=0)[0]
#             clase = int(np.argmax(prediccion))

#             # Agregar el dígito detectado a la lista de resultados
#             resultados.append(clase)

#     # Devolver la respuesta en formato JSON
#     return JSONResponse(content={"digitos_detectados": resultados})


import numpy as np
import tensorflow as tf
from PIL import Image
import io
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker

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

# Configuración de la base de datos
DATABASE_URL = "mysql+pymysql://usr_ia_lf_2025:5sr_31_lf_2025@www.server.daossystem.pro:3301/bd_ia_lf_2025"

# Crear motor y sesión de SQLAlchemy
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
metadata = MetaData()

# Cargar el modelo previamente entrenado para reconocimiento de dígitos manuscritos
model = tf.keras.models.load_model("api/modelo_digitos_manuscritos.h5")

# Endpoint para procesar y predecir los dígitos manuscritos desde una imagen
@app.post("/predecir")
async def predecir_imagen(file: UploadFile = File(...)):
    # Leer el archivo enviado como binario
    contents = await file.read()

    # Convertir a imagen en escala de grises con PIL
    img = preprocesar_foto_real(contents)
    # Invertir colores si el fondo es claro (blanco), para mantener compatibilidad con MNIST (fondo negro)
    if np.mean(img) > 127:
        img = 255 - img

        # Binarizar
    _, thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    # Detectar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return JSONResponse(content={"digitos_detectados": [], "mensaje": "No se encontraron dígitos"})

    # Recorte ajustado global (todos los contornos juntos)
    todos_puntos = np.concatenate(contornos)
    x, y, w, h = cv2.boundingRect(todos_puntos)
    img_recortada = img[y:y+h, x:x+w]

    # Volver a binarizar y buscar dígitos individuales
    _, rec_bin = cv2.threshold(img_recortada, 100, 255, cv2.THRESH_BINARY)
    contornos_digitos, _ = cv2.findContours(rec_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_digitos = sorted(contornos_digitos, key=lambda c: cv2.boundingRect(c)[0])

    resultados = []

    # Recorrer cada contorno detectado
    for contorno in contornos_digitos:
        x, y, w, h = cv2.boundingRect(contorno)
        if w > 10 and h > 10:
            # Ajuste cuadrado centrado
            lado = max(w, h)
            cx, cy = x + w // 2, y + h // 2
            x_ini = max(cx - lado // 2, 0)
            y_ini = max(cy - lado // 2, 0)
            x_fin = x_ini + lado
            y_fin = y_ini + lado
            digito = img_recortada[y_ini:y_fin, x_ini:x_fin]

            # Redimensionar y centrar
            digito = cv2.resize(digito, (20, 20), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - 20) // 2
            y_offset = (28 - 20) // 2
            canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digito

            entrada = np.expand_dims(canvas / 255.0, axis=(0, -1))
            prediccion = model.predict(entrada, verbose=0)[0]
            resultados.append(int(np.argmax(prediccion)))

    # Obtener el número como una cadena de dígitos
    numero_detectado = ''.join(map(str, resultados))
    
    # Insertar el número detectado en la base de datos
    insertar_en_base_de_datos(numero_detectado)

    # Devolver la respuesta en formato JSON
    return JSONResponse(content={"digitos_detectados": resultados, "numero": numero_detectado})
from pydantic import BaseModel

class InsertData(BaseModel):
    numero: str  # El número que has predicho y que deseas insertar
    factorial: str  # El factorial que calculas
    nombre_estudiante: str  # El nombre del estudiante

# Función para insertar datos en la base de datos
@app.post("/datos")
def insertar_en_base_de_datos(data: InsertData):
    # Crear sesión con la base de datos
    db = SessionLocal()

    try:
        # Conectar a la tabla "resultado"
        tabla = Table("segundo_parcial", metadata, autoload_with=engine)

        # Realizar el insert con los tres campos
        db.execute(tabla.insert().values(
            valor=data.numero,           # Número predicho
            factorial=data.factorial,    # Factorial calculado
            nombre_estudiante=data.nombre_estudiante  # Nombre del estudiante
        ))
        db.commit()
        return {"message": "Datos insertados correctamente"}
    except Exception as e:
        db.rollback()
        print(f"Error al insertar en la base de datos: {e}")
        return {"error": "Error al insertar en la base de datos"}
    finally:
        db.close()


def preprocesar_foto_real(file_bytes: bytes):
    # Convertir bytes a imagen OpenCV
    imagen_color = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Escala de grises
    gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Invertir si fondo es claro (como hoja blanca)
    if np.mean(gris) > 127:
        gris = 255 - gris

    # Aplicar desenfoque para reducir ruido
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # Binarizar con umbral adaptativo Otsu
    _, binarizada = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilatar para cerrar trazos sueltos
    kernel = np.ones((3, 3), np.uint8)
    binarizada = cv2.dilate(binarizada, kernel, iterations=1)

    return binarizada






# Importar TensorFlow y el conjunto de datos MNIST
import tensorflow as tf 
from tensorflow.keras.datasets import mnist

# Cargar el conjunto de datos MNIST
# Contiene imágenes de dígitos escritos a mano de 28x28 píxeles en escala de grises
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar los valores de píxeles a un rango [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Redimensionar las imágenes para que tengan forma (28, 28, 1) en lugar de (28, 28)
# Esto es necesario porque las capas Conv2D esperan un canal adicional (profundidad)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Definición del modelo secuencial con capas convolucionales y densas
model = tf.keras.models.Sequential([
    
    # Primera capa convolucional: 32 filtros de 3x3, función de activación ReLU
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),

    # Capa de agrupamiento (pooling) para reducir dimensionalidad
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Aplanar la salida para alimentar a la red neuronal densa
    tf.keras.layers.Flatten(),

    # Capa totalmente conectada (densa) con 128 neuronas y activación ReLU
    tf.keras.layers.Dense(128, activation='relu'),

    # Capa Dropout para reducir overfitting (elimina aleatoriamente un 20% de las conexiones)
    tf.keras.layers.Dropout(0.2),

    # Capa de salida con 10 neuronas (una para cada dígito del 0 al 9)
    tf.keras.layers.Dense(10)
])

# Compilar el modelo con:
# - Optimizador Adam
# - Función de pérdida: entropía cruzada (softmax implícito)
# - Métrica de precisión (accuracy)
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Entrenar el modelo durante 10 épocas
# Se valida el rendimiento al final de cada época usando los datos de prueba
model.fit(
    x_train, y_train,
    epochs=10,
    validation_data=(x_test, y_test)
)

# Evaluar el modelo final en los datos de prueba
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Acurácia nos dados de teste: {test_accuracy*100:.2f}%')

# Guardar el modelo entrenado en formato HDF5 (.h5) para usarlo en producción (FastAPI)
model.save("api/modelo_digitos_manuscritos.h5")

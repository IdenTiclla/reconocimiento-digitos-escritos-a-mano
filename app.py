import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

# SE CARGA EL CONJUNTO DE DATOS MNIST DESDE TENSORFLOW
mnist = tf.keras.datasets.mnist
# Dividimos los datos en conjuntos de entrenamiento y prueba
(x_train, y_train), (x_test, y_test) = mnist.load_data()  

# NORMALIZAMOS LOS DATOS
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# SE DEFINE Y COMPILA EL MODELO DE LA RED NEURONAL
# Crea un modelo  de red neuronal secuencial
model = tf.keras.models.Sequential()
# Agregando las capas
# Agregando capas de aplanamiento Flatten
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# Agregando capas densamente conectadas
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) # numbers of nurones
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# Agregando capa de activacion softmax para la clasificacion de 10 clases
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Configura el modelo para el entrenamiento
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# SE ENTRENA EL MODELO
model.fit(x_train, y_train, epochs=3)

# SE EVALUA EL MODELO CON LOS DATOS DE PRUEBA
loss, accuracy = model.evaluate(x_test, y_test)

print(accuracy)
print(loss)

# SE GUARDA EL MODELO ENTRENADO
model.save('digits.model')

# REALIZAMOS LA PREDICCION PARA CADA IMAGEN DE PRUEBA
for x in range(10):
    img = cv.imread(f'digits-for-test/{x}.png')[:, :, 0] # cargamos la imagen png y la convierte en grises
    img = np.invert(np.array([img])) # Invertimos los valores de los pixeles

    prediction  = model.predict(img) # Realiza una prediccion con el MODELO ENTRENADO
    print(f'The result is probably: {np.argmax(prediction)}') # determina la clase predicha (digito) con mas probabilidad


    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
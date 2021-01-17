import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
loss, accuracy = model.evaluate(x_test, y_test)

model.save("digit.h5")


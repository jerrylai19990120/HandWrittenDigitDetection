import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model.fit(x_train, y_train, epochs=18)
loss, accuracy = model.evaluate(x_test, y_test)

model.save("deepLearningModel.model")


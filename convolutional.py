import tensorflow as tf
import pickle
import numpy as np

from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard


pickle_in = open("features.pickle","rb")
features = pickle.load(pickle_in)

pickle_in = open("label.pickle","rb")
label = pickle.load(pickle_in)

features = features / 255

model = tf.keras.models.Sequential()

model.add(Conv2D(64, (3, 3), input_shape=features.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs/{}".format("Cats-vs-dogs-CNN"))

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
            )

model.fit(np.array(features), np.array(label), batch_size=32, epochs=10, validation_split=0.1, callbacks=[tensorboard])
model.save('64x3-CNN.model')

import numpy as np
import os
import cv2
import pickle

from matplotlib import pyplot as plt


IMG_SIZE = 50
training_data = []
folder_names_list = ['Cat', 'Dog']

for folder_name in folder_names_list:
    path = os.path.join(os.getcwd(), folder_name)
    class_num = folder_names_list.index(folder_name)

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_img, class_num])
        except Exception as e:
            pass

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("features.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("label.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

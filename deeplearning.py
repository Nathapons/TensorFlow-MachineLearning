import tensorflow as tf
import cv2

def prepare(filepath):
    IMG_SIZE = 50  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1) 


CATEGORIES = ['Cat', 'Dog']

try:  
    model = tf.keras.models.load_model("64x3-CNN.model")
    prediction = model.predict([prepare('C:\\Users\\SWD\Desktop\\LearnProgram\\TensorFlow\\Cat_test\\cat8.jpg')])
    print(f'Machine answer: {CATEGORIES[int(prediction[0][0])]}')
except Exception as e:
    print(f'{str(e) = }')

import os
import random
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt

MODEL_PATH = "../lib/model/korean_car_model_512_384.hdf5"
TEST_IMAGE_DIRECTORY = "/Users/pigmong0202/KoreanCar_DataSets/512_384/test"
TRAIN_IMAGE_DIRECTORY = "/Users/pigmong0202/KoreanCar_DataSets/512_384/train"

model = tf.keras.models.load_model(MODEL_PATH)
class_labels = sorted(os.listdir(TRAIN_IMAGE_DIRECTORY))

selected_class_name = os.path.join(TEST_IMAGE_DIRECTORY, random.sample(os.listdir(TEST_IMAGE_DIRECTORY), 1)[0])
test_image_path = os.path.join(selected_class_name, random.sample(os.listdir(selected_class_name), 1)[0])
# test_image_path = os.path.join(TEST_IMAGE_DIRECTORY, "그랜저 HG240_2016", "현대_그랜저HG240_2016_10시대_H0_V0.JPG")
print("test image path :", test_image_path)

test_image = image.image_utils.load_img(test_image_path, target_size=(384, 512))
test_image = image.image_utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0


prediction = model.predict(test_image)
predicted_class = np.argmax(prediction)
print("prediction :", class_labels[predicted_class+1])

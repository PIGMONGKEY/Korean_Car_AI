import os
import random
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt

MODEL_PATH = "../lib/model/korean_car_model_1024_768.hdf5"
TEST_IMAGE_DIRECTORY = "/Users/pigmong0202/KoreanCar_DataSets/1024_768/test"
TRAIN_IMAGE_DIRECTORY = "/Users/pigmong0202/KoreanCar_DataSets/1024_768/train"

model = tf.keras.models.load_model(MODEL_PATH)
class_labels = sorted(os.listdir(TRAIN_IMAGE_DIRECTORY))

test_image_count = 0
correct_prediction_count = 0

for folder_name in os.listdir(TEST_IMAGE_DIRECTORY):
    if folder_name == ".DS_Store":
        continue
    folder_path = os.path.join(TEST_IMAGE_DIRECTORY, folder_name)
    for file_name in os.listdir(folder_path):
        if file_name == ".DS_Store":
            continue
        test_image_path = os.path.join(folder_path, file_name)
        test_image = image.image_utils.load_img(test_image_path, target_size=(768, 1024))
        test_image = image.image_utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        answer_class = folder_name
        prediction = model.predict(test_image)
        predicted_class = np.argmax(prediction)

        print("input :", answer_class)
        print("prediction :", class_labels[predicted_class+1])

        if answer_class == class_labels[predicted_class+1]:
            correct_prediction_count += 1
        test_image_count += 1

print(f"{test_image_count} / {correct_prediction_count}")

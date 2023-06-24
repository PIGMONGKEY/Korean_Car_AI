import os
import shutil

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

MODEL_PATH = "../lib/model/korean_car_model_1024_768.hdf5"
TEST_IMAGE_DIRECTORY = "/Users/pigmong0202/KoreanCar_DataSets/1024_768/test"
TRAIN_IMAGE_DIRECTORY = "/Users/pigmong0202/KoreanCar_DataSets/1024_768/train"

PREDICTION_CORRECT_PATH = "/Users/pigmong0202/KoreanCar_DataSets/1024_768/prediction_correct"

model = tf.keras.models.load_model(MODEL_PATH)
class_labels = sorted(os.listdir(TRAIN_IMAGE_DIRECTORY))

test_image_count = 0
correct_prediction_count = 0
correct_prediction_image_path = list()

for folder_name in sorted(os.listdir(TEST_IMAGE_DIRECTORY)):  # 클래스 이름
    if folder_name == ".DS_Store":
        continue
    folder_path = os.path.join(TEST_IMAGE_DIRECTORY, folder_name)
    for file_name in sorted(os.listdir(folder_path)):  # 이미지 파일 이름
        if file_name == ".DS_Store":
            continue

        test_image_path = os.path.join(folder_path, file_name)  # 이미지 처리
        test_image = image.image_utils.load_img(test_image_path, target_size=(768, 1024))
        test_image = image.image_utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        answer_class = folder_name  # 폴더명이 정답 클래스 이름
        prediction = model.predict(test_image)  # 모델 예측
        predicted_class = np.argmax(prediction)  # 예측 값 중 가장 큰 값의 인덱스 받기

        test_image_count += 1

        print(f"input : {answer_class} prediction : {class_labels[predicted_class + 1]} (5790 / {test_image_count})")

        if answer_class == class_labels[predicted_class + 1]:
            correct_prediction_count += 1
            correct_prediction_image_path.append(test_image_path)
            shutil.copy(test_image_path, PREDICTION_CORRECT_PATH)  # prediction_correct 파일에 정답 파일 복리

print(f"{test_image_count} / {correct_prediction_count}")

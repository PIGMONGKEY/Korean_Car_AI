import os
import random
import shutil
import cv2

# 파일이 존재하는 디렉토리 경로
directory_path = "/Users/pigmong0202/KoreanCar_DataSets/training/Renault Samsung/SM6/2017"

# temp 디렉토리 경로
temp_directory_path_1 = directory_path.replace("training", "validation")

temp_directory_path_2 = directory_path.replace("training", "test")

# 일정 비율로 파일 선택하는 함수
def select_files(file_list, ratio):
    num_files = len(file_list)
    num_selected = int(num_files * ratio)
    return random.sample(file_list, num_selected)

# 주어진 디렉토리에서 파일 목록 가져오기
file_list = os.listdir(directory_path)

# 일정 비율로 파일 선택
selection_ratio = 0.1  # 파일 선택 비율 설정
selected_files = select_files(file_list, selection_ratio)

for file_name in selected_files:
    file_path = os.path.join(directory_path, file_name)
    dest_path = os.path.join(temp_directory_path_1, file_name)
    shutil.move(file_path, dest_path)
    print(f"Moved file: {file_name} to temp directory.")

# 주어진 디렉토리에서 파일 목록 가져오기
file_list = os.listdir(directory_path)

# 일정 비율로 파일 선택
selection_ratio = 0.1  # 파일 선택 비율 설정
selected_files = select_files(file_list, selection_ratio)

for file_name in selected_files:
    file_path = os.path.join(directory_path, file_name)
    dest_path = os.path.join(temp_directory_path_2, file_name)
    shutil.move(file_path, dest_path)
    print("Moved file: {file_name} to temp_2 directory.")
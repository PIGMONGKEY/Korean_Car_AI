import os
import cv2

MAX_COUNT = 64879

directory_path = "/Users/pigmong0202/KoreanCar_DataSets"
file_list = os.listdir(os.path.join(directory_path, "원본"))

count = 0

for file_name_1 in file_list:  # test / training / validation
    if file_name_1 == ".DS_Store":
        continue

    path_1 = os.path.join(os.path.join(directory_path, "원본"), file_name_1)
    file_list_1 = os.listdir(path_1)

    for file_name_2 in file_list_1:  # Hyndai / Kia
        if file_name_2 == ".DS_Store":
            continue

        path_2 = os.path.join(path_1, file_name_2)
        file_list_2 = os.listdir(path_2)

        for file_name_3 in file_list_2:  # 차종
            if file_name_3 == ".DS_Store":
                continue

            path_3 = os.path.join(path_2, file_name_3)
            file_list_3 = os.listdir(path_3)

            for file_name_4 in file_list_3:  # 연식
                if file_name_4 == ".DS_Store":
                    continue

                path_4 = os.path.join(path_3, file_name_4)
                file_list_4 = os.listdir(path_4)

                for file_name_5 in file_list_4:
                    if file_name_5 == ".DS_Store":
                        continue

                    count += 1

                    image_path = os.path.join(directory_path, "원본", file_name_1, file_name_2, file_name_3,
                                              file_name_4, file_name_5)
                    resized_image_path = os.path.join(directory_path, "1024_768", file_name_1, file_name_2, file_name_3,
                                                      file_name_4, file_name_5)

                    image = cv2.imread(image_path)
                    resized_image = cv2.resize(image, (1024, 768))
                    cv2.imwrite(resized_image_path, resized_image)
                    print(file_name_5, "({0} / {1})".format(count, MAX_COUNT))

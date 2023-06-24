import os
import shutil

directory_path = "/Users/pigmong0202/KoreanCar_DataSets/1024_768/"
file_list = os.listdir(directory_path)

for file_name in file_list:                             #train / test / validation
    if file_name == ".DS_Store":
        continue
    path_1 = os.path.join(directory_path, file_name)
    file_list_1 = os.listdir(path_1)

    for file_name_1 in file_list_1:                     #브랜드
        if file_name_1 == ".DS_Store":
            continue
        path_2 = os.path.join(path_1, file_name_1)
        file_list_2 = os.listdir(path_2)

        for file_name_2 in file_list_2:                 #차종
            if file_name_2 == ".DS_Store":
                continue
            path_3 = os.path.join(path_2, file_name_2)
            file_list_3 = os.listdir(path_3)

            for file_name_3 in file_list_3:             #연식
                if file_name_3 == ".DS_Store":
                    continue

                if file_name == "test":
                    new_folder_path = os.path.join("/Users/pigmong0202/KoreanCar_DataSets/temp/test",
                                                   file_name_2 + "_" + file_name_3)
                    if not os.path.exists(new_folder_path):
                        os.mkdir(new_folder_path)
                elif file_name == "validation":
                    new_folder_path = os.path.join("/Users/pigmong0202/KoreanCar_DataSets/temp/validation",
                                                   file_name_2 + "_" + file_name_3)
                    if not os.path.exists(new_folder_path):
                        os.mkdir(new_folder_path)
                else:
                    new_folder_path = os.path.join("/Users/pigmong0202/KoreanCar_DataSets/temp/train",
                                                   file_name_2 + "_" + file_name_3)
                    if not os.path.exists(new_folder_path):
                        os.mkdir(new_folder_path)

                path_4 = os.path.join(path_3, file_name_3)
                file_list_4 = os.listdir(path_4)

                for file_name_4 in file_list_4:         #사진파일
                    if file_name_4 == ".DS_Store":
                        continue
                    image_path = os.path.join(path_4, file_name_4)
                    dest_path = os.path.join(new_folder_path, file_name_4)
                    shutil.move(image_path, dest_path)
                    print(image_path, "->", dest_path)

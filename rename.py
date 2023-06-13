import os
import params as params


def getFileNames(path):
    file_names = os.listdir(path)
    return list(filter(lambda x: x.endswith(".jpg"), file_names))


main_data_directory = params.images_main_directory
dirs = os.listdir(main_data_directory)

print("[Start] Renaming datasets...")


for dir in dirs:
    source_dir_path = os.path.join(main_data_directory, dir).__str__()
    print("Proccesing " + source_dir_path + "...")
    original_file_names = getFileNames(source_dir_path)
    count = 1
    for file in original_file_names:
        file_path = os.path.join(source_dir_path, file).__str__()
        new_file_name = dir.__str__() + '_' + str(count) + '.jpg'
        new_file_path = os.path.join(source_dir_path, new_file_name).__str__()
        os.rename(file_path, new_file_path)
        count = count + 1

print("[Stop] Renaming datasets...")

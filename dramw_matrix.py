import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import copyUtils.copyDirectoryUtils as copy_utils
import numpy as np
import pandas as pd
import params as params
import seaborn as sn


def filter_files(path):
    file_names = os.listdir(path)
    return list(filter(lambda x: "." not in x, file_names))


main_data_directory = "path/to/data"
labels_data_directory = "path/to/directory/with/labels"
dirs = filter_files(main_data_directory)
classes = filter_files(labels_data_directory)

if params.verbose is True:
    print(dirs)
    print(classes)

print("[Start] Clear clusters...")

# initialization confusion matrix
d = {}
for dir in dirs:
    d[dir] = [0 for x in range(len(classes))]

counts = pd.DataFrame(d, index=classes, columns=dirs)

# counting confusion matrix
for dir in dirs:
    source_dir_path = os.path.join(main_data_directory, dir).__str__()
    print("Counting " + source_dir_path + "...")
    file_names = copy_utils.get_file_names(source_dir_path)
    cluster_counts = {}
    for file in file_names:
        last_char_index = file.rfind("_")
        label = file[:last_char_index]
        counts[dir][label] += 1

print(counts)

# SOLUTION TO SMALL CM
# classes_map = {}
# i = 0
# for clazz in classes:
#     classes_map[clazz] = str(i)
#     i += 1
#
# int_dirs = np.sort([int(dir) for dir in dirs])
# int_dirs = [str(dir) for dir in int_dirs]
# matrix = pd.DataFrame(d, index=int_dirs, columns=int_dirs)
# for dir in dirs:
#     for clazz in classes:
#         matrix[dir][classes_map[clazz]] = str(counts[dir][clazz])
#
# print(matrix)
#
# plt.figure()
# sn.heatmap(matrix, annot=True, fmt='g')
# plt.show()


# SOLUTION TO BIG CM
classes_map = {}
i = 0
for clazz in classes:
    classes_map[clazz] = i
    i += 1

int_dirs = np.sort([int(dir) for dir in dirs])

matrix = pd.DataFrame(d, index=int_dirs, columns=int_dirs)
for dir in dirs:
    for clazz in classes:
        matrix[int(dir)][classes_map[clazz]] = int(counts[dir][clazz])

print(matrix)

matrix = np.array(matrix, dtype='f')
plt.imshow(matrix, interpolation='none', cmap='plasma') #'viridis', 'plasma', 'inferno', 'magma', 'cividis'
plt.colorbar(cmap='plasma')
# plt.xticks(int_dirs)
# plt.yticks(int_dirs)
plt.show()
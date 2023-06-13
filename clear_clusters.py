import os
import copyUtils.copyDirectoryUtils as copy_utils
import numpy as np
import pandas as pd
import params as params
import shutil


def filter_files(path):
    file_names = os.listdir(path)
    return list(filter(lambda x: "." not in x, file_names))


main_data_directory = params.previous_predicted_clusters_directory
labels_data_directory = params.images_main_directory
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

if params.verbose is True:
    print(counts)

# counting the best classes for clusters
used_classes = []
used_clusters = []

assignment = {}

for i in range(len(dirs)):
    max = 0
    max_dir = dirs[0]
    max_class = classes[0]
    good = False
    for cluster in dirs:
        for label in classes:
            if counts[cluster][label] > max and cluster not in used_clusters and label not in used_classes:
                max = counts[cluster][label]
                max_dir = cluster
                max_class = label
                good = True

    if good is True:
        used_classes.append(max_class)
        used_clusters.append(max_dir)
        assignment[max_dir] = (max_class, max)

if params.verbose is True:
    print(used_classes)
    print(used_clusters)
    for key in assignment:
        print(key, '->', assignment[key])


# removing bad images
for cluster in dirs:
    if cluster in used_clusters:
        proper_class, _ = assignment[cluster]
        source_dir_path = os.path.join(main_data_directory, cluster).__str__()
        print("Cleaning " + source_dir_path + "...")
        file_names = copy_utils.get_file_names(source_dir_path)
        for file in file_names:
            if not file.startswith(proper_class):
                file_path = os.path.join(source_dir_path, file).__str__()
                junk_path = os.path.join(source_dir_path, params.junk_directory).__str__()
                shutil.move(file_path, os.path.join(junk_path, file).__str__())


#summary
not_used_classes = []
not_used_clusters = []

for cluster in dirs:
    if cluster not in used_clusters:
        not_used_clusters.append(cluster)
    if cluster in used_clusters:
        max_class_ass, max_count_ass = assignment[cluster]
        print(cluster, '->', max_class_ass, ', ', max_count_ass)


for label in classes:
    if label not in used_classes:
        not_used_classes.append(label)

if len(not_used_classes) > 0:
    print("Classes not used: ")
    print(not_used_classes)

if len(not_used_clusters) > 0:
    print("Clusters not cleaned: ")
    print(not_used_clusters)
    print("You need to clean those clusters manually.")

if len(not_used_clusters) == 0 and len(not_used_classes) == 0:
    print("All clusters were cleaned.")

print("[Stop] Clear clusters...")
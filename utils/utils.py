import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import params as params


def save_confusion_matrix(original_labels, predicted_labels, path):
    sns.set(font_scale=3)
    confusion_matrix = metrics.confusion_matrix(original_labels, predicted_labels)
    if params.clusters > 50:
        plt.figure(figsize=(38, 30))
    else:
        plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.savefig(os.path.join(path, 'conf_matrix.png').__str__())


def draw_confusion_matrix(original_labels, predicted_labels):
    sns.set(font_scale=3)
    confusion_matrix = metrics.confusion_matrix(original_labels, predicted_labels)
    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()


def convert_to_numeric_labels(x):
    d = {}
    count = 0
    for i in x:
        if i not in d:
            d[i] = count
            count += 1

    new_x = [d[i] for i in x]

    return new_x
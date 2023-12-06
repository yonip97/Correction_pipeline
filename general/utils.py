import re
import string
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
    return re.sub(' +', ' ', text).strip()


def iter_list(list_, batch_size):
    num_of_batches = math.ceil(len(list_) / batch_size)
    for i in range(num_of_batches):
        yield list_[i * batch_size:(i + 1) * batch_size]


def plot_confusion_matrix(df, col1, col2, classes, title,
                          normalize=False,
                          cmap='gray_r',
                          linecolor='k'):
    cm = confusion_matrix(df[col1].tolist(),
                          df[col2].tolist(), labels=[0, 1])
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_title = 'Confusion matrix, with normalization'
    else:
        cm_title = title

    fmt = '.3f' if normalize else 'd'
    sns.heatmap(cm, fmt=fmt, annot=True, square=True,
                xticklabels=classes, yticklabels=classes,
                cmap=cmap, vmin=0, vmax=0,
                linewidths=0.5, linecolor=linecolor,
                cbar=False)
    sns.despine(left=False, right=False, top=False, bottom=False)

    plt.title(cm_title)
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.tight_layout()
    plt.show()


def remove_punctuation(input_string):
    # Create a translation table
    translation_table = str.maketrans("", "", string.punctuation)

    # Use translate method to remove punctuation
    result_string = input_string.translate(translation_table)

    return result_string


def add_None_for_None(annotated_list, list_of_objects):
    full_list = [None for i in range(len(annotated_list))]
    counter = 0
    for i in range(len(annotated_list)):
        if annotated_list[i] is not None:
            full_list[i] = list_of_objects[counter]
            counter += 1
    return full_list


def add_None_for_one(annotated_list, list_of_objects):
    full_list = [None for i in range(len(annotated_list))]
    counter = 0
    for i in range(len(annotated_list)):
        if annotated_list[i] == 0:
            full_list[i] = list_of_objects[counter]
            counter += 1
    return full_list

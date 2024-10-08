import re
import string
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
import os
import torch.distributed as dist


def get_latest_directory(path):
    # Get all directories in the given path
    all_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not all_dirs:
        print("No directories found.")
        return

    # Get the creation time of each directory
    dir_creation_times = {d: os.path.getctime(os.path.join(path, d)) for d in all_dirs}

    # Find the directory with the latest creation time
    latest_directory = max(dir_creation_times, key=dir_creation_times.get)

    return os.path.join(path, latest_directory)


def find_largest_numbered_dir(root_dir):
    largest_number = -1  # Initialize the largest number to a very small value
    largest_dir = None

    # Iterate over each entry in the root directory
    for entry in os.listdir(root_dir):
        if os.path.isdir(os.path.join(root_dir, entry)):
            if entry.isdigit():  # Check if entry name is a digit
                entry_number = int(entry)
                if entry_number > largest_number:
                    largest_number = entry_number
                    largest_dir = entry
    if largest_dir is None:
        return -1
    return int(largest_dir)


def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
    return re.sub(' +', ' ', text).strip()


def iter_list(list_, batch_size):
    num_of_batches = math.ceil(len(list_) / batch_size)
    for i in range(num_of_batches):
        yield list_[i * batch_size:(i + 1) * batch_size]


class RevisionDataset(Dataset):
    def __init__(self, texts, summaries, revised_summaries):
        self.texts = texts
        self.summaries = summaries
        self.revised_summaries = revised_summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'summary': self.summaries[item],
                'revised_summary': self.revised_summaries[item]}

class RevisionDatasetWithFeedback(Dataset):
    def __init__(self, texts, summaries, revised_summaries, feedback):
        self.texts = texts
        self.summaries = summaries
        self.revised_summaries = revised_summaries
        self.feedback = feedback

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'summary': self.summaries[item],
                'revised_summary': self.revised_summaries[item], 'feedback': self.feedback[item]}
class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries):
        self.texts = texts
        self.summaries = summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'summary': self.summaries[item]}

class SummarizationDatasetWithFeedback(Dataset):
    def __init__(self, texts, summaries, feedback):
        self.texts = texts
        self.summaries = summaries
        self.feedback = feedback

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'summary': self.summaries[item], 'feedback': self.feedback[item]}
class SummarizationDatasetwithLogits(Dataset):
    def __init__(self, texts, summaries, logits):
        self.texts = texts
        self.summaries = summaries
        self.logits = logits

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'summary': self.summaries[item], 'logits': self.logits[item]}


class ExplanationsDataset(Dataset):
    def __init__(self, texts, summaries, explanations):
        self.texts = texts
        self.summaries = summaries
        self.explanations = explanations

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'summary': self.summaries[item], 'explanations': self.explanations[item]}


class InstructionsDataset(Dataset):
    def __init__(self, texts, summaries, instructions):
        self.texts = texts
        self.summaries = summaries
        self.instructions = instructions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {'text': self.texts[item], 'summary': self.summaries[item], 'instructions': self.instructions[item]}


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


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    print("Destroying process group...")
    dist.destroy_process_group()

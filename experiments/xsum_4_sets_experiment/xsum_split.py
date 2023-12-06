from torch.utils.data import Dataset, Subset
from datasets import load_dataset
import random
import os
import pickle


def split_xsum_train(dataset, split, path_to_documents_for_summarization_indices, num_of_documents_for_summarization,
                     seed):
    if os.path.exists(path_to_documents_for_summarization_indices):
        documents_for_summarization_indices = pickle.load(open(path_to_documents_for_summarization_indices, 'rb'))
        if split == 'train_model':
            indices = [i for i in range(len(dataset)) if i not in documents_for_summarization_indices]
        else:
            indices = documents_for_summarization_indices

    else:
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:num_of_documents_for_summarization]
        pickle.dump(indices, open(path_to_documents_for_summarization_indices, 'wb'))
        if split == 'train_model':
            indices = [i for i in range(len(dataset)) if i not in indices]
    return Subset(dataset, indices)


def split_xsum_dataset(split, path_to_documents_for_summarization_indices, num_of_documents_for_summarization=None,
                       seed=42):
    if split == 'validation_model':
        dataset = load_dataset('xsum', split='validation')
    elif split == 'factuality_test':
        dataset = load_dataset('xsum', split='test')
    elif split in ['train_model', 'documents_for_summarization']:
        dataset = load_dataset('xsum', split='train')
        dataset = split_xsum_train(dataset, split, path_to_documents_for_summarization_indices,
                                   num_of_documents_for_summarization,
                                   seed)
    else:
        raise ValueError(f'Unexpected split: {split}')
    return dataset

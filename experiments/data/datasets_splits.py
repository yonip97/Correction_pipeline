from torch.utils.data import Subset
from datasets import load_dataset
import random
import os
import pickle
import json


def split_train(dataset, split, path_to_documents_for_summarization_indices, num_of_documents_for_summarization,
                num_of_documents_for_revision, seed, additional_train_filter_indices):
    print(path_to_documents_for_summarization_indices)
    if os.path.exists(path_to_documents_for_summarization_indices):
        documents_indices = json.load(open(path_to_documents_for_summarization_indices, 'r'))
        if split == 'train_model':
            indices = documents_indices['train_indices']
        elif split == 'revision_documents':
            indices = documents_indices['revision_indices']
        else:
            indices = documents_indices['summarization_indices']
    else:
        print("Should I create a new split? (y/n)")
        answer = input()
        while answer not in ['y', 'n']:
            if answer == 'n':
                raise ValueError("wrong path")
            if answer == 'y':
                break
            print("Should I create a new split? (y/n)")
            answer = input()
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        revision_indices = indices[:num_of_documents_for_revision]
        summarization_indices = indices[
                                num_of_documents_for_revision:num_of_documents_for_summarization + num_of_documents_for_revision]
        train_indices = indices[num_of_documents_for_summarization + num_of_documents_for_revision:]
        with open(path_to_documents_for_summarization_indices, 'w') as file:
            json.dump({'revision_indices': revision_indices, 'summarization_indices': summarization_indices,
                       'train_indices': train_indices}, file)
        if split == 'train_model':
            indices = [i for i in range(len(dataset)) if
                       (i not in revision_indices) and (i not in summarization_indices)]
        elif split == 'revision_documents':
            indices = revision_indices
        else:
            indices = summarization_indices
    if additional_train_filter_indices is not None:
        indices = [i for i in indices if i in additional_train_filter_indices]
    return Subset(dataset, indices)


def split_cnndm_dataset(split, path_to_documents_for_summarization_indices, num_of_documents_for_summarization,
                        num_of_documents_for_revision, seed):
    if 'cnndm' not in path_to_documents_for_summarization_indices:
        raise ValueError("wrong path")
    if split == 'validation_model':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='validation').rename_columns(
            {'highlights': 'summary', 'article': 'text'})
    elif split == 'factuality_test':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test').rename_columns(
            {'highlights': 'summary', 'article': 'text'})
    elif split in ['train_model', 'summarization_documents', 'revision_documents']:
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='train').rename_columns(
            {'highlights': 'summary', 'article': 'text'})
        dataset = split_train(dataset, split, path_to_documents_for_summarization_indices,
                              num_of_documents_for_summarization, num_of_documents_for_revision, seed)
    else:
        raise ValueError(f'Unexpected split: {split}')
    return dataset


def split_xsum_dataset(split, path_to_documents_for_summarization_indices, num_of_documents_for_summarization,
                       num_of_documents_for_revision, seed, additional_train_filter_indices=None):
    if split == 'validation_model':
        dataset = load_dataset('xsum', split='validation').rename_column('document', 'text')
    elif split == 'factuality_test':
        dataset = load_dataset('xsum', split='test').rename_column('document', 'text')
    elif split in ['train_model', 'summarization_documents', 'revision_documents']:
        if 'xsum' not in path_to_documents_for_summarization_indices:
            raise ValueError("wrong path")
        dataset = load_dataset('xsum', split='train').rename_column('document', 'text')
        if num_of_documents_for_revision + num_of_documents_for_summarization == 0:
            return dataset
        dataset = split_train(dataset, split, path_to_documents_for_summarization_indices,
                              num_of_documents_for_summarization, num_of_documents_for_revision, seed,
                              additional_train_filter_indices)
    else:
        raise ValueError(f'Unexpected split: {split}')
    return dataset

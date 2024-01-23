import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from datasets import load_dataset
from Seahorse_metrics.metrics import Seahorse_metrics
import torch
from general.utils import iter_list
from experiments.ablations.ctrl.utils import save_list_to_file, load_list_from_file


# Function to save a list to a file


def main():
    output_path = 'experiments/ablations/ctrl/outputs/scored_xsum_train_dataset.pkl'
    if os.path.exists(output_path):
        scores = load_list_from_file(output_path)
    else:
        scores = []
    print("len(scores): ", len(scores))
    print("score mean: ", np.mean(scores))
    train_xsum_full = load_dataset('xsum', split='train')
    texts = [train_xsum_full[i]['document'] for i in range(len(train_xsum_full))]
    summaries = [train_xsum_full[i]['summary'] for i in range(len(train_xsum_full))]
    factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                         device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16,
                                         return_none=True)
    indexes = list(range(len(scores), len(texts)))
    for batch_indexes in iter_list(indexes, 100):
        print("first index: ", batch_indexes[0])
        batch_texts = [texts[i] for i in batch_indexes]
        batch_summaries = [summaries[i] for i in batch_indexes]
        batch_scores = factuality_metric.score(summaries=batch_summaries, texts=batch_texts)
        scores += batch_scores
        save_list_to_file(scores, output_path)


if __name__ == "__main__":
    main()

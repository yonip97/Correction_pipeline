import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import pickle

import pandas as pd
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
import numpy as np
import evaluate
from general.utils import RevisionDataset
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset


def main():
    rouge_metric =evaluate.load('rouge')
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    original_summaries = [xsum_dataset[i]['summary'] for i in range(len(xsum_dataset))]
    df = pd.read_csv("experiments/ablations/better_summaries/data/gpt_3.5_summaries_with_indices.csv",index_col=0)
    df['original_summary'] = original_summaries
    df = df[df['summary'].notna()]
    df['text'] = [str(x) for x in df['text'].tolist()]
    results = Fragments().score(metrics=['density', 'coverage'], texts=df['text'].tolist(),
                                summaries=df['summary'].tolist())
    df['density'] = results['density']
    df['coverage'] = results['coverage']
    df['length'] = [len(word_tokenize(x)) for x in df['text'].tolist()]
    rouge_results = rouge_metric.compute(predictions = df['summary'].tolist(), references = df['original_summary'].tolist())
    for key in rouge_results.keys():
        df[key] = rouge_results[key]
    df.to_csv("experiments/ablations/better_summaries/data/gpt_3.5_summaries_with_indices.csv", index=False)
def rel():
    df = pd.read_csv("experiments/ablations/better_summaries/data/gpt_3.5_summaries_with_indices.csv")
    with open("experiments/ablations/better_summaries/data/chosen_xsum_indices.pkl", 'rb') as file:
        chosen_indices = pickle.load(file)
    print(df['density'].mean())
    print(df['coverage'].mean())
    print(df['rougeL'].mean())
    df = df[df['indices'].isin(chosen_indices)]
    print(df['density'].mean())
    print(df['coverage'].mean())
    print(df['rougeL'].mean())
    print(len(df))
def check():
    import json
    with open("/data/home/yehonatan-pe/Correction_pipeline/experiments/ablations/better_summaries/outputs/summaries.json",'r') as file:
        results = json.load(file)
    for key in results.keys():
        if key in ['hyperparameters','predictions','method']:
            continue
        else:
            print(key)
            print(np.mean(results[key]))
            print()




if __name__ == '__main__':
    check()
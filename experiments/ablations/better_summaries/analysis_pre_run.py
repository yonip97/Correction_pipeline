import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from scipy.stats import rankdata
import pandas as pd
from general.fragments_metrics import Fragments


def read_data():
    path = "experiments/ablations/better summaries/A_Systematic_Study_and_Comprehensive_Evaluation_of_ChatGPT/with_chatgpt_response_xsum_unrestricted.xlsx"
    df = pd.read_excel(path)
    texts = [str(x) for x in df['document'].tolist()]
    summaries = [str(x) for x in df['chatgpt_response'].tolist()]
    metric = Fragments()
    results = metric.score(metrics=['density', 'coverage'], texts=texts, summaries=summaries)
    for key in results:
        print(key, np.mean(results[key]))


def get_element_ranks(data):
    ranks = rankdata(data, method='max')
    return ranks


def chose_summaries_for_example():
    df = pd.read_csv("experiments/ablations/better_summaries/data/20000_summaries_scored.csv", index_col=0)
    density = df['density'].tolist()
    coverage = df['coverage'].tolist()
    scores = df['score'].tolist()
    rel_df = df[(df['density'] < 1) & (df['score'] > 0.8)]
    from nltk.tokenize import word_tokenize
    rel_df['length'] = [len(word_tokenize(x)) for x in rel_df['text'].tolist()]
    # rel_df = rel_df[rel_df['length'] > 200]
    rel_df.sort_values(by=['score'], inplace=True, ascending=False)

    for i in range(20):
        print("summary:")
        print(rel_df.iloc[i]['summary'])
        print("text:")
        print(rel_df.iloc[i]['text'])
        print(rel_df.iloc[i]['density'])
        print(rel_df.iloc[i]['coverage'])
        print(rel_df.iloc[i]['score'])
        print(rel_df.iloc[i]['length'])
        print('-------------------')


def check_results_of_prompt_check():
    from general.fragments_metrics import Fragments
    from Seahorse_metrics.metrics import Seahorse_metrics
    import torch
    dir = "experiments/ablations/better_summaries/prompt_testing"
    factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                         tokenizer_name='google/seahorse-xxl-q4',
                                         device='auto', batch_size=1, torch_dtype=torch.float16,
                                         max_length=2048, return_none=True)
    files = [file for file in os.listdir(dir) if file.endswith('.csv')]
    all_results = {}
    for file in files:
        df = pd.read_csv(dir + '/' + file)
        all_results[file] = {}
        all_results[file]['prompt'] = df['prompt'][0]
        metric = Fragments()
        results = metric.score(metrics=['density', 'coverage'], texts=df['text'].tolist(),
                               summaries=df['summary'].tolist())
        results['seahorse'] = factuality_metric.score(texts=df['text'].tolist(), summaries=df['summary'].tolist())
        from nltk.tokenize import word_tokenize
        results['length'] = [len(word_tokenize(x)) for x in df['summary'].tolist()]
        for res in results:
            print(res)
            all_results[file][res] = np.mean(results[res])
    for key in all_results:
        print(key)
        for metric in all_results[key]:
            print(metric, all_results[key][metric])
    import json
    with open('experiments/ablations/better_summaries/prompt_testing/results.json', 'w') as f:
        json.dump(all_results, f)
def automatic_analysis_of_prompt_check():
    import json
    densities = []
    scores = []
    prompts = []
    with open('experiments/ablations/better_summaries/prompt_testing/results.json', 'r') as f:
        results = json.load(f)
    for key in results:
        print(key)
        prompts.append(key)
        for metric in results[key]:
            print(metric, results[key][metric])
            if metric == 'density':
                densities.append(results[key][metric])
            if metric == 'seahorse':
                scores.append(results[key][metric])
        print('---------------------------------------------------------')
    import matplotlib.pyplot as plt
    plt.scatter(densities, scores)
    for (density, score, name) in zip(densities, scores, prompts):
        plt.text(density, score, name)
    plt.show()
def manual_analysis_of_prompt_check():
    dir = "experiments/ablations/better_summaries/prompt_testing"
    files = [file for file in os.listdir(dir) if file.endswith('.csv')]
    import random
    random.seed(42)
    random_indexes = random.sample(range(0, 200), 10)
    print([(files[i],i+1) for i in range(len(files))])
    file = files[11]
    df = pd.read_csv(dir + '/' + file)
    print(file)
    print(df['prompt'][0])
    for index in random_indexes:
        print("text:")
        print(df['text'][index])
        print("summary:")
        print(df['summary'][index])
        print()
        print('---------------------------------------------------------')
    # df = pd.read_csv(dir + '/' + file)
    # for index in random_indexes:
    #     print(df['summary'][index])
    #     print()
    # print('---------------------------------------------------------')

    # for file in files:
    #     print(file)
    #     df = pd.read_csv(dir + '/' + file)
    #     for index in random_indexes:
    #         print(df['summary'][index])
    #         print()
    #     print('---------------------------------------------------------')

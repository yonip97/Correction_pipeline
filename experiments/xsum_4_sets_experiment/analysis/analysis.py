import time

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
import json

import evaluate
import matplotlib.pyplot as plt
from sklearn.feature_selection import r_regression
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset, split_cnndm_dataset
from tqdm import tqdm


def get_data(file, data_dir='experiments/xsum_4_sets_experiment'):
    # with open('experiments/xsum_4_sets_experiment/test_summaries_all_methods_scored.json', 'r') as file:
    #     results = json.load(file)
    with open(os.path.join(data_dir, file), 'r') as file:
        results = json.load(file)
    results.pop('hyperparameters')
    from nltk import word_tokenize
    for key in results:
        results[key]['length'] = [len(word_tokenize(text)) for text in results[key]['summaries']]
    # with open('experiments/xsum_4_sets_experiment/baseline_beam_search_4_summaries_scored.json', 'r') as file:
    #     baseline_results = json.load(file)
    # results = {}
    # results['baseline'] = baseline_results['baseline_new_decoding']
    # with open('experiments/xsum_4_sets_experiment/all_methods__beam_search_4_summaries_scored.json', 'r') as file:
    #     new_results = json.load(file)
    return results


def compute_rouge(texts1, texts2):
    rouge_metric = evaluate.load('rouge')
    scores = rouge_metric.compute(predictions=texts1, references=texts2, use_aggregator=False)
    return scores


def compare_histograms(scores1, scores2, method1, method2):
    plt.hist(scores1, bins=20, label=method1, alpha=0.5)
    plt.hist(scores2, bins=20, label=method2, alpha=0.5)
    plt.legend()
    plt.title(f'{method1} vs {method2}')
    plt.show()


def diff_histogram(scores1, scores2, method1, method2):
    plt.hist(scores1 - scores2, bins=20)
    plt.title(f'{method1} - {method2}')
    plt.show()


def rouge_vs_score(rouge_scores, scores, method):
    plt.scatter(rouge_scores, scores)
    plt.title(f'{method}')
    plt.show()


def look_at_examples(baseline_texts, method_texts, baseline_factuality_scores, method_factuality_scores, rouge_scores,
                     method_name):
    diff_factuality_scores = method_factuality_scores - baseline_factuality_scores
    ordered_factuality_scores = np.argsort(diff_factuality_scores)[::-1]
    ordered_rouge_scores = np.argsort(rouge_scores)[::-1]
    percentile = int(0.1 * len(ordered_rouge_scores))
    best = set(ordered_factuality_scores[:percentile]).intersection(set(ordered_rouge_scores[:percentile]))
    worst = set(ordered_factuality_scores[-percentile:]).intersection(set(ordered_rouge_scores[-percentile:]))
    print("Length of best is ", len(best))
    for i in best:
        print(f"Baseline: {baseline_texts[i]}")
        print(f"Method: {method_texts[i]}")
        print(f"baseline factuality score: {baseline_factuality_scores[i]}")
        print(f"Method factuality score: {method_factuality_scores[i]}")
        print(f"Rouge score: {rouge_scores[i]}")
        print("---------------------------------------------")
        print()
    print("Length of worst is ", len(worst))
    for i in worst:
        print(f"Baseline: {baseline_texts[i]}")
        print(f"Method: {method_texts[i]}")
        print(f"baseline factuality score: {baseline_factuality_scores[i]}")
        print(f"Method factuality score: {method_factuality_scores[i]}")
        print(f"Rouge score: {rouge_scores[i]}")
        print("---------------------------------------------")
        print()


def add_rouge_to_baseline(data):
    rouge_metric = evaluate.load('rouge')
    baseline_summaries = data['baseline']['summaries']
    for method in data.keys():
        if method == 'baseline':
            continue
        method_summaries = data[method]['summaries']
        scores = rouge_metric.compute(predictions=baseline_summaries, references=method_summaries,
                                      use_aggregator=False)
        for key in scores:
            data[method]['to_baseline_' + key] = scores[key]
    return data


def add_rouge_to_original_summaries(data):
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_10000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=10000,
                                      seed=42)
    xsum_summaries = [xsum_dataset[i]['summary'] for i in range(len(xsum_dataset))]
    rouge_metric = evaluate.load('rouge')
    for method in data.keys():
            method_summaries = data[method]['summaries']
            scores = rouge_metric.compute(predictions=method_summaries, references=xsum_summaries,
                                          use_aggregator=False)

            for key in scores:
                data[method]['to_original_' + key] = scores[key]
    return data


def add_coverage_and_density(data):
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    xsum_texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    from general.fragments_metrics import Fragments
    scorer = Fragments()
    for method in data.keys():
            method_summaries = data[method]['summaries']
            xsum_results = scorer.score(metrics=['coverage', 'density'], summaries=method_summaries,
                                          texts=xsum_texts)
            for key in xsum_results:
                data[method][key] = xsum_results[key]
            print(np.mean(xsum_results['coverage']))
            print(np.median(xsum_results['coverage']))
            print(np.std(xsum_results['coverage']))
            print(np.mean(xsum_results['density']))
            print(np.median(xsum_results['density']))
            print(np.std(xsum_results['density']))

    return data


def create_result_table(data, output_dir, output_prefix, save=False):
    methods = list(data.keys())
    results = {}
    for method in methods:
        results[method] = {}
        for key in data[method].keys():
            if 'summaries' in key:
                continue
            if key == 'density' or key == 'coverage':
                results[method][key+'_mean'] = np.mean(data[method][key])
                results[method][key+'_median'] = np.median(data[method][key])
            else:
                results[method][key] = np.mean(
                    [x for x in data[method][key] if x is not None])
    df = pd.DataFrame(results)
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.precision', 4,
                           ):
        print()
        print(df.transpose().to_string())
        print()
        print("---------------------------------------------------")
        if save:
            html_table = df.to_html()
            output_path = output_dir + '/' + output_prefix + '.html'
            with open(output_path, 'w') as f:
                f.write(html_table)


def main():
    # results ={}
    # for method in ['baseline','all','classifier','classifier_and_rouge_threshold']:
    #     with open(f'experiments/xsum_4_sets_experiment/{method}_beam_search_4_summaries.json', 'r') as file:
    #         temp = json.load(file)
    #         results[method] = {}
    #         results[method]['xsum'] = {}
    #         results[method]['cnndm'] = {}
    #         results[method]['xsum']['summaries'] = temp['xsum']
    #         results[method]['cnndm']['summaries'] = temp['cnndm']
    # results = add_coverage_and_density(results)
    # results = add_rouge_to_baseline(results)
    # results = add_rouge_to_original_summaries(results)
    # create_result_table(results,'experiments/xsum_4_sets_experiment/outputs','results_beam_search_4_decoding_',save=False)

    results = get_data(file="revision_by_flan_large_nli_scored.json")
    # results = get_data(file= "test_summaries_all_methods_scored.json")

    # results = add_coverage_and_density(results)
    # results = add_rouge_to_original_summaries(results)
    # results = add_rouge_to_baseline(results)
    create_result_table(results, 'experiments/xsum_4_sets_experiment/outputs', 'results_beam_search_4_decoding_revision_by_flan_large',
                        save=True)
    # xsum_baseline_texts = data['baseline']['xsum']['summaries']
    # baseline_trueteacher_scores = np.array(data['baseline']['xsum']['seahorse_xxl_scores'])
    # method = 'classifier'
    # xsum_method_texts = data[method]['xsum']['summaries']
    # all_true_teacher_scores = np.array(data[method]['xsum']['seahorse_xxl_scores'])
    # rouge_scores = compute_rouge(xsum_method_texts, xsum_baseline_texts)['rougeL']
    # previous = 0
    # print(np.median(rouge_scores))
    # print(np.mean(rouge_scores))
    # for i in np.linspace(0, 1, 11):
    #     print(i)
    #     curr = len([s for s in rouge_scores if s < i])
    #     print((curr-previous)/len(rouge_scores))
    #     previous = curr
    #     print("--------------------------------------")
    #
    # plt.hist(rouge_scores, bins=20)
    # plt.xlim(0, 1)
    # plt.show()
    # look_at_examples(baseline_texts=xsum_baseline_texts, method_texts=xsum_method_texts,
    #                  baseline_factuality_scores=baseline_trueteacher_scores,
    #                  method_factuality_scores=all_true_teacher_scores, rouge_scores=rouge_scores, method_name="all")
    # methods = list(data.keys())
    # for dataset_name in ['xsum', 'cnndm']:
    #     print(dataset_name)
    #     for i in range(len(methods)):
    #         method1 = methods[i]
    #         summaries1 = data[method1][dataset_name]['summaries']
    #         seahorse_scores1 = np.array(data[method1][dataset_name]['seahorse_xxl_scores']).reshape((-1, 1))
    #         for j in range(i + 1, len(methods)):
    #             method2 = methods[j]
    #             print(f"{method1} vs {method2}")
    #             summaries2 = data[method2][dataset_name]['summaries']
    #             rouge_scores = compute_rouge(summaries1, summaries2)
    #             seahorse_scores2 = np.array(data[method2][dataset_name]['seahorse_xxl_scores']).reshape((-1, 1))
    #             print(r_regression(np.array(rouge_scores['rouge1']).reshape((-1, 1)),
    #                                seahorse_scores2 - seahorse_scores1))
    # rouge_vs_score(rouge_scores['rouge1'], scores=seahorse_scores2 - seahorse_scores1,
    #                method=f'{method1} vs {method2}')


if __name__ == '__main__':
    main()

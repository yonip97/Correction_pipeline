import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import numpy as np

from Seahorse_metrics.metrics import Seahorse_metrics
from summac.model_summac import SummaCZS
from TrueTeacher.inference import TrueTeacher
import pandas as pd
from data.factuality_datasets import TRUE_dataset
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, roc_auc_score
from matplotlib import pyplot as plt


def get_classifiers_auc_roc(datasets, texts, summaries, labels):
    # if not os.path.exists('data/true_data/seahorse'):
    #     print(os.getcwd())
    #     raise Exception('Please create dir')
    # seahorse_small = Seahorse_metrics('google/seahorse-large-q4', 'google/seahorse-large-q4', device='cuda:1')
    # scores = seahorse_small.score(texts, summaries)
    # seahorse_small.model.to('cpu')
    # pd.DataFrame({'datasets': datasets, 'scores': scores, 'labels': labels}).to_csv(
    #     'data/true_data/seahorse/small_seahorse_scores.csv')
    model = TrueTeacher('google/t5_11b_trueteacher_and_anli', 'google/t5_11b_trueteacher_and_anli', device='auto',
                        batch_size=1,
                        torch_dtype=torch.bfloat16, return_none=True)
    scores = model.score(texts, summaries)
    pd.DataFrame({'datasets': datasets, 'scores': scores, 'labels': labels}).to_csv(
        'data/true_data/TrueTeacher/TrueTeacher11b_16float.csv')
    seahorse_large = Seahorse_metrics('google/seahorse-xxl-q4', 'google/seahorse-xxl-q4', device='auto', batch_size=1)
    scores = seahorse_large.score(texts, summaries)
    pd.DataFrame({'datasets': datasets, 'scores': scores, 'labels': labels}).to_csv(
        'data/true_data/seahorse/large_seahorse_scores_16float.csv')
    # summac_classifier = SummaCZS(granularity="document", model_name="vitc", imager_load_cache=True,
    #                              device="cuda:1", use_con=False)
    # scores = summac_classifier.score(texts, summaries)['scores']
    # pd.DataFrame({'datasets': datasets, 'scores': scores, 'labels': labels}).to_csv(
    #     'data/true_data/summac/summac_scores.csv')


def compare_other_metrics():
    with open('metric_comparison.txt', 'w') as file:
        thresholds = np.linspace(0, 1, 101)
        seahorse_small = pd.read_csv('data/true_data/seahorse/small_seahorse_scores.csv', index_col=0)
        # seahorse_large = pd.read_csv('data/seahorse/large_seahorse_scores.csv', index_col=0)
        summac = pd.read_csv('data/true_data/summac/summac_scores.csv', index_col=0)
        true_teacher = pd.read_csv('data/true_data/TrueTeacher/TrueTeacher_predictions.csv', index_col=0)
        chatgpt = pd.read_csv('data/true_data/chatgpt/True_chatgpt_classification.csv', index_col=0)
        chatgpt = chatgpt[~chatgpt['prediction'].isna()]
        for df, name in zip([seahorse_small, summac], ['seahorse_small', 'summac']):
            file.write(f'{name}:\n')
            for dataset in df['datasets'].unique():
                file.write(f'{dataset}:\n')
                scores = df[df['datasets'] == dataset]['scores'].tolist()
                labels = df[df['datasets'] == dataset]['labels'].tolist()
                for threshold in thresholds:
                    predictions = [1 if score > threshold else 0 for score in scores]
                    file.write("Threshold: " + str(threshold) + "\n")
                    file.write("F1 score: " + str(f1_score(labels, predictions)) + "\n")
                    file.write("Balanced accuracy score: " + str(balanced_accuracy_score(labels, predictions)) + "\n")
                    file.write("Accuracy score: " + str(accuracy_score(labels, predictions)) + "\n")
                    file.write("\n")
        file.write('true_teacher:\n')
        for dataset in true_teacher['dataset'].unique():
            file.write(f'{dataset}:\n')
            df = true_teacher[true_teacher['dataset'] == dataset]
            predictions = df['prediction'].tolist()
            labels = df['label'].tolist()
            file.write('F1 score: ' + str(f1_score(labels, predictions)) + "\n")
            file.write(
                "Balanced accuracy score: " + str(balanced_accuracy_score(labels, predictions)) + "\n")
            file.write("Accuracy score: " + str(accuracy_score(labels, predictions)) + "\n")
            file.write("\n")
        file.write('chatgpt:\n')
        for dataset in chatgpt['dataset'].unique():
            file.write(f'{dataset}:\n')
            df = chatgpt[chatgpt['dataset'] == dataset]
            predictions = df['prediction'].tolist()
            labels = df['label'].tolist()
            file.write('F1 score: ' + str(f1_score(labels, predictions)) + "\n")
            file.write(
                "Balanced accuracy score: " + str(balanced_accuracy_score(labels, predictions)) + "\n")
            file.write("Accuracy score: " + str(accuracy_score(labels, predictions)) + "\n")
            file.write("\n")


def find_best_threshold(results):
    best_threshold = None
    balanced_acc = 0
    for threshold in results:
        if results[threshold]['balanced_accuracy'] > balanced_acc:
            balanced_acc = results[threshold]['balanced_accuracy']
            best_threshold = threshold
    return best_threshold, balanced_acc


def main():
    dataset = TRUE_dataset('data/true_data', ['summarization'])
    texts = dataset.df['grounding'].tolist()
    summaries = dataset.df['generated_text'].tolist()
    labels = dataset.df['label'].tolist()
    datasets = dataset.df['dataset'].tolist()
    get_classifiers_auc_roc(datasets=datasets, texts=texts, summaries=summaries, labels=labels)
    # dataset = pd.read_csv('data/true_data/seahorse/small_seahorse_scores.csv', index_col=0)
    # for dataset_name in dataset['datasets'].unique():
    #     df = dataset[dataset['datasets'] == dataset_name]
    #     scores = df['scores'].tolist()
    #     labels = df['labels'].tolist()
    #     print(dataset_name)
    #     print(roc_auc_score(labels, scores))
    c = 1
    # get_classifiers_auc_roc(datasets=datasets, texts=texts, summaries=summaries, labels=labels)
    # compare_other_metrics()
    # results = {}
    # current_model = None
    # threshold = None
    # with open('metric_comparison.txt', 'r') as file:
    #     txt = file.readlines()
    #     for line in txt:
    #         if 'seahorse_small' in line:
    #             current_model = 'seahorse_small'
    #         elif 'summac' in line:
    #             current_model = 'summac'
    #         elif 'true_teacher' in line:
    #             current_model = 'true_teacher'
    #         elif 'chatgpt' in line:
    #             current_model = 'chatgpt'
    #         if current_model not in results:
    #             results[current_model] = {}
    #         if current_model in ['seahorse_small', 'summac']:
    #             if 'Threshold' in line:
    #                 threshold = float(line.split(' ')[1])
    #                 results[current_model][threshold] = {}
    #             elif 'F1 score' in line:
    #                 results[current_model][threshold]['f1'] = float(line.split(' ')[2])
    #             elif 'Balanced accuracy score' in line:
    #                 results[current_model][threshold]['balanced_accuracy'] = float(line.split(' ')[3])
    #             elif 'Accuracy score' in line:
    #                 results[current_model][threshold]['accuracy'] = float(line.split(' ')[2])
    #         else:
    #             if 'F1 score' in line:
    #                 results[current_model]['f1'] = float(line.split(' ')[2])
    #             elif 'Balanced accuracy score' in line:
    #                 results[current_model]['balanced_accuracy'] = float(line.split(' ')[3])
    #             elif 'Accuracy score' in line:
    #                 results[current_model]['accuracy'] = float(line.split(' ')[2])
    # seahorse_results = results['seahorse_small']
    # threshold, res = find_best_threshold(seahorse_results)
    # print(threshold, res)
    # summac_results = results['summac']
    # threshold, res = find_best_threshold(summac_results)
    # print(threshold, res)
    # print(results['true_teacher'])
    # print(results['chatgpt'])


def get_all_results(scores, labels, metric, thresholds):
    results = []
    for threshold in thresholds:
        predictions = [1 if score > threshold else 0 for score in scores]
        if metric == 'accuracy':
            curr_result = accuracy_score(labels, predictions)
        elif metric == 'balanced_accuracy':
            curr_result = balanced_accuracy_score(labels, predictions)
        elif metric == 'f1':
            curr_result = f1_score(labels, predictions)
        else:
            raise Exception('metric not supported')
        results.append(curr_result)
    return results


def best_threshold(scores, labels, metric, thresholds):
    results = get_all_results(scores, labels, metric, thresholds)
    index = np.argmax(results)
    return thresholds[index], results[index]

def find_rank(list_,element):
    list_ = sorted(list_,reverse=True)
    for i,x in enumerate(list_):
        if x < element:
            return i

thresholds = np.linspace(0, 1, 101)
data_prefix = 'data/true_data'
for metric in ['accuracy', 'balanced_accuracy', 'f1']:
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    df = pd.read_csv(data_prefix + '/chatgpt/True_chatgpt_classification.csv', index_col=0)
    df = df[~df['prediction'].isna()]
    predictions = df['prediction'].tolist()
    labels = df['label'].tolist()
    chatgpt_ranks = {}
    if metric == 'accuracy':
        chatgpt_result = accuracy_score(labels, predictions)
    elif metric == 'balanced_accuracy':
        chatgpt_result = balanced_accuracy_score(labels, predictions)
    else:
        chatgpt_result = f1_score(labels, predictions)
    for index, path in enumerate(['/summac/summac_scores.csv', '/seahorse/small_seahorse_scores.csv',
                                  '/TrueTeacher/TrueTeacher11b_16float_temp.csv',
                                  '/seahorse/large_seahorse_scores_16float.csv']):
        row = index // 2
        col = index % 2
        df = pd.read_csv(data_prefix + path, index_col=0)
        scores = df['scores'].tolist()
        labels = df['labels'].tolist()
        results = get_all_results(scores, labels, metric, thresholds)
        rank = find_rank(results,chatgpt_result)
        chatgpt_ranks[path] = rank

        axs[row, col].hist(results, bins=20)
        axs[row, col].title.set_text(path.split('/')[1])
        axs[row, col].set_xlim((0, 1))
        axs[row, col].set_ylim((0, 100))
    plt.suptitle(metric)
    plt.tight_layout()
    plt.show()
    print(chatgpt_ranks)
    print(chatgpt_result)
    # threshold, res = best_threshold(scores, labels, metric, thresholds)
    # print(f'{path} {metric}: threshold: {threshold} result:{res}')

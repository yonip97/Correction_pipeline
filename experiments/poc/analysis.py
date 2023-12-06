import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')

import pandas as pd
import matplotlib.pyplot as plt
from general.metrics import Rouge, word_wise_f1_score, preserve_lev
import math
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, PrecisionRecallDisplay
from general.utils import plot_confusion_matrix
from general.metrics import add_similarity_metrics_scores


def show_factuality_scores_change_histograms(df, col_pre_revision, col_post_revision, title):
    df[col_pre_revision].hist(bins=24, alpha=0.8, label='pre revision scores')
    df[col_post_revision].hist(bins=24, alpha=0.8, label='post revision scores')
    plt.legend()
    plt.xlim((-0.1, 1.1))
    plt.title(title)
    plt.show()


def plot_factuality_scores_change(df, col_pre_revision, col_post_revision, title):
    plt.plot(df[col_pre_revision], df[col_post_revision], 'o')
    plt.xlim((-0.1, 1.1))
    plt.ylim((-0.1, 1.1))
    plt.xlabel('pre revision scores')
    plt.ylabel('post revision scores')
    plt.title(title)
    plt.show()


def plot_text_similarity_metrics(df, metrics, title):
    metrics_num = len(metrics)
    rows_num = math.ceil(metrics_num / 2)
    fig, axes = plt.subplots(rows_num, 2, figsize=(10, 10))
    # fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    counter = 0
    for i in range(rows_num):
        for j in range(2):
            axes[i, j].hist(df[metrics[i * 2 + j]], bins=20, range=(0, 1))
            axes[i, j].set_title(metrics[i * 2 + j])
            counter += 1
            if counter == metrics_num:
                break
    plt.suptitle(title)
    plt.show()


def transform_scores_to_predictions(scores, threshold=0.5):
    new_scores = []
    for s in scores:
        if s is None:
            new_scores.append(None)
        elif s >= threshold:
            new_scores.append(1)
        else:
            new_scores.append(0)
    return new_scores


def plot_auc_roc(columns, df):
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    for i, pre in enumerate(columns):
        row = i // 2
        if i % 2 == 0:
            col = 0
        else:
            col = 1
        # plot_confusion_matrix(df, 'pre_correction_label', pre, classes=[0, 1], title=f"{pre} performance")
        print(pre)
        valid = df[df[pre].notnull()]
        display = PrecisionRecallDisplay.from_predictions(
            valid['pre_correction_label'].tolist(), valid[pre].tolist(), name=pre, ax=ax[row, col]
        )
        _ = display.ax_.set_title(pre)
    plt.show()


def main():
    df = pd.read_csv('data/poc_results_full_classification.csv', index_col=0)

    df = add_similarity_metrics_scores(df)
    threshold = 0.5
    df['q_squared_summary_nli_predictions'] = transform_scores_to_predictions(df['q_squared_summary_nli_scores'],
                                                                              threshold=threshold)
    df['q_squared_summary_f1_predictions'] = transform_scores_to_predictions(df['q_squared_summary_f1_scores'],
                                                                             threshold=threshold)
    df['q_squared_revised_summary_nli_predictions'] = transform_scores_to_predictions(
        df['q_squared_revised_summary_nli_scores'], threshold=threshold)
    df['q_squared_revised_summary_f1_predictions'] = transform_scores_to_predictions(
        df['q_squared_revised_summary_f1_scores'], threshold=threshold)
    pre_revision_scores = ['true_teacher_summary_scores', 'q_squared_summary_nli_scores', 'q_squared_summary_f1_scores',
                           'factcc_summary_scores']
    post_revision_scores = ['true_teacher_revised_summary_scores', 'q_squared_revised_summary_nli_scores',
                            'q_squared_revised_summary_f1_scores', 'factcc_revised_summary_scores']
    pre_revision_predictions = ['true_teacher_predictions', 'q_squared_summary_nli_predictions',
                                'q_squared_summary_f1_predictions', 'factcc_summary_predictions']
    post_revision_predictions = ['true_teacher_revised_predictions', 'q_squared_revised_summary_nli_predictions',
                                 'q_squared_revised_summary_f1_predictions', 'factcc_revised_summary_predictions']
    # pre_revision_scores = ['true_teacher_summary_scores']
    # post_revision_scores = ['true_teacher_revised_summary_scores']
    # pre_revision_predictions = ['true_teacher_predictions']
    # post_revision_predictions = ['true_teacher_revised_predictions']
    plot_text_similarity_metrics(df, ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'preserve_lev'],
                                 "text similarity metrics")

    plot_auc_roc(pre_revision_predictions, df)
    for pre, post in zip(pre_revision_scores, post_revision_scores):
        show_factuality_scores_change_histograms(df, pre, post, title=pre)
        plot_factuality_scores_change(df, pre, post, title=pre)
    for pre, post in zip(pre_revision_predictions, post_revision_predictions):
        plot_confusion_matrix(df, pre, post, classes=[0, 1], title=f"{pre} performance")




if __name__ == "__main__":
    main()

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
# from transformers import Pipeline,pipeline
from TrueTeacher.inference import TrueTeacher

from data.factuality_datasets import BERTS2S_TConvS2S_xsum_trained_dataset
from knowledge_distillation.text_correction import Summerization_correction_model
import pandas as pd
import time
from tqdm import tqdm
import csv
from general.metrics import Rouge, word_wise_f1_score, preserve_lev
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
import math


class mock_model():
    def __call__(self, text):
        return text[:20]



def main():
    csv_file_path = "data/poc_results.csv"
    if os.path.exists(csv_file_path):
        # os.remove(csv_file_path)
        raise ValueError("File already exists")

    prompt = """The following summary is factually inconsistent w.r.t the document. Please make it factually consistent while making minimal changes to the summary. Output only the corrected summary and nothing more."""
    # API_KEY = "dc2cfae938eb46a6a78f0594715a4006"
    # API_KEY = None
    dataset = BERTS2S_TConvS2S_xsum_trained_dataset()
    model = Summerization_correction_model(prompt_text=prompt, model='gpt-4', API_KEY=API_KEY)
    # model = mock_model()
    # records = []
    counter = 0
    with open(csv_file_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ('index', 'document', 'summary', 'revised_summary', 'pre_correction_label', 'revision_successful'))
        for i in tqdm(range(len(dataset))):
            item = dataset[i]
            summary = item['hypothesis']
            doc = item['premise']
            label = item['label']
            try:
                # revised_summary = model(summary)
                revised_summary = model(document=doc, summary=summary)
                csvwriter.writerow((i, doc, summary, revised_summary, label, 1))

                time.sleep(2)
            except:
                counter += 1
                csvwriter.writerow((i, doc, summary, None, label, 0))
                print(f"There is a problem with index {i}")
                print(f"There are {counter} faulty examples")
                time.sleep(10)

            # records.append((i,doc, summary, revised_summary, label))
            # time.sleep(2)
    print(f"There are {counter} faulty examples")


def add_similarity_metrics_scores(df):
    rouge_metric = Rouge()
    rouge_results = rouge_metric(df['revised_summary'].tolist(), df['summary'].tolist())
    df['rouge1'] = rouge_results['rouge1']
    df['rouge2'] = rouge_results['rouge2']
    df['rougeL'] = rouge_results['rougeL']
    df['rougeLsum'] = rouge_results['rougeLsum']
    df['preserve_lev'] = df.apply(lambda x: preserve_lev(x['summary'], x['revised_summary']), axis=1)
    f1_scores = []
    for i, row in df.iterrows():
        summary = row['summary']
        revised_summary = row['revised_summary']
        f1_scores.append(word_wise_f1_score(revised_summary, summary))
    df['f1_score'] = f1_scores
    return df


def plot_metrics(df, metrics, title):
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


def analysis():
    df = pd.read_csv('data/poc_results.csv')
    df = df[~df['revised_summary'].isna()]
    df['revised_summary'] = df['revised_summary'].apply(lambda x: x.lower())
    df['summary'] = df['summary'].apply(lambda x: x.lower())
    df = add_similarity_metrics_scores(df)
    plot_metrics(df, ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'preserve_lev'],"text similarity metrics")


def true_teacher_classification():
    df = pd.read_csv('data/poc_results.csv')
    df = df[~df['revised_summary'].isna()]
    device = 'cpu'
    # model_path = '/data/home/yehonatan-pe/Correction_pipeline/TrueTeacher/results/run name_2023-10-11 00:31:57/checkpoint-23000'
    model_path = 'google/t5_11b_trueteacher_and_anli'
    tokenizer_name = 'google/t5_11b_trueteacher_and_anli'
    # tokenizer_name = 't5-base'
    model = TrueTeacher(model_path=model_path, tokenizer_name=tokenizer_name, device=device, batch_size=4,
                        max_length=2048)
    texts = df['document'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    predictions = model.apply(summaries, texts)
    df['true_teacher_predictions'] = predictions
    revised_predictions = model.apply(revised_summaries, texts)
    df['true_teacher_revised_predictions'] = revised_predictions
    df.to_csv('data/poc_results_with_true_teacher_classification.csv')


def cut_to_similar(df):
    rouge_metric = Rouge()
    rouge_results = rouge_metric(df['revised_summary'].tolist(), df['summary'].tolist())
    df['rouge2'] = rouge_results['rouge2']
    return df[df['rouge2'] > 0.6]


def analyse_true_teacher():
    df_big = pd.read_csv('data/poc_results_with_true_teacher_classification_big_model.csv', index_col=0)
    # df_big = cut_to_similar(df_big)
    big_pred = df_big['true_teacher_predictions'].tolist()
    big_revised_pred = df_big['true_teacher_revised_predictions'].tolist()
    where_correct = df_big[df_big['true_teacher_predictions'] == 1]
    where_incorrect = df_big[df_big['true_teacher_predictions'] == 0]
    were_1_predicted_0 = where_correct['true_teacher_predictions'].tolist()

    cm = confusion_matrix(df_big['pre_correction_label'].tolist(),
                          df_big['true_teacher_predictions'].tolist(), labels=[0, 1])
    plot_confusion_matrix(cm, classes=[0, 1], title="True teacher performance")

    # print(sum(were_1_predicted_0))
    # print(len(where_correct))
    # were_0_predicted_1 = where_incorrect['true_teacher_predictions'].tolist()
    # print(sum(were_0_predicted_1))
    # print(len(where_incorrect))


def check_differences_between_changed_and_unchanged():
    df = pd.read_csv('data/poc_results_with_true_teacher_classification_big_model.csv', index_col=0)
    df = add_similarity_metrics_scores(df)
    # factual_df = df[df['true_teacher_predictions'] == 1]
    unfactual_df = df[df['true_teacher_predictions'] == 0]
    unfactual_df_unchanged = unfactual_df[unfactual_df['true_teacher_revised_predictions'] == 0]
    unfactual_df_changed = unfactual_df[unfactual_df['true_teacher_revised_predictions'] == 1]
    plot_metrics(unfactual_df_unchanged, ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'f1_score', 'preserve_lev'],
                 title='unchanged predictions')
    plot_metrics(unfactual_df_changed, ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'f1_score', 'preserve_lev'],
                 title='changed predictions')
    print(unfactual_df_unchanged['rouge1'].mean())
    print(unfactual_df_changed['rouge1'].mean())


def plot_confusion_matrix(cm, classes, title,
                          normalize=False,
                          cmap='gray_r',
                          linecolor='k'):
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
    plt.ylabel('Actual')
    plt.xlabel('predicted')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    analysis()
    # true_teacher_classification()
    # analyse_true_teacher()
    # check_differences_between_changed_and_unchanged()

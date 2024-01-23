import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

import pandas as pd
from TrueTeacher.inference import TrueTeacher
from factCC.inference import Factcc_classifier
from q_squared.inference import Q_squared_classifier
from data.factuality_datasets import TRUE_dataset
from Seahorse_metrics.metrics import Seahorse_metrics
import evaluate

def true_teacher_classification(summaries, texts, device='cpu', batch_size=4):
    # df = pd.read_csv('data/poc_results.csv')
    # df = pd.read_csv(data_path)
    # df = df[~df['revised_summary'].isna()]
    # device = 'cpu'
    # model_path = '/data/home/yehonatan-pe/Correction_pipeline/TrueTeacher/results/run name_2023-10-11 00:31:57/checkpoint-23000'
    model_path = 'google/t5_11b_trueteacher_and_anli'
    tokenizer_name = 'google/t5_11b_trueteacher_and_anli'
    # tokenizer_name = 't5-base'
    model = TrueTeacher(model_path=model_path, tokenizer_name=tokenizer_name, device=device, batch_size=batch_size,
                        max_length=2048)
    # texts = df['document'].tolist()
    # summaries = df['summary'].tolist()
    # revised_summaries = df['revised_summary'].tolist()
    predictions = model.classify(summaries, texts)
    # df['true_teacher_predictions'] = predictions
    # revised_predictions = model.classify(revised_summaries, texts)
    return predictions
    # df['true_teacher_revised_predictions'] = revised_predictions
    # df.to_csv('data/poc_results_with_true_teacher_classification.csv')


def factcc_classification(summaries, texts, device='cpu', batch_size=4):
    model_name = 'bert-base-uncased'
    checkpoint = "/data/home/yehonatan-pe/Correction_pipeline/factCC/checkpoints/factcc-checkpoint"
    classifier = Factcc_classifier(checkpoint_path=checkpoint, backbone_model_name=model_name, device=device)
    predictions = classifier.classify(texts=texts, summaries=summaries, eval_batch_size=batch_size)
    return predictions


def factcc_scoring(summaries, texts, device='cpu', batch_size=4):
    model_name = 'bert-base-uncased'
    checkpoint = "/data/home/yehonatan-pe/Correction_pipeline/factCC/checkpoints/factcc-checkpoint"
    classifier = Factcc_classifier(checkpoint_path=checkpoint, backbone_model_name=model_name, device=device)
    predictions = classifier.score(texts=texts, summaries=summaries, eval_batch_size=batch_size)
    return predictions


def q_squared_scoring(summaries, texts, device='cpu'):
    similarity_metric = 'nli'
    classifier = Q_squared_classifier(device=device, similarity_metric=similarity_metric)
    nli_scores, f1_scores = \
        classifier.score(summaries=summaries, texts=texts, remove_personal=True)
    return nli_scores, f1_scores


def check_q_squared():
    dataset = TRUE_dataset("data/true_data", ['summarization'])
    texts = dataset.df['grounding'].tolist()
    summaries = dataset.df['generated_text'].tolist()
    labels = dataset.df['label'].tolist()
    nli_scores, f1_scores = q_squared_scoring(summaries=summaries, texts=texts, device='cuda:1')
    pd.DataFrame({'nli_scores': nli_scores, 'f1_scores': f1_scores, 'labels': labels}).to_csv(
        'data/q_squared_classification/q_squared_scores_true_dataset.csv')


def add_classification_scores(data_path, output_path):
    df = pd.read_csv(data_path, index_col=0)
    summaries = df['summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    texts = df['document'].tolist()
    # df['q_squared_summary_nli_scores'], df['q_squared_summary_f1_scores'] = q_squared_scoring(
    #     summaries=summaries, texts=texts, device='cuda:1')
    # df['q_squared_revised_summary_nli_scores'], df['q_squared_revised_summary_f1_scores'] = q_squared_scoring(
    #     summaries=revised_summaries, texts=texts, device='cuda:1')
    # df.to_csv('data/poc_results_full_classification.csv')
    # df['true_teacher_summary_scores'] = trueteacher_scoring(summaries=summaries, texts=texts,
    #                                                         device='cpu', batch_size=4)
    # df['true_teacher_revised_summary_scores'] = trueteacher_scoring(summaries=revised_summaries,
    #                                                                 texts=texts, device='cpu',
    #                                                                 batch_size=4)
    # df['true_teacher_summary_predictions'] = true_teacher_classification(summaries=summaries, texts=texts,
    #                                                                 device='cpu', batch_size=4)
    # df['true_teacher_revised_summary_predictions'] = true_teacher_classification(summaries=revised_summaries,
    #                                                                         texts=texts, device='cpu',
    #                                                                         batch_size=4)
    df['factcc_summary_scores'] = factcc_scoring(summaries=summaries, texts=texts,
                                                 device='cuda:1', batch_size=4)
    df['factcc_revised_summary_scores'] = factcc_scoring(summaries=revised_summaries,
                                                         texts=texts, device='cuda:1',
                                                         batch_size=4)
    df['factcc_summary_predictions'] = factcc_classification(summaries=summaries, texts=texts,
                                                             device='cuda:1', batch_size=4)
    df['factcc_revised_summary_predictions'] = factcc_classification(summaries=revised_summaries,
                                                                     texts=texts, device='cuda:1',
                                                                     batch_size=4)
    df.to_csv(output_path)


def trueteacher_scoring(summaries, texts, device='cpu', batch_size=4):
    model_path = 'google/t5_11b_trueteacher_and_anli'
    tokenizer_name = 'google/t5_11b_trueteacher_and_anli'
    model = TrueTeacher(model_path=model_path, tokenizer_name=tokenizer_name, device=device, batch_size=batch_size,
                        max_length=2048, torch_dtype=torch.float16)
    scores = model.score(texts = texts, summaries=summaries)
    return scores


def seahorse_scoring(summaries, texts, device='cpu', batch_size=4):
    model_path = 'google/seahorse-xxl-q4'
    tokenizer_name = 'google/seahorse-xxl-q4'
    model = Seahorse_metrics(model_path=model_path, tokenizer_name=tokenizer_name, device=device, batch_size=batch_size,
                             max_length=2048,torch_dtype=torch.float16)
    scores = model.score(texts=texts, summaries=summaries)
    return scores
def add_rouge(summaries, revised_summaries):
    rouge_metric = evaluate.load('rouge')
    scores = rouge_metric.compute(predictions=revised_summaries, references=summaries,use_aggregator=False)
    return scores

def factuality_change(df):
    df['true_teacher_summary_scores'].hist(bins=24, alpha=0.8, label='pre revision trueteacher scores')
    df['true_teacher_revised_summary_scores'].hist(bins=24, alpha=0.8, label='post revision trueteacher scores')
    plt.legend()

    plt.xlim((-0.1, 1.1))
    plt.title('Trueteacher factuality scores change after revision')
    plt.show()
    df['q_squared_summary_nli_scores'].hist(bins=24, alpha=0.8, label='pre revision q_squared scores')
    df['q_squared_revised_summary_nli_scores'].hist(bins=24, alpha=0.8, label='post revision q_squared scores')
    plt.legend()
    plt.xlim((-0.1, 1.1))
    plt.title('q_squared nli factuality scores change after revision')
    plt.show()
    df['q_squared_summary_f1_scores'].hist(bins=24, alpha=0.8, label='pre revision q_squared scores')
    df['q_squared_revised_summary_f1_scores'].hist(bins=24, alpha=0.8, label='post revision q_squared scores')
    plt.legend()
    plt.xlim((-0.1, 1.1))
    plt.title('q_squared f1 factuality scores change after revision')
    plt.show()


def main():
    df = pd.read_csv('data/poc/gpt_4_turbo_results.csv')
    df = df[~df['revised_summary'].isna()]
    df['true_teacher_summary_scores'] = trueteacher_scoring(summaries=df['summary'].tolist(), texts=df['text'].tolist(),
                                                            device='auto', batch_size=1)
    df['true_teacher_revised_summary_scores'] = trueteacher_scoring(summaries=df['revised_summary'].tolist(),
                                                                    texts=df['text'].tolist(), device='auto',
                                                                    batch_size=1)
    df['seahorse_xxl_summary_scores'] = seahorse_scoring(summaries=df['summary'].tolist(), texts=df['text'].tolist(),
                                                         device='auto', batch_size=1)
    df['seahorse_xxl_revised_summary_scores'] = seahorse_scoring(summaries=df['revised_summary'].tolist(),
                                                                 texts=df['text'].tolist(), device='auto',
                                                                 batch_size=1)
    rouge_scores = add_rouge(df['summary'].tolist(), df['revised_summary'].tolist())
    for key in rouge_scores.keys():
        df[key] = rouge_scores[key]
    df.to_csv('data/poc/gpt_4_turbo_results_with_scores.csv')


if __name__ == "__main__":
    main()

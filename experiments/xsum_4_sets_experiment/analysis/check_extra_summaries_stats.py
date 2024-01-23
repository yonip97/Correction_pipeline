import json
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
import torch

import numpy as np
import pandas as pd
import evaluate
from general.utils import RevisionDataset
import pickle
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset


def create_dataset(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, method,
                   train_size=0.85, seed=42, classifier_threshold=0.5, rouge_threshold=0.5, diff_threshold=0.4,
                   density_threshold=3.5):
    np.random.seed(seed)
    if method == 'all':
        train_indices = np.random.choice(len(texts), int(len(texts) * train_size), replace=False)
        val_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
    elif method == 'classifier' or method == 'classifier_and_rouge_threshold':
        properly_revised = [i for i in range(len(pre_revision_scores)) if
                            (pre_revision_scores[i] < classifier_threshold and post_revision_scores[
                                i] >= classifier_threshold)]
        if method == 'classifier_and_rouge_threshold':
            rouge_metric = evaluate.load('rouge')
            rouge_scores = \
                rouge_metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)[
                    'rougeL']
            indices_above_threshold = [i for i in range(len(rouge_scores)) if rouge_scores[i] >= rouge_threshold]
            # properly revised means both factuality increase and high rouge score
            properly_revised = list(set(properly_revised).intersection(indices_above_threshold))
        train_indices = np.random.choice(properly_revised, int(len(properly_revised) * train_size), replace=False)
        val_indices = np.array(list(set(properly_revised) - set(train_indices)))
    elif method == 'diff' or method == 'diff_and_rouge_threshold' or method == 'diff_and_density_threshold':
        properly_revised = [i for i in range(len(pre_revision_scores)) if
                            (post_revision_scores[i] - pre_revision_scores[i] >= diff_threshold)]
        if method == 'diff_and_rouge_threshold':
            rouge_metric = evaluate.load('rouge')
            rouge_scores = \
                rouge_metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)[
                    'rougeL']
            indices_above_threshold = [i for i in range(len(rouge_scores)) if rouge_scores[i] >= rouge_threshold]
            # properly revised means both factuality increase and high rouge score
            properly_revised = list(set(properly_revised).intersection(indices_above_threshold))
        elif method == 'diff_and_density_threshold':
            from general.fragments_metrics import Fragments
            fragments_metric = Fragments()
            density_scores = fragments_metric.score(metrics=['density'], texts=texts, summaries=summaries)['density']
            indices_below_threshold = [i for i in range(len(density_scores)) if density_scores[i] < density_threshold]
            properly_revised = list(set(properly_revised).intersection(indices_below_threshold))
        train_indices = np.random.choice(properly_revised, int(len(properly_revised) * train_size), replace=False)
        val_indices = np.array(list(set(properly_revised) - set(train_indices)))
    else:
        raise NotImplementedError

    train_texts, train_summaries, train_revised_summaries = np.array(texts)[train_indices].tolist(), \
                                                            np.array(summaries)[train_indices].tolist(), \
                                                            np.array(revised_summaries)[train_indices].tolist()
    train_dataset = RevisionDataset(train_texts, train_summaries, train_revised_summaries)
    print(f"Train size: {len(train_dataset)}")
    if len(val_indices) == 0:
        return train_dataset, None
    else:
        val_texts, val_summaries, val_revised_summaries = np.array(texts)[val_indices].tolist(), np.array(summaries)[
            val_indices].tolist(), np.array(revised_summaries)[val_indices].tolist()
        val_dataset = RevisionDataset(val_texts, val_summaries, val_revised_summaries)
        return train_dataset, val_dataset


def args_parser():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--train_size", type=float, default=1)
    args.add_argument("--test_save_path", type=str)
    args.add_argument("--test_save_dir", type=str, default="experiments/xsum_4_sets_experiment/outputs")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--rouge_threshold", type=float, default=0.7)
    args.add_argument("--classifier_threshold", type=float, default=0.5)
    args.add_argument("--diff_threshold", type=float, default=0.4)
    args.add_argument("--density_threshold", type=float, default=2)
    args.add_argument('--method', type=str, default='classifier_and_rouge_threshold')
    return args.parse_args()


def create_full_dataset(revised_dataset_path):
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    xsum_indices_to_text = {xsum_dataset.indices[i]: xsum_dataset[i]['text'] for i in range(len(xsum_dataset))}

    # path = 'experiments/xsum_4_sets_experiment/revision_results_seahorse_threshold.csv'
    revised_dataset = pd.read_csv(revised_dataset_path)
    revised_dataset = revised_dataset.dropna()
    revised_dataset['text'] = revised_dataset.apply(
        lambda row: xsum_indices_to_text[row['indices']], axis=1)
    return revised_dataset


def main():
    args = args_parser()
    revised_dataset_path = 'experiments/xsum_4_sets_experiment/documents_for_summarization_fully_scored_with_revised.csv'
    df = create_full_dataset(revised_dataset_path)
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    revised_summaries = df['revised_summary_4_beam'].tolist()
    pre_revision_scores = df['pre_revision_seahorse_xxl_score'].tolist()
    post_revision_scores = df['post_revision_seahorse_xxl_score_4_beam'].tolist()

    train_dataset, _ = create_dataset(texts=texts, summaries=summaries, revised_summaries=revised_summaries,
                                      pre_revision_scores=pre_revision_scores,
                                      post_revision_scores=post_revision_scores,
                                      method='diff_and_rouge_threshold', train_size=args.train_size, seed=args.seed,
                                      classifier_threshold=args.classifier_threshold,
                                      rouge_threshold=args.rouge_threshold, diff_threshold=args.diff_threshold,
                                      density_threshold=args.density_threshold)
    print(f"Train size: {len(train_dataset)}")
    texts = [train_dataset[i]['text'] for i in range(len(train_dataset))]
    revised_summaries = [train_dataset[i]['revised_summary'] for i in range(len(train_dataset))]
    from TrueTeacher.inference import TrueTeacher
    factuality_metric = TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                                    tokenizer_name='google/t5_11b_trueteacher_and_anli',
                                    device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16,
                                    return_none=True)
    scores = factuality_metric.score(summaries=revised_summaries, texts=texts)
    results = {'args':args.__dict__,'scores': scores, 'texts': texts, 'revised_summaries': revised_summaries}
    with open(os.path.join(args.test_save_dir, args.test_save_path), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    main()

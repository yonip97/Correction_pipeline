import json
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import torch

import numpy as np
import pandas as pd
import evaluate
from general.utils import RevisionDataset
import pickle


def create_dataset(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, xsum_indices,
                   chosen_indices, method,
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
    elif method == 'chosen_indices':
        properly_revised = [i for i in range(len(texts)) if xsum_indices[i] in chosen_indices]
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
    args.add_argument("--test_save_dir", type=str, default="experiments/ablations/revision_on_original/outputs")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--rouge_threshold", type=float, default=0.7)
    args.add_argument("--classifier_threshold", type=float, default=0.5)
    args.add_argument("--diff_threshold", type=float, default=0.4)
    args.add_argument("--density_threshold", type=float, default=2)
    args.add_argument('--method', type=str, default='diff_and_rouge_threshold')
    args.add_argument('--chosen_indices_path', type=str)
    return args.parse_args()


def main():
    args = args_parser()
    revised_dataset_path = "experiments/ablations/revision_on_original/data/revision_results_original_flan_large_results.csv"
    df = pd.read_csv(revised_dataset_path, index_col=0)
    texts = df['text'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['revised_summaries'].tolist()
    pre_revision_scores = df['pre_revision_score_seahorse'].tolist()
    post_revision_scores = df['post_revision_score_seahorse'].tolist()
    xsum_indices = df['indices'].tolist()
    method = args.method
    if method == 'chosen_indices':
        with open(args.chosen_indices_path, 'rb') as f:
            chosen_indices = pickle.load(f)
    train_dataset, _ =create_dataset(texts, summaries, revised_summaries, pre_revision_scores,
                                      post_revision_scores, xsum_indices, chosen_indices, method, train_size=args.train_size,
                                      seed=args.seed,
                                      rouge_threshold=args.rouge_threshold, classifier_threshold=args.classifier_threshold,
                                      diff_threshold=args.diff_threshold)
    texts = [train_dataset[i]['text'] for i in range(len(train_dataset))]
    revised_summaries = [train_dataset[i]['revised_summary'] for i in range(len(train_dataset))]
    from TrueTeacher.inference import TrueTeacher
    factuality_metric = TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                                    tokenizer_name='google/t5_11b_trueteacher_and_anli',
                                    device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16,
                                    return_none=True)
    scores = factuality_metric.score(summaries=revised_summaries, texts=texts)
    results = {'args':args.__dict__,'scores':scores,'texts':texts,'revised_summaries':revised_summaries}
    with open(os.path.join(args.test_save_dir, args.test_save_path), 'w') as file:
        json.dump(results, file)


if __name__ == '__main__':
    main()

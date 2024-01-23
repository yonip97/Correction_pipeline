from experiments.data.datasets_splits import split_xsum_dataset

import pandas as pd
import numpy as np
import evaluate
from general.utils import RevisionDataset
from experiments.scoring import score
from general.fragments_metrics import Fragments


def create_full_dataset(revised_dataset_path, args):
    revised_dataset = pd.read_csv(revised_dataset_path)
    revised_dataset = revised_dataset.dropna()
    revision_num = args.num_of_documents_for_revision
    summarization_num = args.num_of_documents_for_summarization
    if 'text' not in revised_dataset.columns:
        xsum_dataset = split_xsum_dataset(split='summarization_documents',
                                          path_to_documents_for_summarization_indices=f"experiments/data/datasets_splits/xsum_summarization_{summarization_num}_revsion_{revision_num}_seed_42.pkl",
                                          num_of_documents_for_summarization=summarization_num,
                                          num_of_documents_for_revision=revision_num,
                                          seed=42)
        xsum_indices_to_text = {xsum_dataset.indices[i]: xsum_dataset[i]['text'] for i in range(len(xsum_dataset))}
        revised_dataset['text'] = revised_dataset.apply(
            lambda row: xsum_indices_to_text[row['indices']], axis=1)

    if 'pre_revision_factuality_score' not in revised_dataset.columns:
        pre_revision_scores = \
            score(texts=revised_dataset['text'].tolist(), summaries=revised_dataset['model_summary'].tolist(),
                  metrics=['seahorse'])[
                'seahorse']
        revised_dataset['pre_revision_factuality_score'] = pre_revision_scores
        revised_dataset.to_csv(revised_dataset_path)
    if 'post_revision_factuality_score' not in revised_dataset.columns:
        post_revision_scores = \
            score(texts=revised_dataset['text'].tolist(), summaries=revised_dataset['revised_summary'].tolist(),
                  metrics=['seahorse'])[
                'seahorse']
        revised_dataset['post_revision_factuality_score'] = post_revision_scores
        revised_dataset.to_csv(revised_dataset_path)
    if 'pre_revision_density' not in revised_dataset.columns:
        metric = Fragments()
        pre_revision_density = metric.score(texts=revised_dataset['text'].tolist(),
                                            summaries=revised_dataset['model_summary'].tolist(),
                                            metrics=['density'])['density']
        revised_dataset['pre_revision_density'] = pre_revision_density
        revised_dataset.to_csv(revised_dataset_path)
    if 'post_revision_density' not in revised_dataset.columns:
        metric = Fragments()
        post_revision_density = metric.score(texts=revised_dataset['text'].tolist(),
                                             summaries=revised_dataset['revised_summary'].tolist(),
                                             metrics=['density'])['density']
        revised_dataset['post_revision_density'] = post_revision_density
        revised_dataset.to_csv(revised_dataset_path)
    return revised_dataset['text'].tolist(), revised_dataset['model_summary'].tolist(), revised_dataset[
        'revised_summary'].tolist(), revised_dataset[
               'pre_revision_factuality_score'].tolist(), revised_dataset['post_revision_factuality_score'].tolist(), \
           revised_dataset[
               'pre_revision_density'].tolist(), revised_dataset['post_revision_density'].tolist()


def create_dataset(df, method, args):
    if 'classifier' in method:
        df = df[
            (df['pre_revision_factuality_score'] < args.factuality_threshold) & (
                    df['post_revision_factuality_score'] >= args.factuality_threshold)]
    if 'rouge_threshold' in method:
        rouge_metric = evaluate.load('rouge')
        df['rougeL'] = \
            rouge_metric.compute(predictions=df['revised_summary'], references=df['model_summary'],
                                 use_aggregator=False)['rougeL']
        df = df[df['rougeL'] > args.rouge_threshold]
    if 'factuality_diff' in method:
        df = df[
            df['post_revision_factuality_score'] - df['pre_revision_factuality_score'] >=
            args.factuality_diff_threshold]
    if 'density_threshold' in method:
        df = df[df['post_revision_density'] >= args.density_threshold]
    if 'density_diff' in method:
        df = df[df['post_revision_density'] - df['pre_revision_density'] >= args.density_diff_threshold]
    train_dataset = RevisionDataset(df['text'].tolist(), df['model_summary'].tolist(), df['revised_summary'].tolist())
    print(f"Train size: {len(train_dataset)}")
    return train_dataset

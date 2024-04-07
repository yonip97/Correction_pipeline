import os
import sys
import time

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from experiments.scoring import score
from argparse import ArgumentParser
import pandas as pd
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
import evaluate
from experiments.data.datasets_splits import split_xsum_dataset


def parse():
    parser = ArgumentParser()
    parser.add_argument('-output_path')
    parser.add_argument('-revision_data_file', type=str)
    parser.add_argument('-revision_data_dir', type=str)
    parser.add_argument('-should_split_output', action='store_true')
    parser.add_argument('-delimiter', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse()
    df = pd.read_csv(os.path.join(args.revision_data_dir, args.revision_data_file) + '.csv', index_col=0)
    df = df[~df['revised_summary'].isna()]
    dataset = split_xsum_dataset(split='revision_documents',
                                 path_to_documents_for_summarization_indices=
                                 "experiments/data/datasets_splits/xsum_summarization_0_revision_50000_seed_42.json",
                                 num_of_documents_for_summarization=0, num_of_documents_for_revision=50000, seed=42)
    indices_to_summaries = {dataset.indices[i]: dataset[i]['summary'] for i in range(len(dataset))}
    df['original_summary'] = [indices_to_summaries[i] for i in df['indices']]
    if args.should_split_output:
        df['revised_summary_full_text'] = df['revised_summary']
        df['revised_summary'] = df['revised_summary'].apply(lambda x: x.split(args.delimiter)[-1].strip())


    print(f"Scoring {len(df)} examples")
    texts = [str(text) for text in df['text'].tolist()]
    summaries = [str(x) for x in df['model_summary'].tolist()]
    revised_summaries = [str(x) for x in df['revised_summary'].tolist()]
    scores = score(texts, revised_summaries, metrics=['trueteacher', 'seahorse'])
    df['post_revision_factuality_score'] = scores['seahorse']
    df['post_revision_trueteacher'] = scores['trueteacher']
    scores = score(texts, summaries, metrics=['trueteacher'])
    df['pre_revision_trueteacher'] = scores['trueteacher']
    metric = Fragments()
    scores = metric.score(metrics=['density', 'coverage'], texts=texts, summaries=revised_summaries)
    df['post_revision_density'] = scores['density']
    df['post_revision_coverage'] = scores['coverage']
    df['post_revision_length'] = [len(word_tokenize(summary)) for summary in revised_summaries]
    scores = metric.score(metrics=['density', 'coverage'], texts=texts, summaries=summaries)
    df['pre_revision_density'] = scores['density']
    df['pre_revision_coverage'] = scores['coverage']
    df['pre_revision_length'] = [len(word_tokenize(summary)) for summary in summaries]
    metric = evaluate.load('rouge')
    scores = metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)
    df['rougeL_revised_to_base'] = scores['rougeL']
    scores = metric.compute(predictions=revised_summaries, references=df['original_summary'].tolist(), use_aggregator=False)
    df['rougeL_revised_to_original'] = scores['rougeL']
    scores = metric.compute(predictions=summaries, references=df['original_summary'].tolist(), use_aggregator=False)
    df['rougeL_base_to_original'] = scores['rougeL']
    df.to_csv(os.path.join(args.revision_data_dir, args.output_path) + '.csv')


if __name__ == "__main__":
    main()

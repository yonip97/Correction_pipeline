import os
import sys
import time

os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))

from experiments.scoring import score
from argparse import ArgumentParser
import pandas as pd
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
import evaluate


def parse():
    parser = ArgumentParser()
    parser.add_argument('-output_path')
    parser.add_argument('-revision_data_file', type=str)
    parser.add_argument('-revision_data_dir', type=str)
    parser.add_argument('-should_split_output', action='store_true')
    parser.add_argument('-delimiter', type=str)
    parser.add_argument('-llama', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse()
    df = pd.read_csv(os.path.join(args.revision_data_dir, args.revision_data_file) + '.csv', index_col=0)
    df.rename(columns={'revised_summary_full_text': 'revised_summary'}, inplace=True)
    df = df[~df['revised_summary'].isna()]
    if args.llama:
        df['revised_summary'] = df['revised_summary'].apply(lambda x: x.split('assistant\n')[-1].strip())
    if args.should_split_output:
        df['revised_summary_full_text'] = df['revised_summary']
        revised_summaries = df['revised_summary'].tolist()
        for i in range(len(revised_summaries)):
            if args.delimiter in revised_summaries[i]:
                revised_summaries[i] = revised_summaries[i].split(args.delimiter)[-1].strip()
            else:
                revised_summaries[i] = df['model_summary'].tolist()[i]
        df['revised_summary'] = revised_summaries
    print(f"Scoring {len(df)} examples")
    texts = [str(text) for text in df['text'].tolist()]
    summaries = [str(x) for x in df['model_summary'].tolist()]
    revised_summaries = [str(x) for x in df['revised_summary'].tolist()]
    scores = score(texts, revised_summaries, metrics=['trueteacher', 'seahorse'])
    df['revised_summary_seahorse'] = scores['seahorse']
    df['revised_summary_trueteacher'] = scores['trueteacher']
    metric = Fragments()
    scores = metric.score(metrics=['density', 'coverage'], texts=texts, summaries=revised_summaries)
    df['revised_summary_density'] = scores['density']
    df['revised_summary_coverage'] = scores['coverage']
    df['revised_summary_length'] = [len(word_tokenize(summary)) for summary in revised_summaries]
    scores = metric.score(metrics=['density', 'coverage'], texts=texts, summaries=summaries)
    df['model_summary_density'] = scores['density']
    df['model_summary_coverage'] = scores['coverage']
    df['model_summary_length'] = [len(word_tokenize(summary)) for summary in summaries]
    metric = evaluate.load('rouge')
    scores = metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)
    df['rougeL_revised_to_base'] = scores['rougeL']
    # scores = metric.compute(predictions=revised_summaries, references=df['original_summary'].tolist(), use_aggregator=False)
    # df['rougeL_revised_to_original'] = scores['rougeL']
    df.to_csv(os.path.join(args.revision_data_dir, args.output_path+'_temp') + '.csv')


if __name__ == "__main__":
    main()

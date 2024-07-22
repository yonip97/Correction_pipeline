import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import pandas as pd
from nltk.tokenize import word_tokenize
from general.LLMS import SummarizationModel
import argparse

from experiments.scoring import score
from general.fragments_metrics import Fragments
import evaluate


def print_revision_stats():
    df = pd.read_csv(
        "experiments/ablations/better_summaries/data/base_model_outputs_below_0.5_text_length_above_65_100_samples_prompt_checking.csv",
        index_col=0)
    print(df.columns)
    to_match_df = pd.read_csv(
        "experiments/ablations/better_summaries/data/base_model_outputs_below_0.5_text_length_above_65_100_samples_post_revision.csv",
        index_col=0)
    cols = to_match_df.columns
    pre_rel_cols = sorted([col for col in cols if "pre" in col])
    post_rel_cols = sorted([col for col in cols if "post" in col])
    pre_to_match = to_match_df[pre_rel_cols].rename(
        columns={col: col.replace("pre_revision_", '') for col in pre_rel_cols})
    post_to_match = to_match_df[post_rel_cols].rename(
        columns={col: col.replace("post_revision_", '') for col in post_rel_cols})
    x = pd.concat([pre_to_match.mean(), post_to_match.mean()], axis=1)
    print(x)


def parseargs_llms():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--prompt_dir", type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--API_KEY", type=str, default=None)
    parser.add_argument("--azure", action='store_true')
    parser.add_argument("--max_generation_length", type=int, default=1000)
    parser.add_argument("--save_name", type=str)
    args = parser.parse_args()
    args.prompt = open(args.prompt_dir + '/prompt.txt', 'r').read().strip()
    return args


def llm_summarization(args, texts):
    summarization_model = SummarizationModel(
        temp_save_dir='experiments/ablations/better_summaries/data/contingency_tables', prompt=args.prompt,
        model=args.model, API_KEY=args.API_KEY, azure=args.azure)
    summaries, errors = summarization_model.summarize(texts, max_generation_length=args.max_generation_length)
    return summaries, errors


def score_prompt_summaries(df):
    texts = df['text'].tolist()
    summaries = df['model_summary_llm'].tolist()
    scores = score(texts=texts, summaries=summaries, metrics=['trueteacher', 'seahorse'])
    df['model_summary_llm_trueteacher'] = scores['trueteacher']
    df['model_summary_llm_seahorse'] = scores['seahorse']
    fragments_metric = Fragments()
    scores = fragments_metric.score(metrics=['coverage', 'density'], summaries=summaries, texts=texts)
    df['model_summary_llm_density'] = scores['density']
    df['model_summary_llm_coverage'] = scores['coverage']
    rouge_metric = evaluate.load('rouge')
    scores = rouge_metric.compute(predictions=summaries, references=df['original_summary'].tolist(),
                                  use_aggregator=False)
    df['rougeL_llm_to_original'] = scores['rougeL']
    scores = rouge_metric.compute(predictions=summaries, references=df['model_summary'].tolist(), use_aggregator=False)
    df['rougeL_llm_to_base'] = scores['rougeL']
    df['model_summary_llm_length'] = [len(word_tokenize(summary)) for summary in summaries]
    return df


def main():
    args = parseargs_llms()
    df = pd.read_csv(args.data_path+'.csv', index_col=0)
    texts = df['text'].tolist()
    summaries, errors = llm_summarization(args, texts)
    df['model_summary_llm'] = summaries
    df['error'] = errors
    df.to_csv(os.path.join(args.prompt_dir, args.save_name + '_raw.csv'))
    df = df[df['model_summary_llm'].notna()]
    df = score_prompt_summaries(df)
    df.to_csv(os.path.join(args.prompt_dir, args.save_name + '_scored.csv'))


if __name__ == '__main__':
    main()

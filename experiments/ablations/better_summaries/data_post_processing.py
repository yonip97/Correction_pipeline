import pandas as pd
import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('/')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('/')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('/')
from experiments.scoring import score
from general.fragments_metrics import Fragments
import evaluate
from nltk.tokenize import word_tokenize


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
    df['model_summary_llm_length'] = [len(word_tokenize(summary)) for summary in summaries]
    return df


def post_process(df):
    need_correction = df['model_summary_llm'].str.contains("Summary:")
    fixed = score_prompt_summaries(df[need_correction])
    df.loc[need_correction, fixed.columns] = fixed
    return df


def main():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/experiments/ablations/better_summaries/data/prompts/4/base_model_outputs_below_0.5_text_length_above_65_1000_samples_summrized_by_chatgpt_scored.csv",
        index_col=0)
    df = post_process(df)
    df.to_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/experiments/ablations/better_summaries/data/prompts/4/base_model_outputs_below_0.5_text_length_above_65_1000_samples_summrized_by_chatgpt_scored_post_processed.csv")
    print(df.columns)


if __name__ == '__main__':
    main()
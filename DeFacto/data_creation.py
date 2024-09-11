import os
import sys

os.chdir('../')
sys.path.append(os.getcwd())

import pandas as pd
import json
from general.fragments_metrics import Fragments
from experiments.scoring import score
from collections import Counter

def create_and_score_datasets():
    for name in ['test']:
        original_summaries = []
        revised_summaries = []
        instructions = []
        explanations = []
        texts = []
        errors = []
        intrinsics = []
        extrinsics = []
        with open(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}.jsonl") as f:
            data = [json.loads(line) for line in f]
            for x in data:
                errors.append(x['has_error'])
                intrinsics.append(x['intrinsic_error'])
                extrinsics.append(x['extrinsic_error'])
                if x['has_error'] == True:
                    texts.append(str(x['article']))
                    revised_summaries.append(str(x['feedback']['summary']))
                    original_summaries.append(str(x['candidate']))
                    instructions.append(str(x['feedback']['instruction']))
                    explanations.append(str(x['feedback']['explanation']))
                else:
                    texts.append(str(x['article']))
                    revised_summaries.append(str(x['candidate']))
                    original_summaries.append(str(x['candidate']))
                    instructions.append('')
                    explanations.append('')
            df = pd.DataFrame(
                {'text': texts, 'model_summary': original_summaries, 'revised_summary': revised_summaries,
                 'error_in_model_summary': errors, 'intrinsic_error': intrinsics, 'extrinsic_error': extrinsics,
                 'instruction': instructions, 'explanation': explanations})
            from general.fragments_metrics import Fragments
            fragments = Fragments()
            original_summaries_metrics = fragments.score(metrics=['density', 'coverage'], texts=texts,
                                                         summaries=original_summaries)
            df['model_summary_density'] = original_summaries_metrics['density']
            df['model_summary_coverage'] = original_summaries_metrics['coverage']
            revised_summaries_metrics = fragments.score(metrics=['density', 'coverage'], texts=texts,
                                                        summaries=revised_summaries)
            df['revised_summary_density'] = revised_summaries_metrics['density']
            df['revised_summary_coverage'] = revised_summaries_metrics['coverage']
            revised_summaries_metrics = score(texts=texts, summaries=revised_summaries,
                                              metrics=['seahorse', 'trueteacher'])
            df['revised_summary_seahorse'] = revised_summaries_metrics['seahorse']
            df['revised_summary_trueteacher'] = revised_summaries_metrics['trueteacher']
            original_summaries_metrics = score(texts=texts, summaries=original_summaries,
                                               metrics=['seahorse', 'trueteacher'])
            df['model_summary_seahorse'] = original_summaries_metrics['seahorse']
            df['model_summary_trueteacher'] = original_summaries_metrics['trueteacher']
            import evaluate
            rouge_metric = evaluate.load('rouge')
            scores = rouge_metric.compute(predictions=revised_summaries, references=original_summaries,
                                          use_aggregator=False)
            df['revised_summary_rougeL_to_base'] = scores['rougeL']
            from nltk.tokenize import word_tokenize
            df['model_summary_length'] = [len(word_tokenize(x)) for x in df['model_summary']]
            df['revised_summary_length'] = [len(word_tokenize(x)) for x in df['revised_summary']]
            df.to_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}_scores.csv")


def create_subsets():
    dfs = []
    for name in ['train', 'val', 'test']:
        df = pd.read_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}_scores.csv", index_col=0)
        df['data_split'] = name
        dfs.append(df)
    df = pd.concat(dfs)
    df_with_errors = df[df['error_in_model_summary'] == True]
    df_without_errors = df[df['error_in_model_summary'] == False]
    df.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/all_data.csv")
    df_with_errors.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/summaries_with_errors.csv")
    df_without_errors.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/summaries_without_errors.csv")
    for x in [4,30, 50, 100]:
        df_with_errors_sample = df_with_errors.iloc[:x]
        df_with_errors_sample.to_csv(
            f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/summaries_with_errors_{x}.csv")

def merge_train_and_val_with_errors_for_training():
    train = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/train_with_errors_for_training.csv", index_col=0)
    val = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/val_with_errors_for_training.csv", index_col=0)
    train_val = pd.concat([train, val])
    train_val.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/train_val_with_errors_for_training.csv")
def each_set_for_training():
    for name in ['train', 'val', 'test']:
        df = pd.read_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}_scores.csv", index_col=0)
        df_with_errors = df[df['error_in_model_summary'] == True]
        rel_cols = ['text', 'model_summary', 'instruction', 'explanation', 'revised_summary']
        df_with_errors = df_with_errors[rel_cols]
        if name =='test':
            df_with_errors.rename(columns={'revised_summary': 'gt_revised_summary'}, inplace=True)
        df_with_errors.to_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}_with_errors_for_training.csv")
def main():
    # create_and_score_datasets()
    create_subsets()
    # each_set_for_training()
    # merge_train_and_val_with_errors_for_training()

if __name__ == "__main__":
    main()

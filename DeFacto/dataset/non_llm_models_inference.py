import os
import sys

os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
from TrueTeacher.inference import TrueTeacher
from Seahorse_metrics.metrics import Seahorse_metrics
import torch
import pandas as pd
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-data_path')
    args.add_argument('-output_path')
    args.add_argument('-model', type=str)
    args.add_argument('-batch_size', type=int, default=1)
    args.add_argument('-max_length', type=int, default=2048)
    args = args.parse_args()
    return args


def chose_model(args):
    if args.model == 'TrueTeacher':
        return TrueTeacher(model_path="google/t5_11b_trueteacher_and_anli",
                           tokenizer_name='google/t5_11b_trueteacher_and_anli', device='auto',
                           batch_size=args.batch_size,
                           max_length=args.max_length,
                           torch_dtype=torch.bfloat16, return_none=False)
    elif args.model == 'Seahorse':
        return Seahorse_metrics(model_path="google/seahorse-xxl-q4", tokenizer_name="google/seahorse-xxl-q4",
                                device='auto'
                                , batch_size=args.batch_size, max_length=args.max_length,
                                torch_dtype=torch.bfloat16, return_none=False)
    else:
        raise ValueError(f'Unknown model: {args.model}')


def get_data(args):
    df = pd.read_csv(args.data_path)
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    descriptions = df['descriptions'].tolist()
    descriptions = [eval(x) for x in descriptions]
    binary = [0 if len(x) > 0 else 1 for x in descriptions]
    return binary, texts, summaries


def score_summaries():
    args = parse_args()
    model = chose_model(args)
    binary, texts, summaries = get_data(args)
    scores = model.score(texts, summaries)
    df = pd.DataFrame({'text': texts, 'model_summary': summaries, 'gold_label': binary, 'scores': scores})
    df.to_csv(args.output_path)
def check_with_prompt():
    #Not final annotation, i want t see if taking the orcal annotation or model annotation is better,
    df =pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/summary_classification/fine_tuned_models/results_trueteacher.csv")
    df2 = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/results/claude-3-5-sonnet-20241022/cot_1_maybe_consistent_gpt-4o-2024-11-20/test_not_chosen/results.csv")
    df['label'] = df['score'].apply(lambda x: 1 if x > 0.5 else 0)
    final = df2.merge(df, on=['text', 'model_summary'], how='inner')
    final['precision'].fillna('{}', inplace=True)
    final['recall'].fillna('{}', inplace=True)
    prection_dicts = final['precision'].tolist()
    prection_dicts = [eval(x) for x in prection_dicts]
    recall_dicts = final['recall'].tolist()
    recall_dicts = [eval(x) for x in recall_dicts]
    #This is not orcal, need to remove all consistent
    final_prec = sum([sum(x.values()) for x in prection_dicts])/sum([len(x) for x in prection_dicts])
    final_recall = sum([sum(x.values()) for x in recall_dicts])/sum([len(x) for x in recall_dicts])
    print(f"Final precision: {final_prec}")
    print(f"Final recall: {final_recall}")
    filtered_precition_dicts = []


def main():
    score_summaries()
    #check_with_prompt()


if __name__ == '__main__':
    main()

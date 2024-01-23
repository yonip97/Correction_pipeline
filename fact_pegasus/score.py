import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

import time
import torch
import gc
from Seahorse_metrics.metrics import Seahorse_metrics
from TrueTeacher.inference import TrueTeacher
import pandas as pd
import evaluate
from general.fragments_metrics import Fragments
import numpy as np


def score(texts, summaries, metrics):
    results = {}
    for metric in metrics:
        if 'seahorse' in metrics:
            factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                                 tokenizer_name='google/seahorse-xxl-q4',
                                                 device='auto', batch_size=1, torch_dtype=torch.float16,
                                                 max_length=2048, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['seahorse'] = scores
            del factuality_metric
            gc.collect()
        elif 'teacher' in metric:
            factuality_metric = TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                                            tokenizer_name="google/t5_11b_trueteacher_and_anli",
                                            device='auto', batch_size=1, max_length=2048,
                                            torch_dtype=torch.float16, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['trueteacher'] = scores
            del factuality_metric
            gc.collect()
        elif 'nli' in metric:
            from nli.nli_metric import NLI
            factuality_metric = NLI(batch_size=1, torch_dtype=torch.bfloat16, max_length=2048, device='auto',
                                    return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['nli'] = scores
            del factuality_metric
            gc.collect()
        elif 'q_squared' in metric:
            from q_squared.run_nli import scores_with_nli, aggregate_per_response
            from q_squared.prep_sys_experiment import cross_annotated_scores
            for_system = 'fact_pegasus/outputs/fact_pegasus_logger.txt'
            df = cross_annotated_scores(texts, summaries, out_path=None, save=False, for_system=for_system)
            df = scores_with_nli(in_path=None, df=df)
            df = aggregate_per_response(df=df, out_path=None, save=False)
            results['Q2'] = df['Q2'].tolist()
            # if 'nli' in metric:
            #     factuality_metric = Q_squared_classifier(device_generation='cuda:0',device_answering='cuda:1', similarity_metric='nli', threshold=0.5, remove_personal=True)
            # else:
            #     factuality_metric = Q_squared_classifier(device_generation='cuda:0',device_answering='cuda:1', similarity_metric='f1', threshold=0.5, remove_personal=True)
            # scores = factuality_metric.score(texts=texts, summaries=summaries)
            # if 'nli' in metric:
            #     results['q_squared_nli'] = scores
            # else:
            #     results['q_squared_f1'] = scores

        time.sleep(30)
        torch.cuda.empty_cache()
    return results


def main():
    with open('fact_pegasus/data/generated_predictions.txt', 'r') as file:
        fact_pegasus_summaries = [line.strip() for line in file.readlines()]
    with open('fact_pegasus/data/xsum_test_references.txt', 'r') as file:
        summaries = [line.strip() for line in file.readlines()]
    with open('fact_pegasus/data/xsum_test_documents.txt', 'r') as file:
        texts = [line.strip() for line in file.readlines()]
    results = {}
    results['text'] = texts
    results['summary'] = fact_pegasus_summaries
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(predictions=fact_pegasus_summaries, references=summaries, use_aggregator=False)
    for res in rouge_results.keys():
        results[res] = rouge_results[res]
    fragments_metric = Fragments()
    fragments_results = fragments_metric.score(metrics=['density', 'coverage'], texts=texts,
                                               summaries=fact_pegasus_summaries)
    for res in fragments_results.keys():
        results[res] = fragments_results[res]
    scoring_results = score(texts=texts, summaries=fact_pegasus_summaries, metrics=['trueteacher','q_squared'])
    for res in scoring_results.keys():
        results[res] = scoring_results[res]
    df = pd.DataFrame(data=results)
    df.to_csv('fact_pegasus/outputs/fact_pegasus_results.csv')


if __name__ == '__main__':
    print("scoring fact pegasus")
    main()

import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')


import pandas as pd

from data.cost_etimation import Cost_estimator
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset


# need to summarize and un TrueTeacher to check exacly how many to summarize

def estimate():
    print(os.getcwd())
    # checkpoint_path ="/data/home/yehonatan-pe/Correction_pipeline/experiments/xsum_4_sets_experiment/checkpoints/t5_base_both_10_12_2023_08_54_06/checkpoint-115000"
    # model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    # tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    estimator = Cost_estimator('gpt-3.5-turbo', 0.001, 0.002)
    path_to_documents_for_summarization_indices_xsum = "/data/home/yehonatan-pe/Correction_pipeline/experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl"
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_xsum,
                                      num_of_documents_for_summarization=20000, seed=42)
    prompt = """generate a concise and abstractive one-sentence summary that captures the key information, main ideas, and nuances of the provided text. limit yourself to 20 words."""
    # for i in range(len(dataset_xsum)):
    #     print(i)
    #     text = dataset_xsum[i]['text']
    #     summary = dataset_xsum[i]['summary']
    #     texts.append(text)
    #     summaries.append(summary)
    texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    summaries = [xsum_dataset[i]['summary'] for i in range(len(xsum_dataset))]
    estimation = estimator.estimate_for_summarization(prompt=prompt, texts=texts, summaries=summaries)

    print("xsum estimation", estimation)
estimate()
import sys
import os

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
import evaluate
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from general.utils import iter_list

def generate_summaries():
    device = 'cuda:1'
    config = AutoConfig.from_pretrained(
        '/data/home/yehonatan-pe/Correction_pipeline/fact_pegasus/factpegasus_xsum_comb/config.json')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        '/data/home/yehonatan-pe/Correction_pipeline/fact_pegasus/factpegasus_xsum_comb', config=config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        '/data/home/yehonatan-pe/Correction_pipeline/fact_pegasus/factpegasus_xsum_comb', use_fast=True)
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    predictions = []
    with torch.no_grad():
        for batch in tqdm(iter_list(texts, 64)):
            batch = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            batch_outputs = model.generate(**batch, num_beams=6, early_stopping=True)
            batch_predictions = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            predictions += batch_predictions
    with open('fact_pegasus/generated_predictions_mine.txt', 'w') as file:
        file.writelines([pred + '\n' for pred in predictions])
def add_length(df):
    from nltk.tokenize import word_tokenize
    df['length'] =[len(word_tokenize(summary)) for summary in df['summary']]
    return df
def main():
    df = pd.read_csv('fact_pegasus/outputs/fact_pegasus_results.csv',index_col=0)
    df = add_length(df)
    for col in df.columns:
        if 'text' in col or 'summary' in col:
            continue
        else:
            temp_df = df[df[col].notnull()]
            print(f"{col} mean: ",np.mean(temp_df[col].tolist()))
            print(f"{col} median: ",np.median(temp_df[col].tolist()))



if __name__ == "__main__":
    main()
import os
import sys

import evaluate

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from experiments.scoring import score
from experiments.data.datasets_splits import split_xsum_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from general.t5_trainer import t5_summarize
import pandas as pd
import gc
import torch
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize


def main():
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices=
                                      "experiments/data/datasets_splits/xsum_summarization_20000_revision_10000_seed_42.json",
                                      num_of_documents_for_summarization=20000, num_of_documents_for_revision=10000,
                                      seed=42)
    dir = "/data/home/yehonatan-pe/Correction_pipeline/experiments/baseline_model/checkpoints/t5-base_xsum_22_01_2024_20_04_25"
    model_checkpoints = os.listdir(dir)
    model_checkpoints = [checkpoint for checkpoint in model_checkpoints if checkpoint.startswith('checkpoint')]
    texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    original_summaries = [xsum_dataset[i]['summary'] for i in range(len(xsum_dataset))]
    device = 'cuda:0'
    model_checkpoints = sorted(model_checkpoints,key=lambda x: int(x.split('-')[-1]))
    for checkpoint in model_checkpoints:

        if os.path.exists(os.path.join(dir, checkpoint, "model_outputs.csv")):
            print("skipping",checkpoint)
            continue
        else:
            print("processing",checkpoint)
        model = T5ForConditionalGeneration.from_pretrained(os.path.join(dir, checkpoint)).to(device)
        tokenizer = T5Tokenizer.from_pretrained(os.path.join(dir, checkpoint))
        summaries = t5_summarize(texts=texts, model=model, tokenizer=tokenizer,
                                 prompt="summarize: ",
                                 device=device, max_generation_length=128,
                                 batch_size=16,
                                 beam_size=4,
                                 early_stopping=True,
                                 length_penalty=0.6,max_encoding_length=2048)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        df = pd.DataFrame.from_dict({'text': texts, 'model_summary': summaries})
        rouge_metric = evaluate.load('rouge')
        rouge_scores = rouge_metric.compute(predictions=summaries, references=original_summaries, use_aggregator=False)
        for rouge in rouge_scores:
            df[rouge] = rouge_scores[rouge]
        fragments = Fragments()
        fragments_scores = fragments.score(texts=texts, summaries=summaries, metrics=['density', 'coverage'])
        df['density'] = fragments_scores['density']
        df['coverage'] = fragments_scores['coverage']
        df['length'] = [len(word_tokenize(summary)) for summary in summaries]
        df.to_csv(os.path.join(dir, checkpoint, "model_outputs.csv"))
        scores = score(texts=texts, summaries=summaries, metrics=['trueteacher'])
        df['trueteacher'] = scores['trueteacher']
        df.to_csv(os.path.join(dir, checkpoint, "model_outputs.csv"))
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

import os
import sys

import evaluate

os.chdir("/data/home/yehonatan-pe/Correction_pipeline")
sys.path.append(os.getcwd())
from experiments.scoring import score
from experiments.data.datasets_splits import split_xsum_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from general.t5_trainer import t5_summarize_mp_main, t5_summarize
import pandas as pd
import gc
import torch
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
import argparse


def scoring_args():
    args = argparse.ArgumentParser()
    args.add_argument("--model_dir", type=str, required=True)
    args.add_argument("--summarization_device", type=str, default='cuda:0')
    args.add_argument("--batch_size", type=int, default=6)
    args.add_argument("--max_encoding_length", type=int, default=2048)
    args.add_argument("--max_generation_length", type=int, default=128)
    args.add_argument("--beam_size", type=int, default=4)
    args.add_argument("--early_stopping", action='store_true')
    args.add_argument("--length_penalty", type=float, default=0.6)
    args.add_argument('--summarize', action='store_true')
    args.add_argument('--seahorse', action='store_true')
    args.add_argument('--trueteacher', action='store_true')
    return args.parse_args()


def main():
    torch.multiprocessing.set_start_method('spawn')
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices=
                                      "experiments/data/datasets_splits/xsum_summarization_0_revision_50000_seed_42.json",
                                      num_of_documents_for_summarization=None, num_of_documents_for_revision=None,
                                      seed=None)
    args = scoring_args()
    checkpoint = args.model_dir
    texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    original_summaries = [xsum_dataset[i]['summary'] for i in range(len(xsum_dataset))]

    summarization_device = args.summarization_device
    if args.summarize:
        if os.path.exists(os.path.join(checkpoint, "model_outputs.csv")):
            print("skipping", checkpoint)
        else:
            print("processing", checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(
                summarization_device)

            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            # summaries = t5_summarize(texts=texts, model=model, tokenizer=tokenizer, device='cuda:0',
            #              prompt="summarize: ",
            #              max_generation_length=args.max_generation_length,
            #              batch_size=args.batch_size,
            #              beam_size=args.beam_size,
            #              early_stopping=args.early_stopping,
            #              length_penalty=args.length_penalty,
            #              max_encoding_length=args.max_encoding_length,
            #              )
            summaries = t5_summarize_mp_main(texts=texts, model=model, tokenizer=tokenizer,
                                             out_dir=checkpoint,
                                             prompt="summarize: ",
                                             max_generation_length=args.max_generation_length,
                                             batch_size=args.batch_size,
                                             beam_size=args.beam_size,
                                             early_stopping=args.early_stopping,
                                             length_penalty=args.length_penalty,
                                             max_encoding_length=args.max_encoding_length,
                                             )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            df = pd.DataFrame.from_dict({'text': texts, 'model_summary': summaries})
            rouge_metric = evaluate.load('rouge')
            rouge_scores = rouge_metric.compute(predictions=summaries, references=original_summaries,
                                                use_aggregator=False)
            for rouge in rouge_scores:
                df['model_summary_' + rouge] = rouge_scores[rouge]
            fragments = Fragments()
            fragments_scores = fragments.score(texts=texts, summaries=summaries, metrics=['density', 'coverage'])
            df['model_summary_density'] = fragments_scores['density']
            df['model_summary_coverage'] = fragments_scores['coverage']
            df['model_summary_length'] = [len(word_tokenize(summary)) for summary in summaries]
            df.to_csv(os.path.join(checkpoint, "model_outputs.csv"))
    if os.path.exists(os.path.join(checkpoint, "model_outputs.csv")):
        df = pd.read_csv(os.path.join(checkpoint, "model_outputs.csv"))
        if args.trueteacher:
            print("scoring using trueteacher", checkpoint)
            summaries = df['model_summary'].tolist()
            scores = score(texts=texts, summaries=summaries, metrics=['trueteacher'])
            df['model_summary_trueteacher'] = scores['trueteacher']
        if args.seahorse:
            print("scoring using seahorse", checkpoint)
            summaries = df['model_summary'].tolist()
            scores = score(texts=texts, summaries=summaries, metrics=['seahorse'])
            df['model_summary_seahorse'] = scores['seahorse']
        df.to_csv(os.path.join(checkpoint, "model_outputs.csv"))


if __name__ == '__main__':
    main()

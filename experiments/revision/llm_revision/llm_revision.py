import gc
import os
import sys

import numpy as np

os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
import json
from datetime import datetime
import argparse
import os
import pandas as pd
from tqdm import tqdm
from general.revision_pipeline import chose_revision_model
from Seahorse_metrics.metrics import Seahorse_metrics
import torch
from experiments.scoring import score
from experiments.data.datasets_splits import split_xsum_dataset
from general.t5_trainer import t5_summarize
from transformers import T5ForConditionalGeneration, T5Tokenizer


def parseargs_llms():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_path')
    parser.add_argument('-revision_model_name', type=str, default='mock')
    parser.add_argument('-revision_prompt_path', type=str)

    parser.add_argument('-API_KEY_revision_model', type=str, default=None)
    parser.add_argument('-temp_dir_path', type=str, default='contingency_tables')
    parser.add_argument('-revision_summaries', type=int, default=10000)
    parser.add_argument('-summarization_documents', type=int, default=20000)
    parser.add_argument('-base_models_dir', type=str, default="experiments/baseline_model/checkpoints")
    parser.add_argument('-max_generation_length', type=int, default=128)
    parser.add_argument('-num_beams', type=int, default=4)
    parser.add_argument('-length_penalty', type=float, default=0.6)
    parser.add_argument('-base_model_checkpoint', type=str)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-revision_data_dir', type=str, default="experiments/revision/data")
    parser.add_argument('-revision_data_file', type=str)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-azure', action='store_true')
    parser.add_argument('-revision_max_length', type=int, default=1000)
    parser.add_argument('-seahorse_threshold', type=float, default=0.5)
    args = parser.parse_args()
    with open(args.revision_prompt_path, 'r') as file:
        args.revision_prompt = file.read()
        args.revision_prompt = args.revision_prompt.strip()
    args.temp_dir_path = os.path.join(args.revision_data_dir, args.temp_dir_path)
    if not os.path.exists(args.temp_dir_path):
        os.makedirs(args.temp_dir_path)
    model_prices = {'gpt-4-turbo': {'input': 0.01, 'output': 0.03}}
    args.input_price = model_prices[args.revision_model_name]['input']
    args.output_price = model_prices[args.revision_model_name]['output']
    return args


def get_data(args):
    if not os.path.exists(args.revision_data_dir):
        os.makedirs(args.revision_data_dir)
    if not os.path.exists(os.path.join(args.revision_data_dir, args.revision_data_file)):
        with open(os.path.join(args.revision_data_dir, "args.json"), 'w') as file:
            json.dump(args.__dict__, file)
        dataset = split_xsum_dataset(split='revision_documents',
                                     path_to_documents_for_summarization_indices=f"experiments/data/datasets_splits/xsum_summarization_{args.summarization_documents}_revision_{args.revision_summaries}_seed_{args.seed}.json",
                                     num_of_documents_for_summarization=args.summarization_documents,
                                     num_of_documents_for_revision=args.revision_summaries,
                                     seed=args.seed)
        texts = [dataset[i]['text'] for i in range(len(dataset))]
        summarization_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(args.base_models_dir, args.base_model_checkpoint)).to(args.device)
        tokenizer = T5Tokenizer.from_pretrained(os.path.join(args.base_models_dir, args.base_model_checkpoint))
        model_summaries = t5_summarize(texts=texts, model=summarization_model, tokenizer=tokenizer,
                                       prompt="summarize: ",
                                       device=args.device, max_generation_length=args.max_generation_length,
                                       batch_size=args.batch_size,
                                       beam_size=args.num_beams,
                                       early_stopping=True,
                                       length_penalty=args.length_penalty)
        del summarization_model
        gc.collect()
        torch.cuda.empty_cache()
        df = pd.DataFrame.from_dict({'text': texts, 'model_summary': model_summaries})
        df.to_csv(os.path.join(args.revision_data_dir, args.revision_data_file))
    else:
        df = pd.read_csv(os.path.join(args.revision_data_dir, args.revision_data_file), index_col=0)
    for col in df.columns:
        if 'model_summary_seahorse' == col:
            rel_df = df[(df['model_summary_seahorse'] < args.seahorse_threshold)]
            summaries = rel_df['model_summary'].tolist()
            texts = rel_df['text'].tolist()
            scores = rel_df['model_summary_seahorse'].tolist()
            return texts, summaries, scores,
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    factuality_scores = score(texts=texts, summaries=summaries, metrics=['seahorse'])['seahorse']
    df = pd.DataFrame({'model_summary_seahorse': factuality_scores,
                       'text': texts,
                       'model_summary': summaries})
    df.to_csv(os.path.join(args.revision_data_dir, args.revision_data_file))
    rel_df = df[(df['model_summary_seahorse'] < args.seahorse_threshold)]
    scores = rel_df['model_summary_seahorse'].tolist()
    summaries = rel_df['model_summary'].tolist()
    texts = rel_df['text'].tolist()
    return texts, summaries, scores

def llm_revision():
    args = parseargs_llms()
    texts, summaries, scores = get_data(args)
    revision_model = chose_revision_model(args)
    revised_summaries, errors, prices = [], [], []
    for text, summary in tqdm(zip(texts, summaries)):
        revised_summary, error, price = revision_model.revise_single(text=text, summary=summary,
                                                                     max_length=args.revision_max_length)
        revised_summaries.append(revised_summary)
        errors.append(error)
        prices.append(price)
        # time.sleep(2)
    pd.DataFrame.from_dict(
        {'text': texts, 'model_summary': summaries, 'revised_summary': revised_summaries,
         'model_summary_seahorse': scores, 'error': errors, 'price': prices}).to_csv(
        args.output_path + '.csv')
    print(f"Generation cost was {sum(prices)}")


if __name__ == "__main__":
    llm_revision()

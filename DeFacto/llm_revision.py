import gc
import os
import sys

import numpy as np

os.chdir('../')
sys.path.append(os.getcwd())
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
    parser.add_argument('-past_text_prompt_path', type=str)
    parser.add_argument('-API_KEY_revision_model', type=str, default=None)
    parser.add_argument('-temp_dir_path', type=str, default='contingency_tables')
    parser.add_argument('-max_generation_length', type=int, default=128)
    parser.add_argument('-revision_data_dir', type=str, default="DeFacto/revision/data")
    parser.add_argument('-revision_data_file', type=str)
    parser.add_argument('-azure', action='store_true')
    parser.add_argument('-groq', action='store_true')
    parser.add_argument('-revision_max_length', type=int, default=1000)
    parser.add_argument('-instructions', action='store_true')
    parser.add_argument('-output', type=str, default='revised_summary')
    args = parser.parse_args()
    with open(args.revision_prompt_path, 'r') as file:
        args.revision_prompt = file.read()
        args.revision_prompt = args.revision_prompt.strip()
    if args.past_text_prompt_path is not None:
        with open(args.past_text_prompt_path, 'r') as file:
            args.past_text_prompt = file.read()
            args.past_text_prompt = args.past_text_prompt.strip()
    else:
        args.past_text_prompt = ''
    args.temp_dir_path = os.path.join(args.revision_data_dir, args.temp_dir_path)
    if not os.path.exists(args.temp_dir_path):
        os.makedirs(args.temp_dir_path)
    model_prices = {'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
                    'llama-3.1': {'input': 0.59 / 1000, 'output': 0.79 / 1000}}
    args.input_price = model_prices[args.revision_model_name]['input']
    args.output_price = model_prices[args.revision_model_name]['output']
    return args


def get_data(args):
    df = pd.read_csv(os.path.join(args.revision_data_dir, args.revision_data_file), index_col=0)
    texts = df['text'].tolist()
    model_summaries = df['model_summary'].tolist()
    instructions = df['instruction'].tolist()
    return texts, model_summaries, instructions


def llm_revision():
    args = parseargs_llms()
    texts, summaries, instructions = get_data(args)
    revision_model = chose_revision_model(args)
    revised_summaries, errors, prices = [], [], []
    for text, summary, instruction in tqdm(zip(texts, summaries, instructions)):
        if args.instructions:
            revised_summary, error, price = revision_model.revise_single(text=text, summary=summary,
                                                                         instructions=instruction,
                                                                         max_length=args.revision_max_length
                                                                         )
        else:
            revised_summary, error, price = revision_model.revise_single(text=text, summary=summary,
                                                                         max_length=args.revision_max_length
                                                                         )
        revised_summaries.append(revised_summary)
        errors.append(error)
        prices.append(price)
    pd.DataFrame.from_dict(
        {'text': texts, 'model_summary': summaries, args.output: revised_summaries,
         'error': errors, 'price': prices}).to_csv(
        args.output_path + '.csv')
    print(f"Generation cost was {sum(prices)}")


if __name__ == "__main__":
    llm_revision()

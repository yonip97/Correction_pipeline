import gc
import os
import sys
import time

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
import torch
from experiments.scoring import score
from experiments.data.datasets_splits import split_xsum_dataset
from general.t5_trainer import t5_summarize
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer


def parseargs_llms():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_path')
    parser.add_argument('-model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    parser.add_argument('-revision_prompt_path', type=str)
    parser.add_argument('-past_text_prompt_path', type=str)
    parser.add_argument('-seahorse_threshold', type=float)
    parser.add_argument('-revision_data_dir', type=str, default="experiments/revision/data")
    parser.add_argument('-revision_data_file', type=str)
    parser.add_argument('-revision_max_length', type=int, default=1000)
    parser.add_argument('-revision_beam_size', type=int)
    parser.add_argument('-revision_top_p', type=float)
    parser.add_argument('-revision_temperature', type=float)
    parser.add_argument('-revision_early_stopping', action='store_true')
    parser.add_argument('-revision_sample', action='store_true')
    parser.add_argument('-batch_size', type=int, default=4)
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
    return args


def get_data(args):
    df = pd.read_csv(os.path.join(args.revision_data_dir, args.revision_data_file), index_col=0)
    if args.seahorse_threshold:
        rel_df = df[(df['model_summary_seahorse'] <= args.seahorse_threshold)]
    else:
        rel_df = df
    summaries = rel_df['model_summary'].tolist()
    texts = rel_df['text'].tolist()
    return texts, summaries

def llm_revision():
    access_token = "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"
    args = parseargs_llms()
    texts, summaries = get_data(args)
    model_id = args.model
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto", token=access_token
    )

    revisions = []
    times = []
    all_inputs = [
        args.revision_prompt + '\n' + 'Summary: \n' + summary + '\n' + "Document: \n" + args.past_text_prompt + text for
        text, summary in
        zip(texts, summaries)]
    start = time.time()
    for i in range(len(all_inputs)):
        messages = [
            {"role": "user", "content": all_inputs[i]}]
        outputs = pipeline(
            messages,
            max_new_tokens=args.revision_max_length)
        full_text_revision = outputs[0]["generated_text"][-1]['content']
        revisions.append(full_text_revision)
        print(f"Finished {i + 1} out of {len(all_inputs)} in {time.time() - start} seconds")
        times.append(time.time() - start)
        start = time.time()
        df = pd.DataFrame(
            {'text': texts[:len(revisions)], 'model_summary': summaries[:len(revisions)], 'revised_summary_full_text': revisions, 'time': times})
        df.to_csv(args.output_path)
    df = pd.DataFrame(
        {'text': texts, 'model_summary': summaries, 'revised_summary_full_text': revisions, 'time': times})
    df.to_csv(args.output_path)


if __name__ == "__main__":
    llm_revision()

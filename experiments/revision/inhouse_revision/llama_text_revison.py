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
    parser.add_argument('-batch_size', type=int, default=8)
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
    # pipeline = transformers.pipeline(
    #     "text-generation",
    #     model=model_id,
    #     model_kwargs={"torch_dtype": torch.bfloat16},
    #     device_map="auto", token=access_token
    # )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', token=access_token,torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    revisions = []
    times = []
    all_inputs = [
        args.revision_prompt + '\n' + 'Summary: \n' + summary + '\n' + "Document: \n" + args.past_text_prompt + text for
        text, summary in
        zip(texts, summaries)]
    start = time.time()
    batch_size = args.batch_size
    for i in range(0, len(all_inputs), batch_size):
        messages = [[
            {"role": "user", "content": all_inputs[index]},
        ] for index in range(i, min(i + batch_size, len(all_inputs)))]
        tokenizer.pad_token_id = tokenizer.eos_token_id

        messages = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, tokenize=False
        )
        tokenizer.pad_token_id = tokenizer.eos_token_id
        inputs = tokenizer(messages, padding="longest", return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=args.revision_max_length, pad_token_id=tokenizer.eos_token_id,
                                 eos_token_id=terminators)
        full_text_revision = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # temp_texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        # full_text_revision = [i[len(temp_texts[idx]):].strip() for idx, i in enumerate(full_text_revision)]

        revisions += full_text_revision
        print(f"Finished {i + 1} out of {len(all_inputs)} in {time.time() - start} seconds")
        batch_time = time.time() - start
        times += [batch_time/batch_size] * batch_size
        start = time.time()
        df = pd.DataFrame(
            {'text': texts[:len(revisions)], 'model_summary': summaries[:len(revisions)], 'revised_summary_full_text': revisions, 'time': times})
        df.to_csv(args.output_path)
    df = pd.DataFrame(
        {'text': texts, 'model_summary': summaries, 'revised_summary_full_text': revisions, 'time': times})
    df.to_csv(args.output_path)


if __name__ == "__main__":
    llm_revision()

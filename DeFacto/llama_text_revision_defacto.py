import gc
import os
import sys
import time

import numpy as np

os.chdir('/home/yehonatan-pe/Correction_pipeline')
sys.path.append(os.getcwd())
import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    parser.add_argument('-resume', action='store_true')
    parser.add_argument('-feedback', action='store_true')
    parser.add_argument('-instructions', action='store_true')
    parser.add_argument('-output_name', type=str, default='revised_summary')
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
    if args.feedback:
        if args.instructions:
            instructions = rel_df['instruction']
            instructions = [x.split('assistant\n\n')[-1] for x in instructions]
            return texts, summaries, instructions
        else:
            explanations = rel_df['explanation']
            return texts, summaries, explanations
    return texts, summaries, None


def llm_revision():
    access_token = "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"
    args = parseargs_llms()
    texts, summaries, feedbacks = get_data(args)

    if args.resume:
        df = pd.read_csv(args.output_path, index_col=0)
        revisions = df['revised_summary_full_text'].tolist()
        times = df['time'].tolist()
        curr_texts = texts[len(revisions):]
        cur_summaries = summaries[len(revisions):]
        if feedbacks is not None:
            if args.instructions:
                all_inputs = [
                    args.revision_prompt + '\n' + 'Document:\n' + text + '\n' + 'Summary:\n' + summary + '\n' + 'Instructions:\n' + instruction +'\n' +args.past_text_prompt
                    for
                    text, summary, instruction in
                    zip(curr_texts, cur_summaries, feedbacks)]
            else:
                all_inputs = [
                    args.revision_prompt + '\n' + 'Document:\n' + text + '\n' + 'Summary:\n' + summary + '\n' + 'Explanation:\n' + explanation + '\n'
                    +args.past_text_prompt
                    for
                    text, summary, explanation in
                    zip(curr_texts, cur_summaries, feedbacks)]
        else:
            all_inputs = [
                args.revision_prompt + '\n' + 'Document:\n' + text + '\n' + 'Summary:\n' + summary +'\n' +args.past_text_prompt
                for
                text, summary in
                zip(curr_texts, cur_summaries)]

    else:
        revisions = []
        times = []
        if args.feedback:
            if args.instructions:
                all_inputs = [
                    args.revision_prompt + '\n' + 'Document:\n' + text + '\n' + 'Summary:\n' + summary + '\n' + 'Instructions:\n' + instruction +'\n' +args.past_text_prompt
                    for
                    text, summary, instruction in
                    zip(texts, summaries, feedbacks)]
            else:
                all_inputs = [
                    args.revision_prompt + '\n' + 'Document:\n' + text + '\n' + 'Summary:\n' + summary + '\n' + 'Explanation:\n' + explanation + '\n'
                    +args.past_text_prompt
                    for
                    text, summary, explanation in
                    zip(texts, summaries, feedbacks)]
        else:
            all_inputs = [
                args.revision_prompt + '\n' + 'Document:\n' + text + '\n' + 'Summary:\n' + summary +'\n' +args.past_text_prompt
                for
                text, summary in
                zip(texts, summaries)
            ]
    model_id = args.model

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', token=access_token,
                                                 torch_dtype=torch.bfloat16)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    start = time.time()
    batch_size = args.batch_size
    with torch.no_grad():
        for i in range(0, len(all_inputs), batch_size):
            try:
                messages = [[
                    {"role": "user", "content": all_inputs[index]},
                ] for index in range(i, min(i + batch_size, len(all_inputs)))]
                tokenizer.pad_token_id = tokenizer.eos_token_id

                messages = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True, tokenize=False
                )
                tokenizer.pad_token_id = tokenizer.eos_token_id
                inputs = tokenizer(messages, padding="longest", return_tensors="pt", max_length=4096).to('cuda')
                outputs = model.generate(**inputs, max_new_tokens=args.revision_max_length,
                                         pad_token_id=tokenizer.eos_token_id,
                                         eos_token_id=terminators)
                full_text_revision = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                revisions += full_text_revision
                batch_time = time.time() - start
                times += [batch_time / batch_size] * batch_size

            except Exception as e:
                print(f"Error occurred: {str(e)}")
                revisions += ['' for _ in range(i, min(i + batch_size, len(all_inputs)))]
                times += [0] * batch_size
            finally:
                print(f"Finished {i + 1} out of {len(all_inputs)} in {time.time() - start} seconds")
                start = time.time()
                df = pd.DataFrame(
                    {'text': texts[:len(revisions)], 'model_summary': summaries[:len(revisions)],
                     args.output_name: revisions, 'time': times})
                df.to_csv(args.output_path)

    df = pd.DataFrame(
        {'text': texts, 'model_summary': summaries, args.output_name: revisions, 'time': times})
    df.to_csv(args.output_path)


if __name__ == "__main__":
    llm_revision()

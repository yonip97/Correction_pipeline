import os
import sys

import matplotlib.pyplot as plt

os.chdir("/data/home/yehonatan-pe/Correction_pipeline")
sys.path.append(os.getcwd())

import evaluate
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from transformers import Seq2SeqTrainingArguments
from general.utils import ExplanationsDataset, InstructionsDataset, find_largest_numbered_dir
from general.t5_trainer import T5_Trainer, collate_fn_instructions
import argparse
import json


def log_results_to_dir(args, trainer):
    current_logs = trainer.state.log_history
    eval_logs = []
    train_logs = []
    for log in current_logs:
        if 'eval_loss' in log:
            eval_logs.append(log)
        else:
            train_logs.append(log)
    with open(args.output_path + '/train_logs.json', 'w') as f:
        json.dump(train_logs, f)
    with open(args.output_path + '/eval_logs.json', 'w') as f:
        json.dump(eval_logs, f)


def parseargs_train_model():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-output_path')
    parser.add_argument('-data_path', type=str)
    parser.add_argument('-model_name', type=str, default='flan-t5-xl')
    parser.add_argument('-pretrained_model_path', type=str, default=None)
    parser.add_argument('-pretrained_adapter_path', type=str, default=None)
    parser.add_argument('-save_dir', type=str)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-train_batch_size', type=int, default=1)
    parser.add_argument('-eval_batch_size', type=int, default=2)
    parser.add_argument('-evaluation_strategy', type=str, default='no')
    parser.add_argument('-save_strategy', type=str, default='no')
    parser.add_argument('-eval_steps', type=float, default=5.0)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-dataset', type=str, default='xsum')
    parser.add_argument('-beam_size', type=int, default=4)
    parser.add_argument('-length_penalty', type=float, default=0.6)
    parser.add_argument('-max_generation_length', type=int, default=128)
    parser.add_argument('-min_generation_length', type=int, default=0)
    parser.add_argument('-max_encoding_length', type=int, default=1024)
    parser.add_argument('-revised_factuality_threshold', type=float, default=None)
    parser.add_argument('-revised_factuality_diff', type=float, default=None)
    parser.add_argument('-revised_rouge_threshold', type=float, default=None)
    parser.add_argument('-revised_density_threshold', type=float, default=None)
    parser.add_argument('-revised_density_diff', type=float, default=None)
    parser.add_argument('-revised_tokens_diff', type=float, default=None)
    parser.add_argument('-use_lora', action='store_true')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-train', action='store_true', default=False)
    parser.add_argument('-eval', action='store_true', default=False)
    parser.add_argument('-train_instructions', action='store_true', default=False)
    parser.add_argument('-train_size', type=float, default=None)
    parser.add_argument('-prompt', type=str,
                        default='You will be given a summary and a corresponding text. The summary contains factual '
                                'inconsistencies, which is information that cannot be verified by the text. '
                                'Create instructions to revise the summary so that it is factually consistent with the text')
    parser.add_argument('-save', action='store_true', default=False)
    parser.add_argument('-wandb', action='store_true', default=False)
    parser.add_argument('-dataset_name', type=str)
    parser.add_argument('-unwanted_instructions_path', type=str, default=None,
                        help='Path to a file with words we dont not want to appear in the instructions')
    parser.add_argument('-wanted_instructions_path', type=str, default=None,
                        help='Path to a file with words we want to appear in the instructions')
    parser.add_argument('-preprocessing', action='store_true', default=False)
    args = parser.parse_args()
    if args.pretrained_model_path is None:
        model_name_to_path = {'flan-t5-large': 'google/flan-t5-large', 'flan-t5-xl': 'google/flan-t5-xl'}

        args.model_path = model_name_to_path[args.model_name]
    else:
        args.model_path = args.pretrained_model_path
    if args.unwanted_instructions_path is not None:
        with open(args.unwanted_instructions_path, 'r') as f:
            args.unwanted_instructions = f.read().splitlines()
    elif args.wanted_instructions_path is not None:
        with open(args.wanted_instructions_path, 'r') as f:
            args.wanted_instructions = f.read().splitlines()
    return args


def compute_metrics(p, tokenizer):
    rouge_metric = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [str(x) for x in predictions]
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = [str(x) for x in labels]
    rouge = rouge_metric.compute(predictions=predictions, references=labels)
    return rouge


def filter_based_on_revised_summary(revised_df, args):
    if args.revised_factuality_threshold is not None:
        revised_df = revised_df[
            (revised_df['model_summary_seahorse'] < args.revised_factuality_threshold) & (
                    revised_df['revised_summary_seahorse'] >= args.revised_factuality_threshold)]
    if args.revised_rouge_threshold is not None:
        rouge_metric = evaluate.load('rouge')
        revised_df['rougeL'] = \
            rouge_metric.compute(predictions=revised_df['revised_summary'], references=revised_df['model_summary'],
                                 use_aggregator=False)['rougeL']
        revised_df = revised_df[revised_df['rougeL'] > args.revised_rouge_threshold]
    if args.revised_factuality_diff is not None:
        revised_df = revised_df[
            revised_df['revised_summary_seahorse'] - revised_df['model_summary_seahorse'] >=
            args.revised_factuality_diff]
    if args.revised_density_threshold is not None:
        revised_df = revised_df[revised_df['revised_summary_density'] <= args.revised_density_threshold]
    if args.revised_density_diff is not None:
        revised_df = revised_df[
            revised_df['revised_summary_density'] - revised_df['model_summary_density'] <= args.revised_density_diff]
    if args.revised_tokens_diff is not None:
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        revised_lengths = [len(tokenizer(x)['input_ids']) for x in revised_df['revised_summary']]
        model_lengths = [len(tokenizer(x)['input_ids']) for x in revised_df['model_summary']]
        diffs = [x - y for x, y in zip(revised_lengths, model_lengths)]
        revised_df = revised_df[[x >= args.revised_tokens_diff for x in diffs]]
    return revised_df


def get_indices_of_good_lists(list_of_lists, predefined_words):
    good_indices = []
    for i, sublist in enumerate(list_of_lists):
        if all(any(word in string.split() for word in predefined_words) for string in sublist):
            good_indices.append(i)
    return good_indices


def filter_based_on_bad_instructions(revised_df, args):
    if args.unwanted_instructions_path is not None:
        for word in args.unwanted_instructions:
            revised_df = revised_df[~revised_df['instruction'].str.contains(word)]
    if args.wanted_instructions_path is not None:
        instructions = revised_df['instruction'].tolist()
        #TOdo: was '\n' before
        instructions = [x.lower().split('. ') for x in instructions]
        indices = get_indices_of_good_lists(instructions, args.wanted_instructions)
        revised_df = revised_df.iloc[indices]
    return revised_df


def get_revised_model_data(revised_df):
    revised_summary_full_text = revised_df['revised_summary_full_text'].tolist()
    explanations = [x.split('Steps:')[0].replace('Explanation:', "", 1) for x in revised_summary_full_text]
    steps = []
    for x in revised_summary_full_text:
        if "Steps:" in x:
            curr_steps = x.split('Steps:')[1].split('Corrected:')[0].replace('Steps:', "", 1).strip()
            steps.append(curr_steps)
        elif "Steps to correct the summary:" in x:
            curr_steps = x.split('Steps to correct the summary:')[1].split('Corrected:')[0].replace('Steps:', "",
                                                                                                    1).strip()
            steps.append(curr_steps)
        else:
            steps.append(None)
    revised_df['explanation'] = explanations
    revised_df['instruction'] = steps
    revised_df = revised_df.dropna(subset=['model_summary', 'revised_summary', 'instruction', 'explanation'])
    return revised_df


def get_revised_data(args):
    revised_df = pd.read_csv(args.data_path, index_col=0)
    if args.preprocessing:
        revised_df = get_revised_model_data(revised_df)
    revised_df = filter_based_on_revised_summary(revised_df, args)
    revised_df = filter_based_on_bad_instructions(revised_df, args)
    texts = revised_df['text'].tolist()
    original_model_summaries = revised_df['model_summary'].tolist()
    instructions = revised_df['instruction'].tolist()
    explanations = revised_df['explanation'].tolist()
    return texts, original_model_summaries, instructions, explanations


def train_and_evaluate_model(args):
    torch.multiprocessing.set_start_method('spawn')
    if args.train:
        model = T5ForConditionalGeneration.from_pretrained(args.model_path, device_map='auto')
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        generation_config = model.generation_config
        generation_config.max_length = args.max_generation_length
        generation_config.early_stopping = True
        generation_config.length_penalty = args.length_penalty
        generation_config.num_beams = args.beam_size
        generation_config.min_length = args.min_generation_length
        model.generation_config = generation_config
        lr = args.lr
        train_batch_size = args.train_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        epochs = args.epochs
        weight_decay = args.weight_decay
        texts, original_model_summaries, instructions, explanations = get_revised_data(args)
        if args.train_instructions:
            if args.eval:
                train_size = int(len(texts) * args.train_size)
                train_dataset = InstructionsDataset(texts=texts[:train_size],
                                                    summaries=original_model_summaries[:train_size],
                                                    instructions=instructions[:train_size])
                eval_dataset = InstructionsDataset(texts=texts[train_size:],
                                                   summaries=original_model_summaries[train_size:],
                                                   instructions=instructions[train_size:])
            else:
                train_dataset = InstructionsDataset(texts=texts, summaries=original_model_summaries,
                                                    instructions=instructions)
                eval_dataset = None
        else:
            if args.eval:
                train_size = int(len(texts) * args.train_size)
                train_dataset = ExplanationsDataset(texts=texts[:train_size],
                                                    summaries=original_model_summaries[:train_size],
                                                    explanations=explanations[:train_size])
                eval_dataset = ExplanationsDataset(texts=texts[train_size:],
                                                   summaries=original_model_summaries[train_size:],
                                                   explanations=explanations[train_size:])
            else:
                train_dataset = ExplanationsDataset(texts=texts, summaries=original_model_summaries,
                                                    explanations=explanations)
                eval_dataset = None
        if not os.path.exists(os.path.join(args.save_dir, args.dataset_name)):
            os.makedirs(os.path.join(args.save_dir, args.dataset_name))
        args.save_dir = os.path.join(args.save_dir, args.dataset_name)
        model_name = args.model_name
        if not os.path.exists(os.path.join(args.save_dir, model_name)):
            os.makedirs(os.path.join(args.save_dir, model_name))
        model_path = os.path.join(args.save_dir, model_name)
        dir_num = find_largest_numbered_dir(model_path) + 1
        output_path = os.path.join(model_path, str(dir_num))
        os.makedirs(output_path)
        args.output_path = output_path
        with open(output_path + '/args.txt', 'w') as f:
            f.write(str(args))
        train_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(output_path, 'checkpoints'),
            do_train=args.train, do_eval=args.eval,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs, predict_with_generate=True,
            evaluation_strategy=args.evaluation_strategy,
            save_strategy=args.save_strategy, save_total_limit=5,
            eval_steps=args.eval_steps, eval_accumulation_steps=30,
            weight_decay=weight_decay,
            no_cuda=False, logging_steps=0.01, report_to=["none"], save_only_model=True)
        max_length = args.max_encoding_length
        if args.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            if args.pretrained_adapter_path is not None:
                model.load_adapter(args.pretrained_adapter_path)
            else:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    task_type=TaskType.SEQ_2_SEQ_LM
                )
                model = get_peft_model(model, lora_config)
        if eval_dataset is not None:
            trainer = T5_Trainer(collate_fn=collate_fn_instructions, model=model, tokenizer=tokenizer, args=train_args,
                                 train_dataset=train_dataset, eval_dataset=eval_dataset,
                                 max_length_train=max_length, max_length_eval=max_length,
                                 prompt_train=args.prompt, prompt_eval=args.prompt,
                                 compute_metrics=lambda p: compute_metrics(p, tokenizer))
        else:
            trainer = T5_Trainer(collate_fn=collate_fn_instructions, model=model, tokenizer=tokenizer, args=train_args,
                                 train_dataset=train_dataset,
                                 max_length_train=max_length, max_length_eval=max_length,
                                 prompt_train=args.prompt)
        trainer.train()
        if args.save:
            trainer.save_model()
        log_results_to_dir(args, trainer)


def split_instructions(instruction_list):
    all_instructions = []
    import re
    for instructions in instruction_list:
        split_instr = re.split(r'\d+\.\s', instructions)[1:]  # Split and remove empty string at the start
        for i in range(len(split_instr)):
            split_instr[i] = split_instr[i].strip()
        all_instructions.append(split_instr)
    return all_instructions


def analysis():
    args = parseargs_train_model()
    revised_df = pd.read_csv(args.data_path, index_col=0)
    revised_summary_full_text = revised_df['revised_summary_full_text'].tolist()
    explanations = [x.split('Steps:')[0].replace('Explanation:', "", 1) for x in revised_summary_full_text]
    steps = []
    for x in revised_summary_full_text:
        if "Steps:" in x:
            curr_steps = x.split('Steps:')[1].split('Corrected:')[0].replace('Steps:', "", 1).strip()
            steps.append(curr_steps)
        elif "Steps to correct the summary:" in x:
            curr_steps = x.split('Steps to correct the summary:')[1].split('Corrected:')[0].replace('Steps:', "",
                                                                                                    1).strip()
            steps.append(curr_steps)
        else:
            steps.append(None)
    revised_df['explanation'] = explanations
    revised_df['instruction'] = steps
    revised_df = revised_df.dropna(subset=['model_summary', 'revised_summary', 'instruction', 'explanation'])
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    tokens = [len(tokenizer(x)['input_ids']) for x in revised_df['instruction']]
    plt.hist(tokens)
    plt.show()
    splited = split_instructions(revised_df['instruction'])
    words = []
    for instruction_set in splited:
        for instruction in instruction_set:
            instruction = instruction.split()
            if len(instruction) == 0:
                continue
            words.append(instruction[0])
    from collections import Counter
    counter = Counter(words)
    counter = [x for x in dict(counter).items()]
    sorted_counter = sorted(counter, key=lambda x: x[1], reverse=True)
    total_num = sum([x[1] for x in sorted_counter])
    top_words = []
    other = 0
    for word, count in sorted_counter:
        if count / total_num > 0.01:
            top_words.append((word, count))
        else:
            other += count
    plt.bar([x[0] for x in top_words] + ['other'], [x[1] / total_num for x in top_words] + [other / total_num])
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    args = parseargs_train_model()
    train_and_evaluate_model(args)

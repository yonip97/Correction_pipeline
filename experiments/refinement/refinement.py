import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import json
import gc

import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer
from datetime import datetime
import numpy as np
from nltk.tokenize import word_tokenize
from general.fragments_metrics import Fragments
import evaluate
from nltk.tokenize import word_tokenize
from general.utils import SummarizationDataset
from Seahorse_metrics.metrics import Seahorse_metrics
import argparse
from optuna import Trial, create_study, study
import copy


def parse():
    args = argparse.ArgumentParser()
    args.add_argument('-revised_data_file', type=str)
    args.add_argument('-unrevised_data_file', type=str)
    args.add_argument('-eval_data_file', type=str)
    args.add_argument('-model_checkpoint', type=str)
    args.add_argument('-train_batch_size_per_device', type=int, default=2)
    args.add_argument('-eval_batch_size_per_device', type=int, default=4)
    args.add_argument('-encoder_max_length', type=int, default=2048)
    args.add_argument('-max_generation_length', type=int, default=128)
    args.add_argument('-lr', type=float, default=1e-3)
    args.add_argument('-beam_size', type=int, default=4)
    args.add_argument('-length_penalty', type=float, default=0.6)
    args.add_argument('-epochs', type=int, default=1)
    args.add_argument('-gradient_accumulation_steps', type=int, default=8)
    args.add_argument('-evaluation_strategy', type=str, default='steps')
    args.add_argument('-eval_steps', type=float, default=0.19)
    args.add_argument('-output_dir', type=str, default='experiments/refinement/data')
    args.add_argument('-output_path', type=str)
    args.add_argument('-optim', type=str, default='adafactor')
    args.add_argument('-metric_for_best_checkpoint', type=str)
    args.add_argument('-factuality_threshold_revised', type=float)
    args.add_argument('-factuality_threshold_unrevised', type=float)
    args.add_argument('-factuality_diff', type=float)
    args.add_argument('-density_threshold_revised', type=float)
    args.add_argument('-density_threshold_unrevised', type=float)
    args.add_argument('-density_diff', type=float)
    args.add_argument('-rouge_to_base_threshold', type=float)
    args.add_argument('-rouge_to_original_threshold_revised', type=float)
    args.add_argument('-rouge_to_original_threshold_unrevised', type=float)
    args.add_argument('-number_of_unrevised_samples', type=int)
    args.add_argument('-ratio_revised_to_unrevised', type=float)
    args.add_argument('-strategy', type=str, default='revised_and_unrevised')
    args = args.parse_args()
    return args


def compute_metrics(p, tokenizer, eval_texts, base_seahorse_scores, base_density, base_model_summaries,
                    original_dataset_summaries):
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [str(x) for x in predictions]
    results = {}
    fragments_metric = Fragments()
    density_scores = fragments_metric.score(metrics=['density'], texts=eval_texts, summaries=predictions)['density']
    results['first_500_density'] = np.mean(density_scores[:500])
    results['last_500_density'] = np.mean(density_scores[-500:])
    gc.collect()
    torch.cuda.empty_cache()
    seahorse_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                       device='auto', batch_size=1, torch_dtype=torch.float16, max_length=2048,
                                       return_none=True)
    seahorse_scores = seahorse_metric.score(texts=eval_texts, summaries=predictions)
    seahorse_scores_first_500 = seahorse_scores[:500]
    seahorse_scores_last_500 = seahorse_scores[-500:]
    base_seahorse_scores_first_500 = base_seahorse_scores[:500]
    base_seahorse_scores_last_500 = base_seahorse_scores[-500:]
    del seahorse_metric
    gc.collect()
    torch.cuda.empty_cache()
    results['first_500_seahorse'] = np.mean([x for x in seahorse_scores_first_500 if x is not None])
    results['last_500_seahorse'] = np.mean([x for x in seahorse_scores_last_500 if x is not None])
    seahorse_diff_first_500 = [x - y for x, y in zip(seahorse_scores_first_500, base_seahorse_scores_first_500) if
                               x is not None]
    seahorse_diff_last_500 = [x - y for x, y in zip(seahorse_scores_last_500, base_seahorse_scores_last_500) if
                              x is not None]
    results['first_500_seahorse_diff'] = np.mean([x for x in seahorse_diff_first_500 if x is not None])
    results['last_500_seahorse_diff'] = np.mean([x for x in seahorse_diff_last_500 if x is not None])
    density_diff = [x - y for x, y in zip(density_scores, base_density)]
    results['first_500_density_diff'] = np.mean(density_diff[:500])
    results['last_500_density_diff'] = np.mean(density_diff[-500:])
    rouge_metric = evaluate.load('rouge')
    rouge_scores_to_original = \
    rouge_metric.compute(predictions=predictions, references=original_dataset_summaries, use_aggregator=False)['rougeL']
    rouge_scores_to_base = \
    rouge_metric.compute(predictions=predictions, references=base_model_summaries, use_aggregator=False)['rougeL']
    results['first_500_rouge_to_original'] = np.mean(rouge_scores_to_original[:500])
    results['last_500_rouge_to_original'] = np.mean(rouge_scores_to_original[-500:])
    results['first_500_rouge_to_base'] = np.mean(rouge_scores_to_base[:500])
    results['last_500_rouge_to_base'] = np.mean(rouge_scores_to_base[-500:])
    results['predicted_summaries'] = predictions
    return results


def preprocess_data(df):
    df = df[~df['revised_summary_full_text'].str.contains('No correction')]
    return df


def create_filtered_train_data_revised(df, args):
    if args.factuality_threshold_revised is not None:
        df = df[df['post_revision_factuality_score'] >= args.factuality_threshold_revised]
    if args.factuality_diff is not None:
        df = df[df['post_revision_factuality_score'] - df['pre_revision_factuality_score'] >= args.factuality_diff]
    if args.density_threshold_revised is not None:
        df = df[df['post_revision_density'] <= args.density_threshold_revised]
    if args.density_diff is not None:
        df = df[df['post_revision_density'] - df['pre_revision_density'] <= args.density_diff]
    if args.rouge_to_base_threshold is not None:
        df = df[df['rougeL_revised_to_base'] >= args.rouge_to_base_threshold]
    if args.rouge_to_original_threshold_revised is not None:
        df = df[df['rougeL_revised_to_original'] >= args.rouge_to_original_threshold_revised]
    print(f"revised dataset length is {len(df)}")
    return df


def create_filtered_train_data_unrevised(df, args):
    if args.factuality_threshold_unrevised is not None:
        df = df[df['factuality_score'] >= args.factuality_threshold_unrevised]
    if args.density_threshold_unrevised is not None:
        df = df[df['density'] <= args.density_threshold_unrevised]
    if args.rouge_to_original_threshold_unrevised is not None:
        df = df[df['rougeL_base_to_original'] >= args.rouge_to_original_threshold_unrevised]
    if args.number_of_unrevised_samples is not None:
        df = df[:args.number_of_unrevised_samples]
    print(f"unrevised dataset length is {len(df)}")
    return df


def train_collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + ':' + row['text'] for row in batch]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def train(args):
    revised_texts = []
    revised_summaries = []
    unrevised_texts = []
    unrevised_summaries = []

    if args.revised_data_file is not None:
        revised_df = pd.read_csv(args.revised_data_file + '.csv', index_col=0)
        revised_df = preprocess_data(revised_df)
        revised_df = create_filtered_train_data_revised(revised_df, args)
        revised_texts = revised_df['text'].tolist()
        revised_summaries = revised_df['revised_summary'].tolist()
    if args.unrevised_data_file is not None:
        unrevised_summaries = pd.read_csv(args.unrevised_data_file + '.csv', index_col=0)
        unrevised_summaries = create_filtered_train_data_unrevised(unrevised_summaries, args)
        unrevised_texts = unrevised_summaries['text'].tolist()
        unrevised_summaries = unrevised_summaries['model_summary'].tolist()
    if args.ratio_revised_to_unrevised is not None:
        curr_ratio = len(revised_texts) / len(unrevised_texts)
        needed_upsampling = int(1 / curr_ratio * args.ratio_revised_to_unrevised)
        revised_texts = revised_texts * needed_upsampling
        revised_summaries = revised_summaries * needed_upsampling
    texts = revised_texts + unrevised_texts
    summaries = revised_summaries + unrevised_summaries
    if len(texts) == 0:
        raise ValueError("No data to train on")
    train_dataset = SummarizationDataset(texts=texts, summaries=summaries)
    eval_df = pd.read_csv(args.eval_data_file + '.csv', index_col=0)
    eval_dataset = SummarizationDataset(texts=eval_df['text'].tolist(), summaries=eval_df['model_summary'].tolist())
    eval_texts = eval_df['text'].tolist()
    eval_pre_revision_factuality_scores = eval_df['pre_revision_factuality_score'].tolist()
    eval_pre_revision_density = eval_df['pre_revision_density'].tolist()
    eval_base_model_summaries = eval_df['model_summary'].tolist()
    eval_original_dataset_summaries = eval_df['original_summary'].tolist()
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
    generation_config = model.generation_config
    generation_config.max_length = args.max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.length_penalty
    model.generation_config = generation_config
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    model_path = os.path.join(args.output_path, run_name)
    train_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        do_train=True, do_eval=True,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.eval_batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, evaluation_strategy=args.evaluation_strategy,
        save_strategy="no",
        eval_steps=args.eval_steps, eval_accumulation_steps=30,
        metric_for_best_model=args.metric_for_best_checkpoint, no_cuda=False, predict_with_generate=True,
        generation_num_beams=args.beam_size,
        optim=args.optim, overwrite_output_dir=False, logging_steps=0.01)
    max_length_train = args.encoder_max_length
    trainer = T5_Trainer(collate_fn=train_collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer, eval_texts,
                                                                   eval_pre_revision_factuality_scores,
                                                                   eval_pre_revision_density,
                                                                   eval_base_model_summaries,
                                                                   eval_original_dataset_summaries),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    return trainer, model


def tune(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        with open(os.path.join(args.output_path, 'logs.json'), 'w') as f:
            json.dump({}, f)
    trainer, model = train(args)
    current_log = trainer.state.log_history
    with open(os.path.join(args.output_path, 'logs.json'), 'r') as f:
        logs = json.load(f)
        if len(logs) == 0:
            logs = {0: {"logs": current_log, 'args': args.__dict__}}
        else:
            logs[len(logs)] = {"logs": current_log, 'args': args.__dict__}
    with open(os.path.join(args.output_path, 'logs.json'), 'w') as f:
        json.dump(logs, f)
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    os.environ["WANDB_DISABLED"] = "true"
    original_args = parse()
    original_args.output_path = os.path.join(original_args.output_dir, original_args.strategy)
    if not os.path.exists(original_args.output_path):
        os.makedirs(original_args.output_path)
        with open(os.path.join(original_args.output_path, 'individual_logs.json'), 'w') as f:
            json.dump({}, f)
    trainer, model = train(original_args)
    current_log = trainer.state.log_history
    with open(os.path.join(original_args.output_path, 'individual_logs.json'), 'r') as f:
        logs = json.load(f)
        if len(logs) == 0:
            logs = {0: {"logs": current_log, 'args': original_args.__dict__}}
        else:
            logs[len(logs)] = {"logs": current_log, 'args': original_args.__dict__}
    with open(os.path.join(original_args.output_path, 'individual_logs.json'), 'w') as f:
        json.dump(logs, f)
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def check_hyperparameters():
    os.environ["WANDB_DISABLED"] = "true"
    original_args = parse()
    factuality_diffs = [0.3, 0.4, 0.5, 0.6]
    density_diffs = [0.5, 1, 1.5, 2]
    rouge_thresholds = [0.4, 0.5, 0.6, 0.7]
    strategy = original_args.strategy
    for lr in [1e-5, 1e-4, 1e-3]:
        for epochs in [1, 3, 5]:
            run_args = copy.deepcopy(original_args)
            run_args.lr = lr
            run_args.epochs = epochs
            if strategy == 'no_revised':
                run_args.train_data_file = None
                run_args.output_path = os.path.join(run_args.output_dir, strategy)
                tune(run_args)
            elif strategy == 'no_unrevised':
                run_args.unrevised_data_file = None
                run_args.output_path = os.path.join(run_args.output_dir, strategy)
                tune(run_args)
            elif strategy == 'revised_and_unrevised':
                run_args.output_path = os.path.join(run_args.output_dir, strategy)
                tune(run_args)
            elif strategy == 'factuality_diff':
                run_args.output_path = os.path.join(run_args.output_dir, strategy)
                for factuality_diff in factuality_diffs:
                    run_args.factuality_diff = factuality_diff
                    tune(run_args)
            elif strategy == 'factuality_diff_and_rouge_threshold':
                run_args.output_path = os.path.join(run_args.output_dir, strategy)
                for factuality_diff in factuality_diffs:
                    run_args.factuality_diff = factuality_diff
                    for rouge_threshold in rouge_thresholds:
                        run_args.rouge_to_base_threshold = rouge_threshold
                        tune(run_args)
            elif strategy == 'factuality_diff_and_density_diff':
                run_args.output_path = os.path.join(run_args.output_dir, strategy)
                for factuality_diff in factuality_diffs:
                    run_args.factuality_diff = factuality_diff
                    for density_diff in density_diffs:
                        run_args.density_diff = density_diff
                        tune(run_args)
            elif strategy == 'factuality_diff_and_density_diff_and_rouge_threshold':
                run_args.output_path = os.path.join(run_args.output_dir,
                                                    strategy)
                for factuality_diff in factuality_diffs:
                    run_args.factuality_diff = factuality_diff
                    for density_diff in density_diffs:
                        run_args.density_diff = density_diff
                        for rouge_threshold in rouge_thresholds:
                            run_args.rouge_to_base_threshold = rouge_threshold
                            tune(run_args)
            else:
                raise ValueError("wrong strategy")


if __name__ == '__main__':
    main()

import pickle
import numpy as np
import os
import sys

import pandas as pd

os.chdir('../')
sys.path.append(os.getcwd())
os.chdir('../')
sys.path.append(os.getcwd())
os.chdir('../')
sys.path.append(os.getcwd())
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer
from datasets import load_dataset
from torch.utils.data import Dataset
import evaluate
from nltk.tokenize import word_tokenize
from general.fragments_metrics import Fragments
from experiments.scoring import score
from general.utils import find_largest_numbered_dir
import json
import torch
import gc

class XsumDatasetScored(Dataset):
    def __init__(self, xsum_dataset, scores, factuality_threshold, tokens_map):
        super(XsumDatasetScored, self).__init__()
        self.xsum_dataset = xsum_dataset
        self.scores = scores
        self.factuality_threshold = factuality_threshold
        self.tokens_map = tokens_map

    def __len__(self):
        return len(self.xsum_dataset)

    def __getitem__(self, item):
        item_score = self.scores[item]
        if item_score <= self.factuality_threshold:
            token = self.tokens_map[0]
        else:
            token = self.tokens_map[1]
        return {'text': self.xsum_dataset[item]['document'], 'summary': self.xsum_dataset[item]['summary'],
                'begin_token': token}


class EvalDataset(Dataset):
    def __init__(self, xsum_dataset, tokens_map):
        super(EvalDataset, self).__init__()
        self.xsum_dataset = xsum_dataset
        self.tokens_map = tokens_map

    def __len__(self):
        return len(self.xsum_dataset)

    def __getitem__(self, item):
        return {'text': self.xsum_dataset[item]['document'], 'summary': self.xsum_dataset[item]['summary'],
                'begin_token': self.tokens_map[1]}


def create_train_dataset(args):
    df = pd.read_csv(args.train_data_scores + '.csv', index_col=0)
    xsum_dataset = load_dataset('xsum', split='train')
    return XsumDatasetScored(xsum_dataset, df['seahorse_score'].tolist(), args.factuality_threshold, tokens_map=args.tokens_map)


def create_eval_dataset(args):
    xsum_dataset = load_dataset('xsum', split='validation')
    return EvalDataset(xsum_dataset, tokens_map=args.tokens_map)


def compute_metrics(p, tokenizer, eval_texts, args):
    rouge_metric = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = {'summary': predictions}
    rouge_scores = rouge_metric.compute(predictions=predictions, references=labels, use_aggregator=False)
    results['rougeL'] = rouge_scores['rougeL']
    fragments_metric = Fragments()
    density_scores = fragments_metric.score(metrics=['density'], texts=eval_texts, summaries=predictions)['density']
    results['density'] = density_scores
    results['density_mean'] = np.mean(density_scores)
    results['length'] = [len(word_tokenize(x)) for x in predictions]
    results['length_mean'] = np.mean(results['length'])
    if args.trueteacher:
        trueteacher_scores = score(texts=eval_texts, summaries=predictions, metrics=['trueteacher'])['trueteacher']
        results['trueteacher'] = trueteacher_scores
        results['trueteacher_mean'] = np.mean(trueteacher_scores)
    if args.seahorse:
        seahorse_scores = score(texts=eval_texts, summaries=predictions, metrics=['seahorse'])['seahorse']
        results['seahorse'] = seahorse_scores
        results['seahorse_mean'] = np.mean(seahorse_scores)
    return results


def parseargs():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument('--train_data_scores', type=str)
    args.add_argument('--model', type=str)
    args.add_argument('--output_path', type=str)
    args.add_argument('--factuality_threshold',type=float,default=0.5)
    args.add_argument('--max_generation_length',type=int,default=128)
    args.add_argument('--length_penalty',type=float,default=0.6)
    args.add_argument('--beam_size',type=int,default=4)
    args.add_argument('--train_batch_size_per_device',type=int,default=2)
    args.add_argument('--eval_batch_size_per_device',type=int,default=4)
    args.add_argument('--gradient_accumulation_steps',type=int,default=8)
    args.add_argument('--lr',type=float,default=5e-4)
    args.add_argument('--epochs',type=int,default=8)
    args.add_argument('--save_limit',type=int,default=None)
    args.add_argument('--evaluation_strategy',type=str,default='steps')
    args.add_argument('--save_strategy',type=str,default='no')
    args.add_argument('--eval_steps',type=float)
    args.add_argument('--optim',type=str,default='adafactor')
    args.add_argument('--encoder_max_length',type=int,default=2048)
    args.add_argument('--trueteacher',action='store_true')
    args.add_argument('--seahorse',action='store_true')
    args.add_argument()

    args = args.parse_args()
    args.tokens_map = {0:'inconsistent', 1:'consistent'}
    return args


def collate_fn(batch, tokenizer, max_length):
    begin_tokens = [begin_token for begin_token in batch['begin_token']]
    documents = [begin_tokens[i] + ' summarize: ' + batch['texts'][i] for i in range(len(batch['texts']))]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}




def train(args, train_dataset, eval_dataset):
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    generation_config = model.generation_config
    generation_config.max_length = args.max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.length_penalty
    model.generation_config = generation_config
    train_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_path, 'checkpoints'),
        do_train=True, do_eval=True,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.eval_batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, save_total_limit=args.save_limit,
        load_best_model_at_end=True, evaluation_strategy=args.evaluation_strategy, save_strategy=args.save_strategy,
        eval_steps=args.eval_steps, save_steps=args.eval_steps, eval_accumulation_steps=30,
        metric_for_best_model='rougeL', predict_with_generate=True, generation_num_beams=args.beam_size,
        optim=args.optim, overwrite_output_dir=False)
    max_length_train = args.encoder_max_length
    eval_texts = [row['text'] for row in eval_dataset]
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer,
                         args=train_args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer, eval_texts, args),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    return trainer, model


def main():
    args = parseargs()
    train_dataset = create_train_dataset(args)
    eval_dataset = create_eval_dataset(args)
    eval_texts = [row['text'] for row in eval_dataset]
    eval_original_summaries = [row['summary'] for row in eval_dataset]
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        latest_name = 0
    else:
        latest_name = find_largest_numbered_dir(args.output_path)
        latest_name += 1
    args.output_path = os.path.join(args.output_path, str(latest_name))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    trainer, model = train(args, train_dataset, eval_dataset)
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
    temp_df = pd.DataFrame({'text': eval_texts, 'original_summary': eval_original_summaries})
    for i, log in enumerate(eval_logs):
        for key in log:
            if isinstance(log[key], list):
                temp_df[key] = log[key]
        temp_df.to_csv(os.path.join(args.output_path, f"eval_{i}.csv"))
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
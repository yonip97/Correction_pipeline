import os
from datetime import datetime
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer
from experiments.ablations.ctrl.utils import load_list_from_file, SummariesscoredDataset
from datasets import load_dataset
import evaluate


def compute_metrics(p, tokenizer):
    metric = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = metric.compute(predictions=predictions, references=labels)
    return results


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_scores_path', type=str)
    parser.add_argument('--val_scores_path', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=float)
    parser.add_argument('--num_epochs', type=float)
    parser.add_argument('--max_length_train', type=float, default=512)
    parser.add_argument('--max_length_eval', type=float, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--generation_max_length', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='experiments/ablations/ctrl/model_checkpoints')
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument("--evaluation_strategy", type=str, default='steps')
    parser.add_argument("--save_strategy", type=str, default='steps')
    parser.add_argument("--eval_steps", type=int, default=0.1)
    parser.add_argument("--save_limit", type=int, default=2)
    args = parser.parse_args()
    return args


def create_train_data(args):
    scores = load_list_from_file(args.train_scores_path)
    predictions = [1 if score > 0.5 else 0 for score in scores]
    xsum_train_set = load_dataset('xsum', split='train')
    texts = [xsum_train_set[i]['document'] for i in range(len(xsum_train_set))]
    summaries = [xsum_train_set[i]['summary'] for i in range(len(xsum_train_set))]
    train_dataset = SummariesscoredDataset(texts, summaries, predictions)
    return train_dataset


def create_val_data(args):
    scores = load_list_from_file(args.val_scores_path)
    predictions = [1 if score > 0.5 else 0 for score in scores]
    xsum_train_set = load_dataset('xsum', split='validation')
    texts = [xsum_train_set[i]['document'] for i in range(len(xsum_train_set))]
    summaries = [xsum_train_set[i]['summary'] for i in range(len(xsum_train_set))]
    val_dataset = SummariesscoredDataset(texts, summaries, predictions)
    return val_dataset


def collate_fn(batch, tokenizer, max_length):
    texts = [row['text'] for row in batch]
    summaries = [row['summary'] for row in batch]
    predictions = [row['prediction'] for row in batch]
    inputs = tokenizer.encode_plus(
        ["consistent: " if prediction == 1 else "inconsistent: " for prediction in predictions],
        ["summarize: " + text for text in texts], padding=True, truncation='second_only', max_length=max_length,
        return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def train():
    args = parseargs()
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    train_dataset = create_train_data(args)
    val_dataset = create_val_data(args)
    run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_save_path = os.path.join(args.output_dir, run_name)
    train_args = Seq2SeqTrainingArguments(
        output_dir=run_save_path,
        do_train=True, do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps, logging_strategy="steps", save_steps=args.eval_steps, save_total_limit=args.save_limit,
        eval_accumulation_steps=30, weight_decay=args.weight_decay,
        metric_for_best_model='rougeL', no_cuda=args.no_cuda, predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.generation_max_length, logging_steps=0.01)
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer),
                         max_length_train=args.max_length_train, max_length_eval=args.max_length_train)
    trainer.train()


if __name__ == '__main__':
    train()

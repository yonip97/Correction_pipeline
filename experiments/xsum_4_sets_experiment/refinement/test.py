import json
import os
import sys
import time

import torch

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')

from datetime import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer, t5_summarize
import numpy as np
import pandas as pd
import evaluate
from general.utils import RevisionDataset
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
from experiments.xsum_4_sets_experiment.datasets_splits import split_cnndm_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from experiments.scoring import score
from experiments.xsum_4_sets_experiment.refinement.refinement_utils import create_dataset, create_full_dataset


def compute_metrics(p, tokenizer):
    rouge = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.compute(predictions=predictions, references=labels)
    return results


def collate_fn(batch, tokenizer, max_length):
    revised_summaries = [row['revised_summary'] for row in batch]
    texts_inputs = ["summarize: " + row['text'] for row in batch]
    inputs = tokenizer(texts_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def collate_fn_test(batch, tokenizer, max_length):
    texts_inputs = ["summarize: " + row['text'] for row in batch]
    inputs = tokenizer(texts_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


def args_parser():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args.add_argument("--weight_decay", type=float, default=0)
    args.add_argument("--train_size", type=float, default=1)
    args.add_argument("--max_encoding_length", type=int, default=512)
    args.add_argument("--model_checkpoint", type=str)
    args.add_argument("--model_dir", type=str)
    args.add_argument("--test_batch_size", type=int, default=32)
    args.add_argument("--test_max_encoding_length", type=int, default=512)
    args.add_argument("--beam_size", type=int, default=4)
    args.add_argument("--generation_max_length", type=int, default=128)
    args.add_argument("--length_penalty", type=float, default=0.6)
    args.add_argument("--test_save_path", type=str)
    args.add_argument("--test_save_dir", type=str, default="experiments/xsum_4_sets_experiment/refinement")
    args.add_argument('--factuality_threshold', type=float, default=0.5)
    args.add_argument('--factuality_diff_threshold', type=float, default=0.5)
    args.add_argument("--rouge_threshold", type=float, default=0.7)
    args.add_argument("--classifier_threshold", type=float, default=0.5)
    args.add_argument("--density_threshold", type=float, default=2)
    args.add_argument("--density_diff_threshold", type=float, default=0.5)
    args.add_argument('--method', type=str, default='classifier_and_rouge_threshold')
    args.add_argument('--device', type=str, default='auto')
    args.add_argument('--rerun', action='store_true')
    args.add_argument('--run_name', type=str, default=None)

    args.add_argument('--revised_dataset_path', type=str,
                      default='experiments/xsum_4_sets_experiment/documents_for_summarization_fully_scored_with_revised.csv')
    return args.parse_args()


def refine(train_dataset, args):
    torch.cuda.empty_cache()
    # hyperparameters = get_best_hyperparameters("all")[1][1]['hyperparameters']
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    max_length_train = args.max_encoding_length
    num_beams = args.beam_size
    generation_max_length = args.generation_max_length
    device = args.device
    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.environ["WANDB_DISABLED"] = "true"
    model_checkpoint = args.model_checkpoint
    models_dir = args.model_dir
    model_path = os.path.join(models_dir, model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    train_args = Seq2SeqTrainingArguments(
        output_dir=f'experiments/xsum_4_sets_experiment/runs/t5_base_{run_name}',
        do_train=True, do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr, num_train_epochs=epochs, evaluation_strategy='no', save_strategy='no',
        eval_accumulation_steps=30, weight_decay=weight_decay,
        metric_for_best_model='rougeL', no_cuda=no_cuda, predict_with_generate=True, generation_num_beams=num_beams,
        generation_max_length=generation_max_length, logging_steps=0.01)
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    return model, tokenizer


def test(model, tokenizer, args):
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    predictions = t5_summarize(texts=texts, model=model, tokenizer=tokenizer, prompt='summarize: ', device=args.device,
                               batch_size=args.test_batch_size,
                               max_generation_length=args.generation_max_length, beam_size=args.beam_size,
                               early_stopping=True,
                               length_penalty=args.length_penalty)
    return predictions


def create_summaries():
    args = args_parser()
    df = create_full_dataset(args.revised_dataset_path,args)
    test_save_path = args.test_save_path + '/method_' + args.method + '.json'
    test_save_dir = args.test_save_dir
    method = args.method
    train_dataset = create_dataset(df, method, args)
    model, tokenizer = refine(train_dataset, args)
    predictions = test(model, tokenizer, args)
    with open(os.path.join(test_save_dir, test_save_path), 'w') as file:
        print(f"Saving results to {os.path.join(test_save_dir, test_save_path)}")
        json.dump(predictions, file)


def main():
    create_summaries()


if __name__ == '__main__':
    main()

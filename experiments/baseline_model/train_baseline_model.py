import gc
import os
import sys
import time

import evaluate
import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from general.t5_trainer import T5_Trainer, t5_summarize
from experiments.data.datasets_splits import split_xsum_dataset, split_cnndm_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import os
from datetime import datetime
import numpy as np
import torch
from datasets import concatenate_datasets
import argparse
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
import gc
from experiments.scoring import score
from general.utils import get_latest_directory


# from general.utils import get_latest_directory


def parserargs():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default='xsum')
    args.add_argument("--model", type=str, default='t5-base')
    args.add_argument("--train_batch_size_per_device", type=int, default=8)
    args.add_argument("--eval_batch_size_per_device", type=int, default=4)
    args.add_argument('--test_batch_size', type=int, default=4)
    args.add_argument("--eval_steps", type=float, default=0.099)
    args.add_argument("--save_limit", type=int, default=2)
    args.add_argument("--encoder_max_length", type=int, default=512)
    args.add_argument("--max_generation_length", type=int, default=128)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--beam_size", type=int, default=4)
    args.add_argument("--length_penalty", type=float, default=0.6)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--gradient_accumulation_steps", type=int, default=2)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--num_of_documents_for_summarization_xsum", type=int, default=20000)
    args.add_argument("--num_of_documents_for_revision_xsum", type=int, default=10000)
    args.add_argument("--num_of_documents_for_summarization_cnndm", type=int, default=20000)
    args.add_argument("--num_of_documents_for_revision_cnndm", type=int, default=10000)
    args.add_argument("--model_checkpoints_dir", type=str, default="experiments/baseline_model/checkpoints")
    args.add_argument("--checkpoint_path", type=str, default=None)
    args.add_argument("--evaluation_strategy", type=str, default='steps')
    args.add_argument("--save_strategy", type=str, default='steps')
    args.add_argument("--device", type=str, default='cuda')
    args.add_argument('--output_dir', type=str, default='experiments/baseline_model/outputs')
    args.add_argument('--output_path', type=str)
    args.add_argument('--optim', type=str, default='adamw_torch')
    args.add_argument('--resume_from_checkpoint', action='store_true')
    args.add_argument('--checkpoints_path', type=str, default=None)
    args = args.parse_args()
    return args


def compute_metrics(p, tokenizer, dev_texts):
    rouge_metric = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge_metric.compute(predictions=predictions, references=labels)
    fragments_metric = Fragments()
    density_scores = fragments_metric.score(metrics=['density'], texts=dev_texts, summaries=predictions)['density']
    results['density'] = np.mean(density_scores)
    results['length'] = np.mean([len(word_tokenize(x)) for x in predictions])
    return results


def collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + ':' + row['text'] for row in batch]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def train(train_dataset, val_dataset, dataset, args):
    if args.resume_from_checkpoint and args.checkpoints_path is not None:
        checkpoints_path = args.checkpoints_path
    else:
        run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        checkpoints_path = os.path.join(args.model_checkpoints_dir, f'{args.model}_{dataset}_{run_name}')
        os.mkdir(checkpoints_path)
        with open(os.path.join(checkpoints_path, 'args.txt'), 'w') as f:
            f.write(str(args))
    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    generation_config = model.generation_config
    generation_config.max_length = args.max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.length_penalty
    model.generation_config = generation_config
    train_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_path,
        do_train=True, do_eval=True,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.eval_batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, save_total_limit=args.save_limit,
        load_best_model_at_end=True, evaluation_strategy=args.evaluation_strategy, save_strategy=args.save_strategy,
        eval_steps=args.eval_steps, save_steps=args.eval_steps, eval_accumulation_steps=30,
        metric_for_best_model='rougeL', no_cuda=False, predict_with_generate=True, generation_num_beams=args.beam_size,
        optim=args.optim, overwrite_output_dir=False)
    val_texts = [row['text'] for row in val_dataset]
    max_length_train = args.encoder_max_length
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer, val_texts),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_xsum(args):
    num_of_documents_for_summarization = args.num_of_documents_for_summarization_xsum
    num_of_documents_for_revision = args.num_of_documents_for_revision_xsum
    seed = args.seed
    path_to_documents_for_summarization_indices = f'experiments/data/datasets_splits/xsum_summarization_{num_of_documents_for_summarization}' \
                                                  f'_revision_{num_of_documents_for_revision}_seed_{seed}.json'
    train_dataset = split_xsum_dataset(split='train_model',
                                       path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                       num_of_documents_for_summarization=num_of_documents_for_summarization,
                                       num_of_documents_for_revision=num_of_documents_for_revision, seed=seed)
    val_dataset = split_xsum_dataset(split='validation_model',
                                     path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                     num_of_documents_for_summarization=num_of_documents_for_summarization,
                                     num_of_documents_for_revision=num_of_documents_for_revision, seed=seed)
    os.environ["WANDB_DISABLED"] = "true"
    train(train_dataset, val_dataset, dataset='xsum', args=args)


def train_cnndm(args):
    num_of_documents_for_summarization = args.num_of_documents_for_summarization_cnndm
    num_of_documents_for_revision = args.num_of_documents_for_revision_cnndm
    seed = args.seed
    path_to_documents_for_summarization_indices = f'experiments/xsum_4_sets_experiment/datasets_splits/' \
                                                  f'cnndm_summarization_{num_of_documents_for_summarization}_revision_{num_of_documents_for_revision}_seed_{seed}.json'
    train_dataset = split_cnndm_dataset(split='train_model',
                                        path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                        num_of_documents_for_summarization=num_of_documents_for_summarization,
                                        num_of_documents_for_revision=num_of_documents_for_revision,
                                        seed=seed)
    val_dataset = split_cnndm_dataset(split='validation_model',
                                      path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                      num_of_documents_for_summarization=num_of_documents_for_summarization,
                                      num_of_documents_for_revision=num_of_documents_for_revision, seed=seed)

    os.environ["WANDB_DISABLED"] = "true"
    train(train_dataset, val_dataset, dataset='cnndm', args=args)


def train_both(args):
    num_of_documents_for_summarization_xsum = args.num_of_documents_for_summarization_xsum
    num_of_documents_for_revision_xsum = args.num_of_documents_for_revision_xsum
    num_of_documents_for_summarization_cnndm = args.num_of_documents_for_summarization_cnndm
    num_of_documents_for_revision_cnndm = args.num_of_documents_for_revision_cnndm
    seed = args.seed
    path_to_documents_for_summarization_indices_xsum = f'experiments/xsum_4_sets_experiment/datasets_splits/' \
                                                       f'xsum_summarization_{num_of_documents_for_summarization_xsum}_revision_{num_of_documents_for_revision_xsum}_seed_{seed}.json'
    path_to_documents_for_summarization_indices_cnndm = f'experiments/xsum_4_sets_experiment/datasets_splits/' \
                                                        f'cnndm_summarization_{num_of_documents_for_summarization_cnndm}_revision_{num_of_documents_for_revision_cnndm}_seed_{seed}.json'
    train_dataset_xsum = split_xsum_dataset(split='train_model',
                                            path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_xsum,
                                            num_of_documents_for_summarization=num_of_documents_for_summarization_xsum,
                                            num_of_documents_for_revision=num_of_documents_for_revision_xsum, seed=seed)
    val_dataset_xsum = split_xsum_dataset(split='validation_model',
                                          path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_xsum,
                                          num_of_documents_for_summarization=num_of_documents_for_summarization_xsum,
                                          num_of_documents_for_revision=num_of_documents_for_revision_xsum, seed=seed)
    train_dataset_cnndm = split_cnndm_dataset(split='train_model',
                                              path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_cnndm,
                                              num_of_documents_for_summarization=num_of_documents_for_summarization_cnndm,
                                              num_of_documents_for_revision=num_of_documents_for_revision_cnndm,
                                              seed=seed)
    val_dataset_cnndm = split_cnndm_dataset(split='validation_model',
                                            path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_cnndm,
                                            num_of_documents_for_summarization=num_of_documents_for_summarization_cnndm,
                                            num_of_documents_for_revision=num_of_documents_for_revision_cnndm,
                                            seed=seed)
    train_dataset = train_dataset_cnndm + train_dataset_xsum
    val_dataset = concatenate_datasets([val_dataset_xsum, val_dataset_cnndm], axis=0)
    os.environ["WANDB_DISABLED"] = "true"
    train(train_dataset, val_dataset, dataset='both', args=args)


def evaluate_baseline_model(args):
    device = args.device
    model_path = args.model_checkpoints_dir
    if args.checkpoint_path is None:
        model_path = os.path.join(model_path, get_latest_directory(model_path))
        model_path = os.path.join(model_path, get_latest_directory(model_path))
    else:
        model_path = os.path.join(model_path, args.checkpoint_path)
    if args.dataset == 'xsum':
        num_of_documents_for_summarization_xsum = args.num_of_documents_for_summarization_xsum
        num_of_documents_for_revision_xsum = args.num_of_documents_for_revision_xsum
        seed = args.seed
        path_to_documents_for_summarization_indices = f"experiments/data/datasets_splits/xsum_summarization" \
                                                      f"_{num_of_documents_for_summarization_xsum}_revision_{num_of_documents_for_revision_xsum}_seed_{seed}.json"
        test_dataset = split_xsum_dataset(split='factuality_test',
                                          path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                          num_of_documents_for_summarization=num_of_documents_for_summarization_xsum,
                                          num_of_documents_for_revision=num_of_documents_for_revision_xsum, seed=seed)

        texts = [row['text'] for row in test_dataset]
        original_summaries = [row['summary'] for row in test_dataset]
        test_datasets = ['xsum'] * len(texts)
    elif args.dataset == 'cnndm':
        num_of_documents_for_summarization_cnndm = args.num_of_documents_for_summarization_cnndm
        num_of_documents_for_revision_cnndm = args.num_of_documents_for_revision_cnndm
        seed = args.seed
        path_to_documents_for_summarization_indices = f"experiments/data/datasets_splits/cnndm_summarization" \
                                                      f"_{num_of_documents_for_summarization_cnndm}_revision_{num_of_documents_for_revision_cnndm}_seed_{seed}.json"
        test_dataset = split_cnndm_dataset(split='factuality_test',
                                           path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                           num_of_documents_for_summarization=num_of_documents_for_summarization_cnndm,
                                           num_of_documents_for_revision=num_of_documents_for_revision_cnndm,
                                           seed=seed)
        texts = [row['text'] for row in test_dataset]
        original_summaries = [row['summary'] for row in test_dataset]
        test_datasets = ['cnndm'] * len(texts)
    elif args.dataset == 'both':
        num_of_documents_for_summarization_xsum = args.num_of_documents_for_summarization_xsum
        num_of_documents_for_revision_xsum = args.num_of_documents_for_revision_xsum
        num_of_documents_for_summarization_cnndm = args.num_of_documents_for_summarization_cnndm
        num_of_documents_for_revision_cnndm = args.num_of_documents_for_revision_cnndm
        seed = args.seed
        path_to_documents_for_summarization_indices_xsum = f"experiments/xsum_4_sets_experiment/datasets_splits/xsum_summarization" \
                                                           f"_{num_of_documents_for_summarization_xsum}_revision_{num_of_documents_for_revision_xsum}_seed_{seed}.json"
        path_to_documents_for_summarization_indices_cnndm = f"experiments/xsum_4_sets_experiment/datasets_splits/cnndm_summarization" \
                                                            f"_{num_of_documents_for_summarization_cnndm}_revision_{num_of_documents_for_revision_cnndm}_seed_{seed}.json"

        test_dataset_xsum = split_xsum_dataset(split='factuality_test',
                                               path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_xsum,
                                               num_of_documents_for_summarization=num_of_documents_for_summarization_xsum,
                                               num_of_documents_for_revision=num_of_documents_for_revision_xsum,
                                               seed=seed)
        texts = [row['text'] for row in test_dataset_xsum]
        original_summaries = [row['summary'] for row in test_dataset_xsum]
        test_datasets = ['xsum'] * len(texts)
        test_dataset_cnndm = split_cnndm_dataset(split='factuality_test',
                                                 path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_cnndm,
                                                 num_of_documents_for_summarization=num_of_documents_for_summarization_cnndm,
                                                 num_of_documents_for_revision=num_of_documents_for_revision_cnndm,
                                                 seed=seed)
        texts += [row['text'] for row in test_dataset_cnndm]
        original_summaries += [row['summary'] for row in test_dataset_cnndm]
        test_datasets += ['cnndm'] * len(test_dataset_cnndm)
    else:
        raise ValueError("No such dataset")
    if device == 'auto':
        model = T5ForConditionalGeneration.from_pretrained(model_path, device_map=device)
        device = 'cuda'
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    test_summaries = t5_summarize(texts=texts, model=model, tokenizer=tokenizer, prompt='summarize: ',
                                  device=device,
                                  batch_size=args.test_batch_size,
                                  max_generation_length=args.max_generation_length,
                                  beam_size=args.beam_size,
                                  early_stopping=True, length_penalty=args.length_penalty,
                                  max_encoding_length=args.encoder_max_length)
    del model
    time.sleep(5)
    gc.collect()
    torch.cuda.empty_cache()
    fragments_metric = Fragments()
    extractivness_scores = fragments_metric.score(metrics=['density', 'coverage'], texts=texts,
                                                  summaries=test_summaries)
    print("density: ", np.mean(extractivness_scores['density']))
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=test_summaries, references=original_summaries,
                                        use_aggregator=False)
    print("rougeL: ", np.mean(rouge_scores['rougeL']))
    df = pd.DataFrame.from_dict(
        {'dataset': test_datasets, 'summary': test_summaries, 'density': extractivness_scores['density'],
         'coverage': extractivness_scores['coverage']})
    for key in rouge_scores.keys():
        df[key] = rouge_scores[key]
    df.to_csv(os.path.join(args.output_dir, args.output_path))
    test_scores = score(texts=texts, summaries=test_summaries, metrics=['trueteacher'])['trueteacher']
    df['trueteacher'] = test_scores
    df.to_csv(os.path.join(args.output_dir, args.output_path))
    print("density: ", df['density'].mean())
    print("coverage: ", df['coverage'].mean())
    print("trueteacher: ", df['trueteacher'].mean())
    print("rougeL: ", df['rougeL'].mean())


def main():
    args = parserargs()
    print(args)
    train_xsum(args)
    # evaluate_baseline_model(args)


if __name__ == "__main__":
    main()

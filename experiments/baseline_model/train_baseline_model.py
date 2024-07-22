import os
import sys
import evaluate
import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from general.t5_trainer import T5_Trainer, SummarizationDataset
from experiments.data.datasets_splits import split_xsum_dataset, split_cnndm_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import os
from datetime import datetime
import numpy as np
import torch
from datasets import concatenate_datasets
import argparse
from nltk.tokenize import word_tokenize
import gc
import wandb
from experiments.scoring import score
from general.fragments_metrics import Fragments


def parserargs():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default='xsum')
    args.add_argument("--model", type=str, default='t5-base')
    args.add_argument("--train_batch_size_per_device", type=int, default=4)
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
    args.add_argument("--num_of_documents_for_summarization_xsum", type=int, default=0)
    args.add_argument("--num_of_documents_for_revision_xsum", type=int, default=50000)
    args.add_argument("--num_of_documents_for_summarization_cnndm", type=int, default=20000)
    args.add_argument("--num_of_documents_for_revision_cnndm", type=int, default=10000)
    args.add_argument("--model_checkpoints_dir", type=str, default="experiments/baseline_model/checkpoints")
    args.add_argument("--checkpoint_path", type=str, default=None)
    args.add_argument("--evaluation_strategy", type=str, default='steps')
    args.add_argument("--save_strategy", type=str, default='steps')
    args.add_argument('--optim', type=str, default='adamw_torch')
    args.add_argument('--resume_from_checkpoint', action='store_true')
    args.add_argument('--checkpoints_path', type=str, default=None)
    args.add_argument('--seahorse_train_xsum_path', type=str,
                      default='experiments/ablations/ctrl/data/scored_xsum_train_dataset.csv')
    args.add_argument('--train_factuality_threshold', type=float)
    args.add_argument('--data_path', type=str)
    args.add_argument('--density_threshold', type=float)
    args.add_argument('--wandb', action='store_true')
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
    scores = fragments_metric.score(metrics=['density', 'coverage'], texts=dev_texts, summaries=predictions)
    results['density'] = np.mean(scores['density'])
    results['coverage'] = np.mean(scores['coverage'])
    results['length'] = np.mean([len(word_tokenize(x)) for x in predictions])
    return results


def collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + row['text'] for row in batch]
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
    generation_config.num_beams = args.beam_size
    model.generation_config = generation_config
    train_args = Seq2SeqTrainingArguments(
        output_dir=checkpoints_path,
        do_train=True, do_eval=not val_dataset is None,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.eval_batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, save_total_limit=args.save_limit,
        load_best_model_at_end=False, evaluation_strategy=args.evaluation_strategy, save_strategy=args.save_strategy,
        eval_steps=args.eval_steps, save_steps=args.eval_steps, eval_accumulation_steps=30,
        predict_with_generate=True,
        optim=args.optim, overwrite_output_dir=False, logging_steps=0.01)
    max_length_train = args.encoder_max_length
    if val_dataset is not None:
        val_texts = [row['text'] for row in val_dataset]
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer, val_texts),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        if args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()
    else:
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             max_length_train=max_length_train)
        trainer.train()
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()


def train_from_llm(args):
    df = pd.read_csv(args.data_path, index_col=0)
    texts = df['text'].tolist()
    summaries = df['model_summary_llm']
    if args.train_factuality_threshold is not None:
        if 'model_summary_llm_seahorse' not in df.columns:
            df['model_summary_llm_seahorse'] = score(texts, summaries, metrics=['seahorse'])['seahorse']
        df = df[df['model_summary_llm_seahorse'] >= args.train_factuality_threshold]
    if args.density_threshold is not None:
        if 'model_summary_llm_density' not in df.columns:
            fragments = Fragments()
            df['model_summary_llm_density'] = fragments.score(metrics=['density'], texts=texts, summaries=summaries)[
                'density']
        df = df[df['model_summary_llm_density'] <= args.density_threshold]
    texts = df['text'].tolist()
    summaries = df['model_summary_llm'].tolist()
    dataset = SummarizationDataset(texts, summaries)
    train(dataset, None, 'xsum', args)


def train_xsum_filtered_for_factuality(args):
    seahorse_scores_for_train = pd.read_csv(args.seahorse_train_xsum_path, index_col=0)
    chosen_indices = seahorse_scores_for_train[
        seahorse_scores_for_train['seahorse_score'] > args.train_factuality_threshold]['indices'].tolist()
    num_of_documents_for_summarization = args.num_of_documents_for_summarization_xsum
    num_of_documents_for_revision = args.num_of_documents_for_revision_xsum
    seed = args.seed
    path_to_documents_for_summarization_indices = f'experiments/data/datasets_splits/xsum_summarization_{num_of_documents_for_summarization}' \
                                                  f'_revision_{num_of_documents_for_revision}_seed_{seed}.json'
    train_dataset = split_xsum_dataset(split='train_model',
                                       path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                       num_of_documents_for_summarization=num_of_documents_for_summarization,
                                       num_of_documents_for_revision=num_of_documents_for_revision, seed=seed,
                                       additional_train_filter_indices=chosen_indices)
    val_dataset = split_xsum_dataset(split='validation_model',
                                     path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                     num_of_documents_for_summarization=num_of_documents_for_summarization,
                                     num_of_documents_for_revision=num_of_documents_for_revision, seed=seed)
    # os.environ["WANDB_DISABLED"] = "true"
    print(f"Training on {len(train_dataset)} documents")
    train(train_dataset, val_dataset, dataset='xsum', args=args)


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
    # os.environ["WANDB_DISABLED"] = "true"
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

    # os.environ["WANDB_DISABLED"] = "true"
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
    # os.environ["WANDB_DISABLED"] = "true"
    train(train_dataset, val_dataset, dataset='both', args=args)


def main():
    args = parserargs()
    if args.wandb:
        wandb.init(project='baseline_model')
    else:
        os.environ["WANDB_DISABLED"] = "true"
    print(args)
    # train_xsum_filtered_for_factuality(args)
    train_from_llm(args)
    if args.wandb:
        wandb.finish()
    # evaluate_baseline_model(args)


if __name__ == "__main__":
    main()

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
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset, split_cnndm_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import os
from datetime import datetime
from general.metrics import Rouge
import numpy as np
import torch
from datasets import concatenate_datasets
import argparse
def parserargs():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default='xsum')
    args.add_argument("--model", type=str, default='t5-base')
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--max_length", type=int, default=512)
    args.add_argument("--num_train_epochs", type=int, default=1)
    args.add_argument("--learning_rate", type=float, default=1e-3)
    args.add_argument("--gradient_accumulation_steps", type=int, default=2)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--num_of_documents_for_summarization_xsum", type=int, default=20000)
    args.add_argument("--train_size", type=float, default=1)
    args.add_argument("--test_save_path", type=str)
    args.add_argument("--test_save_dir", type=str, default="experiments/ablations/better_summaries/outputs")

def compute_metrics(p, tokenizer,dev_texts):
    rouge_metric = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge_metric.compute(predictions=predictions, references=labels)
    from general.fragments_metrics import Fragments
    fragments_metric = Fragments()
    density_scores = fragments_metric.score(metrics=['density'], texts=dev_texts, summaries=predictions)['density']
    results['density'] = np.mean(density_scores)
    from nltk.tokenize import word_tokenize
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


def train(train_dataset, val_dataset, dataset):
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")

    generation_config = model.generation_config
    generation_config.max_length = 128
    generation_config.early_stopping = True
    generation_config.num_beams = 4
    generation_config.length_penalty = 0.6
    model.generation_config = generation_config
    #model_params = model.config.task_specific_params
    #model_params['summarization']['length_penalty'] = 0.6
    #model_params['summarization']['max_length'] = 128
    #model.config.task_specific_params =  model_params

    # generation_config = GenerationConfig(max_length=128, min_length=0, early_stopping=True,
    #                                      num_beams=4, no_repeat_ngram_size=3, length_penalty=0.6)
    # model.config.task_specific_params['summarization']['length_penalty'] = 0.6
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    args = Seq2SeqTrainingArguments(
        output_dir=f'experiments/xsum_4_sets_experiment/checkpoints/t5_base_{dataset}_{run_name}',
        do_train=True, do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3, num_train_epochs=5, save_total_limit=2,optim = 'adafactor',
        load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
        eval_steps=0.099, save_steps=0.099, eval_accumulation_steps=30,
        metric_for_best_model='rougeL', no_cuda=False, predict_with_generate=True,logging_steps=0.01)
    val_texts = [row['text'] for row in val_dataset]
    max_length_train = 512
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=args,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer,val_texts),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    del trainer
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def train_xsum():
    #args = parserargs()
    num_of_documents_for_summarization = 20000
    seed = 42
    path_to_documents_for_summarization_indices = f'experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_{num_of_documents_for_summarization}_indices_seed_{seed}.pkl'
    train_dataset = split_xsum_dataset(split='train_model',
                                       path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                       num_of_documents_for_summarization=num_of_documents_for_summarization, seed=seed)
    val_dataset = split_xsum_dataset(split='validation_model',
                                     path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                     num_of_documents_for_summarization=num_of_documents_for_summarization, seed=seed)
    os.environ["WANDB_DISABLED"] = "true"
    train(train_dataset, val_dataset, dataset='xsum')


def train_cnndm():
    args = parserargs()
    num_of_documents_for_summarization = 20000
    seed = 42
    path_to_documents_for_summarization_indices = f'experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_{num_of_documents_for_summarization}_indices_seed_{seed}.pkl'
    train_dataset = split_cnndm_dataset(split='train_model',
                                        path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                        num_of_documents_for_summarization=num_of_documents_for_summarization,
                                        seed=seed)
    val_dataset = split_cnndm_dataset(split='validation_model',
                                      path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                      num_of_documents_for_summarization=num_of_documents_for_summarization, seed=seed)

    os.environ["WANDB_DISABLED"] = "true"
    train(train_dataset, val_dataset, dataset='cnndm')


def train_both():
    args = parserargs()
    num_of_documents_for_summarization_xsum = 10000
    num_of_documents_for_summarization_cnndm = 10000
    seed = 42
    path_to_documents_for_summarization_indices_xsum = f'experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_{num_of_documents_for_summarization_xsum}_indices_seed_{seed}.pkl'
    path_to_documents_for_summarization_indices_cnndm = f'experiments/xsum_4_sets_experiment/datasets_splits/cnndm_docs_for_summarization_{num_of_documents_for_summarization_cnndm}_indices_seed_{seed}.pkl'
    train_dataset_xsum = split_xsum_dataset(split='train_model',
                                            path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_xsum,
                                            num_of_documents_for_summarization=num_of_documents_for_summarization_xsum,
                                            seed=seed)
    val_dataset_xsum = split_xsum_dataset(split='validation_model',
                                          path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_xsum,
                                          num_of_documents_for_summarization=num_of_documents_for_summarization_xsum,
                                          seed=seed)
    train_dataset_cnndm = split_cnndm_dataset(split='train_model',
                                              path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_cnndm,
                                              num_of_documents_for_summarization=num_of_documents_for_summarization_cnndm,
                                              seed=seed)
    val_dataset_cnndm = split_cnndm_dataset(split='validation_model',
                                            path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_cnndm,
                                            num_of_documents_for_summarization=num_of_documents_for_summarization_cnndm,
                                            seed=seed)
    train_dataset = train_dataset_cnndm + train_dataset_xsum
    val_dataset = concatenate_datasets([val_dataset_xsum, val_dataset_cnndm], axis=0)
    os.environ["WANDB_DISABLED"] = "true"
    train(train_dataset, val_dataset, dataset='both')



def get_latest_directory(path):
    # Get all directories in the given path
    all_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    if not all_dirs:
        print("No directories found.")
        return

    # Get the creation time of each directory
    dir_creation_times = {d: os.path.getctime(os.path.join(path, d)) for d in all_dirs}

    # Find the directory with the latest creation time
    latest_directory = max(dir_creation_times, key=dir_creation_times.get)

    return os.path.join(path, latest_directory)


def summarize_and_score():
    device = 'cuda:1'
    path = 'experiments/xsum_4_sets_experiment/checkpoints'
    model_dir = get_latest_directory(path)
    model_path = get_latest_directory(model_dir)
    # model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_xsum_26_12_2023_16_03_37/checkpoint-15534"
    path_to_documents_for_summarization_indices = "experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl"
    # model = T5ForConditionalGeneration.from_pretrained(model_path)
    # tokenizer = T5Tokenizer.from_pretrained(model_path)
    from transformers import AutoTokenizer, BartForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained("morenolq/bart-base-xsum")
    model = BartForConditionalGeneration.from_pretrained("morenolq/bart-base-xsum").to(device)
    test_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                      num_of_documents_for_summarization=20000, seed=42)
    test_texts = [row['text'] for row in test_dataset]
    test_original_summaries = [row['summary'] for row in test_dataset]
    test_summaries = t5_summarize(texts=test_texts, model=model, tokenizer=tokenizer, prompt='summarize: ',
                                  device=device,
                                  batch_size=32, max_generation_length=128, beam_size=4,
                                  early_stopping=True, length_penalty=0.6)
    del model
    time.sleep(30)
    gc.collect()
    torch.cuda.empty_cache()
    from general.fragments_metrics import Fragments
    fragments_metric = Fragments()
    extractivness_scores = fragments_metric.score(metrics=['density', 'coverage'], texts=test_texts,
                                                  summaries=test_summaries)
    print("density: ", np.mean(extractivness_scores['density']))
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=test_summaries, references=test_original_summaries)
    print("rougeL: ", np.mean(rouge_scores['rougeL']))
    from TrueTeacher.inference import TrueTeacher
    factuality_metric = TrueTeacher(model_path="google/t5_11b_trueteacher_and_anli",
                                    tokenizer_name="google/t5_11b_trueteacher_and_anli", device="auto"
                                    , batch_size=1, max_length=2048, torch_dtype=torch.float16,
                                    return_none=True)
    test_scores = factuality_metric.score(texts=test_texts, summaries=test_summaries)
    del factuality_metric
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=test_summaries, references=test_original_summaries,
                                        use_aggregator=False)
    df = pd.DataFrame.from_dict(
        {'summary': test_summaries,
         'trueteacher': test_scores, 'density': extractivness_scores['density'],
         'coverage': extractivness_scores['coverage']})
    for key in rouge_scores.keys():
        df[key] = rouge_scores[key]

    df.to_csv(
        'experiments/xsum_4_sets_experiment/outputs/documents_for_summarization_true_teacher_scored_two_epoch.csv')
    print("density: ",df['density'].mean())
    print("coverage: ", df['coverage'].mean())
    print("trueteacher: ", df['trueteacher'].mean())
    print("rougeL: ", df['rougeL'].mean())


def main():
    # summarize_and_score()
    train_xsum()
    #summarize_and_score()
    # train_cnndm()
    # train_both()


if __name__ == "__main__":
    main()
    # check_model()

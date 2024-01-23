import json

import torch
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import pandas as pd
from datetime import datetime
from general.t5_trainer import T5_Trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import evaluate
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
import time
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
import pickle
import gc


class SummariesDataset(torch.utils.data.Dataset):
    def __init__(self, texts, summaries):
        self.texts = texts
        self.summaries = summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'summary': self.summaries[idx]}


def create_dataset(texts, summaries):
    dataset = SummariesDataset(texts, summaries)
    print(f"Train dataset size {len(dataset)}")
    return dataset


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
    summaries = [row['summary'] for row in batch]
    texts_inputs = ["summarize: " + row['text'] for row in batch]
    inputs = tokenizer(texts_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def train(texts, summaries, args):
    torch.cuda.empty_cache()
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
    train_dataset = create_dataset(texts, summaries)

    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.environ["WANDB_DISABLED"] = "true"
    model_checkpoint = args.model_checkpoint
    models_dir = args.model_dir
    model_path = os.path.join(models_dir, model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    train_args = Seq2SeqTrainingArguments(
        output_dir=f'experiments/ablations/better_summaries/runs/{run_name}',
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


def collate_fn_test(batch, tokenizer, max_length):
    texts_inputs = ["summarize: " + row['text'] for row in batch]
    inputs = tokenizer(texts_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


def test(model, tokenizer, args):
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    batch_size = args.test_batch_size
    encoding_max_length = args.test_max_encoding_length
    beam_size = args.beam_size
    generation_max_length = args.generation_max_length
    xsum_dataloader = DataLoader(dataset=xsum_dataset, batch_size=batch_size,
                                 collate_fn=lambda x: collate_fn_test(x, tokenizer, encoding_max_length))
    if args.device == 'cpu':
        device = 'cpu'
    elif args.device == 'auto':
        device = 'cuda'
    else:
        device = args.device
    model.eval()
    xsum_predictions = []
    with torch.no_grad():
        print("xsum")
        for batch in tqdm(xsum_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            batch_predictions = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                               max_length=generation_max_length, num_beams=beam_size,
                                               early_stopping=True)
            batch_predictions = tokenizer.batch_decode(batch_predictions, skip_special_tokens=True)
            xsum_predictions.extend(batch_predictions)
    del model
    return xsum_predictions


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
    args.add_argument("--model_dir", type=str, default="experiments/xsum_4_sets_experiment/checkpoints")
    args.add_argument("--test_batch_size", type=int, default=32)
    args.add_argument("--test_max_encoding_length", type=int, default=512)
    args.add_argument("--beam_size", type=int, default=4)
    args.add_argument("--generation_max_length", type=int, default=128)
    args.add_argument("--test_save_path", type=str)
    args.add_argument("--test_save_dir", type=str, default="experiments/ablations/better_summaries/outputs")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--rouge_threshold", type=float, default=0.7)
    args.add_argument("--classifier_threshold", type=float, default=0.5)
    args.add_argument("--diff_threshold", type=float, default=0.4)
    args.add_argument('--device', type=str, default='auto')
    args.add_argument('--method', type=str)
    return args.parse_args()


def create_summaries(args):
    method = args.method
    df = pd.read_csv("experiments/ablations/better_summaries/data/gpt_3.5_summaries_with_indices.csv")
    if method == 'all':
        df = df
    elif method == 'chosen':
        with open("experiments/ablations/better_summaries/data/chosen_xsum_indices.pkl", 'rb') as file:
            chosen_indices = pickle.load(file)
        df = df[df['indices'].isin(chosen_indices)]
    else:
        raise ValueError("No such method")
    print("Train size: ", len(df))
    texts = df['text'].tolist()
    summaries = df['summary'].tolist()
    test_save_path = args.test_save_path
    test_save_dir = args.test_save_dir
    model, tokenizer = train(texts, summaries, args)
    predictions = test(model, tokenizer, args)
    results = {'hyperparameters': args.__dict__, 'predictions': predictions, 'method': method}
    with open(test_save_dir + '/' + test_save_path, 'w') as f:
        json.dump(results, f)
    with open(
            "/data/home/yehonatan-pe/Correction_pipeline/experiments/ablations/better_summaries/dump" + '/' + test_save_path,
            'w') as f:
        json.dump(results, f)
    return results


def score_factuality(texts, summaries, metrics):
    from Seahorse_metrics.metrics import Seahorse_metrics
    from TrueTeacher.inference import TrueTeacher
    results = {}
    for metric in metrics:
        if 'seahorse' in metrics:
            factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                                 tokenizer_name='google/seahorse-xxl-q4',
                                                 device='auto', batch_size=1, torch_dtype=torch.float16,
                                                 max_length=2048, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['seahorse'] = scores
            del factuality_metric
        elif 'teacher' in metric:
            factuality_metric = TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                                            tokenizer_name="google/t5_11b_trueteacher_and_anli",
                                            device='auto', batch_size=1, max_length=2048,
                                            torch_dtype=torch.float16, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['trueteacher'] = scores
            del factuality_metric
        elif 'nli' in metric:
            from nli.nli_metric import NLI
            factuality_metric = NLI(batch_size=1, torch_dtype=torch.bfloat16, max_length=2048, device='auto',
                                    return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['nli'] = scores
            del factuality_metric
        elif 'q_squared' in metric:
            from q_squared.run_nli import scores_with_nli, aggregate_per_response
            from q_squared.prep_sys_experiment import cross_annotated_scores

            df = cross_annotated_scores(texts, summaries, out_path=None, save=False)
            df = scores_with_nli(in_path=None, df=df)
            df = aggregate_per_response(df=df, out_path=None, save=False)
            results['Q2'] = df['Q2'].tolist()
        gc.collect()
        time.sleep(30)
        torch.cuda.empty_cache()
    return results


def non_factuality_metrics(texts, summaries, original_summaries):
    results = {}
    fragment_metric = Fragments()
    extractive_results = fragment_metric.score(metrics=['density', 'coverage'], texts=texts, summaries=summaries)
    for key in extractive_results:
        results[key] = extractive_results[key]
    results['length'] = [len(word_tokenize(x)) for x in summaries]
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(predictions=summaries, references=original_summaries, use_aggregator=False)
    for key in rouge_results:
        results[key] = rouge_results[key]
    return results


def main():
    args = args_parser()
    create_summaries(args)
    with open(args.test_save_dir + '/' + args.test_save_path, 'r') as f:
        results = json.load(f)
    xsum_test_set = split_xsum_dataset(split='factuality_test',
                                       path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                       num_of_documents_for_summarization=20000,
                                       seed=42)
    texts = [xsum_test_set[i]['text'] for i in range(len(xsum_test_set))]
    original_summaries = [xsum_test_set[i]['summary'] for i in range(len(xsum_test_set))]
    summaries = results['predictions']
    non_factuality_results = non_factuality_metrics(texts=texts, summaries=summaries,
                                                    original_summaries=original_summaries)
    for key in non_factuality_results.keys():
        results[key] = non_factuality_results[key]
    with open(args.test_save_dir + '/' + args.test_save_path, 'w') as f:
        json.dump(results, f)
    true_teacher_results = score_factuality(texts=texts, summaries=summaries, metrics=['trueteacher'])
    for key in true_teacher_results.keys():
        results[key] = true_teacher_results[key]
    with open(args.test_save_dir + '/' + args.test_save_path, 'w') as f:
        json.dump(results, f)
    q_squared_results = score_factuality(texts=texts, summaries=summaries, metrics=['q_squared'])
    for key in q_squared_results.keys():
        results[key] = q_squared_results[key]
    with open(args.test_save_dir + '/' + args.test_save_path, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()

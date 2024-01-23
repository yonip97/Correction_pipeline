import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from experiments.data.datasets_splits import split_xsum_dataset
from experiments.data.datasets_splits import split_cnndm_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import t5_summarize, t5_revise, T5_Trainer, collate_fn_revision, collate_fn_revision_test, \
    compute_metric_rouge
import time
import argparse
from datetime import datetime
from Seahorse_metrics.metrics import Seahorse_metrics
import evaluate
from general.utils import RevisionDataset, SummarizationDataset
from experiments.scoring import score
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize



def create_ood_dataset(args):
    df = pd.read_csv(args.ood_test_path, index_col=0)
    texts = df['text'].tolist()
    model_summaries = df['model_summary'].tolist()
    original_summaries = df['original_summary'].tolist()
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(predictions=model_summaries, references=original_summaries, use_aggregator=False)
    for key in rouge_results.keys():
        df[f'pre_revision_{key}'] = rouge_results[key]
    fragments_metric = Fragments()
    fragments_results = fragments_metric.score(texts=texts, summaries=model_summaries, metrics=['density', 'coverage'])
    for key in fragments_results.keys():
        df[f'pre_revision_{key}'] = fragments_results[key]
    if args.ood_factuality_metrics_pre_revision is not None:
        factuality_metrics = args.ood_factuality_metrics_pre_revision.split(',')
        if args.ood_score_pre_revision:
            scores = score(texts=texts, summaries=model_summaries, metrics=factuality_metrics)
            for key in scores:
                df[f'pre_revision_{key}_score'] = scores[key]
    return df



def get_data_to_train_revision_model(args):
    df = pd.read_csv(args.train_data_path, index_col=0)
    if 'pre_revision_factuality_score' not in df.columns:
        pre_revision_scores = \
            score(texts=df['text'].tolist(), summaries=df['model_summary'].tolist(), metrics=['seahorse'])['seahorse']
        df['pre_revision_factuality_score'] = pre_revision_scores
        df.to_csv(args.train_data_path)
    if 'post_revision_factuality_score' not in df.columns:
        post_revision_scores = \
            score(texts=df['text'].tolist(), summaries=df['revised_summary'].tolist(), metrics=['seahorse'])['seahorse']
        df['post_revision_factuality_score'] = post_revision_scores
        df.to_csv(args.train_data_path)
    if 'pre_revision_density' not in df.columns:
        metric = Fragments()
        pre_revision_density = metric.score(texts=df['text'].tolist(),
                                            summaries=df['model_summary'].tolist(),
                                            metrics=['density'])['density']
        df['pre_revision_density'] = pre_revision_density
        df.to_csv(args.train_data_path)
    if 'post_revision_density' not in df.columns:
        metric = Fragments()
        post_revision_density = metric.score(texts=df['text'].tolist(),
                                             summaries=df['revised_summary'].tolist(),
                                             metrics=['density'])['density']
        df['post_revision_density'] = post_revision_density
        df.to_csv(args.train_data_path)
    return df['text'].tolist(), df['model_summary'].tolist(), df['revised_summary'].tolist(), df[
        'pre_revision_factuality_score'].tolist(), df['post_revision_factuality_score'].tolist(), df[
               'pre_revision_density'].tolist(), df['post_revision_density'].tolist()


def create_datasets_for_revision_model(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores,
                                       pre_revision_density, post_revision_density, args):
    method = args.method
    revised_df = pd.DataFrame.from_dict(
        {'text': texts, 'model_summary': summaries, 'revised_summary': revised_summaries,
         'pre_revision_factuality_score': pre_revision_scores,
         'post_revision_factuality_score': post_revision_scores,
         'pre_revision_density': pre_revision_density,
         'post_revision_density': post_revision_density})
    if 'classifier' in method:
        revised_df = revised_df[
            (revised_df['pre_revision_factuality_score'] < args.factuality_threshold) & (
                    revised_df['post_revision_factuality_score'] >= args.factuality_threshold)]
    if 'rouge_threshold' in method:
        rouge_metric = evaluate.load('rouge')
        revised_df['rougeL'] = \
            rouge_metric.compute(predictions=revised_df['revised_summary'], references=revised_df['model_summary'],
                                 use_aggregator=False)['rougeL']
        revised_df = revised_df[revised_df['rougeL'] > args.rouge_threshold]
    if 'factuality_diff' in method:
        revised_df = revised_df[
            revised_df['post_revision_factuality_score'] - revised_df['pre_revision_factuality_score'] >=
            args.factuality_diff_threshold]
    texts = revised_df['text'].tolist()
    original_summaries = revised_df['model_summary'].tolist()
    revised_summaries = revised_df['revised_summary'].tolist()
    np.random.seed(args.seed)
    if args.train_size != 1 and args.train_size != 0:
        train_indices = np.random.choice(len(texts), int(len(texts) * args.train_size), replace=False)
        val_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
        train_dataset = RevisionDataset(np.array(texts)[train_indices].tolist(),
                                        np.array(original_summaries)[train_indices].tolist(),
                                        np.array(revised_summaries)[train_indices].tolist())
        val_dataset = RevisionDataset(np.array(texts)[val_indices].tolist(),
                                      np.array(original_summaries)[val_indices].tolist(),
                                      np.array(revised_summaries)[val_indices].tolist())
        return train_dataset, val_dataset
    elif args.train_size == 1:
        train_dataset = RevisionDataset(texts, original_summaries, revised_summaries)
        return train_dataset, None
    else:
        val_dataset = RevisionDataset(texts, original_summaries, revised_summaries)
    return None, val_dataset

#
# def get_data_for_revision_model_evaluation(args):
#     df = pd.read_csv(args.train_data_path, index_col=0)
#     for col in df.columns:
#         if 'factuality_score' == col:
#             model_summaries = df['model_summary'].tolist()
#             original_summaries = df['original_summary'].tolist()
#             texts = df['text'].tolist()
#             scores = df['factuality_score'].tolist()
#             return texts, original_summaries, model_summaries, scores
#     texts = df['text'].tolist()
#     model_summaries = df['model_summary'].tolist()
#     original_summaries = df['original_summary'].tolist()
#     factuality_scores = score(texts=texts, summaries=model_summaries, metrics=['seahorse'])['seahorse']
#     return texts, original_summaries, model_summaries, factuality_scores


def parseargs_train_revision_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_path')
    parser.add_argument('-train_data_path', type=str)
    parser.add_argument('-model_path', type=str, default='google/flan-t5-large')
    parser.add_argument('-checkpoint_path', type=str)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-train_batch_size', type=int, default=1)
    parser.add_argument('-eval_batch_size', type=int, default=1)
    parser.add_argument('-evaluation_strategy', type=str, default='steps')
    parser.add_argument('-save_strategy', type=str, default='steps')
    parser.add_argument('-gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-method', type=str, default='diff')
    parser.add_argument('-dataset', type=str, default='xsum')
    parser.add_argument('-train_size', type=float, default=1)
    parser.add_argument('-num_beam', type=int, default=4)
    parser.add_argument('-max_generation_length', type=int, default=128)
    parser.add_argument('-max_encoding_length', type=int, default=1024)
    parser.add_argument('-factuality_threshold', type=float, default=0.5)
    parser.add_argument('-factuality_diff_threshold', type=float, default=0.1)
    parser.add_argument('-rouge_threshold', type=float, default=0.5)
    parser.add_argument('-density_threshold', type=float, default=0.5)
    parser.add_argument('-evaluate_ood', action='store_true')
    parser.add_argument('-ood_score_pre_revision', action='store_true')
    parser.add_argument('-ood_test_path', type=str)
    parser.add_argument('-ood_factuality_metrics_pre_revision', type=str)
    parser.add_argument('-ood_factuality_metrics_post_revision', type=str)
    args = parser.parse_args()
    return args


# def parseargs_revise_model_outputs():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-output_path')
#     parser.add_argument('-model_path', type=str)
#     parser.add_argument('-checkpoint_path', type=str)
#     parser.add_argument('-device', type=str)
#     parser.add_argument('-test_batch_size', type=int, default=1)
#     # parser.add_argument('-num_beam', type=int, default=4)
#     # parser.add_argument('-max_generation_length', type=int, default=128)
#     # parser.add_argument('-max_encoding_length', type=int, default=1024)
#     parser.add_argument('-score', action='store_true')
#     args = parser.parse_args()
#     return args


def train_and_evaluate_revision_model():
    os.environ["WANDB_DISABLED"] = "true"
    args = parseargs_train_revision_model()
    revision_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    if args.train:
        lr = args.lr
        train_batch_size = args.train_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        epochs = args.epochs
        weight_decay = args.weight_decay
        method = args.method
        run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, pred_revision_density, \
        post_revision_density = get_data_to_train_revision_model(args)
        train_dataset, val_dataset = create_datasets_for_revision_model(texts=texts, summaries=summaries,
                                                                        revised_summaries=revised_summaries,
                                                                        pre_revision_scores=pre_revision_scores,
                                                                        post_revision_scores=post_revision_scores,
                                                                        pre_revision_density=pred_revision_density,
                                                                        post_revision_density=post_revision_density,
                                                                        args=args)
        checkpoint_name = f'{args.model_path.split("/")[-1]}_{method}_{run_name}'
        output_dir = f'experiments/revision/checkpoints/{checkpoint_name}'
        with open(output_dir + '/args.txt', 'w') as f:
            f.write(str(args))
        train_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            do_train=True, do_eval=False,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs, predict_with_generate=True, generation_num_beams=args.num_beam,
            generation_max_length=args.max_generation_length,
            evaluation_strategy=args.evaluation_strategy, save_strategy=args.save_strategy, eval_accumulation_steps=30,
            weight_decay=weight_decay,
            metric_for_best_model='rougeL', no_cuda=False, logging_steps=0.01, fp16=True, load_best_model_at_end=True)
        max_length = args.max_encoding_length
        trainer = T5_Trainer(collate_fn=collate_fn_revision, model=revision_model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             compute_metrics=lambda p: compute_metric_rouge(p, tokenizer),
                             max_length_train=max_length, max_length_eval=max_length,
                             collate_fn_test=collate_fn_revision_test)
        trainer.train()
    else:
        train_args = Seq2SeqTrainingArguments(
            output_dir="",
            do_train=True, do_eval=False,
            per_device_eval_batch_size=args.eval_batch_size,
            predict_with_generate=True, generation_num_beams=args.num_beam,
            generation_max_length=args.max_generation_length,eval_accumulation_steps=30,
            no_cuda=False,fp16=True)
        max_length = args.max_encoding_length
        trainer = T5_Trainer(collate_fn=collate_fn_revision, model=revision_model, tokenizer=tokenizer, args=train_args,
                             compute_metrics=lambda p: compute_metric_rouge(p, tokenizer),
                             max_length_train=max_length, max_length_eval=max_length,
                             collate_fn_test=collate_fn_revision_test)
    if args.evaluate_ood:
        df = create_ood_dataset(args)
        texts = df['text'].tolist()
        summaries = df['model_summary'].tolist()
        original_summaries = df['original_summary'].tolist()
        predictions = trainer.predict(test_dataset=SummarizationDataset(texts, summaries))
        rouge_metric = evaluate.load('rouge')
        rouge_scores = rouge_metric.compute(predictions=predictions, references=original_summaries)
        for key in rouge_scores.keys():
            df[f'post_revision_{key}'] = rouge_scores[key]
        rouge_scores = rouge_metric.compute(predictions = predictions,references=summaries)
        for key in rouge_scores.keys():
            df[f'post_revision_{key}_to_base_model_output'] = rouge_scores[key]
        fragments_metric = Fragments()
        fragments_scores = fragments_metric.score(texts=texts, summaries=predictions, metrics=['density', 'coverage'])
        for key in fragments_scores.keys():
            df[f'post_revision_{key}'] = fragments_scores[key]
        df.to_csv(os.path.join(output_dir, f"results_{args.ood_test_path}"))
        if args.ood_factuality_metrics_post_revision is not None:
            factuality_metrics = args.ood_factuality_metrics_post_revision.split(',')
            scores = score(texts=texts, summaries=predictions, metrics=factuality_metrics)
            for key in scores.keys():
                df[f'post_revision_{key}_score'] = scores[key]
            df.to_csv(os.path.join(output_dir, f"results_{args.ood_test_path}"))


# def revise_model_outputs(args):
#     model_state_dict = torch.load(args.checkpoint_path)
#     if args.device == 'auto':
#         revision_model = T5ForConditionalGeneration.from_pretrained(args.model_path, device_map='auto',
#                                                                     torch_dtype=torch.float16)
#         args.device = 'cuda'
#     else:
#         revision_model = T5ForConditionalGeneration.from_pretrained(args.model_path,
#                                                                     torch_dtype=torch.float16).to(args.device)
#     tokenizer = T5Tokenizer.from_pretrained(args.model_path)
#     revision_model.load_state_dict(model_state_dict['model'])
#     texts, original_summaries, model_summaries, scores = get_data_for_revision_model_evaluation(args)
#     revised_summaries = t5_revise(texts=texts, summaries=model_summaries, model=revision_model, tokenizer=tokenizer,
#                                   prompt='revise: ',
#                                   device=args.device,
#                                   batch_size=args.test_batch_size, generation_max_length=args.max_generation_length,
#                                   num_beams=args.num_beams, early_stopping=True,
#                                   encoding_max_length=args.max_encoding_length)
#     df = pd.DataFrame.from_dict(
#         {'text': texts, 'original_summaries': original_summaries, 'model_summary': model_summaries,
#          'revised_summary': revised_summaries,
#          'pre_revision_factuality_score': scores})
#     rouge_metric = evaluate.load('rouge')
#     rouge_scores_to_model_outputs = rouge_metric.compute(predictions=revised_summaries, references=model_summaries)
#     for key in rouge_scores_to_model_outputs.keys():
#         df[f'{key}_to_model_output'] = rouge_scores_to_model_outputs[key]
#     rouge_scores_to_original_summaries = rouge_metric.compute(predictions=revised_summaries,
#                                                               references=original_summaries)
#     for key in rouge_scores_to_original_summaries.keys():
#         df[f'{key}_to_original_summary'] = rouge_scores_to_original_summaries[key]
#     extractivness_metric = Fragments()
#     extractivness_scores_to_model_outputs = extractivness_metric.score(texts=texts, summaries=revised_summaries,
#                                                                        metrics=['density', 'coverage'])
#     for key in extractivness_scores_to_model_outputs.keys():
#         df[f'{key}_to_model_output'] = extractivness_scores_to_model_outputs[key]
#     df['length'] = [len(word_tokenize(summary)) for summary in revised_summaries]
#     if args.score:
#         post_revision_scores = score(texts=texts, summaries=revised_summaries, metrics=['seahorse'])
#         df['post_revision_factuality_score'] = post_revision_scores['seahorse']
#     df.to_csv(args.output_path)


def main():
    # args = parseargs_revise_model_outputs()
    # revise_model_outputs(args)
    train_and_evaluate_revision_model()

if __name__ == '__main__':
    main()

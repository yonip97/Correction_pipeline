import os
import sys
import ast

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from general.t5_trainer import T5_Trainer, revise
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import os
from datetime import datetime
import numpy as np
from Seahorse_metrics.metrics import Seahorse_metrics
import torch
import evaluate
from transformers import AutoTokenizer, AutoModel
from experiments.poc.poc_utils import compute_metrics, collate_fn, SummarizationDataset, load_xsum_ood


def evaluate_factuality(classifier, texts_list, summaries_list):
    all_scores = []
    for texts, summaries in zip(texts_list, summaries_list):
        scores = classifier.score(texts, summaries)
        all_scores.append(scores)
    return all_scores


def evaluate_rouge(texts_list, summaries_list):
    rouge_metric = evaluate.load('rouge')
    all_scores = []
    for texts, summaries in zip(texts_list, summaries_list):
        scores = rouge_metric.compute(predictions=summaries, references=texts)
        all_scores.append(scores)
    return all_scores



def evaluate_on_true(model, tokenizer, factuality_metric):
    from data.factuality_datasets import TRUE_dataset
    rouge_metric = evaluate.load('rouge')
    dataset = TRUE_dataset('data/true_data', ['summarization'])
    df = dataset.df
    for dataset_name in df['dataset'].unique():
        temp_df = df[df['dataset'] == dataset_name]
        for model_name in temp_df['model'].unique():
            temp_df2 = temp_df[temp_df['model'] == model_name]
            texts = temp_df2['grounding'].tolist()
            summaries = temp_df2['generated_text'].tolist()
            model_revisions = revise(texts, summaries, model, tokenizer, device='cuda:1', batch_size=8, max_length=128)
            pre_revision_scores = factuality_metric.score(texts, summaries)
            post_revision_scores = factuality_metric.score(texts, model_revisions)
            rouge_scores = rouge_metric.compute(predictions=model_revisions, references=summaries)
            print(dataset_name)
            print(model_name)
            print(np.mean(pre_revision_scores))
            print(np.mean(post_revision_scores))
            print(rouge_scores)


def evaluate_on_frank(model, tokenizer, factuality_metric):
    df = pd.read_json('data/frank_raw/benchmark_data.json')
    for model_name in df['model_name'].unique():
        temp_df = df[df['model_name'] == model_name]
        texts = temp_df['article'].tolist()
        summaries = temp_df['summary'].tolist()
        model_revisions = revise(texts, summaries, model, tokenizer, device='cuda:1', batch_size=8, max_length=128)
        pre_revision_scores = factuality_metric.score(texts, summaries)
        post_revision_scores = factuality_metric.score(texts, model_revisions)
        print(model_name)
        print(np.mean(pre_revision_scores))
        print(np.mean(post_revision_scores))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--models_dir', type=str, default='experiments/poc/models')
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--calculate_factuality_scores', action='store_true')
    parser.add_argument('--calculate_rouge_scores', action='store_true')
    parser.add_argument('--eval_true', action='store_true')
    parser.add_argument('--eval_frank', action='store_true')
    args = parser.parse_args()
    return args


def train_using_all(texts, summaries, revised_summaries):
    args = parse_args()
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    train_dataset = SummarizationDataset(texts, summaries, revised_summaries)
    ood_test_texts, ood_test_summaries = load_xsum_ood(only_low_score=True)

    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.environ["WANDB_DISABLED"] = "true"

    model_name = args.model_name
    print(f"Using {model_name}")
    models_dir = args.models_dir + '/' + 'use_all/'
    if os.path.exists(models_dir + '/' + model_name + '/model.pkl'):
        print(f"Loading model from {models_dir + '/' + model_name}")
        model = AutoModel.from_pretrained(model_name)
        model.load_state_dict(torch.load(models_dir + '/' + model_name + '/model.pkl'))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_args = Seq2SeqTrainingArguments(
            output_dir=f'experiments/poc/checkpoints/t5_base_{run_name}',
            do_train=True, do_eval=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs,
            evaluation_strategy='no', save_strategy='no', eval_accumulation_steps=30, weight_decay=weight_decay,
            metric_for_best_model='rougeL', no_cuda=False)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), models_dir + '/' + model_name + '/model.pkl')

    results = {}
    if args.calculate_factuality_scores or args.calculate_rouge_scores:
        predictions = revise(texts, summaries, model, tokenizer, device='cuda:1', batch_size=8,
                             max_length=128)
        ood_test_predictions = revise(ood_test_texts, ood_test_summaries, model, tokenizer, device='cuda:1',
                                      batch_size=8,
                                      max_length=128)

        if args.calculate_factuality_scores:
            classifier = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                          device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16)

            scores = classifier.score(texts, predictions)
            ood_test_scores = classifier.score(ood_test_texts, ood_test_predictions)
            results = {}
            results['train_`factuality_score'] = np.mean(scores)
            results['ood_test_factuality_score'] = np.mean(ood_test_scores)
            if args.eval_frank:
                results['frank_eval'] = evaluate_on_frank(model, tokenizer, classifier)
            if args.eval_true:
                results['true_eval'] = evaluate_on_true(model, tokenizer, classifier)
        if args.calculate_rouge_scores:
            rouge_metric = evaluate.load('rouge')
            train_rouge_values = rouge_metric.compute(predictions=predictions,
                                                      references=summaries)
            ood_test_rouge_values = rouge_metric.compute(predictions=ood_test_predictions,
                                                         references=ood_test_summaries)
            results['train_rouge'] = train_rouge_values
            results['ood_test_rouge'] = ood_test_rouge_values
    return results


def using_classifier(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores):
    args = parse_args()
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    properly_revised = [i for i in range(len(pre_revision_scores)) if
                        (pre_revision_scores[i] < 0.5 and post_revision_scores[i] > 0.5)]
    no_revision_needed = [i for i in range(len(pre_revision_scores)) if
                          pre_revision_scores[i] > 0.5]
    texts_properly_revised = [texts[i] for i in properly_revised]
    summaries_properly_revised = [summaries[i] for i in properly_revised]
    revised_summaries_properly_revised = [revised_summaries[i] for i in properly_revised]
    texts_no_revision_needed = [texts[i] for i in no_revision_needed]
    summaries_no_revision_needed = [summaries[i] for i in no_revision_needed]
    revised_summaries_no_revision_needed = [summaries[i] for i in no_revision_needed]
    train_dataset = SummarizationDataset(texts_properly_revised + texts_no_revision_needed,
                                         summaries_properly_revised + summaries_no_revision_needed,
                                         revised_summaries_properly_revised + revised_summaries_no_revision_needed)
    ood_test_texts, ood_test_summaries = load_xsum_ood(only_low_score=True)

    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.environ["WANDB_DISABLED"] = "true"
    model_name = args.model_name
    print(f"Using {model_name}")
    models_dir = args.models_dir + '/use_classifier/'
    if os.path.exists(models_dir + '/' + model_name + '/model.pkl'):
        print(f"Loading model from {models_dir + '/' + model_name}")
        model = AutoModel.from_pretrained(model_name)
        model.load_state_dict(torch.load(models_dir + '/' + model_name + '/model.pkl'))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        train_args = Seq2SeqTrainingArguments(
            output_dir=f'experiments/poc/checkpoints/t5_base_{run_name}',
            do_train=True, do_eval=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs,
            evaluation_strategy='no', save_strategy='no', eval_accumulation_steps=30, weight_decay=weight_decay,
            metric_for_best_model='rougeL', no_cuda=False)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), models_dir + '/' + model_name + '/model.pkl')

    results = {}
    if args.calculate_factuality_scores or args.calculate_rouge_scores:
        predictions_properly_revised = revise(texts_properly_revised, summaries_properly_revised,
                                              model,
                                              tokenizer, device='cuda:1', batch_size=8, max_length=128)
        predictions_no_revision_needed = revise(texts_no_revision_needed,
                                                summaries_no_revision_needed,
                                                model, tokenizer, device='cuda:1', batch_size=8, max_length=128)
        ood_test_predictions = revise(ood_test_texts, ood_test_summaries, model, tokenizer, device='cuda:1',
                                      batch_size=8,
                                      max_length=128)

        if args.calculate_factuality_scores:
            classifier = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                          device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16)

            scores_properly_revised = classifier.score(texts_properly_revised,
                                                       predictions_properly_revised)
            scores_no_revision_needed = classifier.score(texts_no_revision_needed,
                                                         predictions_no_revision_needed)
            ood_test_scores = classifier.score(ood_test_texts, ood_test_predictions)
            results = {}
            results['properly_revised_factuality_score'] = np.mean(scores_properly_revised)
            results['no_revision_needed_factuality_score'] = np.mean(scores_no_revision_needed)
            results['ood_test_factuality_score'] = np.mean(ood_test_scores)
            if args.eval_frank:
                results['frank_eval'] = evaluate_on_frank(model, tokenizer, classifier)
            if args.eval_true:
                results['true_eval'] = evaluate_on_true(model, tokenizer, classifier)
        if args.calculate_rouge_scores:
            rouge_metric = evaluate.load('rouge')
            rouge_values_properly_revised = rouge_metric.compute(predictions=predictions_properly_revised,
                                                                 references=summaries_properly_revised)
            rouge_values_no_revision_needed = rouge_metric.compute(
                predictions=predictions_no_revision_needed,
                references=summaries_no_revision_needed)
            ood_test_rouge_values = rouge_metric.compute(predictions=ood_test_predictions,
                                                         references=ood_test_summaries)
            results['properly_revised_rouge'] = rouge_values_properly_revised
            results['no_revision_needed_rouge'] = rouge_values_no_revision_needed
            results['ood_test_rouge'] = ood_test_rouge_values
    return results


def using_classifier_and_rouge_threshold(texts, summaries, revised_summaries, pre_revision_scores,
                                         post_revision_scores):
    args = parse_args()
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)[
        'rougeL']
    indices_above_threshold = [i for i in range(len(rouge_scores)) if rouge_scores[i] > 0.5]
    properly_revised = [i for i in range(len(pre_revision_scores)) if
                        (pre_revision_scores[i] < 0.5 and post_revision_scores[i] > 0.5)]
    # properly revised means both factuality increase and high rouge score
    properly_revised = list(set(properly_revised).intersection(indices_above_threshold))
    no_revision_needed = [i for i in range(len(pre_revision_scores)) if
                          pre_revision_scores[i] > 0.5]
    texts_properly_revised = [texts[i] for i in properly_revised]
    summaries_properly_revised = [summaries[i] for i in properly_revised]
    revised_summaries_properly_revised = [revised_summaries[i] for i in properly_revised]
    texts_no_revision_needed = [texts[i] for i in no_revision_needed]
    summaries_no_revision_needed = [summaries[i] for i in no_revision_needed]
    revised_summaries_no_revision_needed = [summaries[i] for i in no_revision_needed]
    train_dataset = SummarizationDataset(texts_properly_revised + texts_no_revision_needed,
                                         summaries_properly_revised + summaries_no_revision_needed,
                                         revised_summaries_properly_revised + revised_summaries_no_revision_needed)
    ood_test_texts, ood_test_summaries = load_xsum_ood(only_low_score=True)

    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.environ["WANDB_DISABLED"] = "true"
    model_name = args.model_name
    print(f"Using {model_name}")
    models_dir = args.models_dir + '/use_classifier_and_rouge/'
    if os.path.exists(models_dir + '/' + model_name + '/model.pkl'):
        print(f"Loading model from {models_dir + '/' + model_name}")
        model = AutoModel.from_pretrained(model_name)
        model.load_state_dict(torch.load(models_dir + '/' + model_name + '/model.pkl'))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        train_args = Seq2SeqTrainingArguments(
            output_dir=f'experiments/poc/checkpoints/t5_base_{run_name}',
            do_train=True, do_eval=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs,
            evaluation_strategy='no', save_strategy='no', eval_accumulation_steps=30, weight_decay=weight_decay,
            metric_for_best_model='rougeL', no_cuda=False)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
        torch.save(model.state_dict(), models_dir + '/' + model_name + '/model.pkl')

    results = {}
    if args.calculate_factuality_scores or args.calculate_rouge_scores:
        predictions_properly_revised = revise(texts_properly_revised, summaries_properly_revised,
                                              model,
                                              tokenizer, device='cuda:1', batch_size=8, max_length=128)
        predictions_no_revision_needed = revise(texts_no_revision_needed,
                                                summaries_no_revision_needed,
                                                model, tokenizer, device='cuda:1', batch_size=8, max_length=128)
        ood_test_predictions = revise(ood_test_texts, ood_test_summaries, model, tokenizer, device='cuda:1',
                                      batch_size=8,
                                      max_length=128)

        if args.calculate_factuality_scores:
            classifier = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                          device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16)

            scores_properly_revised = classifier.score(texts_properly_revised,
                                                       predictions_properly_revised)
            scores_no_revision_needed = classifier.score(texts_no_revision_needed,
                                                         predictions_no_revision_needed)
            ood_test_scores = classifier.score(ood_test_texts, ood_test_predictions)
            results = {}
            results['properly_revised_factuality_score'] = np.mean(scores_properly_revised)
            results['no_revision_needed_factuality_score'] = np.mean(scores_no_revision_needed)
            results['ood_test_factuality_score'] = np.mean(ood_test_scores)
            if args.eval_frank:
                results['frank_eval'] = evaluate_on_frank(model, tokenizer, classifier)
            if args.eval_true:
                results['true_eval'] = evaluate_on_true(model, tokenizer, classifier)
        if args.calculate_rouge_scores:
            rouge_metric = evaluate.load('rouge')
            rouge_values_properly_revised = rouge_metric.compute(predictions=predictions_properly_revised,
                                                                 references=summaries_properly_revised)
            rouge_values_no_revision_needed = rouge_metric.compute(
                predictions=predictions_no_revision_needed,
                references=summaries_no_revision_needed)
            ood_test_rouge_values = rouge_metric.compute(predictions=ood_test_predictions,
                                                         references=ood_test_summaries)
            results['properly_revised_rouge'] = rouge_values_properly_revised
            results['no_revision_needed_rouge'] = rouge_values_no_revision_needed
            results['ood_test_rouge'] = ood_test_rouge_values
    return results


def main():
    df = pd.read_csv('data/poc/poc_results_full_classification.csv', index_col=0)
    texts = df['document'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    pre_revision_scores = df['true_teacher_summary_scores'].tolist()
    post_revision_scores = df['true_teacher_revised_summary_scores'].tolist()
    using_classifier_and_rouge_threshold(texts, summaries, revised_summaries, pre_revision_scores,
                                            post_revision_scores)
    results = {}
    with open("experiments/poc/hyperparameter_tuning.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            if 'Error' in line:
                continue
            elif 'Trial' in line:
                continue
            elif 'Hyperparameters' in line:
                name = line.split(':')[0]
                value = ast.literal_eval(line.split(':', 1)[1])
            elif 'rouge' in line:
                name = line.split(':')[0]
                value = ast.literal_eval(line.split(':', 1)[1])
                for key in value.keys():
                    suffix = key[5:]
                    if name + suffix not in results:
                        results[name + suffix] = []
                    results[name + suffix].append(value[key])
                continue
            else:
                name = line.split(':')[0]
                value = float(line.split(':')[1])
            if name not in results.keys():
                results[name] = []
            results[name].append(value)
    for key in results.keys():
        if 'Hyperparameters' in key:
            continue
        sorted_indices = np.argsort(results[key])[::-1]

        to_print = [str(k) + ':' + str(v) for k, v in
                    zip(sorted_indices[:10], np.array(results[key])[sorted_indices[:10]])]
        print(f"{key}: {to_print}")
        print(results['Hyperparameters'][sorted_indices[0]])


if __name__ == "__main__":
    main()

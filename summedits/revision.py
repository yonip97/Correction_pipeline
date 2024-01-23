import json
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import torch
from general.t5_trainer import T5_Trainer, t5_revise
from general.utils import RevisionDataset
from datetime import datetime
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import numpy as np
import evaluate
from Seahorse_metrics.metrics import Seahorse_metrics


def compute_metrics(p, tokenizer):
    rouge = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.compute(predictions=predictions, references=labels)
    results = {k: np.mean(v) for k, v in results.items()}
    # results_dict = {}
    # import json
    # file_path = "summedits/results.json"
    # if os.path.exists(file_path):
    #     file = open(file_path, 'r')
    #     data = json.load(file)
    # else:
    #     data = []
    # results_dict['labels'] = labels
    # results_dict['predictions'] = predictions
    # data.append(results_dict)
    # with open(file_path, 'w') as json_file:
    #     json.dump(data, json_file, indent=2)
    return results


def collate_fn(batch, tokenizer, max_length):
    text_inputs = [("revise: summary: " + row['summary'], " document: " + row['document']) for row in batch]
    revised_summaries = [row['revised_summary'] for row in batch]
    inputs = tokenizer.batch_encode_plus(text_inputs, padding=True, truncation='only_second', max_length=max_length,
                                         return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def evaluate_on_true(model, tokenizer, factuality_metric):
    from data.factuality_datasets import TRUE_dataset
    results = {}
    rouge_metric = evaluate.load('rouge')
    dataset = TRUE_dataset('data/true_data', ['summarization'])
    df = dataset.df.reset_index()
    df['pre_revision_scores'] = pd.read_csv('data/true_data/seahorse/large_seahorse_scores_16float.csv')['scores']
    for dataset_name in df['dataset'].unique():
        temp_df = df[df['dataset'] == dataset_name]
        for model_name in temp_df['model'].unique():
            temp_df2 = temp_df[temp_df['model'] == model_name]
            texts = temp_df2['grounding'].tolist()
            summaries = temp_df2['generated_text'].tolist()
            model_revisions = t5_revise(texts, summaries, model, tokenizer, prompt="revise: ", device='cuda:1',
                                        batch_size=8, generation_max_length=128)
            pre_revision_scores = temp_df2['pre_revision_scores'].tolist()
            post_revision_scores = factuality_metric.score(texts, model_revisions)
            rouge_scores = rouge_metric.compute(predictions=model_revisions, references=summaries)
            print(dataset_name)
            print(model_name)
            print(np.mean(pre_revision_scores))
            print(np.mean(post_revision_scores))
            print(rouge_scores)
            results[dataset_name + '_' + model_name + '_pre_revision'] = np.mean(pre_revision_scores)
            results[dataset_name + '_' + model_name + '_post_revision'] = np.mean(post_revision_scores)
            for key in rouge_scores.keys():
                results[dataset_name + '_' + model_name + '_' + key] = rouge_scores[key]
            # results[dataset_name + '_' + model_name + '_rouge'] = rouge_scores
    return results


def tune(trial, texts, summaries, revised_summaries):
    try:
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        epochs = trial.suggest_int('epochs', 1, 5)
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 4)
        weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1, log=True)
        model_name = trial.suggest_categorical("model_name", ['t5-base', 'google/flan-t5-base'])
        train_dataset = RevisionDataset(texts=texts, summaries=summaries, revised_summaries=revised_summaries)
        run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.environ["WANDB_DISABLED"] = "true"

        print(f"Using {model_name}")

        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        train_args = Seq2SeqTrainingArguments(
            output_dir=f'summedits/runs/{model_name.replace("/", "_").replace("-", "_")}_{run_name}',
            do_train=True, do_eval=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs,
            evaluation_strategy='no', save_strategy='no', eval_accumulation_steps=30, weight_decay=weight_decay,
            no_cuda=False,predict_with_generate=True)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
        factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                             tokenizer_name='google/seahorse-xxl-q4'
                                             , device='auto', batch_size=1, max_length=2048,
                                             torch_dtype=torch.float16)
        xsum_results = evaluate_on_t5_xsum(model, tokenizer, factuality_metric)
        for key in trial.params.keys():
            xsum_results[key] = trial.params[key]
        file_path = "summedits/tune_results.json"
        try:
            with open(file_path, 'r') as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(xsum_results)
        with open(file_path, 'w') as json_file:
            json.dump(existing_data, json_file, indent=2)
        return xsum_results['factuality_scores']
    except Exception as e:
        if "CUDA out of memory" in str(e):
            return 0
        else:
            raise e


def train(args, texts, summaries, revised_summaries):
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    np.random.seed(42)
    train_inidices = np.random.choice(len(texts), int(0.85 * len(texts)), replace=False)
    train_dataset = RevisionDataset(texts=[texts[i] for i in train_inidices],
                                    summaries=[summaries[i] for i in train_inidices],
                                    revised_summaries=[revised_summaries[i] for i in train_inidices])
    val_dataset = RevisionDataset(texts=[texts[i] for i in range(len(texts)) if i not in train_inidices],
                                  summaries=[summaries[i] for i in range(len(texts)) if i not in train_inidices],
                                  revised_summaries=[revised_summaries[i] for i in range(len(texts)) if
                                                     i not in train_inidices])

    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.environ["WANDB_DISABLED"] = "true"

    model_name = args.model_name
    print(f"Using {model_name}")
    models_dir = args.models_dir + '/' + 'use_all/'

    if os.path.exists(models_dir + '/' + model_name + '/model.pkl'):
        print(f"Loading model from {models_dir + '/' + model_name}")
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.load_state_dict(torch.load(models_dir + '/' + model_name + '/model.pkl'))
        tokenizer = T5Tokenizer.from_pretrained(model_name)
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
            evaluation_strategy='steps', save_strategy='no', eval_accumulation_steps=30, weight_decay=weight_decay,
            eval_steps=300,
            metric_for_best_model='factuality_scores', no_cuda=False,predict_with_generate=True)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset, eval_dataset=val_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()
        if args.save:
            if not os.path.exists(models_dir + '/' + model_name):
                os.makedirs(models_dir + '/' + model_name)
            torch.save(model.state_dict(), models_dir + '/' + model_name + '/model.pkl')
        factuality_metric = None
        all_results = {}
        if args.evaluate_on_true:
            if factuality_metric is None:
                factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                                     tokenizer_name='google/seahorse-xxl-q4'
                                                     , device='auto', batch_size=1, max_length=2048,
                                                     torch_dtype=torch.float16)
            true_results = evaluate_on_true(model, tokenizer, factuality_metric)
            for key in true_results.keys():
                all_results['true_' + key] = true_results[key]
        if args.evaluate_on_xsum:
            if factuality_metric is None:
                factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                                     tokenizer_name='google/seahorse-xxl-q4'
                                                     , device='auto', batch_size=1, max_length=2048,
                                                     torch_dtype=torch.float16)
                xsum_results = evaluate_on_t5_xsum(model, tokenizer, factuality_metric)
                all_results['xsum_factuality_score'] = xsum_results
        with open("summedits/true_results.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        return model, tokenizer


def parseargs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--models_dir', type=str, default='summedits/models')
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--evaluate_on_true', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parseargs()
    df = get_all_data()
    texts = df['doc'].tolist()
    corrupted_summaries = df['summary'].tolist()
    original_summaries = df['original_summary'].tolist()
    train(args, texts=texts, summaries=corrupted_summaries, revised_summaries=original_summaries)


def get_all_data():
    path = "data/summedits"
    files = os.listdir(path)
    files = [os.path.join(path, file) for file in files]
    dfs = []
    for file in files:
        df = pd.read_json(file)
        dataset_name = file.split('/')[-1].split('.')[0].split('_')[1]
        df['dataset'] = dataset_name
        dfs.append(df)

        # edit_types += df['edit_type'].tolist()
    df = pd.concat(dfs)
    return df


def evaluate_on_t5_xsum(model, tokenizer, factuality_metric):
    df = pd.read_csv('experiments/xsum_4_sets_experiment/xsum_model_summaries.csv', index_col=0)
    df = df.sample(1000, random_state=42)
    print(df['factuality_score_seahorse_xxl'].mean())
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    xsum_indices_to_text = {xsum_dataset.indices[i]: xsum_dataset[i]['text'] for i in range(len(xsum_dataset))}
    # cnndm_dataset = split_cnndm_dataset(split='documents_for_summarization',
    #                                     path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
    #                                     num_of_documents_for_summarization=20000,
    #                                     seed=42)
    # cnndm_indices_to_text = {cnndm_dataset.indices[i]: cnndm_dataset[i]['text'] for i in range(len(cnndm_dataset))}
    df['text'] = df.apply(
        lambda x: xsum_indices_to_text[x['indices']], axis=1)
    revised_summaries = t5_revise(df['text'].tolist(), df['model_summary'].tolist(), model, tokenizer,
                                  prompt="revise: ",
                                  device='cuda:1',
                                  batch_size=8, generation_max_length=128)
    scores = factuality_metric.score(texts=df['text'].tolist(), summaries=revised_summaries)
    results = {}
    results['pre_revision_score'] = df['factuality_score_seahorse_xxl'].mean()
    results['factuality_scores'] = np.mean(scores)
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=revised_summaries, references=df['model_summary'].tolist())
    for key in rouge_scores.keys():
        results[key] = rouge_scores[key]
    return results


def see_if_factuality_improves():
    factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                         tokenizer_name='google/seahorse-xxl-q4'
                                         , device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16)
    rouge_metric = evaluate.load('rouge')
    path = "summedits/results.json"
    file = open(path, 'r')
    data = json.load(file)
    df = get_all_data()
    texts = df['doc'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['original_summary'].tolist()
    np.random.seed(42)
    train_inidices = np.random.choice(len(texts), int(0.85 * len(texts)), replace=False)
    val_texts = [texts[i] for i in range(len(texts)) if i not in train_inidices]
    val_summaries = [summaries[i] for i in range(len(texts)) if i not in train_inidices]
    val_revised_summaries = [revised_summaries[i] for i in range(len(texts)) if i not in train_inidices]
    df = pd.DataFrame.from_dict({'text': val_texts, 'summary': val_summaries, 'revised_summary': val_revised_summaries})
    for entry in data:
        temp_df = pd.DataFrame.from_dict(entry)
        for col in df.columns:
            temp_df[col] = df[col]
        scores = factuality_metric.score(texts=temp_df['text'].tolist(), summaries=temp_df['predictions'].tolist())
        rouge_to_best = rouge_metric.compute(predictions=temp_df['predictions'].tolist(),
                                             references=temp_df['revised_summary'].tolist())
        rouge_to_source = rouge_metric.compute(predictions=temp_df['predictions'].tolist(),
                                               references=temp_df['summary'].tolist())
        print(np.mean(scores))
        print(rouge_to_best)
        print(rouge_to_source)
    # rouge_metric = evaluate.load('rouge')
    # for i in range(len(different_runs_texts)):
    #     print(rouge_metric.compute(predictions=different_runs_texts[i], references=val_summaries))
    #     print(rouge_metric.compute(predictions=different_runs_texts[i], references=val_revised_summaries))


if __name__ == '__main__':
    # rouge_metric = evaluate.load('rouge')
    df = get_all_data()
    texts = df['doc'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['original_summary'].tolist()
    # print(rouge_metric.compute(predictions=df['summary'].tolist(), references=df['original_summary'].tolist()))
    # main()
    import optuna

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: tune(trial, texts=texts, summaries=summaries, revised_summaries=revised_summaries),
                   n_trials=200,
                   timeout=60 * 60 * 20)
    best_params = study.best_params
    best_score = study.best_value
    print("Best Hyperparameters:", best_params)
    print("Best factuality score:", best_score)
    # see_if_factuality_improves()
    # evaluate_on_t5_xsum()

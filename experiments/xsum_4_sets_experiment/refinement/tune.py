import json
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
import numpy as np
import pandas as pd
import torch
from general.utils import RevisionDataset
from datetime import datetime
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer
import evaluate
import optuna


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


def tune(trial, texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, method):
    try:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        epochs = trial.suggest_int("epochs", 1, 5)
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
        train_dataset, val_dataset = create_dataset(texts, summaries, revised_summaries, pre_revision_scores,
                                                    post_revision_scores, method)

        run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.environ["WANDB_DISABLED"] = "true"
        model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_both_10_12_2023_08_54_06/checkpoint-115000"
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        train_args = Seq2SeqTrainingArguments(
            output_dir=f'experiments/xsum_4_sets_experiments/runs/t5_base_{run_name}',
            do_train=True, do_eval=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs, evaluation_strategy='steps', save_strategy='no',
            eval_accumulation_steps=30, weight_decay=weight_decay, eval_steps=500,
            metric_for_best_model='rougeL', no_cuda=False,predict_with_generate = True)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset, eval_dataset=val_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        save_path = f"experiments/xsum_4_sets_experiment/{method}_results.json"
        if os.path.exists(save_path):
            data = json.load(open(save_path, 'r'))
        else:
            data = []
        results = trainer.evaluate()
        run_hyperparameters = {'lr': lr, 'epochs': epochs, 'batch_size': batch_size,
                               'gradient_accumulation_steps': gradient_accumulation_steps,
                               'weight_decay': weight_decay}
        results['hyperparameters'] = run_hyperparameters
        for item in trainer.state.log_history:
            if 'eval_rougeL' in item:
                results[f'eval_step_{item["step"]}'] = item
            else:
                results[f'train_step_{item["step"]}'] = item
        data.append(results)
        json.dump(data, open(save_path, 'w'))
        del trainer
        torch.cuda.empty_cache()
        return results['eval_rougeL']
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("Out of memory. Trying to free up some GPU memory.")
            # Free up memory (adjust as needed)
            torch.cuda.empty_cache()
            return 0.0
        else:
            raise e


def create_dataset(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, method):
    np.random.seed(42)
    if method == 'all':
        train_indices = np.random.choice(len(texts), int(len(texts) * 0.85), replace=False)
        val_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
    elif method == 'classifier' or method == 'classifier_and_rouge_threshold':
        properly_revised = [i for i in range(len(pre_revision_scores)) if
                            (pre_revision_scores[i] < 0.5 and post_revision_scores[i] > 0.5)]
        if method == 'classifier_and_rouge_threshold':
            rouge_metric = evaluate.load('rouge')
            rouge_scores = \
                rouge_metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)[
                    'rougeL']
            indices_above_threshold = [i for i in range(len(rouge_scores)) if rouge_scores[i] > 0.5]
            # properly revised means both factuality increase and high rouge score
            properly_revised = list(set(properly_revised).intersection(indices_above_threshold))
        train_indices = np.random.choice(properly_revised, int(len(properly_revised) * 0.85), replace=False)
        val_indices = np.array(list(set(properly_revised) - set(train_indices)))
    else:
        raise NotImplementedError
    train_texts, train_summaries, train_revised_summaries = np.array(texts)[train_indices].tolist(), \
                                                            np.array(summaries)[train_indices].tolist(), \
                                                            np.array(revised_summaries)[train_indices].tolist()
    val_texts, val_summaries, val_revised_summaries = np.array(texts)[val_indices].tolist(), np.array(summaries)[
        val_indices].tolist(), np.array(revised_summaries)[val_indices].tolist()
    train_dataset = RevisionDataset(train_texts, train_summaries, train_revised_summaries)
    val_dataset = RevisionDataset(val_texts, val_summaries, val_revised_summaries)
    return train_dataset, val_dataset


def create_full_dataset():
    path = 'experiments/xsum_4_sets_experiment/revision_results_seahorse.csv'
    revised_dataset = pd.read_csv(path)
    revised_dataset = revised_dataset[['text', 'model_summary', 'revised_summary',
                                       'seahorse_scores_post_revision', 'true_teacher_scores_post_revision']]
    path = 'experiments/xsum_4_sets_experiment/both_models_summaries.csv'
    all_dataset = pd.read_csv(path, index_col=0)
    all_dataset = all_dataset[all_dataset['text'].notna()]
    df = pd.merge(all_dataset, revised_dataset, on=['model_summary', 'text'], how='inner')
    df = df.drop_duplicates(['dataset', 'indices'])
    return df


def extract_results(results):
    results_dict = {}
    counter = 0
    for entry in results:
        best_eval_rouge_L = None
        train_step = None
        for key in entry.keys():
            if key == 'hyperparameters':
                continue
            if 'eval_rougeL' == key:
                if best_eval_rouge_L is None:
                    best_eval_rouge_L = entry['eval_rougeL']
                else:
                    best_eval_rouge_L = max(best_eval_rouge_L, entry[key])
                train_step = 'Last'
            if isinstance(entry[key], dict) and 'eval_rougeL' in entry[key].keys():
                best_eval_rouge_L = max(best_eval_rouge_L, entry[key]['eval_rougeL'])
                train_step = entry[key]['step']
        results_dict[counter] = {'best_eval_rouge_L': best_eval_rouge_L, 'train_step': train_step,'eval_loss':entry['eval_loss']}
        counter += 1
    return results_dict


def read_results():
    all_results = json.load(open('experiments/xsum_4_sets_experiment/all_results.json', 'r'))
    classifier_results = json.load(open('experiments/xsum_4_sets_experiment/classifier_results.json', 'r'))
    classifier_and_rouge_threshold_results = json.load(
        open('experiments/xsum_4_sets_experiment/classifier_and_rouge_threshold_results.json', 'r'))
    all_results_dict = extract_results(all_results)
    classifier_results_dict = extract_results(classifier_results)
    classifier_and_rouge_threshold_results_dict = extract_results(classifier_and_rouge_threshold_results)
    all_results_final = sorted(all_results_dict.items(), key=lambda x: x[1]['eval_loss'], reverse=False)
    classifier_results_final = sorted(classifier_results_dict.items(), key=lambda x: x[1]['eval_loss'],
                                      reverse=False)
    classifier_and_rouge_threshold_results_final = sorted(classifier_and_rouge_threshold_results_dict.items(),
                                                          key=lambda x: x[1]['eval_loss'], reverse=False)

    return all_results_final, classifier_results_final, classifier_and_rouge_threshold_results_final


def main():
    all_results_final, classifier_results_final, classifier_and_rouge_threshold_results_final = read_results()
    df = create_full_dataset()
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    pre_revision_scores = df['factuality_score_seahorse_xxl'].tolist()
    post_revision_scores = df['seahorse_scores_post_revision'].tolist()
    # all_baseline = check_baseline(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores,
    #                               method='all')
    # print(all_baseline)


if __name__ == '__main__':
    main()

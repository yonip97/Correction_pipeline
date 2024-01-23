import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import numpy as np
import pandas as pd
from datetime import datetime
import torch
from transformers import Seq2SeqTrainingArguments, T5ForConditionalGeneration, T5Tokenizer
from general.t5_trainer import T5_Trainer, t5_revise
from experiments.poc.poc_utils import  collate_fn, compute_metrics, load_xsum_ood
from Seahorse_metrics.metrics import Seahorse_metrics
import evaluate
import optuna
import ast

from general.utils import RevisionDataset
def tune_using_all(trial, texts, summaries, revised_summaries, model_name):
    try:
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        epochs = trial.suggest_int("epochs", 10, 30)
        batch_size = trial.suggest_categorical("batch_size", [4, 8])
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)
        weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)

        train_indices = np.random.choice(len(texts), int(len(texts) * 0.85), replace=False)
        val_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
        train_texts, train_summaries, train_revised_summaries = np.array(texts)[train_indices].tolist(), \
                                                                np.array(summaries)[train_indices].tolist(), \
                                                                np.array(revised_summaries)[train_indices].tolist()
        val_texts, val_summaries, val_revised_summaries = np.array(texts)[val_indices].tolist(), np.array(summaries)[
            val_indices].tolist(), np.array(revised_summaries)[val_indices].tolist()
        train_dataset = RevisionDataset(train_texts, train_summaries, train_revised_summaries)
        val_dataset = RevisionDataset(val_texts, val_summaries, val_revised_summaries)
        ood_test_texts, ood_test_summaries = load_xsum_ood(only_low_score=True)

        run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.environ["WANDB_DISABLED"] = "true"
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        train_args = Seq2SeqTrainingArguments(
            output_dir=f'experiments/poc/checkpoints/t5_base_{run_name}',
            do_train=True, do_eval=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs, evaluation_strategy='no', save_strategy='no',
            eval_accumulation_steps=30, weight_decay=weight_decay,
            metric_for_best_model='rougeL', no_cuda=False,predict_with_generate=True)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             eval_dataset=val_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()

        train_predictions = t5_revise(np.array(texts)[train_indices].tolist(), np.array(summaries)[train_indices].tolist(),
                                      model,
                                      tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        val_predictions = t5_revise(np.array(texts)[val_indices].tolist(),
                                    np.array(summaries)[val_indices].tolist(),
                                    model, tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)

        ood_test_predictions = t5_revise(ood_test_texts, ood_test_summaries, model, tokenizer, prompt="revise: ", device='cuda:1',
                                         batch_size=8,
                                         generation_max_length=128)
        model.to('cpu')
        classifier = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                      device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16)
        train_scores = classifier.score(train_texts, train_predictions)
        val_scores = classifier.score(val_texts, val_predictions)
        ood_test_scores = classifier.score(ood_test_texts, ood_test_predictions)
        rouge_metric = evaluate.load('rouge')
        train_rouge_values = rouge_metric.compute(predictions=train_predictions,
                                                  references=train_summaries)
        val_rouge_values = rouge_metric.compute(predictions=val_predictions,
                                                references=val_summaries)
        ood_test_rouge_values = rouge_metric.compute(predictions=ood_test_predictions, references=ood_test_summaries)

        results = {}
        results['train_factuality_score'] = np.mean(train_scores)
        results['val_factuality_score'] = np.mean(val_scores)
        results['ood_test_factuality_score'] = np.mean(ood_test_scores)
        results['train_rouge'] = train_rouge_values
        results['val_rouge'] = val_rouge_values
        results['ood_test_rouge'] = ood_test_rouge_values
        del classifier
        import time
        time.sleep(5)
        torch.cuda.empty_cache()
        with open(f"experiments/poc/hyperparameter_tuning_all_{model_name.replace('/', '_')}.txt", "a") as f:
            f.write(f"Trial {trial.number}\n")
            f.write(f"Hyperparameters: {trial.params}\n")
            for key in results.keys():
                f.write(f"{key}: {results[key]}\n")
        return results['val_factuality_score']

    except Exception as e:
        if 'out of memory' in str(e):
            with open(f"experiments/poc/hyperparameter_tuning_all_{model_name.replace('/', '_')}.txt", "a") as f:
                f.write(f"Trial {trial.number}\n")
                f.write(f"Hyperparameters: {trial.params}\n")
                f.write(f"{str(e)}")
            return 0
        else:
            raise e


def tune_using_classifier_and_rouge_threshold(trial, texts, summaries, revised_summaries, pre_revision_scores,
                                              post_revision_scores, model_name, seed=42):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 10, 30)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    try:
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
        chosen_indices = properly_revised + no_revision_needed
        np.random.seed(seed)
        train_indices = np.random.choice(chosen_indices, int(len(chosen_indices) * 0.85), replace=False)
        val_indices = np.array(list(set(chosen_indices) - set(train_indices)))
        train_texts_properly_revised, train_summaries_properly_revised, train_revised_summaries_properly_revised = [], [], []
        train_texts_no_revision_needed, train_summaries_no_revision_needed, train_revised_summaries_no_revision_needed = [], [], []
        for i in train_indices:
            if i in properly_revised:
                train_texts_properly_revised.append(texts[i])
                train_summaries_properly_revised.append(summaries[i])
                train_revised_summaries_properly_revised.append(revised_summaries[i])
            else:
                train_texts_no_revision_needed.append(texts[i])
                train_summaries_no_revision_needed.append(summaries[i])
                train_revised_summaries_no_revision_needed.append(summaries[i])
        val_texts_properly_revised, val_summaries_properly_revised, val_revised_summaries_properly_revised = [], [], []
        val_texts_no_revision_needed, val_summaries_no_revision_needed, val_revised_summaries_no_revision_needed = [], [], []
        for i in val_indices:
            if i in properly_revised:
                val_texts_properly_revised.append(texts[i])
                val_summaries_properly_revised.append(summaries[i])
                val_revised_summaries_properly_revised.append(revised_summaries[i])
            else:
                val_texts_no_revision_needed.append(texts[i])
                val_summaries_no_revision_needed.append(summaries[i])
                val_revised_summaries_no_revision_needed.append(summaries[i])

        train_dataset = RevisionDataset(train_texts_properly_revised + train_texts_no_revision_needed,
                                             train_summaries_properly_revised + train_summaries_no_revision_needed,
                                             train_revised_summaries_properly_revised + train_revised_summaries_no_revision_needed)
        ood_test_texts, ood_test_summaries = load_xsum_ood(only_low_score=True)
        run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.environ["WANDB_DISABLED"] = "true"
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
            metric_for_best_model='rougeL', no_cuda=False,predict_with_generate=True)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()

        train_predictions_properly_revised = t5_revise(train_texts_properly_revised, train_summaries_properly_revised,
                                                       model,
                                                       tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        train_predictions_no_revision_needed = t5_revise(train_texts_no_revision_needed,
                                                         train_summaries_no_revision_needed,
                                                         model, tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        val_predictions_properly_revised = t5_revise(val_texts_properly_revised, val_summaries_properly_revised, model,
                                                     tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        val_predictions_no_revision_needed = t5_revise(val_texts_no_revision_needed, val_summaries_no_revision_needed,
                                                       model,
                                                       tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        ood_test_predictions = t5_revise(ood_test_texts, ood_test_summaries, model, tokenizer, prompt="revise: ", device='cuda:1',
                                         batch_size=8,
                                         generation_max_length=128)
        model.to('cpu')
        del model
        import time
        time.sleep(10)
        torch.cuda.empty_cache()
        classifier = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                      device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16)
        train_scores_properly_revised = classifier.score(train_texts_properly_revised,
                                                         train_predictions_properly_revised)
        train_scores_no_revision_needed = classifier.score(train_texts_no_revision_needed,
                                                           train_predictions_no_revision_needed)
        val_scores_properly_revised = classifier.score(val_texts_properly_revised, val_predictions_properly_revised)
        val_scores_no_revision_needed = classifier.score(val_texts_no_revision_needed,
                                                         val_predictions_no_revision_needed)
        ood_test_scores = classifier.score(ood_test_texts, ood_test_predictions)

        train_rouge_values_properly_revised = rouge_metric.compute(predictions=train_predictions_properly_revised,
                                                                   references=train_summaries_properly_revised)
        train_rouge_values_no_revision_needed = rouge_metric.compute(predictions=train_predictions_no_revision_needed,
                                                                     references=train_summaries_no_revision_needed)
        val_rouge_values_properly_revised = rouge_metric.compute(predictions=val_predictions_properly_revised,
                                                                 references=val_summaries_properly_revised)
        val_rouge_values_no_revision_needed = rouge_metric.compute(predictions=val_predictions_no_revision_needed,
                                                                   references=val_summaries_no_revision_needed)
        ood_test_rouge_values = rouge_metric.compute(predictions=ood_test_predictions, references=ood_test_summaries)

        results = {}
        results['train_properly_revised_factuality_score'] = np.mean(train_scores_properly_revised)
        results['train_no_revision_needed_factuality_score'] = np.mean(train_scores_no_revision_needed)
        results['train_factuality_score'] = np.mean(train_scores_properly_revised + train_scores_no_revision_needed)
        results['val_properly_revised_factuality_score'] = np.mean(val_scores_properly_revised)
        results['val_no_revision_needed_factuality_score'] = np.mean(val_scores_no_revision_needed)
        results['val_factuality_score'] = np.mean(val_scores_properly_revised + val_scores_no_revision_needed)
        results['ood_test_factuality_score'] = np.mean(ood_test_scores)
        results['train_properly_revised_rouge'] = train_rouge_values_properly_revised
        results['train_no_revision_needed_rouge'] = train_rouge_values_no_revision_needed
        results['val_properly_revised_rouge'] = val_rouge_values_properly_revised
        results['val_no_revision_needed_rouge'] = val_rouge_values_no_revision_needed
        results['ood_test_rouge'] = ood_test_rouge_values
        del classifier
        import time
        time.sleep(5)
        torch.cuda.empty_cache()
        with open(
                f"experiments/poc/hyperparameter_tuning__using_classifier_and_rouge_threshold_{model_name.replace('/', '_')}.txt",
                "a") as f:
            f.write(f"Trial {trial.number}\n")
            f.write(f"Hyperparameters: {trial.params}\n")
            for key in results.keys():
                f.write(f"{key}: {results[key]}\n")
        return results['val_factuality_score']
    except Exception as e:
        if 'out of memory' in str(e):
            with open(f"experiments/poc/hyperparameter_tuning_{model_name.replace('/', '_')}.txt", "a") as f:
                f.write(f"Trial {trial.number}\n")
                f.write(f"Hyperparameters: {trial.params}\n")
                f.write(f"{str(e)}")
            return 0
        else:
            raise e


def tune_using_classifier(trial, texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores,
                          model_name, seed=42):
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    epochs = trial.suggest_int("epochs", 10, 30)
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)

    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)
    try:
        properly_revised = [i for i in range(len(pre_revision_scores)) if
                            (pre_revision_scores[i] < 0.5 and post_revision_scores[i] > 0.5)]
        no_revision_needed = [i for i in range(len(pre_revision_scores)) if
                              pre_revision_scores[i] > 0.5]
        chosen_indices = properly_revised + no_revision_needed
        np.random.seed(seed)
        train_indices = np.random.choice(chosen_indices, int(len(chosen_indices) * 0.85), replace=False)
        val_indices = np.array(list(set(chosen_indices) - set(train_indices)))
        train_texts_properly_revised, train_summaries_properly_revised, train_revised_summaries_properly_revised = [], [], []
        train_texts_no_revision_needed, train_summaries_no_revision_needed, train_revised_summaries_no_revision_needed = [], [], []
        for i in train_indices:
            if i in properly_revised:
                train_texts_properly_revised.append(texts[i])
                train_summaries_properly_revised.append(summaries[i])
                train_revised_summaries_properly_revised.append(revised_summaries[i])
            else:
                train_texts_no_revision_needed.append(texts[i])
                train_summaries_no_revision_needed.append(summaries[i])
                train_revised_summaries_no_revision_needed.append(summaries[i])
        val_texts_properly_revised, val_summaries_properly_revised, val_revised_summaries_properly_revised = [], [], []
        val_texts_no_revision_needed, val_summaries_no_revision_needed, val_revised_summaries_no_revision_needed = [], [], []
        for i in val_indices:
            if i in properly_revised:
                val_texts_properly_revised.append(texts[i])
                val_summaries_properly_revised.append(summaries[i])
                val_revised_summaries_properly_revised.append(revised_summaries[i])
            else:
                val_texts_no_revision_needed.append(texts[i])
                val_summaries_no_revision_needed.append(summaries[i])
                val_revised_summaries_no_revision_needed.append(summaries[i])

        train_dataset = RevisionDataset(train_texts_properly_revised + train_texts_no_revision_needed,
                                             train_summaries_properly_revised + train_summaries_no_revision_needed,
                                             train_revised_summaries_properly_revised + train_revised_summaries_no_revision_needed)

        ood_test_texts, ood_test_summaries = load_xsum_ood(only_low_score=True)
        run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        os.environ["WANDB_DISABLED"] = "true"
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
            metric_for_best_model='rougeL', no_cuda=False,predict_with_generate=True)
        max_length_train = 512
        trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer),
                             max_length_train=max_length_train, max_length_eval=max_length_train)
        trainer.train()
        del trainer
        torch.cuda.empty_cache()

        train_predictions_properly_revised = t5_revise(train_texts_properly_revised, train_summaries_properly_revised,
                                                       model,
                                                       tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        train_predictions_no_revision_needed = t5_revise(train_texts_no_revision_needed,
                                                         train_summaries_no_revision_needed,
                                                         model, tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        val_predictions_properly_revised = t5_revise(val_texts_properly_revised, val_summaries_properly_revised, model,
                                                     tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        val_predictions_no_revision_needed = t5_revise(val_texts_no_revision_needed, val_summaries_no_revision_needed,
                                                       model,
                                                       tokenizer, prompt="revise: ", device='cuda:1', batch_size=8, generation_max_length=128)
        ood_test_predictions = t5_revise(ood_test_texts, ood_test_summaries, model, tokenizer, prompt="revise: ", device='cuda:1',
                                         batch_size=8,
                                         generation_max_length=128)
        model.to('cpu')
        del model
        import time
        time.sleep(10)
        torch.cuda.empty_cache()
        classifier = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                      device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16)
        train_scores_properly_revised = classifier.score(train_texts_properly_revised,
                                                         train_predictions_properly_revised)
        train_scores_no_revision_needed = classifier.score(train_texts_no_revision_needed,
                                                           train_predictions_no_revision_needed)
        val_scores_properly_revised = classifier.score(val_texts_properly_revised, val_predictions_properly_revised)
        val_scores_no_revision_needed = classifier.score(val_texts_no_revision_needed,
                                                         val_predictions_no_revision_needed)
        ood_test_scores = classifier.score(ood_test_texts, ood_test_predictions)
        del classifier
        torch.cuda.empty_cache()
        results = {}
        rouge_metric = evaluate.load('rouge')
        train_rouge_values_properly_revised = rouge_metric.compute(predictions=train_predictions_properly_revised,
                                                                   references=train_summaries_properly_revised)
        train_rouge_values_no_revision_needed = rouge_metric.compute(predictions=train_predictions_no_revision_needed,
                                                                     references=train_summaries_no_revision_needed)
        val_rouge_values_properly_revised = rouge_metric.compute(predictions=val_predictions_properly_revised,
                                                                 references=val_summaries_properly_revised)
        val_rouge_values_no_revision_needed = rouge_metric.compute(predictions=val_predictions_no_revision_needed,
                                                                   references=val_summaries_no_revision_needed)
        ood_test_rouge_values = rouge_metric.compute(predictions=ood_test_predictions, references=ood_test_summaries)

        results['train_properly_revised_factuality_score'] = np.mean(train_scores_properly_revised)
        results['train_no_revision_needed_factuality_score'] = np.mean(train_scores_no_revision_needed)
        results['train_factuality_score'] = np.mean(train_scores_properly_revised + train_scores_no_revision_needed)
        results['val_properly_revised_factuality_score'] = np.mean(val_scores_properly_revised)
        results['val_no_revision_needed_factuality_score'] = np.mean(val_scores_no_revision_needed)
        results['val_factuality_score'] = np.mean(val_scores_properly_revised + val_scores_no_revision_needed)
        results['ood_test_factuality_score'] = np.mean(ood_test_scores)
        results['train_properly_revised_rouge'] = train_rouge_values_properly_revised
        results['train_no_revision_needed_rouge'] = train_rouge_values_no_revision_needed
        results['val_properly_revised_rouge'] = val_rouge_values_properly_revised
        results['val_no_revision_needed_rouge'] = val_rouge_values_no_revision_needed
        results['ood_test_rouge'] = ood_test_rouge_values
        with open(f"experiments/poc/hyperparameter_tuning_using_classifier_{model_name.replace('/', '_')}.txt",
                  "a") as f:
            f.write(f"Trial {trial.number}\n")
            f.write(f"Hyperparameters: {trial.params}\n")
            for key in results.keys():
                f.write(f"{key}: {results[key]}\n")
        return results['val_factuality_score']
    except Exception as e:
        if 'out of memory' in str(e):
            with open(f"experiments/poc/hyperparameter_tuning_{model_name.replace('/', '_')}.txt", "a") as f:
                f.write(f"Trial {trial.number}\n")
                f.write(f"Hyperparameters: {trial.params}\n")
                f.write(f"Error\n")
            return 0
        else:
            raise e


def tune(trial, method, model_name):
    df = pd.read_csv('data/poc/poc_results_full_classification.csv', index_col=0)
    texts = df['document'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    if method == 'all':
        return tune_using_all(trial, texts, summaries, revised_summaries, model_name)
    elif method == 'classifier_scores':
        pre_revision_scores = df['true_teacher_summary_scores'].tolist()
        post_revision_scores = df['true_teacher_revised_summary_scores'].tolist()
        return tune_using_classifier(trial, texts, summaries, revised_summaries, pre_revision_scores,
                                     post_revision_scores, model_name)
    elif method == 'classifier_and_rouge_scores':
        pre_revision_scores = df['true_teacher_summary_scores'].tolist()
        post_revision_scores = df['true_teacher_revised_summary_scores'].tolist()
        return tune_using_classifier_and_rouge_threshold(trial, texts, summaries, revised_summaries,
                                                         pre_revision_scores,
                                                         post_revision_scores, model_name)
    else:
        raise ValueError('Invalid method')


def do_study():
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: tune(trial, 'all', model_name='google/flan-t5-base'), n_trials=50,
                   timeout=60 * 60 * 8)
    best_params = study.best_params
    best_score = study.best_value
    print("Best Hyperparameters:", best_params)
    print("Best factuality score:", best_score)


def read_results(path):
    results = {}
    with open(path, 'r') as f:
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
        # Chose train configuration based on 4 options
        # highest val factuality score
        # highest rouge score (rougeL)
        # highest ood score
        # Highest rank on all three metrics
    if 'flan' in path:
        model_name = 'google/flan-t5-base'
    else:
        model_name = 't5-base'
    if 'all' in path:
        method = 'all'
    elif 'classifier_and_rouge_threshold' in path:
        method = 'classifier_and_rouge'
    else:
        method = 'classifier'
    write_script_path = "experiments/poc/best_results.txt"
    base = f"python3 train_corrector --method {method} --model_name {model_name} --output_file experiments/poc/best_results/{method}_{model_name.replace('/','_')}.txt --calculate_factuality_scores --calculate_rouge_scores --eval_true --eval_frank"
    scripts = []
    for key in results.keys():
        if key == 'val_factuality_score' or key == "val_properly_revised_factuality_score":
            res = results[key]
            run_hep = results['Hyperparameters'][np.argmax(res)]
            print(f"Best {key}:run {np.argmax(res)} score {np.max(res)}")
            print(f"Hyperparameters: {run_hep}")
            script = base
            for k in run_hep.keys():
                script += f" --{k} {run_hep[k]}"
        elif key == 'val_rougeL' or key == 'val_properly_revised_rougeL':
            res = results[key]
            run_hep = results['Hyperparameters'][np.argmax(res)]
            print(f"Best {key}:run {np.argmax(res)} score {np.max(res)}")
            print(f"Hyperparameters: {run_hep}")
            script = base
            for k in run_hep.keys():
                script += f" --{k} {run_hep[k]}"

        elif key == 'ood_test_factuality_score':
            res = results[key]
            run_hep = results['Hyperparameters'][np.argmax(res)]
            print(f"Best {key}:run {np.argmax(res)} score {np.max(res)}")
            print(f"Hyperparameters: {run_hep}")
            script = base
            for k in run_hep.keys():
                script += f" --{k} {run_hep[k]}"
        else:
            continue
        scripts.append(script)
    print(scripts)
    with open(write_script_path, 'a') as f:
        for script in scripts:
            f.write(script + '\n')

def main():
    main_dir = "experiments/poc"
    paths = ["hyperparameter_tuning_all_t5-base.txt", "hyperparameter_tuning_all_google_flan-t5-base.txt"
        , "hyperparameter_tuning_using_classifier_and_rouge_threshold_google_flan-t5-base.txt",
             "hyperparameter_tuning_using_classifier_and_rouge_threshold_t5.txt",
             "hyperparameter_tuning_t5_using_classifier.txt",
             "hyperparameter_tuning_t5_flan_using_classifier.txt"
             ]
    for path in paths:
        print(path)
        read_results(main_dir + '/' + path)
        print('---------------------------------------------------------------------------------------')
        print()
        print()


if __name__ == '__main__':
    main()

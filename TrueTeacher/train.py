import os
import sys

# sys.path.append(os.path.dirname(os.getcwd()))
# os.chdir('../')

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
import json
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from scipy.special import softmax
from datetime import datetime
from data.data_utils import collate_fn
from data.factuality_datasets import FactCC_dataset, TrueTeacher_anli_dataset, TRUE_dataset
from data.data_utils import tokeinized_collate_fn
from general.t5_trainer import T5_Trainer


def compute_metrics(p, tokenizer, model):
    # last_checkpoint = find_last_checkpoint(output_dir)
    # results_dict = evaluate(None, 'data', model)
    # total = results_dict.pop('total')
    # roc_auc = 0
    # for roc_auc_score_dataset in results_dict.values():
    #     roc_auc += roc_auc_score_dataset['roc auc']
    # avg_auc_roc_true = roc_auc / len(results_dict)

    logits = p.predictions[0]
    probs = softmax(logits, axis=-1)
    one_token_id = tokenizer('1').input_ids[0]
    entailment_prob = probs[:, 0, one_token_id]
    gt = tokenizer.batch_decode(p.label_ids)
    gt = [int(x) for x in gt]
    score = roc_auc_score(gt, entailment_prob)
    print("Roc Auc Score: ", score)
    return {'roc auc': score}
    # return {'roc auc': score, 'avg_auc_roc_true': avg_auc_roc_true}


def train():
    os.environ["WANDB_DISABLED"] = "true"
    eval_dataset = FactCC_dataset("factCC/data/unpaired_annotated_data/")

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    dataset = TrueTeacher_anli_dataset(tokenizer=tokenizer, true_teacher_samples=1e5, seed=1)
    current_datetime = datetime.now()

    current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    run_name = f"run name_{current_datetime_string}"
    args = Seq2SeqTrainingArguments(output_dir=f'TrueTeacher/results/{run_name}', do_train=True, do_eval=True,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=4,
                                    gradient_accumulation_steps=1,
                                    learning_rate=1e-4, num_train_epochs=20, save_total_limit=3,
                                    load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
                                    eval_steps=1000, save_steps=1000, eval_accumulation_steps=100,
                                    metric_for_best_model='roc auc', no_cuda=False)
    max_length_train = 512
    max_length_eval = 2048
    trainer = T5_Trainer(model=model, tokenizer=tokenizer, args=args, train_dataset=dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer, model),
                         max_length_train=max_length_train, max_length_eval=max_length_eval)
    trainer.train()


def evaluate(dataloader, model, tokenizer, device='cpu'):
    model.to(device)
    with torch.no_grad():
        model.eval()
        probs_list = []
        labels = []
        per_dataset_labels_and_probs = {}
        for batch in tqdm(dataloader):
            datasets_names_batch = batch[0]
            decoder_input_ids = torch.tensor([[tokenizer.pad_token_id] * len(batch[1])]).reshape(-1, 1).to(
                device)
            premises = [f"premise: {premise}" for premise in batch[1]]
            hypotheses = [f"hypothesis: {hypothesis}" for hypothesis in batch[2]]
            model_input = tokenizer(text=premises, text_pair=hypotheses, max_length=2048, truncation='only_first',
                                    return_tensors='pt', padding=True).to(device)
            logits = model(**model_input, decoder_input_ids=decoder_input_ids)[0].detach()
            probs = softmax(logits.cpu().numpy(), axis=-1)
            one_token_id = tokenizer('1').input_ids[0]
            entailment_prob = probs[:, 0, one_token_id]
            probs_list += entailment_prob.tolist()
            batch_labels = batch[3]
            for dataset_name, label, prob in zip(datasets_names_batch, batch_labels, entailment_prob):
                if dataset_name not in per_dataset_labels_and_probs:
                    per_dataset_labels_and_probs[dataset_name] = {'labels': [], 'probs': []}
                per_dataset_labels_and_probs[dataset_name]['labels'].append(label)
                per_dataset_labels_and_probs[dataset_name]['probs'].append(prob)
            labels += batch_labels
        results_dict = {}
        for dataset_name in per_dataset_labels_and_probs:
            print(dataset_name)
            roc_auc = roc_auc_score(per_dataset_labels_and_probs[dataset_name]['labels'],
                                    per_dataset_labels_and_probs[dataset_name]['probs'])
            accuracy = sum([1 if (per_dataset_labels_and_probs[dataset_name]['probs'][i] > 0.5 and
                                  per_dataset_labels_and_probs[dataset_name]['labels'][i] == 1) or (
                                         per_dataset_labels_and_probs[dataset_name]['probs'][i] < 0.5 and
                                         per_dataset_labels_and_probs[dataset_name]['labels'][i] == 0) else 0 for
                            i in range(len(per_dataset_labels_and_probs[dataset_name]['labels']))]) / len(
                per_dataset_labels_and_probs[dataset_name]['labels'])
            results_dict[dataset_name] = {'roc auc': roc_auc, 'accuracy': accuracy}
            print("Roc Auc Score: ", roc_auc)
            print("Accuracy: ", accuracy)
        df = pd.DataFrame.from_records(
            [(dataset_name, roc_auc_score(per_dataset_labels_and_probs[dataset_name]['labels'],
                                          per_dataset_labels_and_probs[dataset_name]['probs']))
             for dataset_name in per_dataset_labels_and_probs.keys()], columns=['dataset', 'roc auc'])
        roc_auc = roc_auc_score(labels, probs_list)
        accuracy = sum([1 if (probs_list[i] > 0.5 and labels[i] == 1) or (probs_list[i] < 0.5 and labels[i] == 0) else 0
                        for i in range(len(labels))]) / len(labels)
        results_dict['total'] = {'roc auc': roc_auc, 'accuracy': accuracy}
        print("Roc Auc Score: ", roc_auc)
        print("Accuracy: ", accuracy)
        model.refine()
        return df, results_dict


import os
os.chdir('../')
import sys
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
import evaluate
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from torch import nn


class Summarization_dataset(Dataset):
    def __init__(self, tokenizer, split='train'):
        self.dataset_xsum = load_dataset('xsum', split=split)
        self.dataset_cnn = load_dataset("cnn_dailymail", version="3.0.0", split=split)
        lengths_array = [len(self.dataset_xsum), len(self.dataset_cnn)]
        self.start_indexes = []
        for i in range(len(lengths_array)):
            self.start_indexes.append(sum(lengths_array[:i + 1]))
        self.tokenizer = tokenizer

    def __len__(self):
        #return sum([len(self.dataset_xsum), len(self.dataset_cnn)])
        return 30
    def __getitem__(self, item):
        if item < self.start_indexes[0]:
            row = self.dataset_xsum[item]
            document = row['document']
            summary = row['summary']
        else:
            row = self.dataset_cnn[item - self.start_indexes[0]]
            document = row['article']
            summary = row['highlights']
        return {'document': document, 'summary': summary}


def collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize " + prefix + ':' + row['document'] for row in batch]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


class T5_Trainer(Seq2SeqTrainer):
    def __init__(self, max_length_train=512, max_length_eval=2048, **kwargs):
        self.max_length_train = max_length_train
        self.max_length_eval = max_length_eval
        super(T5_Trainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                          collate_fn=lambda x: collate_fn(x, self.tokenizer, self.max_length_train,
                                                          prefix='consistent '), pin_memory=False)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                          collate_fn=lambda x: collate_fn(x, self.tokenizer, self.max_length_eval,
                                                          prefix='consistent '), pin_memory=False)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        loss, logits, labels = super(T5_Trainer, self).prediction_step(model, inputs, prediction_loss_only,
                                                                       ignore_keys=ignore_keys)
        return (loss, torch.argmax(logits[0], dim=-1), labels)


class Rouge():
    def __init__(self):
        self.rouge = evaluate.load('rouge')

    def compute_metrics(self, pred, tokenizer):
        pred.predictions[pred.predictions == -100] = tokenizer.pad_token_id
        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id
        labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        predictions = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        results = self.rouge.compute(predictions=predictions,
                                     references=labels)
        return results


def main():
    os.environ["WANDB_DISABLED"] = "true"

    model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-base")
    train_dataset = Summarization_dataset(tokenizer, split='train')
    eval_dataset = Summarization_dataset(tokenizer, split='validation')
    current_datetime = datetime.now()

    current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    run_name = f"run name_{current_datetime_string}"
    args = Seq2SeqTrainingArguments(output_dir=f'../knowledge_distillation/summarization_results/{run_name}',
                                    do_train=True, do_eval=True,
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    gradient_accumulation_steps=2,
                                    learning_rate=1e-4, num_train_epochs=4, save_total_limit=3,
                                    load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
                                    eval_steps=1, save_steps=10000, eval_accumulation_steps=30,
                                    metric_for_best_model='rougeL', no_cuda=False)
    max_length_train = 512
    max_length_eval = 2048
    evaluation_metric = Rouge()
    trainer = T5_Trainer(model=model, tokenizer=tokenizer, args=args, train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=lambda p: evaluation_metric.compute_metrics(p, tokenizer),
                         max_length_train=max_length_train, max_length_eval=max_length_eval)
    trainer.train()

    # current_datetime = datetime.now()
    #
    # current_datetime_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # run_name = f"run name_{current_datetime_string}"
    # args = Seq2SeqTrainingArguments(output_dir=f'../TrueTeacher/results/{run_name}', do_train=True, do_eval=True,
    #                                 per_device_train_batch_size=16,
    #                                 per_device_eval_batch_size=4,
    #                                 learning_rate=1e-4, num_train_epochs=20, save_total_limit=1,
    #                                 load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
    #                                 eval_steps=1000, save_steps=1000, eval_accumulation_steps=100,
    #                                 metric_for_best_model='roc auc',no_cuda = False)
    # trainer = T5_Trainer(model=model, tokenizer=tokenizer, args=args, train_dataset=dataset,
    #                      eval_dataset=eval_dataset, compute_metrics=lambda p: compute_metrics(p, tokenizer))
    # trainer.train()


if __name__ == '__main__':
    main()

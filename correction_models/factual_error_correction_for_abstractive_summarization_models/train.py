import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from general.bart_trainer import BartTrainer
from torch.utils.data import Dataset
import json
from datasets import load_dataset
import nltk
import evaluate
from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments


class AugmentedDataCnn(Dataset):
    def __init__(self, data_dir, split):
        data_dir += f'/{split}'
        source_dataset = load_dataset("cnn_dailymail", "3.0.0", split=split)
        self.id_to_article = {i: source_dataset[i]['article'] for i in range(len(source_dataset))}
        self.id_to_summary = {i: source_dataset[i]['highlights'] for i in range(len(source_dataset))}
        self.clean = [(i, self.id_to_summary[i]) for i in range(len(source_dataset))]
        self.numswp = []
        with open(data_dir + '/numswp.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                data_id = data['id']
                claim = data['claim']
                self.numswp.append((data_id, claim))
        self.dateswp = []
        with open(data_dir + '/dateswp.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                data_id = data['id']
                claim = data['claim']
                self.dateswp.append((data_id, claim))
        self.pronoun = []
        with open(data_dir + '/pronoun.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                data_id = data['id']
                claim = data['claim']
                self.pronoun.append((data_id, claim))
        self.entityswap = []
        with open(data_dir + '/entswp.jsonl', 'r') as file:
            for line in file:
                data = json.loads(line)
                data_id = data['id']
                claim = data['claim']
                self.entityswap.append((data_id, claim))

    def __len__(self):
        return len(self.clean) + len(self.numswp) + len(self.dateswp) + len(self.pronoun) + len(self.entityswap)

    def __getitem__(self, idx):
        if idx < len(self.clean):
            item = self.clean[idx]
        elif idx < len(self.clean) + len(self.numswp):
            item = self.numswp[idx - len(self.clean)]
        elif idx < len(self.clean) + len(self.numswp) + len(self.dateswp):
            item = self.dateswp[idx - len(self.clean) - len(self.numswp)]
        elif idx < len(self.clean) + len(self.numswp) + len(self.dateswp) + len(self.pronoun):
            item = self.pronoun[idx - len(self.clean) - len(self.numswp) - len(self.dateswp)]
        else:
            item = self.entityswap[idx - len(self.clean) - len(self.numswp) - len(self.dateswp) - len(self.pronoun)]
        "Return the article, the augmentation, and the summary"
        return self.id_to_article[item[0]], item[1], self.id_to_summary[item[0]]

    def adjustment_to_paper(self):
        num_of_clean = 201644
        num_of_date = 16858
        num_of_num = 35113
        num_of_pronoun = 13408
        num_of_entity = 20204
        used_ids = set()
        temp = self.dateswp
        self.dateswp = []
        for i in range(len(self.clean)):
            id = temp[i][0]
            used_ids.add(id)
            self.dateswp.append(temp[i])
            if len(self.dateswp) == num_of_date:
                break
        temp = self.numswp
        self.numswp = []
        for i in range(len(self.clean)):
            id = temp[i][0]
            if id in used_ids:
                continue
            used_ids.add(id)
            self.numswp.append(temp[i])
            if len(self.numswp) == num_of_num:
                break
        temp = self.pronoun
        self.pronoun = []
        for i in range(len(self.clean)):
            id = temp[i][0]
            if id in used_ids:
                continue
            used_ids.add(id)
            self.pronoun.append(temp[i])
            if len(self.pronoun) == num_of_pronoun:
                break
        temp = self.entityswap
        self.entityswap = []
        for i in range(len(self.clean)):
            id = temp[i][0]
            if id in used_ids:
                continue
            used_ids.add(id)
            self.entityswap.append(temp[i])
            if len(self.entityswap) == num_of_entity:
                break
        temp = self.clean
        self.clean = []
        for i in range(len(temp)):
            id = temp[i][0]
            if id in used_ids:
                continue
            self.clean.append(temp[i])


def collate_fn(batch, tokenizer, max_length):
    torch.cuda.empty_cache()
    articles = [row[0] for row in batch]
    augmentations = [row[1] for row in batch]
    summaries = [row[2] for row in batch]
    text_input = [augmentations[i] + " <sep> " + articles[i] for i in range(len(batch))]
    inputs = tokenizer(text_input, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length,
                       return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def compute_metrics(p, tokenizer, rouge):
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    results = rouge.compute(predictions=predictions, references=labels)
    return results


def train_full():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    rouge = evaluate.load('rouge')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    args = Seq2SeqTrainingArguments(
        output_dir=f'correction_models/factual_error_correction_for_abstractive_summarization_models/checkpoints/bart_base_cnn_full_data',
        do_train=True, do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=3e-5, num_train_epochs=3, save_total_limit=2,
        load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
        eval_steps=50000, save_steps=50000, eval_accumulation_steps=30,
        metric_for_best_model='rougeL', no_cuda=False, generation_max_length=128, predict_with_generate=True)
    data_path = 'data/factual_error_correction_for_abstractive_summarization_models'
    train_dataset = AugmentedDataCnn(data_path, 'train')
    eval_dataset = AugmentedDataCnn(data_path, 'validation')
    trainer = BartTrainer(model=model, tokenizer=tokenizer, args=args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge), collate_fn=collate_fn,
                          max_length=512)
    trainer.train()


def train_by_split():
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    rouge = evaluate.load('rouge')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    args = Seq2SeqTrainingArguments(
        output_dir=f'correction_models/factual_error_correction_for_abstractive_summarization_models/checkpoints/bart_base_cnn_full_data',
        do_train=True, do_eval=True,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=3e-5, num_train_epochs=10, save_total_limit=2,
        load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
        eval_steps=50000, save_steps=50000, eval_accumulation_steps=30,
        metric_for_best_model='rougeL', no_cuda=False, generation_max_length=128, predict_with_generate=True)
    data_path = '/data/factual_error_correction_for_abstractive_summarization_models'
    train_dataset = AugmentedDataCnn(data_path, 'train')
    train_dataset.adjustment_to_paper()
    eval_dataset = AugmentedDataCnn(data_path, 'validation')
    trainer = BartTrainer(model=model, tokenizer=tokenizer, args=args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge), collate_fn=collate_fn,
                          max_length=512)
    trainer.train()

def revise(texts, summaries, model, tokenizer, device='cpu', batch_size=128, max_length=128):
    from general.utils import iter_list
    from tqdm import tqdm
    model_inputs = [summaries[i] + " <sep> " + texts[i] for i in range(len(texts))]
    revised_summaries = []
    model.to(device)
    with torch.no_grad():
        model.eval()
        for batch_model_inputs in tqdm(iter_list(model_inputs, batch_size=batch_size)):
            tokenized = tokenizer(batch_model_inputs, padding=True, truncation=True, max_length=512,
                                  return_tensors='pt').to(device)
            outputs = model.generate(**tokenized, max_length=max_length)
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            revised_summaries += batch_summaries
    return revised_summaries

def eval_on_frank():
    import numpy as np
    model = BartForConditionalGeneration.from_pretrained(
        'correction_models/factual_error_correction_for_abstractive_summarization_models/checkpoints/bart_base_cnn_split_data/checkpoint-100000')
    tokenizer = BartTokenizer.from_pretrained(
        'correction_models/factual_error_correction_for_abstractive_summarization_models/checkpoints/bart_base_cnn_split_data/checkpoint-100000')
    import pandas as pd
    from Seahorse_metrics.metrics import Seahorse_metrics
    df = pd.read_json('data/frank_raw/benchmark_data.json')
    from factCC.inference import Factcc_classifier
    # classifier = Factcc_classifier(checkpoint_path='factCC/checkpoints/factcc-checkpoint',device= 'cuda:1',batch_size=8)
    classifier = Seahorse_metrics(model_path='google/seahorse-large-q4', tokenizer_name='google/seahorse-large-q4',
                                  device='auto', batch_size=8, max_length=2048)
    for model_name in df['model_name'].unique():
        temp_df = df[df['model_name'] == model_name]
        texts = temp_df['article'].tolist()
        summaries = temp_df['summary'].tolist()
        model_revisions = revise(texts, summaries, model, tokenizer, device='cuda:1', batch_size=8, max_length=128)
        pre_revision_scores = classifier.score(texts, summaries)
        post_revision_scores = classifier.score(texts, model_revisions)
        print(model_name)
        print(np.mean(pre_revision_scores))
        print(np.mean(post_revision_scores))


def main():
    eval_on_frank()


print(os.getcwd())
main()

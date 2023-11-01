import os

os.chdir('../')

import pandas as pd
from factCC.inference import Factcc_clasifier
from q_squared.inference import Q_squared_classifier
from TrueTeacher.inference import TrueTeacher
from text_correction import Text_correction_model
from data.factuality_datasets import TRUE_dataset, Dataset_no_labels
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import csv
from kd_utils import Regular_dataset
import os
import torch
from argparse import ArgumentParser
from transformers import T5Tokenizer, T5ForConditionalGeneration


def create_correction_model(args):
    if args.correction_model == 'llm':
        correction_model = Text_correction_model(prompt_path=args.text_correction_prompt_path, model=args.llm_model,
                                                 API_KEY=args.api_key)
        kwargs = {}
        return correction_model, {}
    elif args.correction_model == 'pipeline':
        correction_model, qg_kwargs, qa_kwargs, revision_kwargs = create_correction_model(args)
        kwargs = {'qg_kwargs': qg_kwargs, 'qa_kwargs': qa_kwargs, 'revision_kwargs': revision_kwargs}

    else:
        raise ValueError("No such Correction model!")
    return correction_model, kwargs


def load_dataset(args):
    if args.dataset_name == 'TRUE':
        dataset = TRUE_dataset(args.data_dir_path, ['summarization'])
        # datasets_names = true_topics(['summarization'])
        # dataset.filter_to_datasets(datasets_names)
        dataset = Dataset_no_labels(dataset.df)
    else:
        raise ValueError('No such dataset exist')
    return dataset


def load_model(args):
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    return model, tokenizer


def load_factuality_classifier(args):
    if args.factuality_classifier == 'factCC':
        factuality_classifier = Factcc_clasifier(checkpoint_path=args.path_to_factcc_checkpoint, device=args.device)
    elif args.factuality_classifier == 'q_squared':
        factuality_classifier = Q_squared_classifier(device=args.device, similarity_metric=args.similarity_metric,
                                                     threshold=args.threshold)
    elif args.factuality_classifier == 'TrueTeacher':
        factuality_classifier = TrueTeacher(device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    else:
        raise ValueError('No such factuality classifier!')
    return factuality_classifier


def collate_fn(batch):
    return batch


class Trainer():
    def __init__(self, args):
        self.args = args
        self.correction_model, self.correction_model_kwargs = create_correction_model(args)
        self.model, self.tokenizer = load_model(args)
        self.factuality_classifier = load_factuality_classifier(args)
        dataset = load_dataset(args)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        # self.epochs = args.epochs
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        # self.train_method = args.train_method
        self.loss = CrossEntropyLoss()
        if self.args.train_method == 'control':
            with open(self.args.control_save_path, 'w') as f:
                self.writer = csv.writer(f)
                self.writer.writerow(('text', 'summary', 'factuality score'))

    def train(self):
        for epoch in range(self.args.epochs):
            for original_texts in self.dataloader:
                model_inputs = self.tokenizer(original_texts, padding=True, return_tensors='pt', truncation=True,
                                              max_length=self.args.train_max_length)
                model_scores_outputs, texts_outputs = self.get_model_scores_and_output_texts(model_inputs)
                factuality_scores = self.check_factuality(original_texts, texts_outputs)
                inconsistent_texts = [texts_outputs[~i] for i in factuality_scores]
                corrected_texts = self.correction_model(inconsistent_texts, self.correction_model_kwargs)
                if self.args.train_method == 'distillation':
                    inconsistent_scores_outputs = [model_scores_outputs[~i] for i in factuality_scores]
                    self.calculate_distillation_loss_and_update_model(inconsistent_scores_outputs, corrected_texts)
                elif self.args.train_method == 'control':
                    self.save_for_further_fine_tuning(original_texts, factuality_scores, texts_outputs)
                else:
                    raise ValueError('Such training method does not exist!')
            if self.args.train_method == 'control':
                self.fine_tune_for_control()

    #        self.evalaute_model()

    def calculate_distillation_loss_and_update_model(self, inconsistent_model_scores, corrected_texts):
        self.optimizer.zero_grad()
        tokenized_gt_texts = self.tokenizer(corrected_texts)
        loss = self.loss.forward(inconsistent_model_scores, tokenized_gt_texts)
        loss.backward()
        self.optimizer.step()

    def save_for_further_fine_tuning(self, original_texts, factuality_scores, texts_outputs):
        for text, factuality_score, output in zip(original_texts, factuality_scores, texts_outputs):
            if factuality_score:
                text = self.args.consistent_token + text
            else:
                text = self.args.inconsistent_token + text
            self.writer.writerow((text, output, factuality_score))

    def fine_tune_for_control(self):
        df = pd.read_csv(self.args.control_save_path)
        texts = df['text']
        summaries = df['summary']
        dataset = Regular_dataset(texts=texts, labels=summaries)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=collate_fn)
        for batch in dataloader:
            self.optimizer.zero_grad()
            summaries, labels = batch
            model_input = self.tokenizer(summaries)
            gt = self.tokenizer(labels)
            model_scores_outputs, _ = self.get_model_scores_and_output_texts(model_input)
            loss = self.loss(model_scores_outputs, gt)
            loss.backward()
            self.optimizer.step()
        # if self.args.control_method == 'dual_input':
        #     for inconsistent_label
        # #self.model()

    def reset_csv(self):
        os.remove(self.args.control_save_path)
        with open(self.args.control_save_path, 'w') as f:
            self.writer = csv.writer(f)
            self.writer.writerow(('text', 'summary', 'factuality score'))

    def check_factuality(self, original_texts, texts_outputs):
        factuality_scores = self.factuality_classifier.apply(original_texts, texts_outputs)
        return factuality_scores

    def get_model_scores_and_output_texts(self, model_inputs):
        generation = self.model.generate(**model_inputs, output_scores=True, return_dict_in_generate=True,
                                         max_length=150)
        model_token_output, token_distribution = generation[0], generation[1]
        text_outputs = self.tokenizer.batch_decode(model_token_output, skip_special_tokens=True)
        model_scores_outputs = torch.stack(token_distribution, dim=1)
        # texts_outputs = self.tokenizer.batch_decode(model_scores_outputs, skip_special_tokens=True)
        return model_scores_outputs, text_outputs


def parser_args():
    args = ArgumentParser()
    args.add_argument('-dataset_name', type=str, default='TRUE')
    args.add_argument('-train_max_length', type=int, default=512)
    args.add_argument('-eval_max_length', type=int, default=2048)
    args.add_argument('-batch_size', type=int, default=8)
    args.add_argument('-epochs', type=int, default=10)
    args.add_argument('-lr', type=float, default=1e-5)
    args.add_argument('-tokenizer_path', type=str, default="mrm8488/t5-base-finetuned-summarize-news")
    args.add_argument('-model_path', type=str, default="mrm8488/t5-base-finetuned-summarize-news")
    args.add_argument('-data_dir_path', type=str, default='data')
    args.add_argument('-factuality_classifier', type=str, default='factCC')
    args.add_argument('-path_to_factcc_checkpoint', type=str, default='factCC/checkpoints/factcc-checkpoint')
    args.add_argument('-device', type=str, default='cpu')
    args.add_argument('-similarity_metric', type=str, default='cosine')
    args.add_argument('-threshold', type=float, default=0.5)
    args.add_argument('-train_method', type=str, default='distillation')
    args.add_argument('-control_save_path', type=str, default='control.csv')
    args.add_argument('-consistent_token', type=str, default='consistent')
    args.add_argument('-inconsistent_token', type=str, default='inconsistent')
    args.add_argument('-correction_model', type=str, default='llm')
    args.add_argument('-text_correction_prompt_path', default=None)
    args.add_argument('-llm_model', type=str, default='gpt-4')
    args.add_argument('-api_key', type=str, default=None)
    args = args.parse_args()
    return args


def main():
    args = parser_args()
    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()

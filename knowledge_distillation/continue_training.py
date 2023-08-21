import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (main_directory) and add it to the Python path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd

from correction_pipeline.text_correction_model import Text_correction_model
from transformers import AutoTokenizer, AutoModel
from data.utils import TRUE_dataset, true_topics, Dataset_no_labels
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import csv
from knowledge_distillation.utils import Regular_dataset
import os

def create_correction_model(args):
    if args.correction_model == 'llm':
        correction_model = Text_correction_model(prompt_path=args.text_correction_prompt_path, model=args.llm_model,
                                                 API_KEY=args.api_ley)
        kwargs = {}
        return correction_model, {}
    elif args.correction_model == 'pipeline':
        correction_model, qg_kwargs, qa_kwargs, revision_kwargs = create_correction_model(args)
        kwargs = {'qg_kwargs': qg_kwargs, 'qa_kwargs': qa_kwargs, 'revision_kwargs': revision_kwargs}

    else:
        raise ValueError("No such Correction model!")
    return correction_model, kwargs


def load_dataset(args):
    if args.dataset_name == 'true':
        dataset = TRUE_dataset(args.data_dir_path)
        datasets_names = true_topics(['summarization'])
        dataset.filter_to_datasets(datasets_names)
        dataset = Dataset_no_labels(dataset.df)
    else:
        raise ValueError('No such dataset exist')
    return dataset


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModel.from_pretrained(args.model_path)
    return tokenizer, model


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
                self.writer.writerow(('text','summary','factuality score'))


    def train(self):
        for epoch in self.args.epochs:
            for original_texts in self.dataloader:
                model_inputs = self.tokenizer.batch_encode(original_texts)
                model_scores_outputs, texts_outputs = get_model_scores_and_output_texts(model_inputs)
                factuality_scores = self.check_factuality(original_texts, texts_outputs)
                inconsistent_texts = [texts_outputs[~i] for i in factuality_scores]
                corrected_texts = self.correction_model(inconsistent_texts, self.correction_model_kwargs)
                if self.args.train_method == 'distillation':
                    inconsistent_scores_outputs = [model_scores_outputs[~i] for i in factuality_scores]
                    self.calculate_distillation_loss_and_update_model(inconsistent_scores_outputs, corrected_texts)
                elif self.args.train_method == 'control':
                    self.save_for_further_fine_tuning(original_texts,factuality_scores,texts_outputs)
                else:
                    raise ValueError('Such training method does not exist!')
            if self.args.train_method == 'control':
                self.fine_tune_for_control()
#        self.evalaute_model()

    def calculate_distillation_loss_and_update_model(self, inconsistent_model_scores, corrected_texts):
        tokenized_gt_texts = self.tokenizer(corrected_texts)
        loss = self.loss.forward(inconsistent_model_scores, tokenized_gt_texts)
        loss.backward()
        self.optimizer.step()

    def save_for_further_fine_tuning(self, original_texts,factuality_scores,texts_outputs):
        for text,factuality_score,output in zip(original_texts,factuality_scores,texts_outputs):
            if factuality_score:
                text = self.args.consistent_token + text
            else:
                text = self.args.inconsistent_token + text
            self.writer.writerow((text,output,factuality_score))
    def fine_tune_for_control(self):
        df = pd.read_csv(self.args.control_save_path)
        texts = df['text']
        summaries = df['summary']
        dataset = Regular_dataset(texts= texts,labels=summaries)
        dataloader = DataLoader(dataset,batch_size=self.args.batch_size,collate_fn=collate_fn)
        for batch in dataloader:
            summaries,labels = batch
            model_input = self.tokenizer(summaries)
            gt = self.tokenizer(labels)
            model_scores_outputs, _ = get_model_scores_and_output_texts(model_input)
            loss = self.loss(model_scores_outputs,gt)
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

def train(args):
    # trainer = Trainer(args)
    # m = T5ForConditionalGeneration.from_pretrained("t5-small")
    # t = T5Tokenizer.from_pretrained("t5-small")
    # text = "translate english"
    # model_input = t("translate English to German: The house is wonderful.", return_tensors="pt")
    # output = m.generate(**model_input).logits
    c = 1
    # if args.train_method == "distillation":
    #
    # elif args.train_method == "control":
    #     train_with_control()


train(None)

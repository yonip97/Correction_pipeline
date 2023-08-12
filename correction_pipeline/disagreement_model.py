import numpy as np
import re
import string
from collections import Counter


from bert_score import score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from allennlp.predictors.predictor import Predictor
from utils import iter_list


class Disagreement_model_nli_based():
    def __init__(self, model_type='hf', confidence_cutoff=0.8, batch_size=16, device='cpu'):
        self.model_type = model_type
        self.cutoff = confidence_cutoff
        self.device = device
        self.batch_size = batch_size
        if self.model_type == 'hf':
            hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
            self.tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name).to(device)
        elif self.model_type == 'allennlp':
            self.predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
                predictor_name="textual_entailment")

    def __call__(self, questions, answers_based_on_generated_text, answers_based_on_original_text,**kwargs):
        indexes = []
        nli_scores = self.nli_score(questions, answers_based_on_generated_text, answers_based_on_original_text,**kwargs)
        # This means that if the result was contradiction,
        # strong disagreement was detected, and therefore the result is disagreement
        for i in range(len(questions)):
            if nli_scores[i][2] >= self.cutoff:
                indexes.append(i)
        return indexes

    def clean_text(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b(a|an|the|in|our)\b', ' ', text)
        return re.sub(' +', ' ', text).strip()

    def nli_score(self, questions, preds, reals,**kwargs):
        if self.model_type == 'hf':
            return self.hf_nli_score(preds, reals, questions,**kwargs)
        elif self.model_type == 'allennlp':
            return self.allennlp_nli_score(preds, reals, questions)

    def allennlp_nli_score(self, preds, reals, questions):
        distributions = []
        for q, p, r in zip(questions, preds, reals):
            premise = q + ' ' + r + '.'
            hypothesis = q + ' ' + p + '.'

            res = self.predictor.predict(
                premise=premise,
                hypothesis=hypothesis
            )
            distributions.append(res['probs'])
        return np.stack(distributions)

    def hf_nli_score(self, preds, reals, questions,**kwargs):
        predicted_probabilities = []
        with torch.no_grad():
            for indexes in iter_list(range(len(questions)), self.batch_size):
                model_input = []
                for i in indexes:
                    premise = questions[i] + ' ' + reals[i] + '.'
                    hypothesis = questions[i] + ' ' + preds[i] + '.'
                    model_input.append((premise, hypothesis))
                tokenized_input_seq_pair = self.tokenizer.batch_encode_plus(model_input,
                                                                            return_token_type_ids=True, padding=True)

                input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long()
                token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long()
                attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long()
                outputs = self.model(input_ids.to(self.device),
                                     attention_mask=attention_mask.to(self.device),
                                     token_type_ids=token_type_ids.to(self.device),
                                     labels=None)
                batch_predicted_probabilities = torch.softmax(outputs[0], dim=1)
                predicted_probabilities.append(batch_predicted_probabilities)
        predicted_probabilities = torch.cat(predicted_probabilities)
        return predicted_probabilities

    def f1_score(self, a_gold, a_pred):
        if a_pred == '':
            return 0
        gold_toks = self.clean_text(a_gold).split()
        pred_toks = self.clean_text(a_pred).split()
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def f1_bert_score(self, a_gold, a_pred):
        P, R, F1 = score(a_pred, a_gold, lang="en", verbose=True)
        return F1.mean().item()

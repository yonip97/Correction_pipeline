from transformers import Seq2SeqTrainer
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
from general.utils import iter_list
import evaluate
import numpy as np
from torch.utils.data import Dataset


def compute_metric_rouge(p, tokenizer):
    rouge = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.compute(predictions=predictions, references=labels)
    return results


def collate_fn_revision(batch, tokenizer, max_length):
    text_inputs = [("revise: summary: " + row['summary'], " text: " + row['text']) for row in batch]
    revised_summaries = [row['revised_summary'] for row in batch]
    inputs = tokenizer.batch_encode_plus(text_inputs, padding=True, truncation='only_second', max_length=max_length,
                                         return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def collate_fn_revision_test(batch, tokenizer, max_length):
    text_inputs = [("revise: summary: " + row['summary'], " text: " + row['text']) for row in batch]
    inputs = tokenizer.batch_encode_plus(text_inputs, padding=True, truncation='only_second', max_length=max_length,
                                         return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


class T5_Trainer(Seq2SeqTrainer):
    def __init__(self, collate_fn, max_length_train=512, max_length_eval=512, collate_fn_eval=None,
                 collate_fn_test=None, **kwargs):
        self.collate_fn = collate_fn
        self.max_length_train = max_length_train
        self.max_length_eval = max_length_eval
        self.collate_fn_eval = collate_fn_eval
        self.collate_fn_test = collate_fn_test
        super(T5_Trainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                          collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_train),
                          pin_memory=False)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None:
            if self.collate_fn_eval is None:
                return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)
            else:
                return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn_eval(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)
        else:
            if self.collate_fn_eval is None:
                return DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)
            else:
                return DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn_eval(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if self.collate_fn_test is None:
            return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                              collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval),
                              pin_memory=False)
        else:
            return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                              collate_fn=lambda x: self.collate_fn_test(x, self.tokenizer, self.max_length_eval),
                              pin_memory=False)

    # def prediction_step(
    #         self,
    #         model: nn.Module,
    #         inputs: Dict[str, Union[torch.Tensor, Any]],
    #         prediction_loss_only: bool,
    #         ignore_keys: Optional[List[str]] = None
    # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     loss, logits, labels = super(T5_Trainer, self).prediction_step(model, inputs, prediction_loss_only,
    #                                                                    ignore_keys=ignore_keys)
    #     #return (loss, torch.argmax(logits[0], dim=-1), labels)
    #     return (loss, logits, labels)


def t5_summarize(texts, model, tokenizer, prompt, device='cpu', batch_size=128, max_generation_length=128, beam_size=4,
                 early_stopping=True, length_penalty=0.6, max_encoding_length=512):
    summaries = []
    model_inputs = [(prompt + text) for text in texts]
    model.eval()
    # model.to(device)
    with torch.no_grad():
        model.eval()
        for batch_texts in tqdm(iter_list(model_inputs, batch_size=batch_size)):
            tokenized = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_encoding_length,
                                  return_tensors='pt').to(
                device)
            outputs = model.generate(**tokenized, max_length=max_generation_length, num_beams=beam_size,
                                     early_stopping=early_stopping, length_penalty=length_penalty)
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries += batch_summaries
    model.train()
    return summaries


def t5_revise(texts, summaries, model, tokenizer, prompt, device='cpu', batch_size=128,
              generation_max_length=128, num_beams=1,
              early_stopping=False, encoding_max_length=512):
    model_inputs = [(f'{prompt}: summary: ' + summary, "text: " + text) for text, summary in zip(texts, summaries)]
    revised_summaries = []
    # model.to(device)
    model.eval()
    with torch.no_grad():
        model.eval()
        for batch_model_inputs in tqdm(iter_list(model_inputs, batch_size=batch_size)):
            tokenized = tokenizer.batch_encode_plus(batch_model_inputs, padding=True, truncation="only_second",
                                                    max_length=encoding_max_length, return_tensors='pt').to(
                device)
            outputs = model.generate(**tokenized, max_length=generation_max_length, num_beams=num_beams,
                                     early_stopping=early_stopping)
            batch_revised_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            revised_summaries += batch_revised_summaries
            torch.cuda.empty_cache()
    model.train()
    return revised_summaries

from transformers import Seq2SeqTrainer
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
from general.utils import iter_list


class T5_Trainer(Seq2SeqTrainer):
    def __init__(self, collate_fn, max_length_train=512, max_length_eval=512, **kwargs):
        self.collate_fn = collate_fn
        self.max_length_train = max_length_train
        self.max_length_eval = max_length_eval
        super(T5_Trainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                          collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_train),
                          pin_memory=False)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                          collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval),
                          pin_memory=False)

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


def summarize(texts, model, tokenizer, device='cpu', batch_size=128, max_length=128):
    summaries = []
    model.to(device)
    with torch.no_grad():
        model.eval()
        for batch_texts in tqdm(iter_list(texts, batch_size=batch_size)):
            tokenized = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(
                device)
            outputs = model.generate(**tokenized, max_length=max_length)
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries += batch_summaries
    return summaries


def revise(texts, summaries, model, tokenizer, device='cpu', batch_size=128, max_length=128):
    model_inputs = [('revise: summary: ' + summary, "document: " + text) for text, summary in zip(texts, summaries)]
    revised_summaries = []
    model.to(device)
    with torch.no_grad():
        model.eval()
        for batch_model_inputs in tqdm(iter_list(model_inputs, batch_size=batch_size)):
            tokenized = tokenizer.batch_encode_plus(batch_model_inputs, padding=True, truncation="only_second",
                                                    max_length=512, return_tensors='pt').to(
                device)
            outputs = model.generate(**tokenized, max_length=max_length)
            batch_revised_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            revised_summaries += batch_revised_summaries
    return revised_summaries

import math
import pickle

from transformers import Seq2SeqTrainer
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
from general.utils import iter_list, setup, cleanup, SummarizationDataset
from torch.nn.parallel import DistributedDataParallel as DDP
import evaluate
import numpy as np
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from transformers import TrainerCallback, T5ForConditionalGeneration, T5Tokenizer
import wandb
import os
import shutil
from torch.nn.functional import log_softmax, softmax


class WandbCallback(TrainerCallback):

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # Log the evaluation metrics to wandb
        log_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, list):
                print(f"The key {key} is excluded")
                continue
            log_metrics[key] = value
        wandb.log(log_metrics)


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


def collate_fn_revision(batch, tokenizer, max_length, introduction_prompt="revise: "):
    text_inputs = [(introduction_prompt + "summary: " + row['summary'], " text: " + row['text']) for row in batch]
    revised_summaries = [row['revised_summary'] for row in batch]
    inputs = tokenizer.batch_encode_plus(text_inputs, padding=True, truncation='only_second', max_length=max_length,
                                         return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def collate_fn_revision_test(batch, tokenizer, max_length, introduction_prompt="revise: "):
    text_inputs = [(introduction_prompt + "summary: " + row['summary'], " text: " + row['text']) for row in batch]
    inputs = tokenizer.batch_encode_plus(text_inputs, padding=True, truncation='only_second', max_length=max_length,
                                         return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


def collate_fn_summarization(batch, tokenizer, max_length, introduction_prompt="summarize: "):
    documents = [introduction_prompt + row['text'] for row in batch]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def collate_fn_summarization_distillation(batch, tokenizer, max_length, introduction_prompt="summarize: "):
    documents_with_logits = [introduction_prompt + row['text'] for row in batch if row['logits'] is not None]
    documents_without_logits = [introduction_prompt + row['text'] for row in batch if row['logits'] is None]
    documents = documents_with_logits + documents_without_logits
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    logits = [row['logits'] for row in batch if row['logits'] is not None] + [None] * len(documents_without_logits)
    summaries_for_documents_with_logits = [row['summary'] for row in batch if row['logits'] is not None]
    summaries_for_documents_without_logits = [row['summary'] for row in batch if row['logits'] is None]
    tokenized_summaries = tokenizer(summaries_for_documents_with_logits + summaries_for_documents_without_logits,
                                    padding=True,
                                    truncation=True, max_length=max_length, return_tensors='pt')
    tokenized_summaries[tokenized_summaries == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': tokenized_summaries['input_ids'], 'logits': logits}


def collate_fn_summarization_test(batch, tokenizer, max_length, introduction_prompt="summarize: "):
    documents = [introduction_prompt + row['text'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


class T5_Trainer(Seq2SeqTrainer):
    def __init__(self, collate_fn=None, max_length_train=512, max_length_eval=512, collate_fn_eval=None,
                 collate_fn_test=None, prompt_train=None, prompt_eval=None, prompt_test=None, distillation=False,
                 **kwargs):
        self.collate_fn = collate_fn
        self.max_length_train = max_length_train
        self.max_length_eval = max_length_eval
        self.collate_fn_eval = collate_fn_eval
        self.collate_fn_test = collate_fn_test
        self.prompt_train = prompt_train
        self.prompt_eval = prompt_eval
        self.prompt_test = prompt_test
        self.distillation = distillation
        super(T5_Trainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self.prompt_train is not None:
            return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                              collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_train,
                                                                   introduction_prompt=self.prompt_train),
                              pin_memory=False)
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                          collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_train),
                          pin_memory=False)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if self.collate_fn_eval is None:
            if self.prompt_train is None:
                return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)
            else:
                return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval,
                                                                       self.prompt_train), pin_memory=False)
        else:
            if self.prompt_eval is None:
                return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn_eval(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)
            else:
                return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn_eval(x, self.tokenizer, self.max_length_eval,
                                                                            self.prompt_eval), pin_memory=False)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if self.collate_fn_test is None:
            if self.prompt_train is None:
                return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)
            else:
                return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length_eval,
                                                                       self.prompt_train), pin_memory=False)
        else:
            if self.prompt_test is None:
                return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn_test(x, self.tokenizer, self.max_length_eval),
                                  pin_memory=False)
            else:
                return DataLoader(test_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                                  collate_fn=lambda x: self.collate_fn_test(x, self.tokenizer, self.max_length_eval,
                                                                            self.prompt_test),
                                  pin_memory=False)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.distillation:
            original_logits = inputs.pop('logits')
            outputs = super(T5_Trainer, self).compute_loss(model, inputs, return_outputs=return_outputs)
            ce_loss = outputs[0] if return_outputs else outputs
            ce_loss = ce_loss.mean()
            logits = [torch.Tensor(logit) for logit in original_logits if logit is not None]
            if len(logits) == 0:
                return ce_loss
            logits = torch.cat(logits)
            logits = logits.view(-1, logits.size(-1))
            logits = softmax(logits, dim=-1)
            outputs = model(**inputs)
            predicted_logits = outputs.logits
            predicted_logits = predicted_logits.reshape(-1, predicted_logits.size(-1))
            predicted_logits = torch.cat(
                [predicted_logits[:len(logit)] for logit in original_logits if logit is not None])
            predicted_logits = log_softmax(predicted_logits, dim=-1)
            loss_fct = nn.KLDivLoss(reduction='batchmean')
            distil_loss = loss_fct(predicted_logits, logits.to(predicted_logits.device))
            loss = ce_loss + distil_loss
            return (loss, outputs) if return_outputs else loss
        else:
            return super(T5_Trainer, self).compute_loss(model, inputs, return_outputs=return_outputs)

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
    dataset = SummarizationDataset(model_inputs, ['None'] * len(model_inputs))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_texts = batch['text']
            tokenized = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_encoding_length,
                                  return_tensors='pt').to(device)
            outputs = model.generate(**tokenized, max_length=max_generation_length, num_beams=beam_size,
                                     early_stopping=early_stopping, length_penalty=length_penalty)
            # outputs = model.generate(**tokenized)
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            summaries += batch_summaries
    model.train()
    return summaries


def t5_summarize_mp_main(model, tokenizer, texts, out_dir, prompt, batch_size, max_generation_length, beam_size,
                         early_stopping, length_penalty, max_encoding_length, min_generation_length):
    world_size = torch.cuda.device_count()
    os.makedirs(out_dir + '/summarization_temp')
    processes = [mp.Process(target=t5_summarize_mp, args=(i,
                                                          world_size, out_dir + '/summarization_temp', texts,
                                                          model, tokenizer, prompt,
                                                          batch_size,
                                                          max_generation_length,
                                                          beam_size, early_stopping,
                                                          length_penalty,
                                                          max_encoding_length,
                                                          min_generation_length)) for i in
                 range(world_size)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
    for process in processes:
        process.kill()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    new_model_summaries = []
    print("done")
    files = os.listdir(out_dir + '/summarization_temp')
    files = sorted(files)
    for file in files:
        with open(out_dir + '/summarization_temp/' + file, 'rb') as f:
            summaries = pickle.load(f)
            new_model_summaries += summaries
    shutil.rmtree(out_dir + '/summarization_temp')
    return new_model_summaries


def t5_summarize_mp(rank, world_size, output_dir, texts, model, tokenizer, prompt, batch_size=128,
                    max_generation_length=128,
                    beam_size=4,
                    early_stopping=True, length_penalty=0.6, max_encoding_length=512, min_generation_length=0):
    # setup(rank, world_size)
    print(rank)
    print(world_size)

    try:
        data_per_device_size = math.ceil(len(texts) / world_size)
        print(data_per_device_size)
        texts = texts[rank * data_per_device_size:(rank + 1) * data_per_device_size]
        summaries = []
        model_inputs = [(prompt + text) for text in texts]
        model.to(rank)
        model.eval()
        dataset = SummarizationDataset(model_inputs, ['None'] * len(model_inputs))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch_texts = batch['text']
                tokenized = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_encoding_length,
                                      return_tensors='pt').to(
                    rank)
                outputs = model.generate(**tokenized, max_length=max_generation_length, num_beams=beam_size,
                                         early_stopping=early_stopping, length_penalty=length_penalty,
                                         min_new_tokens=min_generation_length)
                batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                summaries += batch_summaries
        model.train()
        print("putting")
        with open(output_dir + f'/rank_{rank}.pkl', 'wb') as f:
            pickle.dump(summaries, f)
        # cleanup()
    except Exception as e:
        print(e)


def t5_revise(texts, summaries, model, tokenizer, prompt, device='cpu', batch_size=128,
              generation_max_length=128, num_beams=1,
              early_stopping=True, encoding_max_length=512, len_penalty=0.6, return_logits=False):
    model_inputs = [(f'{prompt} summary: ' + summary, "text: " + text) for text, summary in zip(texts, summaries)]
    revised_summaries = []
    logits = []
    model.eval()
    with torch.no_grad():
        for batch_model_inputs in tqdm(iter_list(model_inputs, batch_size=batch_size)):
            tokenized = tokenizer.batch_encode_plus(batch_model_inputs, padding=True, truncation="only_second",
                                                    max_length=encoding_max_length, return_tensors='pt').to(
                device)
            outputs = model.generate(**tokenized, max_length=generation_max_length, num_beams=num_beams,
                                     early_stopping=early_stopping, length_penalty=len_penalty,
                                     output_logits=True, output_scores=True,
                                     return_dict_in_generate=True
                                     )
            batch_logits = get_proper_scores(outputs.logits, outputs.beam_indices.tolist(),
                                             list(outputs.sequences_scores.size())[0])
            batch_revised_summaries = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            revised_summaries += batch_revised_summaries
            if return_logits:
                logits += batch_logits
            torch.cuda.empty_cache()
    model.train()
    if return_logits:
        return revised_summaries, logits
    return revised_summaries


def get_proper_scores(scores, beam_indices, batch_size):
    new_scores = []
    for score in scores:
        new_scores.append(score.cpu().numpy())
    final_logits = []
    for index in range(batch_size):
        sample_final_scores = []
        sample_beam_indices = beam_indices[index]
        sample_beam_indices = [x for x in sample_beam_indices if x != -1]
        for i in range(len(sample_beam_indices)):
            sample_final_scores.append(new_scores[i][sample_beam_indices[i]])
        final_logits.append(np.array(sample_final_scores))
    return final_logits


def t5_revise_mp_main(texts, summaries, args):
    world_size = torch.cuda.device_count()
    os.makedirs(args.output_dir + '/revision_temp')
    processes = [mp.Process(target=t5_revise_mp, args=(i,
                                                       world_size, args.revision_model_checkpoint,
                                                       args.output_dir + '/revision_temp', texts, summaries,
                                                       args.revision_prompt,
                                                       args.revision_batch_size,
                                                       args.revision_max_generation_length, args.revision_beam_size,
                                                       True,
                                                       args.revision_max_encoding_length,
                                                       args.revision_length_penalty,
                                                       args.revision_model_min_length, args.distillation)) for i in
                 range(world_size)]
    for process in processes:
        process.start()

    for process in processes:
        process.join()
    for process in processes:
        process.kill()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    new_model_summaries = []
    new_model_summaries_logits = []
    files = os.listdir(args.output_dir + '/revision_temp')
    files = sorted(files)
    for file in files:
        with open(args.output_dir + '/revision_temp/' + file, 'rb') as f:
            if args.distillation:
                summaries, logits = pickle.load(f)
                new_model_summaries_logits += logits
                new_model_summaries += summaries
            else:
                summaries = pickle.load(f)
                new_model_summaries += summaries
    shutil.rmtree(args.output_dir + '/revision_temp')
    if args.distillation:
        return new_model_summaries, new_model_summaries_logits
    return new_model_summaries, [None] * len(new_model_summaries)


def t5_revise_mp(rank, world_size, revision_model_path, output_dir, texts, summaries, prompt, batch_size=128,
                 generation_max_length=128, num_beams=1, early_stopping=True,
                 encoding_max_length=512, len_penalty=0.6, min_generation_length=0, return_logits=False):
    print("The min generation length is: ", min_generation_length)
    model = T5ForConditionalGeneration.from_pretrained(revision_model_path)
    tokenizer = T5Tokenizer.from_pretrained(revision_model_path)
    model.to(rank)
    data_per_device_size = math.ceil(len(texts) / world_size)
    model_inputs = [(f'{prompt} summary: ' + summary, "text: " + text) for text, summary in zip(texts, summaries)]
    model_inputs = model_inputs[rank * data_per_device_size:(rank + 1) * data_per_device_size]
    revised_summaries = []
    revised_logits = []
    model.eval()
    with torch.no_grad():
        for batch_model_inputs in tqdm(iter_list(model_inputs, batch_size=batch_size)):
            # tokenized = tokenizer.batch_encode_plus(batch_model_inputs, padding=True, truncation="only_second",
            #                                         max_length=encoding_max_length, return_tensors='pt').to(
            #     rank)
            # outputs = model.generate(**tokenized, max_length=generation_max_length, num_beams=num_beams,
            #                          early_stopping=early_stopping, length_penalty=len_penalty,
            #                          min_new_tokens=min_generation_length, output_logits=True, output_scores=True,
            #                          return_dict_in_generate=True)
            # batch_logits = get_proper_scores(outputs.logits, outputs.beam_indices.tolist(),  list(outputs.sequences_scores.size())[0])
            # if return_logits:
            #     revised_logits += batch_logits
            # batch_revised_summaries = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            # revised_summaries += batch_revised_summaries
            # torch.cuda.empty_cache()
            tokenized = tokenizer.batch_encode_plus(batch_model_inputs, padding=True, truncation="only_second",
                                                    max_length=encoding_max_length, return_tensors='pt').to(
                rank)
            generated_outputs = model.generate(**tokenized, max_length=generation_max_length, num_beams=num_beams,
                                               early_stopping=early_stopping, length_penalty=len_penalty,
                                               min_new_tokens=min_generation_length)
            batch_revised_summaries = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            generated_outputs = tokenizer(batch_revised_summaries, padding=True, truncation=True,
                                          max_length=generation_max_length,
                                          return_tensors='pt').to(rank)
            outputs = model(**tokenized, labels=generated_outputs['input_ids'])
            batch_logits = []
            for i in range(outputs.logits.shape[0]):
                mask = generated_outputs['attention_mask'][i] == 1
                logits = outputs.logits[i][mask]
                batch_logits.append(logits.detach().cpu())
            if return_logits:
                revised_logits += batch_logits
            revised_summaries += batch_revised_summaries
            torch.cuda.empty_cache()
    model.train()
    if return_logits:
        with open(output_dir + f'/rank_{rank}.pkl', 'wb') as f:
            pickle.dump((revised_summaries, revised_logits), f)
    else:
        with open(output_dir + f'/rank_{rank}.pkl', 'wb') as f:
            pickle.dump(revised_summaries, f)

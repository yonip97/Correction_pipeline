# import os
# import sys
#
# sys.path.append(os.path.dirname(os.getcwd()))
# os.chdir('../')

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from general.utils import iter_list
from data.factuality_datasets import TRUE_dataset
from tqdm import tqdm
from torch import nn
from accelerate import load_checkpoint_and_dispatch


class TrueTeacher():
    def __init__(self, model_path, tokenizer_name='t5-base', device='cpu', batch_size=8, max_length=2048,
                 torch_dtype=torch.float32, return_none=False):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.one_token = self.tokenizer('1').input_ids[0]
        if self.device == 'auto':
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, device_map='auto',
                                                                    torch_dtype=torch_dtype).eval()
            self.input_device = 'cpu'
        else:
            self.input_device = self.device
            self.model = T5ForConditionalGeneration.from_pretrained(model_path,
                                                                    torch_dtype=torch_dtype).to(self.device).eval()
        self.return_none = return_none

    def classify(self, texts, summaries):
        pairs = []
        results = []
        for summary, original_text in zip(summaries, texts):
            pairs.append(f'premise: {original_text} hypothesis: {summary}')
        with torch.no_grad():
            from tqdm import tqdm
            for batch in tqdm(iter_list(pairs, self.batch_size)):
                try:
                    model_input = self.tokenizer(batch, return_tensors='pt', truncation=True,
                                                 max_length=self.max_length,
                                                 padding=True).to(self.input_device)
                    outputs = self.model.generate(**model_input)
                    batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    for string in batch_results:
                        if string[0] == '1':
                            results.append(1)
                        elif string[0] == '0':
                            results.append(0)
                        else:
                            raise ValueError(f'Unexpected output: {string}')
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.return_none:
                        print("Out of memory. Trying to free up some GPU memory.")
                        # Free up memory (adjust as needed)
                        torch.cuda.empty_cache()
                        results += [None] * len(batch)
                    else:
                        raise e

            return results

    def classify_single(self, text, summary):
        return self.classify([summary], [text])[0]

    def score(self, texts, summaries):
        pairs = []
        results = []
        for summary, original_text in zip(summaries, texts):
            pairs.append(f'premise: {original_text} hypothesis: {summary}')
        with torch.no_grad():
            for batch in tqdm(iter_list(pairs, self.batch_size)):
                try:
                    model_input = self.tokenizer(batch, return_tensors='pt', truncation=True,
                                                 max_length=self.max_length,
                                                 padding=True).to(self.input_device)
                    decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id] * len(batch)]).reshape((-1, 1)).to(
                        self.input_device)
                    outputs = self.model(**model_input, decoder_input_ids=decoder_input_ids)
                    logits = torch.softmax(outputs.logits.float(), dim=-1)
                    torch.cuda.empty_cache()
                    batch_factuality_score = logits.detach().cpu()[:, 0, self.one_token]
                    results += batch_factuality_score.tolist()
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.return_none:
                        print("Out of memory. Trying to free up some GPU memory.")
                        # Free up memory (adjust as needed)
                        torch.cuda.empty_cache()
                        results += [None] * len(batch)
                    else:
                        raise e
        return results

    def score_single(self, text, summary):
        return self.score([summary], [text])[0]


def main():
    device = 'cuda'
    model_path = '/data/home/yehonatan-pe/Correction_pipeline/TrueTeacher/results/run name_2023-10-11 00:31:57/checkpoint-23000'
    # model_path = 'google/t5_11b_trueteacher_and_anli'
    # tokenizer_name = 'google/t5_11b_trueteacher_and_anli'
    tokenizer_name = 't5-base'
    model = TrueTeacher(model_path=model_path, tokenizer_name=tokenizer_name, device=device, batch_size=8,
                        max_length=2048)
    dataset = TRUE_dataset("data/true_data", ['summarization'])
    texts = dataset.df['grounding'].tolist()
    summaries = dataset.df['generated_text'].tolist()
    labels = dataset.df['label'].tolist()
    predictions = model.classify(summaries, texts)
    dataset['predictions'] = predictions
    accuracy = sum([1 if predictions[i] == labels[i] else 0 for i in range(len(labels))]) / len(labels)
    print(accuracy)
    # print(roc_auc_score(labels, predictions))


if __name__ == '__main__':
    pass
    # main()

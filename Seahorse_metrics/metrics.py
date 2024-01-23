import numpy as np
import pandas as pd
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from general.utils import iter_list
from tqdm import tqdm
import matplotlib.pyplot as plt


class Seahorse_metrics():
    def __init__(self, model_path, tokenizer_name, device='cpu', batch_size=8, max_length=2048,
                 torch_dtype=torch.float32, return_none=False):
        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.one_token = self.tokenizer('1').input_ids[0]
        if self.device == 'auto':
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path, device_map='auto',
                                                                     torch_dtype=torch_dtype).eval()
            self.input_device = 'cuda'
        else:
            self.input_device = self.device
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path,
                                                                     torch_dtype=torch_dtype).to(self.device).eval()
        self.return_none = return_none

    def score(self, texts, summaries):
        counter = 1
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
                    batch_factuality_score = logits.detach().cpu()[:, 0, self.one_token]

                    results += batch_factuality_score.tolist()
                    #torch.cuda.empty_cache()
                    if len(results) / 100 > counter:
                        counter += 1
                        print(np.nanmean(results))
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.return_none:
                        print("Out of memory. Trying to free up some GPU memory.")
                        # Free up memory (adjust as needed)
                        torch.cuda.empty_cache()
                        results += [None] * len(batch)
                    else:
                        raise e
        return results

import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from general.utils import iter_list
from tqdm import tqdm


class TrueTeacher():
    def __init__(self, model_path, tokenizer_name='t5-base', device='cpu', batch_size=8, max_length=2048,
                 torch_dtype=torch.float32, return_none=False):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.one_token = self.tokenizer('1').input_ids[0]
        if self.device == 'auto':
            self.model = T5ForConditionalGeneration.from_pretrained(model_path,device_map='auto',
                                                                    torch_dtype=torch_dtype).eval()
            # from accelerate import infer_auto_device_map, dispatch_model
            # device_map = infer_auto_device_map(self.model,
            #                                    max_memory={0: "12GiB",1:"12GB", "cpu": "40GiB"},
            #                                    no_split_module_classes=["T5Block"])
            # self.model = dispatch_model(self.model, device_map)

            self.input_device = 'cuda'
        else:
            self.input_device = self.device
            self.model = T5ForConditionalGeneration.from_pretrained(model_path,
                                                                    torch_dtype=torch_dtype).to(self.device).eval()
        self.return_none = return_none
        self.number_of_errors = 0

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
                        self.number_of_errors += 1
                        print(f'Number of errors: {self.number_of_errors}')
                    else:
                        raise e

            return results

    def classify_single(self, text, summary):
        return self.classify([summary], [text])[0]

    def score(self, texts, summaries):
        counter = 0
        pairs = []
        results = []
        out_of_memory_indexes = []
        for summary, original_text in zip(summaries, texts):
            pairs.append(f'premise: {original_text} hypothesis: {summary}')
        indexes = [i for i in range(len(pairs))]
        with torch.no_grad():
            # for batch in tqdm(iter_list(pairs, self.batch_size)):
            for batch_indexes in tqdm(iter_list(indexes, self.batch_size)):
                batch = [pairs[i] for i in batch_indexes]
                try:
                    model_input = self.tokenizer(batch, return_tensors='pt', truncation=True,
                                                 max_length=self.max_length,
                                                 padding=True).to(self.input_device)
                    decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id] * len(batch)]).reshape((-1, 1)).to(
                        self.input_device)
                    outputs = self.model(**model_input, decoder_input_ids=decoder_input_ids)
                    logits = torch.softmax(outputs.logits.float(), dim=-1)
                    # torch.cuda.empty_cache()
                    batch_factuality_score = logits.detach().cpu()[:, 0, self.one_token]
                    results += batch_factuality_score.tolist()
                    counter += 1
                    if counter % 100 == 0:
                        print(np.mean([r for r in results if r is not None]))
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.return_none:
                        print("Out of memory. Trying to free up some GPU memory.")
                        # Free up memory (adjust as needed)
                        torch.cuda.empty_cache()
                        results += [None] * len(batch)
                        out_of_memory_indexes += batch_indexes
                    else:
                        raise e
        #self.model = self.model.module.to('cpu')
            # print("processing out of memory examples on cpu")
            # out_of_memory_results = []
            # for i in tqdm(out_of_memory_indexes):
            #     batch = [pairs[i]]
            #     model_input = self.tokenizer(batch, return_tensors='pt', truncation=True,
            #                                  max_length=self.max_length,
            #                                  padding=True).to('cpu')
            #     decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id] * len(batch)]).reshape((-1, 1)).to(
            #         'cpu')
            #     outputs = self.model(**model_input, decoder_input_ids=decoder_input_ids)
            #     logits = torch.softmax(outputs.logits.float(), dim=-1)
            #     # torch.cuda.empty_cache()
            #     batch_factuality_score = logits.detach().cpu()[:, 0, self.one_token]
            #     out_of_memory_results += batch_factuality_score
            # for i in range(len(results)):
            #     if i in out_of_memory_indexes:
            #         item_index = out_of_memory_indexes.index(i)
            #         results[i] = out_of_memory_results[item_index]
        return results

    def score_single(self, text, summary):
        return self.score([summary], [text])[0]

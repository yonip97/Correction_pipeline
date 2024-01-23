from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from general.utils import iter_list
from tqdm import tqdm


class NLI():
    def __init__(self, batch_size, torch_dtype, max_length=2048, device='cpu', return_none=True):
        if device == 'auto':
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5_xxl_true_nli_mixture",
                                                                    device_map='auto',
                                                                    torch_dtype=torch_dtype).eval()
            self.input_device = 'cuda'
        else:
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5_xxl_true_nli_mixture",
                                                                    torch_dtype=torch_dtype).to(device).eval()
            self.input_device = device
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5_xxl_true_nli_mixture")
        self.max_length = max_length
        self.one_token = self.tokenizer('1').input_ids[0]
        self.batch_size = batch_size
        self.return_none = return_none

    def score(self, texts, summaries):
        pairs = []
        results = []
        count = 0
        for summary, original_text in zip(summaries, texts):
            pairs.append(("premise: " + original_text, "hypothesis: " + summary))
            # pairs.append(f'premise: {original_text} hypothesis: {summary}')
        with torch.no_grad():
            for batch in tqdm(iter_list(pairs, self.batch_size)):
                try:
                    model_input = self.tokenizer.batch_encode_plus(batch, return_tensors='pt', truncation='only_first',
                                                                   max_length=self.max_length,
                                                                   padding=True).to(self.input_device)
                    decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id] * len(batch)]).reshape((-1, 1)).to(
                        self.input_device)
                    outputs = self.model(**model_input, decoder_input_ids=decoder_input_ids)
                    logits = torch.softmax(outputs.logits.float(), dim=-1)
                    batch_factuality_score = logits.detach().cpu()[:, 0, self.one_token]
                    results += batch_factuality_score.tolist()
                    count += 1
                    if count % 100 == 0:
                        import numpy as np
                        print(np.mean([x for x in results if x is not None]))
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.return_none:
                        print("Out of memory. Trying to free up some GPU memory.")
                        # Free up memory (adjust as needed)
                        torch.cuda.empty_cache()
                        results += [None] * len(batch)
                    else:
                        raise e
                # torch.cuda.empty_cache()
        return results

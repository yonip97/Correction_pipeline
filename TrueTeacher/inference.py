from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
# from correction_pipeline.utils import iter_list
# from correction_pipeline.run_pipeline import TRUE_dataset,true_topics
from accelerate import PartialState

from tqdm import tqdm
class TrueTeacher():
    def __init__(self,device = 'cpu', batch_size=8, max_length=2048):
        model_path = 'google/t5_11b_trueteacher_and_anli'
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

    def apply(self, summaries, original_texts):
        pairs = []
        results = []
        for summary, original_text in zip(summaries, original_texts):
            pairs.append(f'premise: {original_text} hypothesis: {summary}')
        with torch.no_grad():
            for batch in tqdm(iter_list(pairs, self.batch_size)):
                model_input = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=self.max_length,
                                             padding=True).to(self.device)

                outputs = self.model.generate(**model_input)
                batch_results = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results += batch_results
            return results
import time
device = 'cuda'
distributed_state = PartialState()
start = time.time()
model = TrueTeacher(device = device)
model.model.to(distributed_state.device)
print(f"model loading takes {time.time()-start} seconds")
dataset = TRUE_dataset('/data/home/yehonatan-pe/Correction_pipeline/data')
dataset.filter_to_datasets(true_topics(['summarization']))
texts = dataset.df['grounding']
summaries = dataset.df['generated_text']
labels = dataset.df['label']
predictions = model.apply(summaries,texts)
print(predictions)
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


class TrueTeacher():
    def __init__(self, model_path, tokenizer_name='t5-base', device='cpu', batch_size=8, max_length=2048):
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.one_token = self.tokenizer('1').input_ids[0]

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
                batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for string in batch_results:
                    if string[0] == '1':
                        results.append(1)
                    elif string[0] == '0':
                        results.append(0)
                    else:
                        print(f"Big problem the model predicts {string}")
                #results += batch_results
            return results

    def score(self, summaries, original_texts):
        pairs = []
        results = []
        for summary, original_text in zip(summaries, original_texts):
            pairs.append(f'premise: {original_text} hypothesis: {summary}')
        with torch.no_grad():
            for batch in tqdm(iter_list(pairs, self.batch_size)):
                model_input = self.tokenizer(batch, return_tensors='pt', truncation=True, max_length=self.max_length,
                                             padding=True).to(self.device)
                decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id] * len(batch)]).reshape((-1, 1)).to(
                    self.device)
                outputs = self.model(**model_input, decoder_input_ids=decoder_input_ids)
                logits = torch.softmax(outputs.logits, dim=-1)
                batch_factuality_score = logits.detach().cpu()[:, 0, self.one_token]
                results += batch_factuality_score.tolist()
        return results


def main():
    device = 'cuda'
    model_path = '/data/home/yehonatan-pe/Correction_pipeline/TrueTeacher/results/run name_2023-10-11 00:31:57/checkpoint-23000'
    #model_path = 'google/t5_11b_trueteacher_and_anli'
    #tokenizer_name = 'google/t5_11b_trueteacher_and_anli'
    tokenizer_name = 't5-base'
    model = TrueTeacher(model_path=model_path, tokenizer_name=tokenizer_name, device=device, batch_size=8,
                        max_length=2048)
    dataset = TRUE_dataset("data/true_data", ['summarization'])
    texts = dataset.df['grounding'].tolist()
    summaries = dataset.df['generated_text'].tolist()
    labels = dataset.df['label'].tolist()
    predictions = model.apply(summaries, texts)
    dataset['predictions'] = predictions
    accuracy = sum([1 if predictions[i] == labels[i] else 0 for i in range(len(labels))]) / len(labels)
    print(accuracy)
    predictions.to_csv('data/true_data/true_teacher_base_predictions.csv')
    #print(roc_auc_score(labels, predictions))



if __name__ == '__main__':
    main()

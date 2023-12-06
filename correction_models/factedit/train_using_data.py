import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import matplotlib.pyplot as plt
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainingArguments
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import evaluate
import nltk
from data.factuality_datasets import TRUE_dataset,BERTS2S_TConvS2S_xsum_trained_dataset
from Seahorse_metrics.metrics import Seahorse_metrics
import torch
from general.bart_trainer import BartTrainer
from factCC.inference import Factcc_classifier

class FactEditDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.xsum = []
        self.cnndm = []
        self._get_cnndm()
        self._get_xsum()

    def _get_xsum(self):
        print("getting xsum")
        with open(f'data/factedit/xsum_corr_data/infill_cands_0.2/{self.split}.json', 'r') as json_file:
            for line in tqdm(json_file):
                entry = json.loads(line)
                source_article_sentences = entry['source_article_sentences']
                relevant_sentences = entry['relevant_article_sent_indices']
                relevant_passages = self.create_relevant_passages(source_article_sentences, relevant_sentences)
                generated_summary_lines = entry['generated_summary_sentences'][
                                          len(entry['generated_summary_sentences']) // 2:]
                generated_summary = " ".join(generated_summary_lines)
                generated_summary = generated_summary.replace('[blank]', entry['err_span'])
                generated_summary_line = entry['generated_summary_sent']
                original_summary_line = entry['original_summary_sent']
                generated_summary_line = generated_summary_line.replace('[blank]', entry['err_span'])
                self.xsum.append((relevant_passages, generated_summary, generated_summary_line, original_summary_line))

    def _get_cnndm(self):
        print("getting cnndm")
        with open(f'data/factedit/cnndm_corr_data/infill_cands_0.2/{self.split}.json', 'r') as json_file:
            for line in tqdm(json_file):
                entry = json.loads(line)
                source_article_sentences = entry['source_article_sentences']
                relevant_sentences = entry['relevant_article_sent_indices']
                relevant_passages = self.create_relevant_passages(source_article_sentences, relevant_sentences)
                generated_summary_lines = entry['generated_summary_sentences'][
                                          len(entry['generated_summary_sentences']) // 2:]
                generated_summary = " ".join(generated_summary_lines)
                generated_summary_line = entry['generated_summary_sent']
                original_summary_line = entry['original_summary_sent']
                self.cnndm.append((relevant_passages, generated_summary, generated_summary_line, original_summary_line))

    def create_relevant_passages(self, source_article_sentences, relevant_sentences):
        relevant_passages = []
        for i in relevant_sentences:
            start_index = max(0, i - 2)
            end_index = min(len(source_article_sentences), i + 3)
            relevant_passages.append(" ".join(source_article_sentences[start_index:end_index]))
        return " ".join(relevant_passages)

    def __len__(self):
        return len(self.xsum) + len(self.cnndm)

    def __getitem__(self, item):
        if item < len(self.xsum):
            article, generated_summary, incorrect_line, correct_line = self.xsum[item]
        else:
            article, generated_summary, incorrect_line, correct_line = self.cnndm[item - len(self.xsum)]
        return article, generated_summary, incorrect_line, correct_line


def collate_fn(batch, tokenizer, max_length):
    relevant_passages = [row[0] for row in batch]
    summaries = [row[1] for row in batch]
    incorrect_summary_sentence = [row[2] for row in batch]
    correct_summary_sentence = [row[3] for row in batch]
    text_input = [incorrect_summary_sentence[i] + " <sep> " + summaries[i] + " <sep> " + relevant_passages[i] for i in
                  range(len(relevant_passages))]
    inputs = tokenizer(text_input, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(correct_summary_sentence, padding=True, truncation=True, max_length=max_length,
                       return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def compute_metrics(p, tokenizer, rouge):
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    results = rouge.compute(predictions=predictions, references=labels)
    return results







def train():
    os.environ["WANDB_DISABLED"] = "true"
    rouge = evaluate.load('rouge')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    train_dataset = FactEditDataset('train')
    eval_dataset = FactEditDataset('validation')
    args = Seq2SeqTrainingArguments(
        output_dir=f'correction_models/factedit/checkpoints/bart_base_factedit',
        do_train=True, do_eval=True, warmup_steps=1000,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        gradient_accumulation_steps=2,
        learning_rate=3e-5, num_train_epochs=1, save_total_limit=2,
        load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
        eval_steps=5000, save_steps=5000, eval_accumulation_steps=30,
        metric_for_best_model='rougeL', no_cuda=False, generation_max_length=128, predict_with_generate=True)
    trainer = BartTrainer(model=model, tokenizer=tokenizer, args=args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          compute_metrics=lambda p: compute_metrics(p, tokenizer, rouge), collate_fn=collate_fn,
                          max_length=512)
    trainer.train()
    # train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True, collate_fn=collate_fn)
    # for batch in train_dataloader:
    #     input_text, label = batch
    #     model_input = tokenizer.encode(input_text)


def main():
    checkpoint_path = 'correction_models/factedit/checkpoints/bart_base_factedit/checkpoint-15000'
    model = BartForConditionalGeneration.from_pretrained(checkpoint_path).to(device)
    tokenizer = BartTokenizer.from_pretrained(checkpoint_path)
    model.eval()
    dataset = BERTS2S_TConvS2S_xsum_trained_dataset()
    texts = [dataset[i]['text'] for i in range(len(dataset))]
    summaries = [dataset[i]['summary'] for i in range(len(dataset))]
    rouge = evaluate.load('rouge')
    corrected_summaries = []
    for i in tqdm(range(len(dataset))):
        with torch.no_grad():
            item = dataset[i]
            document, summary = item['text'], item['summary']
            corrected_summary = predict(model, tokenizer, document, summary, metric=rouge, device=device)
            corrected_summaries.append(corrected_summary)

    # with open("data/factedit/BERTS2S_TConvS2S_xsum_trained_dataset/corrected_summaries.txt", "w") as file:
    #     for string_item in corrected_summaries:
    #         file.write(string_item + "\n")
    model.to('cpu')

    del model

    torch.cuda.empty_cache()
    factuality_metric = Seahorse_metrics(model_path="google/seahorse-large-q4", tokenizer_name="google/seahorse-large-q4",
                                         device='cpu', batch_size=1, max_length=2048,
                                         return_none=True)
    # corrected_summaries = []
    # with open("data/factedit/BERTS2S_TConvS2S_xsum_trained_dataset/corrected_summaries.txt", "r") as file:
    #     for line in file:
    #         corrected_summaries.append(line.strip())
    #factuality_metric = Factcc_classifier(checkpoint_path ='factCC/checkpoints/factcc-checkpoint',device='cpu')
    pre_scores = factuality_metric.score(texts=texts, summaries=summaries)
    post_scores = factuality_metric.score(texts=texts, summaries=corrected_summaries)
    print(f"The mean score before revision is {np.mean(pre_scores):.4f}")
    print(f"The mean score after revision is {np.mean(post_scores):.4f}")
    plt.hist(pre_scores, bins=20, alpha=0.5, label='pre revision')
    plt.hist(post_scores, bins=20, alpha=0.5, label='post revision')
    plt.legend(loc='upper right')
    plt.xlim(0, 1)
    # plt.savefig('correction_models/factedit/outputs/factuality_scores_factcc.png')
    plt.show()
    # with open("correction_models/factuality/outputs/results.txt",'w') as text_file:
    #     text_file.write(f"The mean score before revision is {np.mean(pre_scores):.4f}\n")
    #     text_file.write(f"The mean score after revision is {np.mean(post_scores):.4f}\n")
    # text_file.close()
    #



if __name__ == '__main__':
    main()
# count = 0
# with open('/data/home/yehonatan-pe/Correction_pipeline/data/factedit/xsum_corr_data/infill_cands_0.2/train.json',
#           'r') as json_file:
#     # Iterate over each line in the JSON file
#     for line in json_file:
#         count += 1
#         entry = json.loads(line)
#         source_sents = entry['source_article_sentences']
#         generated_summary_sentences = entry['generated_summary_sentences']
#         orig_summary_2 = " ".join(generated_summary_sentences[:len(generated_summary_sentences) // 2])
#         original_summary_sentences = entry['original_summary_sentences']
#         orig_summary = " ".join(original_summary_sentences)
#
#         if orig_summary == orig_summary_2:
#             print("same")
#         else:
#             print("not same")
#         err_span = entry['err_span']
#         # print(entry)
# print(count)

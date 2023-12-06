import os
import sys

import numpy as np
import pandas as pd

# sys.path.append(os.path.dirname(os.getcwd()))
# os.chdir('../')
# sys.path.append(os.path.dirname(os.getcwd()))
# os.chdir('../')
from datasets import load_dataset
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from general.metrics import Rouge
from general.t5_trainer import T5_Trainer, summarize
import nltk

def compute_metrics(p, tokenizer):
    metric = Rouge()
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
    results = metric(predictions, labels)
    results = {k: np.mean(v) for k, v in results.items()}
    return results


def collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + ':' + row['document'] for row in batch]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def main():
    train_dataset = load_dataset('xsum', split='train')
    val_dataset = load_dataset('xsum', split='validation')
    os.environ["WANDB_DISABLED"] = "true"

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    from datetime import datetime
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    args = Seq2SeqTrainingArguments(output_dir=f'checkpoints/t5_base_xsum_{run_name}',
                                    do_train=True, do_eval=True,
                                    per_device_train_batch_size=4,
                                    per_device_eval_batch_size=4,
                                    gradient_accumulation_steps=1,
                                    learning_rate=2e-5, num_train_epochs=2, save_total_limit=2,
                                    load_best_model_at_end=True, evaluation_strategy='steps', save_strategy='steps',
                                    eval_steps=5000, save_steps=5000, eval_accumulation_steps=30, weight_decay=0.01,
                                    metric_for_best_model='rougeL', no_cuda=False)
    max_length_train = 512
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=args,
                         train_dataset=train_dataset,
                         eval_dataset=val_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()


if __name__ == "__main__":
    # main()
    from data.factuality_datasets import TRUE_dataset

    dataset = TRUE_dataset('data/true_data', ['summarization'])
    texts = dataset.df['grounding'].drop_duplicates().tolist()
    model = T5ForConditionalGeneration.from_pretrained('checkpoints/t5_base_xsum_12_11_2023_17_51_56/checkpoint-45000')
    tokenizer = T5Tokenizer.from_pretrained('checkpoints/t5_base_xsum_12_11_2023_17_51_56/checkpoint-45000')
    summaries = summarize(texts, model, tokenizer)
    print(summaries)

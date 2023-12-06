import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from general.t5_trainer import T5_Trainer, summarize
from xsum_split import split_xsum_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import os
from datetime import datetime
from general.metrics import Rouge
import numpy as np


def compute_metrics(p, tokenizer):
    metric = Rouge()
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
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
    num_of_documents_for_summarization = 20000
    seed = 42
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    path_to_documents_for_summarization_indices = f'experiments/xsum_4_sets_experiment/xsum_docs_for_summarization_{num_of_documents_for_summarization}_indices_seed_{seed}.pkl'
    train_dataset = split_xsum_dataset(split='train_model',
                                       path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                       num_of_documents_for_summarization=num_of_documents_for_summarization, seed=seed)
    val_dataset = split_xsum_dataset(split='validation_model',
                                     path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices,
                                     num_of_documents_for_summarization=num_of_documents_for_summarization, seed=seed)
    os.environ["WANDB_DISABLED"] = "true"

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    args = Seq2SeqTrainingArguments(
        output_dir=f'experiments/xsum_4_sets_experiment/checkpoints/t5_base_xsum_{run_name}',
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
    main()

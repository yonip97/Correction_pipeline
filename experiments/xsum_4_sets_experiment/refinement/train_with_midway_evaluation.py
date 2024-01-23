import time
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import numpy as np
from general.t5_trainer import T5_Trainer
from datetime import datetime
import os
from general.utils import RevisionDataset
import evaluate
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
import json
from experiments.xsum_4_sets_experiment.refinement.refinement_utils import create_dataset, create_full_dataset
import argparse


def compute_metrics(pred, tokenizer, eval_texts):
    rouge_metric = evaluate.load('rouge')
    results = {}
    predictions = pred.predictions
    labels = pred.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge_scores = rouge_metric.compute(predictions=predictions, references=labels)
    for key in rouge_scores.keys():
        results[key] = rouge_scores[key]
    from general.fragments_metrics import Fragments
    fragments_metric = Fragments()
    density_and_coverage_scores = \
        fragments_metric.score(metrics=['density', 'coverage'], texts=eval_texts, summaries=predictions)
    for key in density_and_coverage_scores.keys():
        results[key] = np.nanmean(density_and_coverage_scores[key])
    print(results['density'])
    # torch.cuda.empty_cache()
    from Seahorse_metrics.metrics import Seahorse_metrics
    seahorse_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                       device='auto', batch_size=2, torch_dtype=torch.float16, max_length=2048,
                                       return_none=True)
    seahorse_scores = seahorse_metric.score(texts=eval_texts, summaries=predictions)
    results['seahorse'] = np.mean([x for x in seahorse_scores if x is not None])
    del seahorse_metric
    import gc
    gc.collect()
    time.sleep(5)
    torch.cuda.empty_cache()
    return results


def collate_fn(batch, tokenizer, max_length):
    revised_summaries = [row['revised_summary'] for row in batch]
    texts_inputs = ["summarize: " + row['text'] for row in batch]
    inputs = tokenizer(texts_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def train(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, pre_revision_density,
          post_revision_density, method, args):
    torch.cuda.empty_cache()
    # hyperparameters = get_best_hyperparameters("all")[1][1]['hyperparameters']
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    max_length_train = args.max_encoding_length
    num_beams = args.beam_size
    generation_max_length = args.generation_max_length
    rouge_threshold = args.rouge_threshold
    classifier_threshold = args.classifier_threshold
    diff_threshold = args.diff_threshold
    density_threshold = args.density_threshold
    density_diff_threshold = args.density_diff_threshold
    seed = args.seed
    device = args.device
    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False
    train_dataset = create_dataset(texts, summaries, revised_summaries, pre_revision_scores,
                                   post_revision_scores, pre_revision_density, post_revision_density, method,
                                   rouge_threshold=rouge_threshold, classifier_threshold=classifier_threshold,
                                   diff_threshold=diff_threshold, density_threshold=density_threshold,
                                   density_diff_threshold=density_diff_threshold)
    eval_dataset = split_xsum_dataset(split='validation_model',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000, seed=42)
    eval_texts = [row['text'] for row in eval_dataset][::50]
    eval_summaries = [row['summary'] for row in eval_dataset][::50]
    eval_dataset = RevisionDataset(eval_texts, eval_summaries, eval_summaries)
    os.environ["WANDB_DISABLED"] = "true"
    model_checkpoint = args.model_checkpoint
    models_dir = args.model_dir
    model_path = os.path.join(models_dir, model_checkpoint)
    # model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_both_10_12_2023_08_54_06/checkpoint-115000"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.run_name,
        do_train=True, do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr, num_train_epochs=epochs, evaluation_strategy='steps', save_strategy='no',
        eval_steps=0.2, logging_strategy="steps",
        eval_accumulation_steps=30, weight_decay=weight_decay,
        metric_for_best_model='seahorse', no_cuda=no_cuda, predict_with_generate=True, generation_num_beams=num_beams,
        generation_max_length=generation_max_length, logging_steps=0.01)
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset, eval_dataset=eval_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer, eval_texts),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    return model, tokenizer, trainer.state.log_history


def args_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float, default=5e-5)
    args.add_argument("--epochs", type=int, default=1)
    args.add_argument("--batch_size", type=int, default=8)
    args.add_argument("--gradient_accumulation_steps", type=int, default=2)
    args.add_argument("--weight_decay", type=float, default=0)
    args.add_argument("--train_size", type=float, default=1)
    args.add_argument("--max_encoding_length", type=int, default=512)
    args.add_argument("--model_checkpoint", type=str)
    args.add_argument("--model_dir", type=str, default="experiments/xsum_4_sets_experiment/checkpoints")
    args.add_argument("--test_batch_size", type=int, default=32)
    args.add_argument("--test_max_encoding_length", type=int, default=512)
    args.add_argument("--beam_size", type=int, default=4)
    args.add_argument("--generation_max_length", type=int, default=128)
    # args.add_argument("--test_save_path", type=str)
    args.add_argument("--test_save_dir", type=str, default="experiments/xsum_4_sets_experiment")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--rouge_threshold", type=float, default=0.7)
    args.add_argument("--classifier_threshold", type=float, default=0.5)
    args.add_argument("--diff_threshold", type=float, default=0.4)
    args.add_argument("--density_threshold", type=float, default=2)
    args.add_argument("--density_diff_threshold", type=float, default=0.75)
    args.add_argument('--device', type=str, default='auto')
    args.add_argument('--method', type=str)
    args.add_argument('--rerun', action='store_true')
    args.add_argument('--run_name', type=str, default=None)
    return args.parse_args()


def main():
    args = args_parser()
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    runs_path = "experiments/xsum_4_sets_experiment/runs_for_train_thresholds_adjustments"
    method = args.method
    method_path = os.path.join(runs_path, method)
    if not os.path.exists(method_path):
        os.makedirs(method_path)

    run_path = os.path.join(method_path, run_name)
    os.makedirs(run_path)
    revised_dataset_path = 'experiments/xsum_4_sets_experiment/documents_for_summarization_fully_scored_with_revised.csv'
    df = create_full_dataset(revised_dataset_path, args)
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    revised_summaries = df['revised_summary_4_beam'].tolist()
    pre_revision_scores = df['pre_revision_seahorse_xxl_score'].tolist()
    post_revision_scores = df['post_revision_seahorse_xxl_score_4_beam'].tolist()
    pre_revision_density = df['pre_revision_density'].tolist()
    post_revision_density = df['revised_density'].tolist()
    args.run_name = run_name
    with open(os.path.join(run_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    print(args.method)
    print()
    _, _, log_history = train(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores,
                              pre_revision_density,
                              post_revision_density, method=args.method, args=args)
    logging_path = os.path.join(run_path, 'log_history.txt')
    with open(logging_path, 'w') as f:
        for log in log_history:
            f.write(str(log))
            f.write('\n')
    print("-----------------------------------------------------")
    print()


if __name__ == '__main__':
    main()

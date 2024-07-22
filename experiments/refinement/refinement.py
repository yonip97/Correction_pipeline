import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import json
import gc
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer, WandbCallback
from general.utils import find_largest_numbered_dir
import numpy as np
from general.fragments_metrics import Fragments
import evaluate
from general.utils import SummarizationDataset
from Seahorse_metrics.metrics import Seahorse_metrics
from TrueTeacher.inference import TrueTeacher
import argparse
import wandb
from nltk.tokenize import word_tokenize
from general.utils import SummarizationDataset
from experiments.scoring import score


def parse():
    args = argparse.ArgumentParser()
    args.add_argument('-revised_data_file', type=str)
    args.add_argument('-revised_inhouse_data_file', type=str)
    args.add_argument('-unrevised_data_file', type=str)
    args.add_argument('-eval_data_file', type=str)
    args.add_argument('-model_checkpoint', type=str)
    args.add_argument('-train_batch_size_per_device', type=int, default=1)
    args.add_argument('-eval_batch_size_per_device', type=int, default=4)
    args.add_argument('-encoder_max_length', type=int, default=2048)
    args.add_argument('-max_generation_length', type=int, default=128)
    args.add_argument('-lr', type=float, default=1e-3)
    args.add_argument('-beam_size', type=int, default=4)
    args.add_argument('-length_penalty', type=float, default=0.6)
    args.add_argument('-epochs', type=int, default=1)
    args.add_argument('-gradient_accumulation_steps', type=int, default=8)
    args.add_argument('-evaluation_strategy', type=str, default='steps')
    args.add_argument('-eval_steps', type=float, default=0.19)
    args.add_argument('-output_dir', type=str, default='experiments/refinement/data')
    args.add_argument('-output_path', type=str)
    args.add_argument('-optim', type=str, default='adafactor')
    args.add_argument('-metric_for_best_checkpoint', type=str)
    args.add_argument('-factuality_threshold_revised', type=float)
    args.add_argument('-factuality_threshold_unrevised', type=float)
    args.add_argument('-factuality_diff', type=float)
    args.add_argument('-density_threshold_revised', type=float)
    args.add_argument('-density_threshold_unrevised', type=float)
    args.add_argument('-density_diff', type=float)
    args.add_argument('-rouge_to_base_threshold', type=float)
    args.add_argument('-rouge_to_original_threshold_revised', type=float)
    args.add_argument('-rouge_to_original_threshold_unrevised', type=float)
    args.add_argument('-number_of_unrevised_samples', type=int)
    args.add_argument('-ratio_revised_to_unrevised', type=float)
    args.add_argument('-strategy', type=str, default='revised_and_unrevised')
    args.add_argument('-data_used', type=str, default='revised_and_unrevised')
    args.add_argument('-eval_compute_metric', type=str)
    args.add_argument('-test', action='store_true')
    args.add_argument('-test_data_path', type=str)
    args.add_argument('-test_seahorse', action='store_true')
    args.add_argument('-test_trueteacher', action='store_true')
    args = args.parse_args()
    return args

def check_model_drift(p, tokenizer, eval_texts, base_seahorse_scores, base_density,
                                         base_model_summaries,
                                         original_dataset_summaries):
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [str(x) for x in predictions]
    rouge_metric = evaluate.load('rouge')
    rouge_scores_to_base = \
        rouge_metric.compute(predictions=predictions, references=base_model_summaries, use_aggregator=False)['rougeL']
    results = {}
    results['predicted_summaries'] = predictions
    results['rouge_to_base'] = rouge_scores_to_base
    results['first_500_rougeL_mean'] = np.mean(rouge_scores_to_base[:500])
    results['last_500_rougeL_mean'] = np.mean(rouge_scores_to_base[-500:])
    print(results['first_500_rougeL_mean'])
    print(results['last_500_rougeL_mean'])
    return results

def compute_metrics_500_500_eval_dataset(p, tokenizer, eval_texts, base_seahorse_scores, base_density,
                                         base_model_summaries,
                                         original_dataset_summaries):
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [str(x) for x in predictions]
    results = {}
    fragments_metric = Fragments()
    density_scores = fragments_metric.score(metrics=['density'], texts=eval_texts, summaries=predictions)['density']
    results['first_500_density'] = np.mean(density_scores[:500])
    results['last_500_density'] = np.mean(density_scores[-500:])
    gc.collect()
    torch.cuda.empty_cache()
    seahorse_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                       device='auto', batch_size=1, torch_dtype=torch.float16, max_length=2048,
                                       return_none=True)
    seahorse_scores = seahorse_metric.score(texts=eval_texts, summaries=predictions)
    seahorse_scores_first_500 = seahorse_scores[:500]
    seahorse_scores_last_500 = seahorse_scores[-500:]
    base_seahorse_scores_first_500 = base_seahorse_scores[:500]
    base_seahorse_scores_last_500 = base_seahorse_scores[-500:]
    del seahorse_metric
    gc.collect()
    torch.cuda.empty_cache()
    results['first_500_seahorse'] = np.mean([x for x in seahorse_scores_first_500 if x is not None])
    results['last_500_seahorse'] = np.mean([x for x in seahorse_scores_last_500 if x is not None])
    seahorse_diff_first_500 = [x - y for x, y in zip(seahorse_scores_first_500, base_seahorse_scores_first_500) if
                               x is not None]
    seahorse_diff_last_500 = [x - y for x, y in zip(seahorse_scores_last_500, base_seahorse_scores_last_500) if
                              x is not None]
    results['first_500_seahorse_diff'] = np.mean([x for x in seahorse_diff_first_500 if x is not None])
    results['last_500_seahorse_diff'] = np.mean([x for x in seahorse_diff_last_500 if x is not None])
    density_diff = [x - y for x, y in zip(density_scores, base_density)]
    results['first_500_density_diff'] = np.mean(density_diff[:500])
    results['last_500_density_diff'] = np.mean(density_diff[-500:])
    rouge_metric = evaluate.load('rouge')
    rouge_scores_to_original = \
        rouge_metric.compute(predictions=predictions, references=original_dataset_summaries, use_aggregator=False)[
            'rougeL']
    rouge_scores_to_base = \
        rouge_metric.compute(predictions=predictions, references=base_model_summaries, use_aggregator=False)['rougeL']
    results['first_500_rouge_to_original'] = np.mean(rouge_scores_to_original[:500])
    results['last_500_rouge_to_original'] = np.mean(rouge_scores_to_original[-500:])
    results['first_500_rouge_to_base'] = np.mean(rouge_scores_to_base[:500])
    results['last_500_rouge_to_base'] = np.mean(rouge_scores_to_base[-500:])
    # wandb.log(results)
    results['predicted_summaries'] = predictions
    results['seahorse_scores'] = seahorse_scores
    results['density_scores'] = density_scores
    results['rouge_to_original'] = rouge_scores_to_original
    results['rouge_to_base'] = rouge_scores_to_base
    return results


def compute_metric(p, tokenizer, eval_texts, base_seahorse_scores, base_density, base_model_summaries,
                   original_dataset_summaries):
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [str(x) for x in predictions]
    results = {}
    fragments_metric = Fragments()
    density_scores = fragments_metric.score(metrics=['density'], texts=eval_texts, summaries=predictions)['density']
    results['density_mean'] = np.mean(density_scores)
    gc.collect()
    torch.cuda.empty_cache()
    trueteacher_metric = TrueTeacher(
        model_path="google/t5_11b_trueteacher_and_anli", tokenizer_name="google/t5_11b_trueteacher_and_anli",
        device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16, return_none=True)
    trueteacher_scores = trueteacher_metric.score(texts=eval_texts, summaries=predictions)
    results['trueteacher_mean'] = np.mean([x for x in trueteacher_scores if x is not None])
    del trueteacher_metric
    gc.collect()
    torch.cuda.empty_cache()
    density_diff = [x - y for x, y in zip(density_scores, base_density)]
    results['density_diff_mean'] = np.mean(density_diff)
    rouge_metric = evaluate.load('rouge')
    rouge_scores_to_original = \
        rouge_metric.compute(predictions=predictions, references=original_dataset_summaries, use_aggregator=False)[
            'rougeL']
    rouge_scores_to_base = \
        rouge_metric.compute(predictions=predictions, references=base_model_summaries, use_aggregator=False)['rougeL']
    results['rouge_to_original_mean'] = np.mean(rouge_scores_to_original)
    results['rouge_to_base_mean'] = np.mean(rouge_scores_to_base)
    # wandb.log(results)
    results['predicted_summaries'] = predictions
    results['trueteacher_scores'] = trueteacher_scores
    results['density_scores'] = density_scores
    results['rouge_to_original'] = rouge_scores_to_original
    results['rouge_to_base'] = rouge_scores_to_base
    return results


def preprocess_data(df):
    df = df[~df['revised_summary_full_text'].str.contains('No correction')]
    return df


def process_data_inhouse_revision(df):
    df.rename(
        columns={"eval_predicted_summaries": 'revised_summary', 'eval_seahorse_scores': 'revised_summary_seahorse',
                 'eval_density_scores': 'revised_summary_density', 'eval_rouge_to_base': 'rougeL_revised_to_base'},
        inplace=True)
    return df


def create_filtered_train_data_revised(df, args):
    if args.factuality_threshold_revised is not None:
        df = df[df['revised_summary_seahorse'] >= args.factuality_threshold_revised]
    if args.factuality_diff is not None:
        df = df[df['revised_summary_seahorse'] - df['model_summary_seahorse'] >= args.factuality_diff]
    if args.density_threshold_revised is not None:
        df = df[df['revised_summary_density'] <= args.density_threshold_revised]
    if args.density_diff is not None:
        df = df[df['revised_summary_density'] - df['model_summary_density'] <= args.density_diff]
    if args.rouge_to_base_threshold is not None:
        df = df[df['rougeL_revised_to_base'] >= args.rouge_to_base_threshold]
    if args.rouge_to_original_threshold_revised is not None:
        df = df[df['rougeL_revised_to_original'] >= args.rouge_to_original_threshold_revised]
    print(f"revised dataset length is {len(df)}")
    return df


def create_filtered_train_data_unrevised(df, args):
    if args.factuality_threshold_unrevised is not None:
        df = df[df['model_summary_seahorse'] >= args.factuality_threshold_unrevised]
    if args.density_threshold_unrevised is not None:
        df = df[df['model_summary_density'] <= args.density_threshold_unrevised]
    if args.rouge_to_original_threshold_unrevised is not None:
        df = df[df['rougeL_base_to_original'] >= args.rouge_to_original_threshold_unrevised]
    if args.number_of_unrevised_samples is not None:
        df = df[:args.number_of_unrevised_samples]
    print(f"unrevised dataset length is {len(df)}")
    return df


def train_collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + row['text'] for row in batch]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def test_collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + row['text'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


def train(args):
    revised_texts = []
    revised_summaries = []
    unrevised_texts = []
    unrevised_summaries = []
    compute_metric_dict = {'500_500_eval': compute_metrics_500_500_eval_dataset, 'all': compute_metric}
    if args.eval_compute_metric == '500_500_eval':
        wandb.init(project='refinement_500_500_dataset', config=args.__dict__)
        eval_compute_metric = compute_metric_dict[args.eval_compute_metric]
    elif args.eval_compute_metric == 'all':
        wandb.init(project='refinement_all_eval_dataset', config=args.__dict__)
        eval_compute_metric = compute_metric_dict[args.eval_compute_metric]
    elif args.eval_compute_metric == '500_500_check_model_drift':
        wandb.init(project='500_500_refinement_check_model_drift', config=args.__dict__)
        eval_compute_metric = check_model_drift
    elif args.eval_compute_metric is None:
        wandb.init(project='refinement_test_dataset', config=args.__dict__)
        eval_compute_metric = None
    else:
        raise ValueError("No such evaluation dataset")
    if args.revised_data_file is not None:
        revised_df = pd.read_csv(args.revised_data_file + '.csv', index_col=0)
        revised_df = preprocess_data(revised_df)
        revised_df = create_filtered_train_data_revised(revised_df, args)
        revised_texts += revised_df['text'].tolist()
        revised_summaries += revised_df['revised_summary'].tolist()
    if args.revised_inhouse_data_file is not None:
        revised_inhouse_df = pd.read_csv(args.revised_inhouse_data_file + '.csv', index_col=0)
        revised_inhouse_df = process_data_inhouse_revision(revised_inhouse_df)
        revised_inhouse_df = create_filtered_train_data_revised(revised_inhouse_df, args)
        revised_texts += revised_inhouse_df['text'].tolist()
        revised_summaries += revised_inhouse_df['revised_summary'].tolist()
    if args.unrevised_data_file is not None:
        unrevised_df = pd.read_csv(args.unrevised_data_file + '.csv', index_col=0)
        unrevised_df = create_filtered_train_data_unrevised(unrevised_df, args)
        unrevised_texts += unrevised_df['text'].tolist()
        unrevised_summaries += unrevised_df['model_summary'].tolist()
    if args.ratio_revised_to_unrevised is not None and len(revised_texts) > 0 and len(unrevised_texts) > 0:
        curr_ratio = len(revised_texts) / len(unrevised_texts)
        needed_upsampling = int(1 / curr_ratio * args.ratio_revised_to_unrevised)
        revised_texts = revised_texts * needed_upsampling
        revised_summaries = revised_summaries * needed_upsampling
    texts = revised_texts + unrevised_texts
    summaries = revised_summaries + unrevised_summaries
    if len(texts) == 0:
        raise ValueError("No data to train on")
    train_dataset = SummarizationDataset(texts=texts, summaries=summaries)
    eval_df = pd.read_csv(args.eval_data_file + '.csv', index_col=0)
    eval_dataset = SummarizationDataset(texts=eval_df['text'].tolist(), summaries=eval_df['model_summary'].tolist())
    eval_texts = eval_df['text'].tolist()
    eval_pre_revision_factuality_scores = eval_df['model_summary_seahorse'].tolist()
    eval_pre_revision_density = eval_df['model_summary_density'].tolist()
    eval_base_model_summaries = eval_df['model_summary'].tolist()
    eval_original_dataset_summaries = eval_df['original_summary'].tolist()
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
    generation_config = model.generation_config
    generation_config.max_length = args.max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.length_penalty
    generation_config.num_beams = args.beam_size
    model.generation_config = generation_config

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_path,
        do_train=True, do_eval=True,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.eval_batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, evaluation_strategy=args.evaluation_strategy,
        save_strategy="no",
        eval_steps=args.eval_steps, eval_accumulation_steps=30,
        metric_for_best_model=args.metric_for_best_checkpoint, no_cuda=False, predict_with_generate=True,
        #generation_num_beams=args.beam_size,
        optim=args.optim, overwrite_output_dir=False, logging_steps=0.01, report_to=['none'])
    max_length_train = args.encoder_max_length
    trainer = T5_Trainer(collate_fn=train_collate_fn, collate_fn_test=test_collate_fn, model=model, tokenizer=tokenizer,
                         args=train_args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         compute_metrics=lambda p: eval_compute_metric(p, tokenizer, eval_texts,
                                                                       eval_pre_revision_factuality_scores,
                                                                       eval_pre_revision_density,
                                                                       eval_base_model_summaries,
                                                                       eval_original_dataset_summaries),
                         max_length_train=max_length_train, max_length_eval=max_length_train,
                         callbacks=[WandbCallback()])
    trainer.train()
    return trainer, model


def score_predictions(texts, summaries, original_dataset_summaries, original_model_summaries, args):
    results = {}
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=summaries, references=original_dataset_summaries,
                                        use_aggregator=False)
    results['rougeL_to_original'] = rouge_scores['rougeL']
    rouge_scores = rouge_metric.compute(predictions=summaries, references=original_model_summaries,
                                        use_aggregator=False)
    results['rougeL_to_base'] = rouge_scores['rougeL']
    fragments_metric = Fragments()
    scores = fragments_metric.score(metrics=['density', 'coverage'], summaries=summaries, texts=texts)
    results['model_summary_density'] = scores['density']
    print("The mean density is", np.mean(scores['density']))
    results['model_summary_coverage'] = scores['coverage']
    results['model_summary_length'] = [len(word_tokenize(summary)) for summary in summaries]
    if args.test_trueteacher:
        results['model_summary_trueteacher'] = score(texts=texts, summaries=summaries, metrics=['trueteacher'])[
            'trueteacher']
    if args.test_seahorse:
        results['model_summary_seahorse'] = score(texts=texts, summaries=summaries, metrics=['seahorse'])['seahorse']
    wandb_dict = {}
    for key in results:
        wandb_dict[key] = np.mean([x for x in results[key] if x is not None])
    wandb.log(wandb_dict)
    return results


def test(trainer, args):
    test_df = pd.read_csv(args.test_data_path + '.csv', index_col=0)
    test_df = test_df[~test_df['text'].isnull()]
    test_texts = test_df['text'].tolist()
    original_dataset_summaries = test_df['original_summary'].tolist()
    original_model_summaries = test_df['model_summary'].tolist()
    test_dataset = SummarizationDataset(texts=test_texts, summaries=original_dataset_summaries)
    predictions = trainer.predict(test_dataset=test_dataset)
    predictions = trainer.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    results = score_predictions(test_texts, predictions, original_dataset_summaries, original_model_summaries, args)
    wandb_dict = {}
    for key in results:
        wandb_dict['test_' + key] = np.mean([x for x in results[key] if x is not None])
    wandb.log(wandb_dict)
    results_df = pd.DataFrame(results)
    results_df['text'] = test_texts
    results_df['summary'] = predictions
    return results_df


def main():
    args = parse()
    eval_df = pd.read_csv(args.eval_data_file + '.csv', index_col=0)
    # eval_dataset = args.eval_data_file.split('/')[-1].replace('base_model_summaries_', '')
    # args.output_dir = os.path.join(args.output_dir, eval_dataset)
    args.output_path = os.path.join(args.output_dir, args.strategy)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        latest_name = 0
    else:
        latest_name = find_largest_numbered_dir(args.output_path)
        latest_name += 1
    args.output_path = os.path.join(args.output_path, str(latest_name))
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(os.path.join(args.output_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    trainer, model = train(args)
    if args.test:
        results_df = test(trainer, args)
        results_df.to_csv(os.path.join(args.output_path, 'test_results.csv'))
    current_logs = trainer.state.log_history
    eval_logs = []
    train_logs = []
    for log in current_logs:
        if 'eval_loss' in log:
            eval_logs.append(log)
        else:
            train_logs.append(log)
    with open(args.output_path + '/train_logs.json', 'w') as f:
        json.dump(train_logs, f)
    with open(args.output_path + '/eval_logs.json', 'w') as f:
        json.dump(eval_logs, f)
    for i, log in enumerate(eval_logs):
        temp_df = eval_df.copy(deep=True)
        for key in log:
            if isinstance(log[key], list):
                temp_df['post_revision_' + key] = log[key]
        temp_df.to_csv(os.path.join(args.output_path, f"eval_{i}.csv"))
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == '__main__':
    main()

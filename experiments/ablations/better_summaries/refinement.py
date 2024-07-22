import os
import sys

import evaluate
import numpy as np
import pandas as pd
import wandb
import gc
import torch
import json
os.chdir('../')
sys.path.append(os.getcwd())
os.chdir('../')
sys.path.append(os.getcwd())
os.chdir('../')
sys.path.append(os.getcwd())
import argparse
from general.utils import SummarizationDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
from experiments.scoring import score
from general.utils import find_largest_numbered_dir


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str)
    parser.add_argument('--revised_data_file', type=str, default=None)
    parser.add_argument('--unrevised_data_file', type=str, default=None)
    parser.add_argument('--test_data_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--model_checkpoint', type=str)
    parser.add_argument('--max_generation_length', type=int, default=128)
    parser.add_argument('--encoder_max_length', type=int, default=2048)
    parser.add_argument('--length_penalty', type=float, default=0.6)
    parser.add_argument('--train_batch_size_per_device', type=int, default=2)
    parser.add_argument('--eval_batch_size_per_device', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--save_strategy', type=str, default='no')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--optim', type=str, default='adafactor')
    parser.add_argument('--filter_strategy', type=str)
    parser.add_argument('--factuality_threshold_llm', type=float, default=None)
    parser.add_argument('--factuality_threshold_revised', type=float, default=None)
    parser.add_argument('--factuality_threshold_unrevised', type=float, default=None)
    parser.add_argument('--factuality_diff_revised', type=float, default=None)
    parser.add_argument('--factuality_diff_llm', type=float, default=None)
    parser.add_argument('--density_threshold_llm', type=float, default=None)
    parser.add_argument('--density_threshold_revised', type=float, default=None)
    parser.add_argument('--density_threshold_unrevised', type=float, default=None)
    parser.add_argument('--density_diff_revised', type=float, default=None)
    parser.add_argument('--density_diff_llm', type=float, default=None)
    parser.add_argument('--rouge_to_base_threshold_llm', type=float, default=None)
    parser.add_argument('--rouge_to_base_threshold_revised', type=float, default=None)
    parser.add_argument('--rouge_to_original_threshold_llm', type=float, default=None)
    parser.add_argument('--rouge_to_original_threshold_revised', type=float, default=None)
    parser.add_argument('--rouge_to_original_threshold_unrevised', type=float, default=None)
    parser.add_argument('--ratio_revised_to_unrevised', type=float, default=None)
    parser.add_argument('--trueteacher', action='store_true')
    parser.add_argument('--seahorse', action='store_true')
    args = parser.parse_args()
    return args


def create_filtered_train_data_generated_by_llm(args):
    df = pd.read_csv(args.train_data_path + '.csv', index_col=0)
    if args.filter_strategy == 'by_revision_filters':
        indices = get_revision_document_indices(args)
        df = df[df['indices'].isin(indices)]
    if args.filter_strategy == 'by_filters':
        df = filter_train_data(df, args)
    return df


def get_revision_document_indices(args):
    revised_df = pd.read_csv(args.revised_data_file + '.csv', index_col=0)
    revised_df = preprocess_data(revised_df)
    revised_df = filter_revision_data(revised_df, args)
    indices = revised_df['indices'].tolist()
    return indices


def preprocess_data(df):
    df = df[~df['revised_summary_full_text'].str.contains('No correction')]
    return df


def filter_train_data(df, args):
    if args.factuality_threshold_llm is not None:
        df = df[df['model_summary_llm_seahorse'] >= args.factuality_threshold_llm]
    if args.factuality_diff_llm is not None:
        df = df[df['model_summary_llm_seahorse'] - df['model_summary_seahorse'] >= args.factuality_diff_llm]
    if args.density_threshold_llm is not None:
        df = df[df['model_summary_llm_density'] <= args.density_threshold_llm]
    if args.density_diff_llm is not None:
        df = df[df['model_summary_llm_density'] - df['model_summary_density'] <= args.density_diff_llm]
    if args.rouge_to_base_threshold_llm is not None:
        df = df[df['rougeL_llm_to_base'] >= args.rouge_to_base_threshold_llm]
    if args.rouge_to_original_threshold_llm is not None:
        df = df[df['rougeL_llm_to_original'] >= args.rouge_to_original_threshold_llm]
    print(f"revised dataset length is {len(df)}")
    return df


def filter_revision_data(df, args):
    if args.factuality_threshold_revised is not None:
        df = df[df['revised_summary_seahorse'] >= args.factuality_threshold_revised]
    if args.factuality_diff_revised is not None:
        df = df[
            df['revised_summary_seahorse'] - df['model_summary_seahorse'] >= args.factuality_diff_revised]
    if args.density_threshold_revised is not None:
        df = df[df['revised_summary_density'] <= args.density_threshold_revised]
    if args.density_diff_revised is not None:
        df = df[df['revised_summary_density'] - df['model_summary_density'] <= args.density_diff_revised]
    if args.rouge_to_base_threshold_revised is not None:
        df = df[df['rougeL_revised_to_base'] >= args.rouge_to_base_threshold_revised]
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
    print(f"unrevised dataset length is {len(df)}")
    return df


def train_collate_fn(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + ':' + row['text'] for row in batch]
    summaries = [row['summary'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def score_predictions(texts, summaries, original_dataset_summaries, original_model_summaries, args):
    results = {}
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=summaries, references=original_dataset_summaries,
                                        use_aggregator=False)
    results['rougeL_llm_to_original'] = rouge_scores['rougeL']
    rouge_scores = rouge_metric.compute(predictions=summaries, references=original_model_summaries,
                                        use_aggregator=False)
    results['rougeL_llm_to_base'] = rouge_scores['rougeL']
    fragments_metric = Fragments()
    scores = fragments_metric.score(metrics=['density', 'coverage'], summaries=summaries, texts=texts)
    results['model_summary_llm_density'] = scores['density']
    print("The mean density is", np.mean(scores['density']))
    results['model_summary_llm_coverage'] = scores['coverage']
    results['model_summary_llm_length'] = [len(word_tokenize(summary)) for summary in summaries]
    if args.trueteacher:
        results['model_summary_llm_trueteacher'] = score(texts=texts, summaries=summaries, metrics=['trueteacher'])['trueteacher']
    if args.seahorse:
        results['model_summary_llm_seahorse'] = score(texts=texts, summaries=summaries, metrics=['seahorse'])['seahorse']
    wandb_dict = {}
    for key in results:
        wandb_dict[key] = np.mean([x for x in results[key] if x is not None])
    wandb.log(wandb_dict)
    return results


def train(args):
    llm_generated_texts = []
    llm_generated_summaries = []
    unrevised_texts = []
    unrevised_summaries = []
    if args.train_data_path is not None:
        llm_generated_df = create_filtered_train_data_generated_by_llm(args)
        llm_generated_texts += llm_generated_df['text'].tolist()
        llm_generated_summaries += llm_generated_df['model_summary_llm'].tolist()
    if args.unrevised_data_file is not None:
        unrevised_df = pd.read_csv(args.unrevised_data_file + '.csv', index_col=0)
        unrevised_df = create_filtered_train_data_unrevised(unrevised_df, args)
        unrevised_texts += unrevised_df['text'].tolist()
        unrevised_summaries += unrevised_df['model_summary'].tolist()
    if args.ratio_revised_to_unrevised is not None and len(llm_generated_texts) > 0 and len(unrevised_texts) > 0:
        curr_ratio = len(llm_generated_texts) / len(unrevised_texts)
        needed_upsampling = int((1 / curr_ratio) * args.ratio_revised_to_unrevised)
        llm_generated_texts = llm_generated_texts * needed_upsampling
        llm_generated_summaries = llm_generated_summaries * needed_upsampling
    texts = llm_generated_texts + unrevised_texts
    summaries = llm_generated_summaries + unrevised_summaries
    texts = texts
    summaries = summaries
    train_dataset = SummarizationDataset(texts=texts, summaries=summaries)
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
    generation_config = model.generation_config
    generation_config.max_length = args.max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.length_penalty
    generation_config.num_beams = args.beam_size
    model.generation_config = generation_config
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_path + '/checkpoints',
        do_train=True, do_eval=True, do_predict=True,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.eval_batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, save_strategy=args.save_strategy,
        eval_accumulation_steps=30,
        predict_with_generate=True,
        optim=args.optim, overwrite_output_dir=False, logging_steps=0.01, report_to=['none'])
    max_length_train = args.encoder_max_length
    trainer = T5_Trainer(collate_fn=train_collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    #torch.save(model.state_dict(), args.output_path + '/model_state_dict')
    return trainer


def test(args, trainer):
    test_df = pd.read_csv(args.test_data_path + '.csv', index_col=0)
    test_df = test_df[test_df['text'].notna()]
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
    results_df = pd.DataFrame(results)
    results_df['text'] = test_texts
    results_df['summary'] = predictions
    return results_df


def main():
    args = parse()
    wandb.init(project="better_summaries_ablation", config=args.__dict__)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    dir_num = find_largest_numbered_dir(args.output_dir)
    args.output_path = os.path.join(args.output_dir, str(dir_num + 1))
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    with open(os.path.join(args.output_path, 'args.json'), 'w') as file:
        json.dump(args.__dict__, file)
    trainer = train(args)
    results_df = test(args, trainer)
    results_df.to_csv(args.output_path + '/results.csv')
    wandb.finish()


if __name__ == '__main__':
    main()

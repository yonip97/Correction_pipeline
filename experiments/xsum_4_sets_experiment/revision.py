import os
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
from experiments.xsum_4_sets_experiment.datasets_splits import split_cnndm_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import t5_summarize, t5_revise, T5_Trainer, collate_fn_revision, compute_metric_rouge
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from general.revision_pipeline import chose_revision_model
from TrueTeacher.inference import TrueTeacher
from Seahorse_metrics.metrics import Seahorse_metrics
import evaluate
from general.utils import RevisionDataset


def create_summaries_both():
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_10000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=10000,
                                      seed=42)
    cnndm_dataset = split_cnndm_dataset(split='documents_for_summarization',
                                        path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/cnndm_docs_for_summarization_10000_indices_seed_42.pkl",
                                        num_of_documents_for_summarization=10000,
                                        seed=42)
    both_model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_both_10_12_2023_08_54_06/checkpoint-115000"
    model = T5ForConditionalGeneration.from_pretrained(both_model_path)
    tokenizer = T5Tokenizer.from_pretrained(both_model_path)
    xsum_texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    cnndm_texts = [cnndm_dataset[i]['text'] for i in range(len(cnndm_dataset))]
    model_summaries_xsum = t5_summarize(xsum_texts, model, tokenizer, 'summarize: ', device='cuda:0', batch_size=256,
                                        max_generation_length=128)
    model_summaries_cnndm = t5_summarize(cnndm_texts, model, tokenizer, 'summarize: ', device='cuda:0', batch_size=256,
                                         max_generation_length=128)
    xsum_df = pd.DataFrame.from_dict({'indices': xsum_dataset.indices, 'model_summary': model_summaries_xsum})
    cnndm_df = pd.DataFrame.from_dict({'indices': cnndm_dataset.indices, 'model_summary': model_summaries_cnndm})
    xsum_df['dataset'] = 'xsum'
    cnndm_df['dataset'] = 'cnndm'
    # pd.concat([xsum_df, cnndm_df], axis=0).to_csv('experiments/xsum_4_sets_experiment/both_models_summaries.csv',
    #                                               index=False)


def create_summaries_xsum():
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    xsum_model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_xsum_19_11_2023_16_44_42/checkpoint-45000"
    model = T5ForConditionalGeneration.from_pretrained(xsum_model_path)
    tokenizer = T5Tokenizer.from_pretrained(xsum_model_path)
    xsum_texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    model_summaries_xsum = t5_summarize(xsum_texts, model, tokenizer, 'summarize: ', device='cuda:1', batch_size=256,
                                        max_generation_length=128)
    xsum_df = pd.DataFrame.from_dict({'indices': xsum_dataset.indices, 'model_summary': model_summaries_xsum})
    xsum_df['dataset'] = 'xsum'
    xsum_df.to_csv('experiments/xsum_4_sets_experiment/xsum_model_summaries.csv',
                   index=False)


def create_summaries_cnndm():
    cnndm_dataset = split_cnndm_dataset(split='documents_for_summarization',
                                        path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/cnndm_docs_for_summarization_20000_indices_seed_42.pkl",
                                        num_of_documents_for_summarization=20000,
                                        seed=42)
    cnndm_model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_cnndm_09_12_2023_23_40_56/checkpoint-65000"
    model = T5ForConditionalGeneration.from_pretrained(cnndm_model_path)
    tokenizer = T5Tokenizer.from_pretrained(cnndm_model_path)
    cnndm_texts = [cnndm_dataset[i]['text'] for i in range(len(cnndm_dataset))]
    model_summaries_cnndm = t5_summarize(cnndm_texts, model, tokenizer, 'summarize: ', device='cuda:1', batch_size=256,
                                         max_generation_length=128)
    cnndm_df = pd.DataFrame.from_dict({'indices': cnndm_dataset.indices, 'model_summary': model_summaries_cnndm})
    cnndm_df['dataset'] = 'cnndm'
    cnndm_df.to_csv('experiments/xsum_4_sets_experiment/cnndm_model_summaries.csv',
                    index=False)


def run_cnndm_model_on_xsum_summaries():
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    cnndm_model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_cnndm_09_12_2023_23_40_56/checkpoint-65000"
    model = T5ForConditionalGeneration.from_pretrained(cnndm_model_path)
    tokenizer = T5Tokenizer.from_pretrained(cnndm_model_path)
    xsum_texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    model_summaries_xsum = t5_summarize(xsum_texts, model, tokenizer, 'summarize: ', device='cuda:0', batch_size=256,
                                        max_generation_length=128)
    xsum_df = pd.DataFrame.from_dict({'indices': xsum_dataset.indices, 'model_summary': model_summaries_xsum})
    xsum_df['dataset'] = 'xsum'
    xsum_df.to_csv('experiments/xsum_4_sets_experiment/cnndm_model_xsum_summaries.csv',
                   index=False)


def run_xsum_model_on_cnndm_summaries():
    cnndm_dataset = split_cnndm_dataset(split='documents_for_summarization',
                                        path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/cnndm_docs_for_summarization_20000_indices_seed_42.pkl",
                                        num_of_documents_for_summarization=20000,
                                        seed=42)
    xsum_model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_xsum_19_11_2023_16_44_42/checkpoint-45000"
    model = T5ForConditionalGeneration.from_pretrained(xsum_model_path)
    tokenizer = T5Tokenizer.from_pretrained(xsum_model_path)
    cnndm_texts = [cnndm_dataset[i]['text'] for i in range(len(cnndm_dataset))]
    model_summaries_cnndm = t5_summarize(cnndm_texts, model, tokenizer, 'summarize: ', device='cuda:1', batch_size=256,
                                         max_generation_length=128)
    cnndm_df = pd.DataFrame.from_dict({'indices': cnndm_dataset.indices, 'model_summary': model_summaries_cnndm})
    cnndm_df['dataset'] = 'cnndm'
    cnndm_df.to_csv('experiments/xsum_4_sets_experiment/xsum_model_on_cnndm_summaries.csv',
                    index=False)


def create_dataset_for_revision_xsum():
    revised_path = 'experiments/xsum_4_sets_experiment/revision_results_seahorse_threshold.csv'
    revised_df = pd.read_csv(revised_path)
    revised_df = revised_df[revised_df['dataset'] == 'xsum']
    full_df = pd.read_csv('experiments/xsum_4_sets_experiment/xsum_model_summaries.csv', index_col=0)
    no_need_to_revise = full_df[full_df['factuality_score_seahorse_xxl'] >= 0.5]
    return revised_df, no_need_to_revise


def create_dataset_for_revision_cnndm():
    revised_path = 'experiments/xsum_4_sets_experiment/revision_results_seahorse_threshold.csv'
    revised_df = pd.read_csv(revised_path)
    revised_df = revised_df[revised_df['dataset'] == 'cnndm']

    full_df = pd.read_csv('experiments/xsum_4_sets_experiment/both_models_summaries.csv', index_col=0)
    full_df = full_df[full_df['dataset'] == 'cnndm']
    no_need_to_revise = full_df[full_df['factuality_score_seahorse_xxl'] >= 0.5]

    return revised_df, no_need_to_revise


def create_dataset_for_revision_combined():
    revised_path = 'experiments/xsum_4_sets_experiment/revision_results_seahorse_threshold.csv'
    revised_df = pd.read_csv(revised_path)

    full_df = pd.read_csv('experiments/xsum_4_sets_experiment/cnndm_model_summaries.csv', index_col=0)
    no_need_to_revise = full_df[full_df['factuality_score_seahorse_xxl'] >= 0.5]

    return revised_df, no_need_to_revise


def create_datasets_for_revision_model(method, dataset, seed=42, train_size=0.85, factuality_threshold=0.5,
                                       rouge_threshold=0.5, diff_threshold=0.4):
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    cnndm_dataset = split_cnndm_dataset(split='documents_for_summarization',
                                        path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/cnndm_docs_for_summarization_20000_indices_seed_42.pkl",
                                        num_of_documents_for_summarization=20000,
                                        seed=42)
    xsum_indices_to_text = {xsum_dataset.indices[i]: xsum_dataset[i]['text'] for i in range(len(xsum_dataset))}
    cnndm_indices_to_text = {cnndm_dataset.indices[i]: cnndm_dataset[i]['text'] for i in range(len(cnndm_dataset))}
    if dataset == 'xsum':
        revised_df, no_need_to_revise = create_dataset_for_revision_xsum()
    elif dataset == 'cnndm':
        revised_df, no_need_to_revise = create_dataset_for_revision_cnndm()
    else:
        revised_df, no_need_to_revise = create_dataset_for_revision_combined()
    revised_df['text'] = revised_df.apply(
        lambda x: xsum_indices_to_text[x['indices']] if x['dataset'] == 'xsum' else cnndm_indices_to_text[x['indices']],
        axis=1)
    no_need_to_revise['text'] = no_need_to_revise.apply(
        lambda x: xsum_indices_to_text[x['indices']] if x['dataset'] == 'xsum' else cnndm_indices_to_text[x['indices']],
        axis=1)
    no_need_revision_texts = no_need_to_revise['text'].tolist()
    no_need_revision_original_summaries = no_need_to_revise['model_summary'].tolist()
    no_need_revision_revised_summaries = no_need_to_revise['model_summary'].tolist()

    if method == 'classifier':
        revised_df = revised_df[
            (revised_df['factuality_score_seahorse_xxl'] < factuality_threshold) & (
                    revised_df['seahorse_scores_post_revision'] >= factuality_threshold)]
    elif method == 'classifier_and_rouge_threshold':
        rouge_metric = evaluate.load('rouge')
        revised_df['rougeL'] = \
            rouge_metric.compute(predictions=revised_df['revised_summary'], references=revised_df['model_summary'],
                                 use_aggregator=False)['rougeL']
        revised_df = revised_df[
            (revised_df['factuality_score_seahorse_xxl'] < factuality_threshold) & (
                    revised_df['seahorse_scores_post_revision'] >= factuality_threshold) & (
                    revised_df['rougeL'] > rouge_threshold)]
    elif method == 'diff':
        revised_df = revised_df[
            revised_df['seahorse_scores_post_revision'] - revised_df['factuality_score_seahorse_xxl'] >=
            diff_threshold]
    elif method == 'diff_and_rouge':
        rouge_metric = evaluate.load('rouge')
        revised_df['rougeL'] = \
            rouge_metric.compute(predictions=revised_df['revised_summary'], references=revised_df['model_summary'],
                                 use_aggregator=False)['rougeL']
        revised_df = revised_df[
            (revised_df['seahorse_scores_post_revision'] - revised_df['factuality_score_seahorse_xxl'] >=
             diff_threshold) & (revised_df['rougeL'] > rouge_threshold)]

    need_revision_texts = revised_df['text'].tolist()
    need_revision_original_summaries = revised_df['model_summary'].tolist()
    need_revision_revised_summaries = revised_df['revised_summary'].tolist()
    # texts = no_need_revision_texts + need_revision_texts
    # original_summaries = no_need_revision_original_summaries + need_revision_original_summaries
    # revised_summaries = no_need_revision_revised_summaries + need_revision_revised_summaries
    texts = need_revision_texts
    original_summaries = need_revision_original_summaries
    revised_summaries = need_revision_revised_summaries
    np.random.seed(seed)
    if train_size != 1 and train_size != 0:
        train_indices = np.random.choice(len(texts), int(len(texts) * train_size), replace=False)
        val_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
        train_dataset = RevisionDataset(np.array(texts)[train_indices].tolist(),
                                        np.array(original_summaries)[train_indices].tolist(),
                                        np.array(revised_summaries)[train_indices].tolist())
        val_dataset = RevisionDataset(np.array(texts)[val_indices].tolist(),
                                      np.array(original_summaries)[val_indices].tolist(),
                                      np.array(revised_summaries)[val_indices].tolist())
        return train_dataset, val_dataset
    elif train_size == 1:
        train_dataset = RevisionDataset(texts, original_summaries, revised_summaries)
        return train_dataset, None
    else:
        val_dataset = RevisionDataset(texts, original_summaries, revised_summaries)
    return None, val_dataset


def parseargs_train_revision_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_path')
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-method', type=str, default='diff')
    parser.add_argument('-dataset', type=str, default='xsum')
    parser.add_argument('-train_size', type=float, default=1)
    parser.add_argument('--save', action='store_true')
    return parser.parse_args()


def train_revision_model():
    os.environ["WANDB_DISABLED"] = "true"
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    args = parseargs_train_revision_model()
    lr = args.lr
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    epochs = args.epochs
    weight_decay = args.weight_decay
    method = args.method
    dataset = args.dataset
    train_size = args.train_size
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    revision_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large')
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-base')
    train_dataset, val_dataset = create_datasets_for_revision_model(method, dataset, train_size=train_size)
    train_args = Seq2SeqTrainingArguments(
        output_dir=f'experiments/xsum_4_sets_experiment/checkpoints/flan_t5_{run_name}',
        do_train=True, do_eval=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr, num_train_epochs=epochs, predict_with_generate=True, generation_num_beams=4,
        generation_max_length=128,
        evaluation_strategy='no', save_strategy='no', eval_accumulation_steps=30,
        weight_decay=weight_decay,
        metric_for_best_model='rougeL', no_cuda=False, logging_steps=0.01, fp16=True)
    max_length_train = 1024
    trainer = T5_Trainer(collate_fn=collate_fn_revision, model=revision_model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         compute_metrics=lambda p: compute_metric_rouge(p, tokenizer),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    if args.save:
        checkpoint_path = 'experiments/xsum_4_sets_experiment/checkpoints/flan_t5_large_final_revision_model.pth'
        checkpoint = {'model': revision_model.state_dict(),
                      'lr': lr, 'epochs': epochs, 'batch_size': batch_size,
                      'gradient_accumulation_steps': gradient_accumulation_steps
            , 'weight_decay': weight_decay, 'method': method, 'dataset': dataset, 'train_size': train_size}
        torch.save(checkpoint, checkpoint_path)
    # need_revision = []
    # for i in range(len(val_dataset)):
    #     item = val_dataset[i]
    #     summary = item['summary']
    #     text = item['text']
    #     revised_summary = item['revised_summary']
    #     if summary != revised_summary:
    #         need_revision.append(i)
    # from torch.utils.data import Subset
    # need_revision_dataset = Subset(val_dataset, need_revision)
    # no_need_revision_dataset = Subset(val_dataset, list(set(range(len(val_dataset))) - set(need_revision)))
    df = pd.read_csv('experiments/xsum_4_sets_experiment/xsum_model_summaries.csv', index_col=0)
    might_contain = split_xsum_dataset(split='documents_for_summarization',
                                       path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_10000_indices_seed_42.pkl",
                                       num_of_documents_for_summarization=10000,
                                       seed=42)
    all_rel = split_xsum_dataset(split='documents_for_summarization',
                                 path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                 num_of_documents_for_summarization=20000,
                                 seed=42)
    xsum_indices_to_text = {all_rel.indices[i]: all_rel[i]['text'] for i in range(len(all_rel))}
    df = df[~df['indices'].isin(might_contain.indices)]
    df = df[df['factuality_score_seahorse_xxl'] < 0.5]
    df['text'] = df.apply(lambda x: xsum_indices_to_text[x['indices']], axis=1)
    texts = df['text'].tolist()[::20]
    summaries = df['model_summary'].tolist()[::20]
    ood_before_scores = df['factuality_score_seahorse_xxl'].tolist()[::20]
    ood_dataset = RevisionDataset(texts=texts, summaries=summaries, revised_summaries=summaries)
    # from torch.utils.data import Subset
    # trainer.train()
    # train_dataset = Subset(train_dataset, list(range(len(train_dataset)))[::10])
    # train_predictions = trainer.predict(test_dataset=train_dataset, max_length=128, num_beams=4)
    # train_predictions = tokenizer.batch_decode(train_predictions.predictions, skip_special_tokens=True)
    #
    # val_predictions = trainer.predict(test_dataset=val_dataset, max_length=128, num_beams=4)
    # val_predictions = tokenizer.batch_decode(val_predictions.predictions, skip_special_tokens=True)

    ood_predictions = trainer.predict(test_dataset=ood_dataset, max_length=128, num_beams=4)
    ood_predictions = tokenizer.batch_decode(ood_predictions.predictions, skip_special_tokens=True)
    del trainer
    del revision_model
    torch.cuda.empty_cache()
    return ood_predictions, ood_before_scores, texts, epochs, lr, args


def score(ood_predictions, ood_before_scores, texts, epochs, lr, scoring_metric):
    # before_train_scores = scoring_metric.score(texts=train_dataset.texts, summaries=train_dataset.summaries)
    # print(np.mean(before_train_scores))
    ood_scores_post_revision = scoring_metric.score(texts=texts, summaries=ood_predictions)
    plt.scatter(ood_before_scores, ood_scores_post_revision)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'experiments/xsum_4_sets_experiment/outputs/full_epochs_{epochs}_lr_{lr}.png')
    plt.show()
    plt.hist(ood_before_scores, bins=20, label='before revision', alpha=0.5)
    plt.hist(ood_scores_post_revision, bins=20, label='after revision', alpha=0.5)
    plt.xlim(0, 1)
    plt.legend()
    plt.savefig(f'experiments/xsum_4_sets_experiment/outputs/full_epochs_{epochs}_lr_{lr}_hist.png')
    plt.show()
    print("lr = ", lr)
    print("epochs = ", epochs)
    print(f"ood scores before revision: {np.mean(ood_before_scores)}")
    print(f"ood scores after revision: {np.mean(ood_scores_post_revision)}")
    print("------------------------------------------------")
    print()


def revise_model_outputs():
    x = torch.load('experiments/xsum_4_sets_experiment/checkpoints/flan_t5_large_final_revision_model.pth')
    revision_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large', device_map='auto',
                                                                torch_dtype=torch.float16)
    revision_model.load_state_dict(x['model'])
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    # xsum_models = pd.read_csv('experiments/xsum_4_sets_experiment/xsum_model_summaries.csv')
    df = pd.read_csv('experiments/xsum_4_sets_experiment/documents_for_summarization_fully_scored.csv', index_col=0)
    # both_summaries_xsum = both_summaries[both_summaries['dataset'] == 'xsum']
    # both_summaries_cnndm = both_summaries[both_summaries['dataset'] == 'cnndm']
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    xsum_indices_to_text = {xsum_dataset.indices[i]: xsum_dataset[i]['text'] for i in range(len(xsum_dataset))}
    rel_df = df[df['seahorse_score'] < 0.5]
    texts = [xsum_indices_to_text[i] for i in rel_df['indices'].tolist()]
    machine_summaries = rel_df['summary'].tolist()

    # del summarizer
    # torch.cuda.empty_cache()
    revised_summaries_4_beams = t5_revise(texts, machine_summaries, revision_model, tokenizer, prompt='revise: ',
                                          device='cuda',
                                          batch_size=12,
                                          generation_max_length=128, num_beams=4, early_stopping=True,
                                          encoding_max_length=1024)
    revised_summaries_1_beams = t5_revise(texts, machine_summaries, revision_model, tokenizer, prompt='revise: ',
                                          device='cuda',
                                          batch_size=16,
                                          generation_max_length=128, num_beams=1, encoding_max_length=1024)
    df['revised_summary_4_beams'] = None
    df['revised_summary_1_beams'] = None
    df.loc[df['seahorse_score'] < 0.5, 'revised_summary_4_beams'] = revised_summaries_4_beams
    df.loc[df['seahorse_score'] < 0.5, 'revised_summary_1_beams'] = revised_summaries_1_beams
    df.to_csv('experiments/xsum_4_sets_experiment/documents_for_summarization_fully_scored_with_revised.csv',
              index=False)
    del revision_model
    time.sleep(30)
    torch.cuda.empty_cache()
    metric = Seahorse_metrics(model_path="google/seahorse-xxl-q4",
                              tokenizer_name="google/seahorse-xxl-q4",
                              device="auto"
                              , batch_size=1, max_length=2048, torch_dtype=torch.float16)
    pre_scores = rel_df['seahorse_score'].tolist()
    post_scores_4_beams = metric.score(texts=texts, summaries=revised_summaries_4_beams)
    df['seahorse_scores_post_revision_4_beams'] = None
    df.loc[df['seahorse_score'] < 0.5, 'seahorse_scores_post_revision_4_beams'] = post_scores_4_beams
    df.to_csv('experiments/xsum_4_sets_experiment/documents_for_summarization_fully_scored_with_revised.csv',
              index=False)
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(predictions=revised_summaries_4_beams, references=machine_summaries)
    print(np.mean(pre_scores))
    print(np.mean(post_scores_4_beams))
    print(rouge_scores)
    post_scores_1_beams = metric.score(texts=texts, summaries=revised_summaries_1_beams)
    df['seahorse_score_revised_1_beams'] = None
    df.loc[df['seahorse_score'] < 0.5, 'seahorse_score_revised_1_beams'] = post_scores_1_beams
    df.to_csv('experiments/xsum_4_sets_experiment/documents_for_summarization_fully_scored_with_revised.csv',
              index=False)
    print(np.mean(pre_scores))
    print(np.mean(post_scores_1_beams))
    print(rouge_scores)


def main():
    revise_model_outputs()
    # ood_predictions, ood_before_scores,texts,epochs,lr,args = train_revision_model()
    # import json
    # res = {}
    # res['ood_texts'] = texts
    # res['method'] = args.method
    # res['ood_predictions'] = ood_predictions
    # res['ood_before_scores'] = ood_before_scores
    # with open('experiments/xsum_4_sets_experiment/ood_predictions_flan_large.json', 'r') as f:
    #     full_results = json.load(f)
    # full_results[f'epochs_{epochs}_lr_{lr}_method_{args.method}'] = res
    # with open('experiments/xsum_4_sets_experiment/ood_predictions_flan_large.json', 'w') as f:
    #     json.dump(full_results, f)
    # time.sleep(30)
    # torch.cuda.empty_cache()
    # import json
    # with open('experiments/xsum_4_sets_experiment/ood_predictions_flan_large.json', 'r') as f:
    #     full_results = json.load(f)
    # from Seahorse_metrics.metrics import Seahorse_metrics
    # factuality_metric = Seahorse_metrics(model_path="google/seahorse-xxl-q4",
    #                                      tokenizer_name="google/seahorse-xxl-q4",
    #                                      device="auto"
    #                                      , batch_size=1, max_length=2048, torch_dtype=torch.float16)
    # rel_res = full_results['epochs_1_lr_0.0001_method_diff']
    # ood_predictions = rel_res['ood_predictions']
    # ood_before_scores = rel_res['ood_before_scores']
    # texts = rel_res['ood_texts']
    # epochs = 1
    # lr = 0.0001
    # score(ood_predictions, ood_before_scores,texts,epochs,lr,scoring_metric=factuality_metric)


if __name__ == '__main__':
    main()


def parseargs_llms():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_path')
    parser.add_argument('-revision_model_name', type=str, default='mock')
    parser.add_argument('-revision_prompt', type=str,
                        default="""I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific  compared to the document, you should not change it. Output only the corrected summary and nothing more.""")

    parser.add_argument('-API_KEY_revision_model', type=str, default=None)
    parser.add_argument('-contingency_file_dir', type=str, default='contingency_tables')

    args = parser.parse_args()
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f'run_{formatted_datetime}'
    args.dir_path = args.contingency_file_dir + '/' + run_name
    os.makedirs(args.dir_path)
    return args


def llm_revision():
    args = parseargs_llms()
    df = pd.read_csv('experiments/xsum_4_sets_experiment/both_models_summaries.csv', index_col=0)
    rel_df = df[(df['factuality_score_seahorse_xxl'] < 0.5)]
    summaries = rel_df['model_summary'].tolist()
    texts = rel_df['text'].tolist()
    print(len(texts))
    revision_model = chose_revision_model(args)
    revised_summaries, errors = [], []
    for text, summary in tqdm(zip(texts, summaries)):
        revised_summary, error = revision_model.revise_single(text=text, summary=summary)
        revised_summaries.append(revised_summary)
        errors.append(error)
        time.sleep(2)
    rel_df['revised_summary'] = revised_summaries
    rel_df['error'] = errors
    rel_df.to_csv(args.output_path + '.csv', index=False)
    return revised_summaries, errors

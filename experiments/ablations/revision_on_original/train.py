import gc
import json
import pickle

import torch
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import pandas as pd
from datetime import datetime
from general.t5_trainer import T5_Trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
import evaluate
import numpy as np
from general.utils import RevisionDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
import time
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize
import gc


def create_dataset(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, xsum_indices,
                   chosen_indices, method,
                   train_size=0.85, seed=42, classifier_threshold=0.5, rouge_threshold=0.5, diff_threshold=0.4,
                   density_threshold=3.5):
    np.random.seed(seed)
    if method == 'all':
        train_indices = np.random.choice(len(texts), int(len(texts) * train_size), replace=False)
        val_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
    elif method == 'classifier' or method == 'classifier_and_rouge_threshold':
        properly_revised = [i for i in range(len(pre_revision_scores)) if
                            (pre_revision_scores[i] < classifier_threshold and post_revision_scores[
                                i] >= classifier_threshold)]
        if method == 'classifier_and_rouge_threshold':
            rouge_metric = evaluate.load('rouge')
            rouge_scores = \
                rouge_metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)[
                    'rougeL']
            indices_above_threshold = [i for i in range(len(rouge_scores)) if rouge_scores[i] >= rouge_threshold]
            # properly revised means both factuality increase and high rouge score
            properly_revised = list(set(properly_revised).intersection(indices_above_threshold))
        train_indices = np.random.choice(properly_revised, int(len(properly_revised) * train_size), replace=False)
        val_indices = np.array(list(set(properly_revised) - set(train_indices)))
    elif method == 'diff' or method == 'diff_and_rouge_threshold' or method == 'diff_and_density_threshold':
        properly_revised = [i for i in range(len(pre_revision_scores)) if
                            (post_revision_scores[i] - pre_revision_scores[i] >= diff_threshold)]
        if method == 'diff_and_rouge_threshold':
            rouge_metric = evaluate.load('rouge')
            rouge_scores = \
                rouge_metric.compute(predictions=revised_summaries, references=summaries, use_aggregator=False)[
                    'rougeL']
            indices_above_threshold = [i for i in range(len(rouge_scores)) if rouge_scores[i] >= rouge_threshold]
            # properly revised means both factuality increase and high rouge score
            properly_revised = list(set(properly_revised).intersection(indices_above_threshold))
        elif method == 'diff_and_density_threshold':
            from general.fragments_metrics import Fragments
            fragments_metric = Fragments()
            density_scores = fragments_metric.score(metrics=['density'], texts=texts, summaries=summaries)['density']
            indices_below_threshold = [i for i in range(len(density_scores)) if density_scores[i] < density_threshold]
            properly_revised = list(set(properly_revised).intersection(indices_below_threshold))
        train_indices = np.random.choice(properly_revised, int(len(properly_revised) * train_size), replace=False)
        val_indices = np.array(list(set(properly_revised) - set(train_indices)))
    elif method == 'chosen_indices':
        properly_revised = [i for i in range(len(texts)) if xsum_indices[i] in chosen_indices]
        train_indices = np.random.choice(properly_revised, int(len(properly_revised) * train_size), replace=False)
        val_indices = np.array(list(set(properly_revised) - set(train_indices)))
    else:
        raise NotImplementedError

    train_texts, train_summaries, train_revised_summaries = np.array(texts)[train_indices].tolist(), \
                                                            np.array(summaries)[train_indices].tolist(), \
                                                            np.array(revised_summaries)[train_indices].tolist()
    train_dataset = RevisionDataset(train_texts, train_summaries, train_revised_summaries)
    print(f"Train size: {len(train_dataset)}")
    if len(val_indices) == 0:
        return train_dataset, None
    else:
        val_texts, val_summaries, val_revised_summaries = np.array(texts)[val_indices].tolist(), np.array(summaries)[
            val_indices].tolist(), np.array(revised_summaries)[val_indices].tolist()
        val_dataset = RevisionDataset(val_texts, val_summaries, val_revised_summaries)
        return train_dataset, val_dataset


def compute_metrics(p, tokenizer):
    rouge = evaluate.load('rouge')
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = rouge.compute(predictions=predictions, references=labels)
    return results


def collate_fn(batch, tokenizer, max_length):
    revised_summaries = [row['revised_summary'] for row in batch]
    texts_inputs = ["summarize: " + row['text'] for row in batch]
    inputs = tokenizer(texts_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def train(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores, xsum_indices, chosen_indices,
          method, args):
    torch.cuda.empty_cache()
    # hyperparameters = get_best_hyperparameters("all")[1][1]['hyperparameters']
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    weight_decay = args.weight_decay
    train_size = args.train_size
    max_length_train = args.max_encoding_length
    num_beams = args.beam_size
    generation_max_length = args.generation_max_length
    rouge_threshold = args.rouge_threshold
    classifier_threshold = args.classifier_threshold
    diff_threshold = args.diff_threshold
    seed = args.seed
    device = args.device
    if device == 'cpu':
        no_cuda = True
    else:
        no_cuda = False
    train_dataset, _ = create_dataset(texts, summaries, revised_summaries, pre_revision_scores,
                                      post_revision_scores, xsum_indices, chosen_indices, method, train_size=train_size,
                                      seed=seed,
                                      rouge_threshold=rouge_threshold, classifier_threshold=classifier_threshold,
                                      diff_threshold=diff_threshold)
    with open('experiments/ablations/revision_on_original/data/all_data_for_same_texts_as_our_method.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)
    run_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    os.environ["WANDB_DISABLED"] = "true"
    model_checkpoint = args.model_checkpoint
    models_dir = args.model_dir
    model_path = os.path.join(models_dir, model_checkpoint)
    # model_path = "experiments/xsum_4_sets_experiment/checkpoints/t5_base_both_10_12_2023_08_54_06/checkpoint-115000"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    train_args = Seq2SeqTrainingArguments(
        output_dir=f'experiments/ablations/revision_on_original/runs/{run_name}',
        do_train=True, do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr, num_train_epochs=epochs, evaluation_strategy='no', save_strategy='no',
        eval_accumulation_steps=30, weight_decay=weight_decay,
        metric_for_best_model='rougeL', no_cuda=no_cuda, predict_with_generate=True, generation_num_beams=num_beams,
        generation_max_length=generation_max_length, logging_steps=0.01)
    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         compute_metrics=lambda p: compute_metrics(p, tokenizer),
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer


def collate_fn_test(batch, tokenizer, max_length):
    texts_inputs = ["summarize: " + row['text'] for row in batch]
    inputs = tokenizer(texts_inputs, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


def test(model, tokenizer, args):
    xsum_dataset = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000,
                                      seed=42)
    batch_size = args.test_batch_size
    encoding_max_length = args.test_max_encoding_length
    beam_size = args.beam_size
    generation_max_length = args.generation_max_length
    xsum_dataloader = DataLoader(dataset=xsum_dataset, batch_size=batch_size,
                                 collate_fn=lambda x: collate_fn_test(x, tokenizer, encoding_max_length))
    if args.device == 'cpu':
        device = 'cpu'
    elif args.device == 'auto':
        device = 'cuda'
    else:
        device = args.device
    model.eval()
    xsum_predictions = []
    with torch.no_grad():
        print("xsum")
        for batch in tqdm(xsum_dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            batch_predictions = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                               max_length=generation_max_length, num_beams=beam_size,
                                               early_stopping=True)
            batch_predictions = tokenizer.batch_decode(batch_predictions, skip_special_tokens=True)
            xsum_predictions.extend(batch_predictions)
    del model
    return xsum_predictions


def args_parser():
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--lr", type=float, default=1e-4)
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--batch_size", type=int, default=16)
    args.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args.add_argument("--weight_decay", type=float, default=0)
    args.add_argument("--train_size", type=float, default=1)
    args.add_argument("--max_encoding_length", type=int, default=512)
    args.add_argument("--model_checkpoint", type=str)
    args.add_argument("--model_dir", type=str, default="experiments/xsum_4_sets_experiment/checkpoints")
    args.add_argument("--test_batch_size", type=int, default=48)
    args.add_argument("--test_max_encoding_length", type=int, default=512)
    args.add_argument("--beam_size", type=int, default=4)
    args.add_argument("--generation_max_length", type=int, default=128)
    args.add_argument("--test_save_path", type=str)
    args.add_argument("--test_save_dir", type=str, default="experiments/ablations/revision_on_original/outputs")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--rouge_threshold", type=float, default=0.7)
    args.add_argument("--classifier_threshold", type=float, default=0.5)
    args.add_argument("--diff_threshold", type=float, default=0.4)
    args.add_argument('--device', type=str, default='auto')
    args.add_argument("--density_threshold", type=float, default=2)
    args.add_argument('--method', type=str, default='diff_and_rouge_threshold')
    args.add_argument("--chosen_indices_path", type=str)
    return args.parse_args()


def create_summaries(args):
    revised_dataset_path = "experiments/ablations/revision_on_original/data/revision_results_original_flan_large_results.csv"
    df = pd.read_csv(revised_dataset_path, index_col=0)
    texts = df['text'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['revised_summaries'].tolist()
    pre_revision_scores = df['pre_revision_score_seahorse'].tolist()
    post_revision_scores = df['post_revision_score_seahorse'].tolist()
    xsum_indices = df['indices'].tolist()
    test_save_path = args.test_save_path
    test_save_dir = args.test_save_dir
    method = args.method
    if method == 'chosen_indices':
        with open(args.chosen_indices_path, 'rb') as f:
            chosen_indices = pickle.load(f)
    print(f"Testing method {method}")
    model, tokenizer = train(texts, summaries, revised_summaries, pre_revision_scores, post_revision_scores,
                             xsum_indices, chosen_indices, method,
                             args)
    predictions = test(model, tokenizer, args)
    results = {'hyperparameters': args.__dict__, 'predictions': predictions, 'method': method}
    with open(test_save_dir + '/' + test_save_path, 'w') as f:
        json.dump(results, f)
    return predictions


def score_factuality(texts, summaries, metrics):
    from Seahorse_metrics.metrics import Seahorse_metrics
    from TrueTeacher.inference import TrueTeacher
    results = {}
    for metric in metrics:
        if 'seahorse' in metrics:
            factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                                 tokenizer_name='google/seahorse-xxl-q4',
                                                 device='auto', batch_size=1, torch_dtype=torch.float16,
                                                 max_length=2048, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['seahorse'] = scores
            del factuality_metric
            gc.collect()
        elif 'teacher' in metric:
            factuality_metric = TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                                            tokenizer_name="google/t5_11b_trueteacher_and_anli",
                                            device='auto', batch_size=1, max_length=2048,
                                            torch_dtype=torch.float16, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['trueteacher'] = scores
            del factuality_metric
            gc.collect()
        elif 'nli' in metric:
            from nli.nli_metric import NLI
            factuality_metric = NLI(batch_size=1, torch_dtype=torch.bfloat16, max_length=2048, device='auto',
                                    return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['nli'] = scores
            del factuality_metric
            gc.collect()
        elif 'q_squared' in metric:
            from q_squared.run_nli import scores_with_nli, aggregate_per_response
            from q_squared.prep_sys_experiment import cross_annotated_scores
            df = cross_annotated_scores(texts, summaries, out_path=None, save=False)
            df = scores_with_nli(in_path=None, df=df)
            df = aggregate_per_response(df=df, out_path=None, save=False)
            results['Q2'] = df['Q2'].tolist()
        time.sleep(30)
        torch.cuda.empty_cache()
    return results


def score(texts, summaries, original_summaries, args, result_dict):
    # results = result_dict
    fragment_metric = Fragments()
    extractive_results = fragment_metric.score(metrics=['density', 'coverage'], texts=texts, summaries=summaries)
    for key in extractive_results:
        result_dict[key] = extractive_results[key]
    result_dict['length'] = [len(word_tokenize(x)) for x in summaries]
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(predictions=summaries, references=original_summaries, use_aggregator=False)
    for key in rouge_results:
        result_dict[key] = rouge_results[key]
    with open(args.test_save_dir + '/' + args.test_save_path, 'w') as f:
        json.dump(result_dict, f)
    factuality_results = score_factuality(texts, summaries, metrics=['trueteacher'])
    for key in factuality_results:
        result_dict[key] = factuality_results[key]
    with open(args.test_save_dir + '/' + args.test_save_path, 'w') as f:
        json.dump(result_dict, f)
    factuality_results = score_factuality(texts, summaries, metrics=['q_squared'])
    for key in factuality_results:
        result_dict[key] = factuality_results[key]
    with open(args.test_save_dir + '/' + args.test_save_path, 'w') as f:
        json.dump(result_dict, f)
    return result_dict


def main():
    args = args_parser()
    create_summaries(args)
    gc.collect()
    torch.cuda.empty_cache()
    with open(args.test_save_dir + '/' + args.test_save_path, 'r') as f:
        run_produced_results = json.load(f)
    predictions = run_produced_results['predictions']
    xsum_test_set = split_xsum_dataset(split='factuality_test',
                                       path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                       num_of_documents_for_summarization=20000,
                                       seed=42)
    texts = [xsum_test_set[i]['text'] for i in range(len(xsum_test_set))]
    original_summaries = [xsum_test_set[i]['summary'] for i in range(len(xsum_test_set))]
    score(texts=texts, summaries=predictions, original_summaries=original_summaries, args=args,
                               result_dict=run_produced_results)
    # for key in evaluation_results:
    #     run_produced_results[key] = evaluation_results[key]
    # with open(args.test_save_dir + '/' + args.test_save_path, 'w') as f:
    #     json.dump(run_produced_results, f)


def no_score():
    args = args_parser()
    with open(args.test_save_dir + '/' + args.test_save_path, 'r') as f:
        run_produced_results = json.load(f)
    predictions = run_produced_results['predictions']
    xsum_test_set = split_xsum_dataset(split='factuality_test',
                                       path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                       num_of_documents_for_summarization=20000,
                                       seed=42)
    texts = [xsum_test_set[i]['text'] for i in range(len(xsum_test_set))]
    original_summaries = [xsum_test_set[i]['summary'] for i in range(len(xsum_test_set))]
    #    evaluation_results = score(texts=texts, summaries=predictions, original_summaries=original_summaries)
    #     for key in evaluation_results:
    #         run_produced_results[key] = evaluation_results[key]
    from general.fragments_metrics import Fragments
    fragment_metric = Fragments()
    results = {}
    extractive_results = fragment_metric.score(metrics=['density', 'coverage'], texts=texts, summaries=predictions)
    for key in extractive_results:
        results[key] = extractive_results[key]
    import evaluate
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(predictions=predictions, references=original_summaries, use_aggregator=False)
    for key in rouge_results:
        results[key] = rouge_results[key]
    results['length'] = [len(word_tokenize(x)) for x in predictions]
    for key in results:
        print(key, np.mean(results[key]))


if __name__ == '__main__':
    main()
    # no_score()

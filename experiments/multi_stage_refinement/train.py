import json
import os
import sys
import time

os.chdir("/home/yehonatan-pe/Correction_pipeline")
sys.path.append(os.getcwd())

import gc
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from transformers import Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer, t5_summarize, t5_revise, collate_fn_summarization, t5_summarize_mp_main, \
    t5_revise_mp_main, collate_fn_summarization_distillation, llama_revise
import torch.multiprocessing as mp
import evaluate
import numpy as np
from general.fragments_metrics import Fragments
from experiments.scoring import score
from general.utils import SummarizationDataset, find_largest_numbered_dir, SummarizationDatasetwithLogits
import torch
import os
import argparse
import wandb
import shutil
from nltk.tokenize import word_tokenize
import traceback

print(os.getcwd())



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')
    parser.add_argument('-wandb', action='store_true')
    parser.add_argument('-num_of_docs', type=int, default=10000)
    parser.add_argument('-summarization_model_checkpoint', type=str)
    parser.add_argument('-revision_model_checkpoint', type=str)
    parser.add_argument('-revision_prompt_path', type=str, default=None)
    parser.add_argument('-revision_model_device', type=str, default='auto')
    parser.add_argument('-revision_batch_size', type=int, default=1)
    parser.add_argument('-revision_max_generation_length', type=int, default=128)
    parser.add_argument('-revision_beam_size', type=int, default=4)
    parser.add_argument('-revision_max_encoding_length', type=int, default=1024)
    parser.add_argument('-revision_length_penalty', type=float, default=1.0)
    parser.add_argument('-summarization_prompt', type=str, default='summarize: ')
    parser.add_argument('-summarization_device', type=str)
    parser.add_argument('-summarization_batch_size', type=int, default=8)
    parser.add_argument('-summarization_max_generation_length', type=int, default=128)
    parser.add_argument('-summarization_beam_size', type=int, default=4)
    parser.add_argument('-summarization_beam_size_test', type=int, default=4)
    parser.add_argument('-summarization_length_penalty', type=float, default=0.6)
    parser.add_argument('-summarization_max_encoding_length', type=int, default=2048)
    parser.add_argument('-train_data_path', type=str)
    parser.add_argument('-test_data_path', type=str)
    parser.add_argument('-output_dir', type=str)
    parser.add_argument('-set_size', type=int)
    parser.add_argument('-iterations', type=int, default=1)
    parser.add_argument('-batches_to_save_model', type=int)
    parser.add_argument('-batches_to_test_model', type=int)
    parser.add_argument('-epochs_to_save_model', type=int, default=1)
    parser.add_argument('-epochs_to_test_model', type=int, default=1)
    parser.add_argument('-need_revision_factuality_threshold', type=float)
    parser.add_argument('-need_revision_density_threshold', type=float)
    parser.add_argument('-need_revision_rougeL_threshold', type=float)
    parser.add_argument('-revision_successful_factuality_threshold', type=float)
    parser.add_argument('-revision_successful_factuality_diff', type=float)
    parser.add_argument('-revision_successful_revised_to_new_rouge_threshold', type=float)
    parser.add_argument('-revision_successful_revised_to_base_rouge_threshold', type=float)
    parser.add_argument('-revision_successful_density_threshold', type=float)
    parser.add_argument('-revision_successful_density_diff_threshold', type=float)
    parser.add_argument('-refinement_epochs', type=int)
    parser.add_argument('-refinement_batch_size', type=int)
    parser.add_argument('-refinement_gradient_accumulation_steps', type=int)
    parser.add_argument('-refinement_lr', type=float)
    parser.add_argument('-refinement_weight_decay', type=float, default=0.00)
    parser.add_argument('-revision_successful_length_diff_of_revised_summary', type=int)
    parser.add_argument('-revision_successful_length_diff_of_original_model_summary', type=int,
                        help="The lower it is, the more we can remove words from the summaries")
    parser.add_argument('-need_revision_length_diff', type=int,
                        help="The lower it is, the more we can remove words from the summaries")
    parser.add_argument('-revision_model_min_length', type=int, default=0)
    parser.add_argument('-summarization_model_min_length', type=int, default=0)
    parser.add_argument('-save_refinement_inputs', action='store_true')
    parser.add_argument('-use_no_revision_needed', action='store_true')
    parser.add_argument('-distillation', action='store_true')
    parser.add_argument('-student_logits', action='store_true')
    parser.add_argument('-test_seahorse', action='store_true')
    parser.add_argument('-test_trueteacher', action='store_true')
    parser.add_argument('-test_metrics_batch_size', type=int, default=1)
    args = parser.parse_args()
    if args.summarization_device == 'all' or args.revision_model_device == 'all':
        torch.multiprocessing.set_start_method('spawn')
    if args.revision_prompt_path is not None:
        with open(args.revision_prompt_path, 'r') as file:
            args.revision_prompt = file.read()
            args.revision_prompt = args.revision_prompt.strip()
    else:
        args.revision_prompt = 'revise: '
    return args


def refine_model_on_revised_summaries(model, tokenizer, set_texts, set_revised_summaries, set_logits, args):
    if args.distillation:
        train_dataset = SummarizationDatasetwithLogits(set_texts, set_revised_summaries, set_logits)
        # if not os.path.exists(args.output_dir + '/train_dataset.pt'):
        #     torch.save(train_dataset, args.output_dir + '/train_dataset.pt')
        collate_fn = collate_fn_summarization_distillation
    else:
        train_dataset = SummarizationDataset(set_texts, set_revised_summaries)
        collate_fn = collate_fn_summarization
    train_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, 'checkpoints'),
        do_train=args.train,
        per_device_train_batch_size=args.refinement_batch_size,
        gradient_accumulation_steps=args.refinement_gradient_accumulation_steps,
        learning_rate=args.refinement_lr, num_train_epochs=args.refinement_epochs,
        evaluation_strategy='no',
        save_strategy='no', eval_accumulation_steps=30,
        weight_decay=args.refinement_weight_decay,
        no_cuda=False, logging_steps=0.01, report_to=["none"])

    trainer = T5_Trainer(collate_fn=collate_fn, model=model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         max_length_train=args.summarization_max_encoding_length,
                         prompt_train=args.summarization_prompt, distillation=args.distillation)
    trainer.train()
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


def revise_model_summaries(texts, summaries, args):
    if args.revision_model_device == 'all':
        revised_summaries, revised_summaries_logits = t5_revise_mp_main(texts, summaries,
                                                                        args.revision_model_checkpoint, args.output_dir,
                                                                        args.revision_prompt,
                                                                        args.revision_batch_size,
                                                                        args.revision_max_generation_length,
                                                                        args.revision_beam_size,
                                                                        args.revision_max_encoding_length,
                                                                        args.revision_length_penalty,
                                                                        args.revision_model_min_length,
                                                                        args.distillation)
    else:
        if 'llama' in args.revision_model_checkpoint:
            access_token ="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"
            model = AutoModelForCausalLM.from_pretrained(args.revision_model_checkpoint, device_map='auto',
                                                         token=access_token,
                                                         torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(args.revision_model_checkpoint, token=access_token)
            device = "cuda"
            revised_summaries, revised_summaries_logits =llama_revise(texts, summaries, model, tokenizer,
                                                                    args.revision_prompt, device,
                                                                    args.revision_batch_size,
                                                                    args.revision_max_generation_length,
                                                                    args.revision_beam_size, True,
                                                                    args.revision_max_encoding_length,
                                                                    len_penalty=args.revision_length_penalty,
                                                                    return_logits=args.distillation)

        else:
            if args.revision_model_device == 'auto' or args.revision_model_device == 'balanced_low_0':
                model = T5ForConditionalGeneration.from_pretrained(args.revision_model_checkpoint,
                                                                   device_map=args.revision_model_device)
                device = "cuda"
            else:
                model = T5ForConditionalGeneration.from_pretrained(args.revision_model_checkpoint
                                                                   ).to(args.revision_model_device)
                device = args.revision_model_device
            tokenizer = T5Tokenizer.from_pretrained(args.revision_model_checkpoint)
            revised_summaries, revised_summaries_logits = t5_revise(texts, summaries, model, tokenizer,
                                                                    args.revision_prompt, device,
                                                                    args.revision_batch_size,
                                                                    args.revision_max_generation_length,
                                                                    args.revision_beam_size, True,
                                                                    args.revision_max_encoding_length,
                                                                    len_penalty=args.revision_length_penalty,
                                                                    return_logits=args.distillation)
        del model
        gc.collect()
    torch.cuda.empty_cache()
    return revised_summaries, revised_summaries_logits


def process(new_model_summaries_seahorse, new_model_summaries, revised_summaries, texts, original_model_summaries,
            args):
    rouge_metric = evaluate.load('rouge')
    rouge_scores_revised_to_new = \
        rouge_metric.compute(predictions=revised_summaries, references=new_model_summaries, use_aggregator=False)[
            'rougeL']
    rouge_scores_revised_to_base = \
        rouge_metric.compute(predictions=revised_summaries, references=original_model_summaries, use_aggregator=False)[
            'rougeL']
    fragments_metric = Fragments()
    fragments_scores_revised_summaries = fragments_metric.score(['density'], summaries=revised_summaries,
                                                                texts=texts)['density']
    fragments_metric = Fragments()
    fragments_scores_new_model_summaries = fragments_metric.score(['density'], summaries=new_model_summaries,
                                                                  texts=texts)['density']
    revised_summaries_seahorse_scores = None
    indices_of_bad_revisions = []
    summaries_lengths = [len(word_tokenize(x)) for x in new_model_summaries]
    revised_lengths = [len(word_tokenize(x)) for x in revised_summaries]
    original_summaries_lengths = [len(word_tokenize(x)) for x in original_model_summaries]
    if args.revision_successful_revised_to_new_rouge_threshold is not None:
        indices_of_bad_revisions += [i for i in range(len(rouge_scores_revised_to_new)) if
                                     rouge_scores_revised_to_new[
                                         i] < args.revision_successful_revised_to_new_rouge_threshold and i not in indices_of_bad_revisions]
    if args.revision_successful_revised_to_base_rouge_threshold is not None:
        indices_of_bad_revisions += [i for i in range(len(rouge_scores_revised_to_base)) if
                                     rouge_scores_revised_to_base[
                                         i] < args.revision_successful_revised_to_base_rouge_threshold and i not in
                                     indices_of_bad_revisions]
    if args.revision_successful_density_threshold is not None:
        indices_of_bad_revisions += [i for i in range(len(fragments_scores_revised_summaries)) if
                                     fragments_scores_revised_summaries[
                                         i] >= args.revision_successful_density_threshold and i not in
                                     indices_of_bad_revisions]
    if args.revision_successful_density_diff_threshold is not None:
        indices_of_bad_revisions += [i for i in range(len(fragments_scores_new_model_summaries)) if
                                     fragments_scores_revised_summaries[i] - fragments_scores_new_model_summaries[i] >=
                                     args.revision_successful_density_diff_threshold and i not in indices_of_bad_revisions]
    if args.revision_successful_length_diff_of_revised_summary is not None:
        indices_of_bad_revisions += [i for i in range(len(fragments_scores_new_model_summaries)) if
                                     revised_lengths[i] - summaries_lengths[i] <=
                                     args.revision_successful_length_diff_of_revised_summary and i not in indices_of_bad_revisions]
    if args.revision_successful_length_diff_of_original_model_summary is not None:
        indices_of_bad_revisions += [i for i in range(len(fragments_scores_new_model_summaries)) if
                                     revised_lengths[i] - original_summaries_lengths[i] <=
                                     args.revision_successful_length_diff_of_original_model_summary and i not in indices_of_bad_revisions]
    if args.revision_successful_factuality_threshold is not None or args.revision_successful_factuality_diff is not None:
        good_indices = [i for i in range(len(new_model_summaries)) if i not in indices_of_bad_revisions]
        possible_texts = [texts[i] for i in good_indices]
        possible_summaries = [revised_summaries[i] for i in good_indices]
        scores = score(possible_texts, possible_summaries, ['seahorse'],batch_size=args.test_metrics_batch_size)['seahorse']
        if args.revision_successful_factuality_threshold is not None:
            indices_of_bad_revisions += [good_indices[i] for i in range(len(scores)) if
                                         scores[i] < args.revision_successful_factuality_threshold]
        if args.revision_successful_factuality_diff is not None:
            indices_of_bad_revisions += [good_indices[i] for i in range(len(scores)) if
                                         scores[i] - new_model_summaries_seahorse[
                                             good_indices[i]] < args.revision_successful_factuality_diff]

    indices_of_good_revisions = [i for i in range(len(new_model_summaries)) if
                                 i not in indices_of_bad_revisions]
    return indices_of_good_revisions


def select_summaries(new_model_summaries, texts, original_model_summaries, args):
    seahorse_scores = score(texts, new_model_summaries, ['seahorse'],batch_size=args.test_metrics_batch_size)['seahorse']
    need_to_revise_indices = []
    if args.need_revision_factuality_threshold is not None:
        need_to_revise_indices += [i for i in range(len(seahorse_scores)) if
                                   seahorse_scores[i] < args.need_revision_factuality_threshold]
    if args.need_revision_density_threshold is not None:
        fragments_metric = Fragments()
        fragments_scores = fragments_metric.score(['density'], summaries=new_model_summaries, texts=texts)['density']
        need_to_revise_indices += [i for i in range(len(fragments_scores)) if
                                   fragments_scores[i] >= args.need_revision_density_threshold]
    if args.need_revision_rougeL_threshold is not None:
        rouge_metric = evaluate.load('rouge')
        scores = rouge_metric.compute(predictions=new_model_summaries, references=original_model_summaries,
                                      use_aggregator=False)
        need_to_revise_indices += [i for i in range(len(scores['rougeL'])) if
                                   scores['rougeL'][i] < args.need_revision_rougeL_threshold]
    if args.need_revision_length_diff is not None:
        summaries_lengths = [len(word_tokenize(x)) for x in new_model_summaries]
        original_lengths = [len(word_tokenize(x)) for x in original_model_summaries]
        need_to_revise_indices += [i for i in range(len(summaries_lengths)) if
                                   summaries_lengths[i] - original_lengths[i] <= args.need_revision_length_diff]
    need_to_revise_indices = list(set(need_to_revise_indices))
    no_revision_needed_indices = [i for i in range(len(texts)) if i not in need_to_revise_indices]
    return no_revision_needed_indices, need_to_revise_indices, seahorse_scores


def produce_summaries(model, tokenizer, texts, args):
    print("producing summaries")
    if args.summarization_device == 'all':
        new_model_summaries, new_model_logits = t5_summarize_mp_main(model, tokenizer, texts, out_dir=args.output_dir,
                                                                     prompt=args.summarization_prompt,
                                                                     batch_size=args.summarization_batch_size,
                                                                     max_generation_length=args.summarization_max_generation_length,
                                                                     beam_size=args.summarization_beam_size,
                                                                     early_stopping=True,
                                                                     length_penalty=args.summarization_length_penalty,
                                                                     max_encoding_length=args.summarization_max_encoding_length,
                                                                     min_generation_length=args.summarization_model_min_length,
                                                                     student_logits=args.student_logits)
        print("num of summaries produced: ", len(new_model_summaries))
    else:
        model.to(args.summarization_device)
        new_model_summaries, new_model_logits = t5_summarize(texts=texts, model=model, tokenizer=tokenizer,
                                                             prompt=args.summarization_prompt,
                                                             device=args.summarization_device,
                                                             batch_size=args.summarization_batch_size,
                                                             max_generation_length=args.summarization_max_generation_length,
                                                             beam_size=args.summarization_beam_size,
                                                             length_penalty=args.summarization_length_penalty,
                                                             max_encoding_length=args.summarization_max_encoding_length)
    new_model_summaries = [str(x) for x in new_model_summaries]
    return new_model_summaries, new_model_logits


def score_and_chose_summaries(texts, new_model_summaries, new_model_summaries_logits, original_model_summaries, args):
    no_revision_needed_indices, revision_needed_indices, seahorse_scores = select_summaries(new_model_summaries, texts,
                                                                                            original_model_summaries,
                                                                                            args)
    no_revision_needed_texts = [texts[i] for i in no_revision_needed_indices]
    no_revision_needed_summaries = [new_model_summaries[i] for i in no_revision_needed_indices]
    if args.student_logits:
        no_revision_needed_logits = [new_model_summaries_logits[i] for i in no_revision_needed_indices]
    else:
        no_revision_needed_logits = [None] * len(no_revision_needed_indices)
    revision_needed_texts = [texts[i] for i in revision_needed_indices]
    revision_needed_summaries = [new_model_summaries[i] for i in revision_needed_indices]
    revision_needed_seahorse = [seahorse_scores[i] for i in revision_needed_indices]
    revision_need_original_model_summaries = [original_model_summaries[i] for i in revision_needed_indices]
    revised_summaries, revised_summaries_logits = revise_model_summaries(revision_needed_texts,
                                                                         revision_needed_summaries, args)
    indices_of_successful_revisions = process(new_model_summaries_seahorse=revision_needed_seahorse,
                                              new_model_summaries=revision_needed_summaries,
                                              revised_summaries=revised_summaries, texts=revision_needed_texts,
                                              original_model_summaries=revision_need_original_model_summaries,
                                              args=args)
    successfully_revised_summaries = [revised_summaries[i] for i in indices_of_successful_revisions]
    successfully_revised_summaries_logits = [revised_summaries_logits[i] for i in indices_of_successful_revisions]
    successfully_revised_summaries_texts = [revision_needed_texts[i] for i in indices_of_successful_revisions]
    summaries_which_were_successfully_revised = [revision_needed_summaries[i] for i in indices_of_successful_revisions]

    if args.save_refinement_inputs:
        with open(os.path.join(args.output_dir, 'refinement_inputs.json'), 'r') as f:
            data = json.load(f)
        new_key = str(len(data.keys()) - 1)
        data[new_key]['no_revision_summaries'] = no_revision_needed_summaries
        data[new_key]['summaries_which_were_revised'] = summaries_which_were_successfully_revised
        data[new_key]['successfully_revised_summaries'] = successfully_revised_summaries
        with open(os.path.join(args.output_dir, 'refinement_inputs.json'), 'w') as f:
            json.dump(data, f)

    if args.use_no_revision_needed:
        summaries = no_revision_needed_summaries + successfully_revised_summaries
        texts = no_revision_needed_texts + successfully_revised_summaries_texts
        logits = no_revision_needed_logits + successfully_revised_summaries_logits
    else:
        summaries = successfully_revised_summaries
        texts = successfully_revised_summaries_texts
        logits = successfully_revised_summaries_logits
    return summaries, texts, logits


def load_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    return df['text'].tolist()


def load_summarization_model_and_tokenizer(args):
    print(args.summarization_model_checkpoint)
    tokenizer = T5Tokenizer.from_pretrained(args.summarization_model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.summarization_model_checkpoint)
    generation_config = model.generation_config
    generation_config.max_length = args.summarization_max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.summarization_length_penalty
    generation_config.num_beams = args.summarization_beam_size
    model.generation_config = generation_config
    return model, tokenizer


def train(args):
    os.makedirs(args.output_dir + '/latest')
    model, tokenizer = load_summarization_model_and_tokenizer(args)
    args.summarization_model_checkpoint = args.output_dir + '/latest'
    model.save_pretrained(args.output_dir + '/latest')
    tokenizer.save_pretrained(args.output_dir + '/latest')
    train_texts = load_data(args.train_data_path)
    train_texts = train_texts[:args.num_of_docs]
    original_model_summaries, _ = produce_summaries(model=model, tokenizer=tokenizer, texts=train_texts, args=args)
    from nltk.tokenize import word_tokenize
    print(np.mean([len(word_tokenize(x)) for x in original_model_summaries]))
    if args.save_refinement_inputs:
        if not os.path.exists(os.path.join(args.output_dir, 'refinement_inputs.json')):
            data = {}
            new_key = 0
        else:
            with open(os.path.join(args.output_dir, 'refinement_inputs.json'), 'r') as f:
                data = json.load(f)
            new_key = len(data.keys())
        data[new_key] = {'original_summaries': original_model_summaries}
        with open(os.path.join(args.output_dir, 'refinement_inputs.json'), 'w') as f:
            json.dump(data, f)
    for iteration in range(args.iterations):
        for batch, i in enumerate(range(0, len(train_texts), args.set_size)):
            print(f'batch {batch}')
            set_texts = train_texts[i:i + args.set_size]
            original_model_summaries_set = original_model_summaries[i:i + args.set_size]
            new_summaries, new_model_logits = produce_summaries(model=model, tokenizer=tokenizer, texts=set_texts,
                                                                args=args)
            if args.save_refinement_inputs:
                with open(os.path.join(args.output_dir, 'refinement_inputs.json'), 'r') as f:
                    data = json.load(f)
                key_num = len(data.keys())
                data[key_num] = {'new_summaries': new_summaries}
                with open(os.path.join(args.output_dir, 'refinement_inputs.json'), 'w') as f:
                    json.dump(data, f)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            successful_summaries, successful_texts, logits = score_and_chose_summaries(set_texts, new_summaries,
                                                                                       new_model_logits,
                                                                                       original_model_summaries_set,
                                                                                       args)
            model, tokenizer = load_summarization_model_and_tokenizer(args)
            refine_model_on_revised_summaries(model, tokenizer, successful_texts, successful_summaries, logits, args)
            model.save_pretrained(args.output_dir + '/latest')
            tokenizer.save_pretrained(args.output_dir + '/latest')
            if args.batches_to_save_model is not None and (batch + 1) % args.batches_to_save_model == 0:
                dir_name = args.output_dir + f'/iter_{iteration}_batch_{batch + 1}'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                model.save_pretrained(dir_name)
                tokenizer.save_pretrained(dir_name)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if args.batches_to_test_model is not None and (batch + 1) % args.batches_to_test_model == 0:
                temp = args.output_dir
                dir_name = args.output_dir + f'/iter_{iteration}_batch_{batch + 1}'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                args.output_dir = dir_name
                test(args)
                args.output_dir = temp
            model, tokenizer = load_summarization_model_and_tokenizer(args)
        if args.epochs_to_save_model is not None and (iteration + 1) % args.epochs_to_save_model == 0:
            dir_name = args.output_dir + f'/iter_{iteration}'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            model.save_pretrained(dir_name)
            tokenizer.save_pretrained(dir_name)
        if args.epochs_to_test_model is not None and (iteration + 1) % args.epochs_to_test_model == 0:
            dir_name = args.output_dir + f'/iter_{iteration}'
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            temp = args.output_dir
            args.output_dir = dir_name
            test(args, model, tokenizer)
            args.output_dir = temp
    shutil.rmtree(args.output_dir + '/latest')
    return model, tokenizer


def test(args, model=None, tokenizer=None):
    df = pd.read_csv(args.test_data_path, index_col=0)
    args.summarization_beam_size = args.summarization_beam_size_test
    if model is None:
        model, tokenizer = load_summarization_model_and_tokenizer(args)
    summaries, _ = produce_summaries(model=model, tokenizer=tokenizer, texts=df['text'].tolist(), args=args)
    df['new_model_summary'] = summaries
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.test_seahorse:
        seahorse_scores = score(texts=df['text'].tolist(), summaries=summaries, metrics=['seahorse'],batch_size=args.test_metrics_batch_size)['seahorse']
        df['new_model_summary_seahorse'] = seahorse_scores
    else:
        seahorse_scores = 0
    if args.test_trueteacher:
        trueteacher_scores = score(texts=df['text'].tolist(), summaries=summaries, metrics=['trueteacher'],batch_size=args.test_metrics_batch_size)[
            'trueteacher']
        df['new_model_summary_trueteacher'] = trueteacher_scores
    else:
        trueteacher_scores = 0
    fragments_metric = Fragments()
    fragments_scores = fragments_metric.score(['density', 'coverage'], summaries=summaries, texts=df['text'].tolist())
    df['new_model_summary_density'] = fragments_scores['density']
    df['new_model_summary_coverage'] = fragments_scores['coverage']
    rouge_metric = evaluate.load('rouge')
    rouge_scores = \
        rouge_metric.compute(predictions=summaries, references=df['model_summary'].tolist(), use_aggregator=False)[
            'rougeL']
    df['new_model_summary_rougeL_to_base'] = rouge_scores
    df['new_model_summary_length'] = [len(word_tokenize(x)) for x in summaries]
    if args.wandb:
        wandb.log({'rougeL': np.mean(rouge_scores), 'density': np.mean(fragments_scores['density']),
                   'coverage': np.mean(fragments_scores['coverage']), 'seahorse': np.mean(seahorse_scores),'true_teacher': np.mean(trueteacher_scores)})
    test_output_path = os.path.join(args.output_dir, 'test_results.csv')
    df.to_csv(test_output_path)
    gc.collect()
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.wandb:
        wandb.init(project="multi_stage_refinement")
        wandb.config.update(args)

    if args.train:
        latest_dir = find_largest_numbered_dir(args.output_dir) + 1
        args.output_dir = os.path.join(args.output_dir, str(latest_dir))
        os.makedirs(args.output_dir)
        with open(args.output_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f)
        try:
            model, tokenizer = train(args)
            if args.test:
                test(args, model, tokenizer)
        except Exception as e:
            traceback.print_exc()
            error_trace = traceback.format_exc()
            with open(args.output_dir + '/error_trace.txt', 'w') as f:
                f.write(error_trace)
            with open(args.output_dir + '/error.txt', 'w') as f:
                f.write(str(e))


    else:
        if args.test:
            test(args)
    if args.wandb:
        wandb.finish()


def only_refine():
    args = parse_args()
    args.output_dir = os.path.join(args.output_dir, str(32))
    model, tokenizer = load_summarization_model_and_tokenizer(args)
    dataset = torch.load(args.output_dir + '/train_dataset.pt')
    successful_texts = dataset.texts
    successful_summaries = dataset.summaries
    logits = dataset.logits
    refine_model_on_revised_summaries(model, tokenizer, successful_texts, successful_summaries, logits, args)
    args.output_dir = os.path.join(args.output_dir,
                                   f'iter_0_refinement_epochs_{args.refinement_epochs}_lr_{args.refinement_lr}')
    os.makedirs(args.output_dir)
    test(args, model, tokenizer)


if __name__ == '__main__':
    # only_refine()
    main()

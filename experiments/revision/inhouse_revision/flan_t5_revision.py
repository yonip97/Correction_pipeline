import os
import sys

import numpy as np
import pandas as pd
import torch

# os.chdir('../')
# sys.path.append(os.path.dirname(os.getcwd()))
# os.chdir('../')
# sys.path.append(os.path.dirname(os.getcwd()))
# os.chdir('../')
# sys.path.append(os.path.dirname(os.getcwd()))
# print(os.getcwd())
os.chdir("/data/home/yehonatan-pe/Correction_pipeline")
sys.path.append(os.getcwd())
from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer, collate_fn_revision, collate_fn_revision_test, \
    WandbCallback, t5_revise_mp
import argparse
from Seahorse_metrics.metrics import Seahorse_metrics
import evaluate
from general.utils import RevisionDataset, SummarizationDataset, find_largest_numbered_dir
from general.fragments_metrics import Fragments
from experiments.scoring import score
import gc
import wandb
import json
import os
import pickle
import shutil
from nltk.tokenize import word_tokenize


def log_results_to_dir(args, trainer):
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
    if args.eval:
        eval_df = pd.read_csv(args.eval_data_path + '.csv', index_col=0)
        for i, log in enumerate(eval_logs):
            temp_df = eval_df.copy(deep=True)
            for key in log:
                if isinstance(log[key], list):
                    temp_df[key] = log[key]
            temp_df.to_csv(os.path.join(args.output_path, f"eval_{i}.csv"))


def compute_metrics(p, tokenizer, eval_texts, base_seahorse_scores, base_density,
                    base_model_summaries, original_dataset_summaries):
    print("Computing metrics")
    predictions = p.predictions
    labels = p.label_ids
    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    predictions = [str(x) for x in predictions]
    results = {}
    fragments_metric = Fragments()
    density_scores = fragments_metric.score(metrics=['density'], texts=eval_texts, summaries=predictions)['density']
    results['density'] = np.mean(density_scores)
    gc.collect()
    torch.cuda.empty_cache()
    seahorse_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                       device='auto', batch_size=1, torch_dtype=torch.float16, max_length=2048,
                                       return_none=True)
    seahorse_scores = seahorse_metric.score(texts=eval_texts, summaries=predictions)
    del seahorse_metric
    gc.collect()
    torch.cuda.empty_cache()
    results['seahorse'] = np.mean([x for x in seahorse_scores if x is not None])
    seahorse_diff = [x - y for x, y in zip(seahorse_scores, base_seahorse_scores) if
                     x is not None]
    results['seahorse_diff'] = np.mean([x for x in seahorse_diff if x is not None])
    density_diff = [x - y for x, y in zip(density_scores, base_density)]
    results['density_diff'] = np.mean(density_diff)

    rouge_metric = evaluate.load('rouge')
    rouge_scores_to_original = \
        rouge_metric.compute(predictions=predictions, references=original_dataset_summaries, use_aggregator=False)[
            'rougeL']
    results['rougeL_to_original'] = np.mean(rouge_scores_to_original)
    rouge_scores_to_base = \
        rouge_metric.compute(predictions=predictions, references=base_model_summaries, use_aggregator=False)['rougeL']
    results['rougeL_to_base'] = np.mean(rouge_scores_to_base)
    # wandb.log(results)
    results['predicted_summaries'] = predictions
    results['seahorse_scores'] = seahorse_scores
    results['density_scores'] = density_scores
    results['rouge_to_original'] = rouge_scores_to_original
    results['rouge_to_base'] = rouge_scores_to_base
    return results


def get_unrevised_train_data(args):
    df = pd.read_csv(args.unrevised_train_data_path, index_col=0)
    if args.unrevised_factuality_threshold is not None:
        df = df[df['model_summary_seahorse'] >= args.unrevised_factuality_threshold]
    if args.unrevised_rouge_threshold is not None:
        rouge_metric = evaluate.load('rouge')
        df['rougeL'] = \
            rouge_metric.compute(predictions=df['model_summary'], references=df['original_summary'],
                                 use_aggregator=False)['rougeL']
        df = df[df['rougeL'] > args.unrevised_rouge_threshold]
    if args.unrevised_density_threshold is not None:
        df = df[df['model_summary_density'] <= args.unrevised_density_threshold]
    texts = df['text'].tolist()
    original_model_summaries = df['model_summary'].tolist()
    return texts, original_model_summaries


def get_revised_train_data(args):
    revised_df = pd.read_csv(args.train_data_path, index_col=0)
    if args.revised_factuality_threshold is not None:
        revised_df = revised_df[
            (revised_df['model_summary_seahorse'] < args.revised_factuality_threshold) & (
                    revised_df['revised_summary_seahorse'] >= args.revised_factuality_threshold)]
    if args.revised_rouge_threshold is not None:
        rouge_metric = evaluate.load('rouge')
        revised_df['rougeL'] = \
            rouge_metric.compute(predictions=revised_df['revised_summary'], references=revised_df['model_summary'],
                                 use_aggregator=False)['rougeL']
        revised_df = revised_df[revised_df['rougeL'] > args.revised_rouge_threshold]
    if args.revised_factuality_diff is not None:
        revised_df = revised_df[
            revised_df['revised_summary_seahorse'] - revised_df['model_summary_seahorse'] >=
            args.revised_factuality_diff]
    if args.revised_density_threshold is not None:
        revised_df = revised_df[revised_df['revised_summary_density'] <= args.revised_density_threshold]
    if args.revised_density_diff is not None:
        revised_df = revised_df[
            revised_df['revised_summary_density'] - revised_df['model_summary_density'] <= args.revised_density_diff]
    if args.revised_tokens_diff is not None:
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        revised_lengths = [len(tokenizer(x)['input_ids']) for x in revised_df['revised_summary']]
        model_lengths = [len(tokenizer(x)['input_ids']) for x in revised_df['model_summary']]
        diffs = [x - y for x, y in zip(revised_lengths, model_lengths)]
        revised_df = revised_df[[x >= args.revised_tokens_diff for x in diffs]]
    texts = revised_df['text'].tolist()
    original_model_summaries = revised_df['model_summary'].tolist()
    revised_summaries = revised_df['revised_summary'].tolist()
    print("Number of revised samples: ", len(revised_df))
    # train_dataset = RevisionDataset(texts, original_model_summaries, revised_summaries)
    return texts, original_model_summaries, revised_summaries


def create_val_dataset(args):
    eval_df = pd.read_csv(args.eval_data_path + '.csv', index_col=0)
    eval_dataset = SummarizationDataset(texts=eval_df['text'].tolist(), summaries=eval_df['model_summary'].tolist())
    eval_texts = eval_df['text'].tolist()
    eval_pre_revision_factuality_scores = eval_df['model_summary_seahorse'].tolist()
    eval_pre_revision_density = eval_df['model_summary_density'].tolist()
    eval_base_model_summaries = eval_df['model_summary'].tolist()
    eval_original_dataset_summaries = eval_df['original_summary'].tolist()
    return eval_dataset, eval_texts, eval_pre_revision_factuality_scores, eval_pre_revision_density, eval_base_model_summaries, eval_original_dataset_summaries


def parseargs_train_revision_model():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-output_path')
    parser.add_argument('-train_data_path', type=str)
    parser.add_argument('-unrevised_train_data_path', type=str)
    parser.add_argument('-model_name', type=str, default='flan-t5-xl')
    parser.add_argument('-pretrained_model_path', type=str, default=None)
    parser.add_argument('-pretrained_adapter_path', type=str, default=None)
    parser.add_argument('-save_dir', type=str, default='experiments/revision/inhouse_revision_model_results')
    parser.add_argument('-dataset_name', type=str, default='500_not_factual')
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-train_batch_size', type=int, default=1)
    parser.add_argument('-eval_batch_size', type=int, default=2)
    parser.add_argument('-evaluation_strategy', type=str, default='no')
    parser.add_argument('-save_strategy', type=str, default='no')
    parser.add_argument('-eval_steps', type=float, default=5.0)
    parser.add_argument('-gradient_accumulation_steps', type=int, default=16)
    parser.add_argument('-epochs', type=int, default=5)
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-dataset', type=str, default='xsum')
    parser.add_argument('-beam_size', type=int, default=4)
    parser.add_argument('-length_penalty', type=float, default=0.6)
    parser.add_argument('-max_generation_length', type=int, default=128)
    parser.add_argument('-min_generation_length', type=int, default=0)
    parser.add_argument('-max_encoding_length', type=int, default=1024)
    parser.add_argument('-revised_factuality_threshold', type=float, default=None)
    parser.add_argument('-unrevised_factuality_threshold', type=float, default=None)
    parser.add_argument('-revised_factuality_diff', type=float, default=None)
    parser.add_argument('-revised_rouge_threshold', type=float, default=None)
    parser.add_argument('-unrevised_rouge_threshold', type=float, default=None)
    parser.add_argument('-revised_density_threshold', type=float, default=None)
    parser.add_argument('-unrevised_density_threshold', type=float, default=None)
    parser.add_argument('-revised_density_diff', type=float, default=None)
    parser.add_argument('-revised_tokens_diff', type=float, default=None)
    parser.add_argument('-eval_data_path', type=str)
    parser.add_argument('-use_lora', action='store_true')
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-train', action='store_true', default=False)
    parser.add_argument('-eval', action='store_true', default=False)
    parser.add_argument('-test', action='store_true', default=False)
    parser.add_argument('-test_data_path', type=str)
    parser.add_argument('-test_output_path', type=str)
    parser.add_argument('-train_prompt', type=str, default=None)
    parser.add_argument('-eval_prompt', type=str, default=None)
    parser.add_argument('-test_prompt', type=str, default=None)
    parser.add_argument('-revision_iterations', type=int, default=1)
    parser.add_argument('-upsample_revised', action='store_true', default=False)
    parser.add_argument('-downsample_unrevised', action='store_true', default=False)
    parser.add_argument('-test_seahorse', action='store_true', default=False)
    parser.add_argument('-test_trueteacher', action='store_true', default=False)
    parser.add_argument('-save', action='store_true', default=False)
    parser.add_argument('-wandb', action='store_true', default=False)
    args = parser.parse_args()
    if args.pretrained_model_path is None:
        model_name_to_path = {'flan-t5-large': 'google/flan-t5-large', 'flan-t5-xl': 'google/flan-t5-xl'}

        args.model_path = model_name_to_path[args.model_name]
    else:
        args.model_path = args.pretrained_model_path
    return args


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def train_and_evaluate_revision_model(args):
    torch.multiprocessing.set_start_method('spawn')
    if args.train:
        revision_model = T5ForConditionalGeneration.from_pretrained(args.model_path, device_map='auto')
        tokenizer = T5Tokenizer.from_pretrained(args.model_path)
        generation_config = revision_model.generation_config
        generation_config.max_length = args.max_generation_length
        generation_config.early_stopping = True
        generation_config.length_penalty = args.length_penalty
        generation_config.num_beams = args.beam_size
        generation_config.min_length = args.min_generation_length
        revision_model.generation_config = generation_config
        lr = args.lr
        train_batch_size = args.train_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        epochs = args.epochs
        weight_decay = args.weight_decay
        texts = []
        original_model_summaries = []
        new_summaries = []
        revised_texts, revised_original_model_summaries, revised_summaries = [], [], []
        unrevised_texts, unrevised_original_model_summaries = [], []
        if args.train_data_path is not None:
            revised_texts, revised_original_model_summaries, revised_summaries = get_revised_train_data(args)
        if args.unrevised_train_data_path is not None:
            unrevised_texts, unrevised_original_model_summaries = get_unrevised_train_data(args)
        if args.upsample_revised and args.train_data_path is not None:
            revised_texts = revised_texts * (len(unrevised_texts) // len(revised_texts))
            revised_original_model_summaries = revised_original_model_summaries * (
                    len(unrevised_texts) // len(revised_texts))
            revised_summaries = revised_summaries * (len(unrevised_texts) // len(revised_texts))
        if args.downsample_unrevised and args.unrevised_train_data_path is not None:
            unrevised_texts = unrevised_texts[:len(revised_texts)]
            unrevised_original_model_summaries = unrevised_original_model_summaries[:len(revised_texts)]
        texts += revised_texts
        original_model_summaries += revised_original_model_summaries
        new_summaries += revised_summaries
        texts += unrevised_texts
        original_model_summaries += unrevised_original_model_summaries
        new_summaries += unrevised_original_model_summaries
        train_dataset = RevisionDataset(texts=texts, summaries=original_model_summaries,
                                        revised_summaries=new_summaries)
        eval_dataset, eval_texts, eval_pre_revision_factuality_scores, eval_pre_revision_density \
            , eval_base_model_summaries, eval_original_dataset_summaries = None, None, None, None, None, None
        if args.eval:
            eval_dataset, eval_texts, eval_pre_revision_factuality_scores, eval_pre_revision_density \
                , eval_base_model_summaries, eval_original_dataset_summaries = create_val_dataset(args)
        dataset_name = args.dataset_name
        if not os.path.exists(os.path.join(args.save_dir, dataset_name)):
            os.makedirs(os.path.join(args.save_dir, dataset_name))
        args.save_dir = os.path.join(args.save_dir, dataset_name)
        model_name = args.model_name
        if not os.path.exists(os.path.join(args.save_dir, model_name)):
            os.makedirs(os.path.join(args.save_dir, model_name))
        model_path = os.path.join(args.save_dir, model_name)
        dir_num = find_largest_numbered_dir(model_path) + 1
        output_path = os.path.join(model_path, str(dir_num))
        os.makedirs(output_path)
        args.output_path = output_path
        args.test_output_path = output_path
        with open(output_path + '/args.txt', 'w') as f:
            f.write(str(args))
        train_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(output_path, 'checkpoints'),
            do_train=args.train, do_eval=args.eval,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr, num_train_epochs=epochs, predict_with_generate=True,
            evaluation_strategy=args.evaluation_strategy,
            save_strategy=args.save_strategy, save_total_limit=5,
            eval_steps=args.eval_steps, eval_accumulation_steps=30,
            weight_decay=weight_decay,
            no_cuda=False, logging_steps=0.01, report_to=["none"], save_only_model=True)
        max_length = args.max_encoding_length
        if args.use_lora:
            from peft import LoraConfig, get_peft_model, TaskType
            if args.pretrained_adapter_path is not None:
                revision_model.load_adapter(args.pretrained_adapter_path)
            else:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    lora_dropout=0.05,
                    task_type=TaskType.SEQ_2_SEQ_LM
                )
                revision_model = get_peft_model(revision_model, lora_config)
                print_trainable_parameters(revision_model)
        trainer = T5_Trainer(collate_fn=collate_fn_revision, model=revision_model, tokenizer=tokenizer, args=train_args,
                             train_dataset=train_dataset, eval_dataset=eval_dataset,
                             compute_metrics=lambda p: compute_metrics(p, tokenizer, eval_texts,
                                                                       eval_pre_revision_factuality_scores,
                                                                       eval_pre_revision_density,
                                                                       eval_base_model_summaries,
                                                                       eval_original_dataset_summaries),
                             max_length_train=max_length, max_length_eval=max_length,
                             collate_fn_test=collate_fn_revision_test, callbacks=[WandbCallback()],
                             prompt_train=args.train_prompt, prompt_eval=args.eval_prompt, prompt_test=args.test_prompt)
        trainer.train()
        log_results_to_dir(args, trainer)
        if args.save:
            trainer.save_model()
        if args.test:
            revision_model.save_pretrained(args.test_output_path + "/trained_model")
            tokenizer.save_pretrained(args.test_output_path + '/trained_model')
            args.model_path = args.test_output_path + "/trained_model"
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pass
        # max_length = args.max_encoding_length
        # output_path = args.test_output_path
        # test_args = Seq2SeqTrainingArguments(
        #     output_dir=os.path.join(output_path, 'checkpoints'),
        #     do_train=args.train, do_eval=args.eval,
        #     per_device_eval_batch_size=args.eval_batch_size,
        #     predict_with_generate=True,
        #     evaluation_strategy=args.evaluation_strategy,
        #     save_strategy=args.save_strategy, save_total_limit=3,
        #     eval_steps=args.eval_steps, eval_accumulation_steps=30,
        #     no_cuda=False, report_to=["none"])
        # trainer = T5_Trainer(collate_fn=collate_fn_revision, model=revision_model, tokenizer=tokenizer, args=test_args,
        #                      train_dataset=None, eval_dataset=None,
        #                      max_length_train=max_length, max_length_eval=max_length,
        #                      collate_fn_test=collate_fn_revision_test, callbacks=[WandbCallback()],
        #                      prompt_train=args.train_prompt, prompt_eval=args.eval_prompt, prompt_test=args.test_prompt)
    if args.test:
        revisions_dict = create_test_revisions(args)
        if os.path.exists(args.test_output_path + '/trained_model'):
            shutil.rmtree(args.test_output_path + '/trained_model')
        # del trainer
        # del revision_model
        # gc.collect()
        # torch.cuda.empty_cache()
        score_and_save_revisions(revisions_dict, args)


def score_summaries(texts, summaries, original_model_summaries, original_dataset_summaries, args):
    results = {}
    rouge_metric = evaluate.load('rouge')
    rouge_scores_to_original = \
        rouge_metric.compute(predictions=summaries, references=original_dataset_summaries, use_aggregator=False)[
            'rougeL']
    results['rougeL_to_original'] = rouge_scores_to_original
    rouge_scores_to_base = \
        rouge_metric.compute(predictions=summaries, references=original_model_summaries, use_aggregator=False)['rougeL']
    results['rougeL_to_base'] = rouge_scores_to_base
    fragments_metric = Fragments()
    extract_scores = fragments_metric.score(metrics=['density', 'coverage'], texts=texts, summaries=summaries)
    results['density'] = extract_scores['density']
    results['coverage'] = extract_scores['coverage']
    results['length'] = [len(word_tokenize(x)) for x in summaries]
    if args.test_seahorse:
        seahorse_scores = score(texts=texts, summaries=summaries, metrics=['seahorse'])['seahorse']
        results['seahorse'] = seahorse_scores
    if args.test_trueteacher:
        trueteacher_scores = score(texts=texts, summaries=summaries, metrics=['trueteacher'])['trueteacher']
        results['trueteacher'] = trueteacher_scores
    return results


def create_test_revisions(args):
    test_df = pd.read_csv(args.test_data_path + '.csv', index_col=0)
    test_df = test_df[~test_df['text'].isnull()]
    test_texts = test_df['text'].tolist()
    original_model_summaries = test_df['model_summary'].tolist()
    results_per_iteration = {i: [] for i in range(args.revision_iterations)}
    for iteration in range(args.revision_iterations):
        import torch.multiprocessing as mp
        world_size = torch.cuda.device_count()
        if os.path.exists(args.test_output_path + '/revision_temp'):
            shutil.rmtree(args.test_output_path + '/revision_temp')
        os.makedirs(args.test_output_path + '/revision_temp')
        processes = [mp.Process(target=t5_revise_mp, args=(i,
                                                           world_size, args.model_path,
                                                           args.test_output_path + '/revision_temp', test_texts,
                                                           original_model_summaries,
                                                           args.test_prompt,
                                                           args.eval_batch_size,
                                                           args.max_generation_length, args.beam_size,
                                                           True,
                                                           args.max_encoding_length,
                                                           args.length_penalty, args.min_generation_length)) for i in
                     range(world_size)]
        for process in processes:
            process.start()

        for process in processes:
            process.join()
        for process in processes:
            process.kill()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        revised_summaries = []
        files = os.listdir(args.test_output_path + '/revision_temp')
        files = sorted(files)
        for file in files:
            with open(args.test_output_path + '/revision_temp/' + file, 'rb') as f:
                summaries = pickle.load(f)
                revised_summaries += summaries
        shutil.rmtree(args.test_output_path + '/revision_temp')
        results_per_iteration[iteration] = revised_summaries
        original_model_summaries = revised_summaries
    return results_per_iteration
    # t5_revise(texts=test_texts, summaries=original_model_summaries, model=revision_model, tokenizer=tokenizer,
    #           prompt='revise: ',
    #           device='cuda:0', batch_size=3, generation_max_length=128, num_beams=4, early_stopping=True,
    #           encoding_max_length=1024, len_penalty=0.6)
    # t5_revise_mp_main(texts=test_texts,summaries=original_model_summaries,revision_model=revision_model,args=args)
    # for iter in range(args.revision_iterations):
    #     test_dataset = RevisionDataset(texts=test_texts, summaries=original_model_summaries,
    #                                    revised_summaries=[None] * len(test_texts))
    #     t5_revise_mp_main()
    #
    #     # predictions = trainer.predict(test_dataset=test_dataset)
    #     # final_summaries = trainer.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    #     results_per_iteration[iter] = final_summaries
    #     original_model_summaries = final_summaries
    # return results_per_iteration


def score_and_save_revisions(results_per_iteration, args):
    test_df = pd.read_csv(args.test_data_path + '.csv', index_col=0)
    test_df = test_df[~test_df['text'].isnull()]
    test_texts = test_df['text'].tolist()
    original_dataset_summaries = test_df['original_summary'].tolist()
    original_model_summaries = test_df['model_summary'].tolist()
    for iter in range(args.revision_iterations):
        iter_summaries = results_per_iteration[iter]
        score_predictions = score_summaries(texts=test_texts, summaries=iter_summaries,
                                            original_model_summaries=original_model_summaries,
                                            original_dataset_summaries=original_dataset_summaries, args=args)
        if args.wandb:
            for score in score_predictions:
                wandb.log({f'{score}_iter_{iter}': score_predictions[score]})
        test_df['revised_summary'] = iter_summaries
        for res in score_predictions:
            test_df['revised_summary_' + res] = score_predictions[res]
        print(args.test_output_path)
        test_df.to_csv(args.test_output_path + f'/test_results_{iter}_len_penalty_{args.length_penalty}.csv')


def main():
    args = parseargs_train_revision_model()
    if args.wandb:
        if args.dataset_name == '500_not_factual':
            wandb.init(project='inhouse_revision_model', config=args.__dict__)
        elif args.dataset_name == 'train_docs_not_revised_by_gpt':
            wandb.init(project='inhouse_revision_model_full_revision_results', config=args.__dict__)
        elif args.dataset_name == '500_not_factual_500_factual':
            wandb.init(project='inhouse_revision_model_500_not_factual_500_factual', config=args.__dict__)
    else:
        os.environ['WANDB_DISABLED'] = 'true'
    train_and_evaluate_revision_model(args)
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

import os

os.chdir('../')
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
import pandas as pd
import numpy as np
import argparse
import wandb
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from general.t5_trainer import T5_Trainer
from general.utils import RevisionDataset, SummarizationDataset
from general.fragments_metrics import Fragments
import gc
import torch
import evaluate
from experiments.scoring import score
from nltk.tokenize import word_tokenize
from general.utils import find_largest_numbered_dir
import json
from peft import LoraModel, LoraConfig


def parse():
    args = argparse.ArgumentParser()
    args.add_argument('-revised_data_file', type=str)
    args.add_argument('-unrevised_data_file', type=str)
    args.add_argument('-model_checkpoint', type=str)
    args.add_argument('-train_batch_size_per_device', type=int, default=2)
    args.add_argument('-eval_batch_size_per_device', type=int, default=10)
    args.add_argument('-encoder_max_length', type=int, default=2048)
    args.add_argument('-max_generation_length', type=int, default=128)
    args.add_argument('-lr', type=float, default=5e-4)
    args.add_argument('-beam_size', type=int, default=4)
    args.add_argument('-length_penalty', type=float, default=0.6)
    args.add_argument('-epochs', type=int, default=1)
    args.add_argument('-gradient_accumulation_steps', type=int, default=8)
    args.add_argument('-evaluation_strategy', type=str, default='steps')

    args.add_argument('-output_dir', type=str, default='experiments/multi_objective/data')
    args.add_argument('-optim', type=str, default='adafactor')
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
    args.add_argument('-test', action='store_true')
    args.add_argument('-test_data_path', type=str)
    args.add_argument('-test_new_seahorse', action='store_true')
    args.add_argument('-test_new_trueteacher', action='store_true')
    args.add_argument('-test_fixed_seahorse', action='store_true')
    args.add_argument('-test_fixed_trueteacher', action='store_true')
    args.add_argument('-test_original_model_revised_seahorse', action='store_true')
    args.add_argument('-test_original_model_revised_trueteacher', action='store_true')
    args.add_argument('-save', action='store_true')
    args.add_argument('-train', action='store_true')
    args.add_argument('-create_new_dir', action='store_true')
    args.add_argument('-wandb', action='store_true')
    args.add_argument('-project_name', type=str)
    args.add_argument('-use_adapter', action='store_true')
    args = args.parse_args()
    return args


#


def preprocess_data(df):
    df = df[~df['revised_summary_full_text'].str.contains('No correction')]
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
    summaries = ['Correct the following summary according to the text:\n ' + 'Summary: ' + row['summary'] + '\n' for row
                 in batch]
    documents = ['Text: ' + row['text'] for row in batch]
    revised_summaries = [row['revised_summary'] for row in batch]
    inputs = tokenizer(summaries, documents, padding=True, truncation="only_second", max_length=max_length,
                       return_tensors='pt')
    labels = tokenizer(revised_summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def test_collate_fn_summarize(batch, tokenizer, max_length, prefix=''):
    documents = ["summarize: " + prefix + ':' + row['text'] for row in batch]
    inputs = tokenizer(documents, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


def test_collate_fn_revise(batch, tokenizer, max_length, prefix=''):
    summaries = ['Correct the following summary according to the text:\n ' + 'Summary: ' + row['summary'] + '\n' for row
                 in batch]
    documents = ['Text: ' + row['text'] for row in batch]
    inputs = tokenizer(summaries, documents, padding=True, truncation="only_second", max_length=max_length,
                       return_tensors='pt')
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}


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


def create_train_dataset(args):
    revised_texts = []
    revised_summaries = []
    unrevised_texts = []
    unrevised_summaries = []
    revised_model_summaries = []
    unrevised_model_summaries = []
    if args.revised_data_file is not None:
        revised_df = pd.read_csv(args.revised_data_file + '.csv', index_col=0)
        revised_df = preprocess_data(revised_df)
        revised_df = create_filtered_train_data_revised(revised_df, args)
        revised_texts += revised_df['text'].tolist()
        revised_summaries += revised_df['revised_summary'].tolist()
        revised_model_summaries += revised_df['model_summary'].tolist()
    if args.unrevised_data_file is not None:
        unrevised_df = pd.read_csv(args.unrevised_data_file + '.csv', index_col=0)
        unrevised_df = create_filtered_train_data_unrevised(unrevised_df, args)
        unrevised_texts += unrevised_df['text'].tolist()
        unrevised_summaries += unrevised_df['model_summary'].tolist()
        unrevised_model_summaries += unrevised_df['model_summary'].tolist()
    if args.ratio_revised_to_unrevised is not None and len(revised_texts) > 0 and len(unrevised_texts) > 0:
        curr_ratio = len(revised_texts) / len(unrevised_texts)
        needed_upsampling = int(1 / curr_ratio * args.ratio_revised_to_unrevised)
        revised_texts = revised_texts * needed_upsampling
        revised_summaries = revised_summaries * needed_upsampling
        revised_model_summaries = revised_model_summaries * needed_upsampling
    texts = revised_texts + unrevised_texts
    summaries = revised_summaries + unrevised_summaries
    model_summaries = revised_model_summaries + unrevised_model_summaries
    return texts, summaries, model_summaries


def train(args):
    texts, summaries, model_summaries = create_train_dataset(args)
    if len(texts) == 0:
        raise ValueError("No data to train on")
    train_dataset = RevisionDataset(texts=texts, summaries=model_summaries, revised_summaries=summaries)
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
    generation_config = model.generation_config
    generation_config.max_length = args.max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.length_penalty
    generation_config.num_beams = args.beam_size
    model.generation_config = generation_config
    if args.use_adapter:
        peft_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.01,
        )
        model = LoraModel(model, peft_config, "default")
    print_trainable_parameters(model)
    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True, do_eval=True,
        per_device_train_batch_size=args.train_batch_size_per_device,
        per_device_eval_batch_size=args.eval_batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, num_train_epochs=args.epochs, evaluation_strategy=args.evaluation_strategy,
        save_strategy="no",
        eval_accumulation_steps=30,
        no_cuda=False, predict_with_generate=True,
        optim=args.optim, overwrite_output_dir=False, logging_steps=0.01, report_to=['none'])
    max_length_train = args.encoder_max_length
    trainer = T5_Trainer(collate_fn=train_collate_fn, model=model, tokenizer=tokenizer,
                         args=train_args,
                         train_dataset=train_dataset,
                         max_length_train=max_length_train, max_length_eval=max_length_train)
    trainer.train()
    if args.save:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    return trainer, model


def test(args, trainer):
    test_df = pd.read_csv(args.test_data_path + '.csv', index_col=0)
    test_df = test_df[~test_df['text'].isnull()]
    test_texts = test_df['text'].tolist()
    original_dataset_summaries = test_df['original_summary'].tolist()
    original_model_summaries = test_df['model_summary'].tolist()
    test_dataset = SummarizationDataset(texts=test_texts, summaries=[None] * len(test_texts))
    trainer.collate_fn_test = test_collate_fn_summarize
    if args.use_adapter:
        trainer.model.disable_adapter_layers()
    predictions = trainer.predict(test_dataset=test_dataset)
    predicted_summaries = trainer.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    trainer.collate_fn_test = test_collate_fn_revise
    if args.use_adapter:
        trainer.model.enable_adapter_layers()
    test_dataset = RevisionDataset(texts=test_texts, summaries=predicted_summaries,
                                   revised_summaries=[None] * len(test_texts))
    predictions = trainer.predict(test_dataset=test_dataset)
    final_predictions = trainer.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    test_dataset = RevisionDataset(texts=test_texts, summaries=original_model_summaries,
                                   revised_summaries=original_dataset_summaries)
    predictions = trainer.predict(test_dataset=test_dataset)
    original_model_summaries_revised = trainer.tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    results = score_predictions(test_texts, predicted_summaries, final_predictions, original_model_summaries_revised,
                                original_dataset_summaries, original_model_summaries,
                                args)
    if args.wandb:
        wandb_dict = {}
        for key in results:
            wandb_dict['test_' + key] = np.mean([x for x in results[key] if x is not None])
        wandb.log(wandb_dict)
    results_df = pd.DataFrame(results)
    results_df['text'] = test_texts
    results_df['original_summary'] = original_dataset_summaries
    results_df['model_summary'] = original_model_summaries
    results_df['new_model_summary'] = predicted_summaries
    results_df['new_fixed_summary'] = final_predictions
    results_df['old_model_summary_fixed'] = original_model_summaries_revised
    return results_df


def score_predictions(texts, new_summaries, fixed_summaries, original_model_summaries_revised,
                      original_dataset_summaries, original_model_summaries, args):
    results = {}
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=new_summaries, references=original_model_summaries,
                                        use_aggregator=False)
    results['rougeL_new_to_base'] = rouge_scores['rougeL']
    rouge_scores = rouge_metric.compute(predictions=fixed_summaries, references=original_model_summaries,
                                        use_aggregator=False)
    results['rougeL_fixed_to_base'] = rouge_scores['rougeL']
    print("The fixed to base rouge is", np.mean(rouge_scores['rougeL']))
    rouge_scores = rouge_metric.compute(predictions=new_summaries, references=original_dataset_summaries,
                                        use_aggregator=False)
    results['rougeL_new_to_original'] = rouge_scores['rougeL']
    rouge_scores = rouge_metric.compute(predictions=fixed_summaries, references=original_dataset_summaries,
                                        use_aggregator=False)
    results['rougeL_fixed_to_original'] = rouge_scores['rougeL']
    rouge_scores = rouge_metric.compute(predictions=fixed_summaries, references=new_summaries,
                                        use_aggregator=False)
    results['rougeL_fixed_to_new'] = rouge_scores['rougeL']
    rouge_scores = rouge_metric.compute(predictions=original_model_summaries_revised,
                                        references=original_model_summaries,
                                        use_aggregator=False)
    results['rougeL_original_model_revised_to_base'] = rouge_scores['rougeL']
    rouge_scores = rouge_metric.compute(predictions=original_model_summaries_revised,
                                        references=original_dataset_summaries,
                                        use_aggregator=False)
    results['rougeL_original_model_revised_to_original'] = rouge_scores['rougeL']
    fragments_metric = Fragments()
    scores = fragments_metric.score(metrics=['density', 'coverage'], summaries=new_summaries, texts=texts)
    results['new_model_summary_density'] = scores['density']
    print("The new summary mean density is", np.mean(scores['density']))
    results['new_model_summary_coverage'] = scores['coverage']
    results['new_model_summary_length'] = [len(word_tokenize(summary)) for summary in new_summaries]
    fragments_metric = Fragments()
    scores = fragments_metric.score(metrics=['density', 'coverage'], summaries=fixed_summaries, texts=texts)
    results['fixed_model_summary_density'] = scores['density']
    print("The fixed summary mean density is", np.mean(scores['density']))
    results['fixed_model_summary_coverage'] = scores['coverage']
    results['fixed_model_summary_length'] = [len(word_tokenize(summary)) for summary in fixed_summaries]
    scores = fragments_metric.score(metrics=['density', 'coverage'], summaries=original_model_summaries_revised,
                                    texts=texts)
    results['original_model_summary_revised_density'] = scores['density']
    print("The original model summary revised mean density is", np.mean(scores['density']))
    results['original_model_summary_revised_coverage'] = scores['coverage']
    results['original_model_summary_revised_length'] = [len(word_tokenize(summary)) for summary in
                                                        original_model_summaries_revised]

    if args.test_new_trueteacher:
        results['new_model_summary_trueteacher'] = score(texts=texts, summaries=new_summaries, metrics=['trueteacher'])[
            'trueteacher']
    if args.test_new_seahorse:
        results['new_model_summary_seahorse'] = score(texts=texts, summaries=new_summaries, metrics=['seahorse'])[
            'seahorse']
    if args.test_fixed_trueteacher:
        results['fixed_model_summary_trueteacher'] = \
        score(texts=texts, summaries=fixed_summaries, metrics=['trueteacher'])[
            'trueteacher']
    if args.test_fixed_seahorse:
        results['fixed_model_summary_seahorse'] = score(texts=texts, summaries=fixed_summaries, metrics=['seahorse'])[
            'seahorse']
    if args.test_original_model_revised_seahorse:
        results['original_model_summary_revised_seahorse'] = \
        score(texts=texts, summaries=original_model_summaries_revised, metrics=['seahorse'])['seahorse']
    if args.test_original_model_revised_trueteacher:
        results['original_model_summary_revised_trueteacher'] = \
        score(texts=texts, summaries=original_model_summaries_revised, metrics=['trueteacher'])['trueteacher']
    return results


def main():
    args = parse()
    if args.wandb:
        wandb.init(project=args.project_name)
    else:
        os.environ["WANDB_DISABLED"] = "true"
    if args.create_new_dir:
        args.output_dir = os.path.join(args.output_dir, args.strategy)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            latest_name = 0
        else:
            latest_name = find_largest_numbered_dir(args.output_dir)
            latest_name += 1
        args.output_dir = os.path.join(args.output_dir, str(latest_name))
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f)
    if args.train:
        trainer, model = train(args)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint)
        tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
        train_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir, do_train=False, do_eval=True,
            per_device_train_batch_size=args.train_batch_size_per_device,
            per_device_eval_batch_size=args.eval_batch_size_per_device, eval_accumulation_steps=30,
            no_cuda=False, predict_with_generate=True, overwrite_output_dir=False)
        trainer = T5_Trainer(model=model, tokenizer=tokenizer, max_length_eval=args.encoder_max_length, args=train_args)
    if args.test:
        results_df = test(args, trainer)
        results_df.to_csv(os.path.join(args.output_dir, 'test_results.csv'))
    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()

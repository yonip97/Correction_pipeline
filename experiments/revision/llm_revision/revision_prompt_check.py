import gc
import json
import os
import sys

import torch

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')

import argparse
import time
import pandas as pd
from datetime import datetime
import os
from general.revision_pipeline import chose_revision_model
from tqdm import tqdm
from experiments.data.datasets_splits import split_xsum_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from general.t5_trainer import t5_summarize
import numpy as np
from general.fragments_metrics import Fragments
from Seahorse_metrics.metrics import Seahorse_metrics
from nltk.tokenize import word_tokenize
import evaluate
from experiments.scoring import score
import matplotlib.pyplot as plt
def add_examples_to_prompt(prompt, examples, connector):
    prompt += '\n' + connector + '\n'
    for example in examples:
        prompt += 'document: ' + '\n' + example['document'] + '\n'
        prompt += 'summary: ' + '\n' + example['summary'] + '\n'
        prompt += 'revised summary: ' + example['revised summary'] + '\n'
    return prompt


def parseargs_llms():
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_path')
    parser.add_argument('-revision_model_name', type=str, default='mock')
    parser.add_argument('-revision_prompt_path', type=str)
    parser.add_argument('-examples_path', type=str, default=None)
    parser.add_argument('-examples_num', type=int, default=0)
    parser.add_argument('-API_KEY_revision_model', type=str, default=None)
    parser.add_argument('-dir_path', type=str, default='experiments/revision/data/prompt_check')
    parser.add_argument('-azure', action='store_true')
    parser.add_argument('-output_path', type=str, default=None)
    args = parser.parse_args()
    return args


def check_all_prompts():
    prompts = {
        "prompt 1": """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will correct it to be factually consistent. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific  compared to the document, you should not change it. Output only the corrected summary and nothing more.""",
        "prompt 2": """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will correct it to be factually consistent. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific compared to the document, you should not change it. Output only the corrected summary and nothing more.""",
        "prompt 3": """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will correct it to be factually consistent. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific compared to the document, you should not change it. Output only the corrected summary and nothing more.""",
        "prompt 4": """The following summary is factually inconsistent w.r.t the document. Please make it factually consistent while making minimal changes to the summary. Output only the corrected summary and nothing more.""",
        "prompt 5": """The following summary is factually inconsistent w.r.t the document. Please make it factually consistent while making minimal changes to the summary. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Output only the corrected summary and nothing more.""",
        "prompt 6": """The following summary is factually inconsistent w.r.t the document. Please make it factually consistent while making minimal changes to the summary.  Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Output only the corrected summary and nothing more.""",
        "prompt 7": """The following summary is factually inconsistent w.r.t the document. Please follow the following to correct it. First, identify the parts in the summary which are factually inconsistent w.r.t the document. Then, find the correct information in the document that should replace the factually inconsistent parts in the summary. Finally, edit the summary by incorporating the correct information from the document, resulting in a precise and factually accurate summary. Output only the corrected summary and nothing more.""",
        "prompt 8": """The following summary is factually inconsistent w.r.t the document. Please follow the following to correct it. First, identify the parts in the summary which are factually inconsistent w.r.t the document. Then, find the correct information in the document that should replace the factually inconsistent parts in the summary. Finally, edit the summary by incorporating the correct information from the document, resulting in a precise and factually accurate summary. Focus on maintaining an abstractive approach and minimize direct copying from the source text.. Output only the corrected summary and nothing more.""",
        "prompt 9": """The following summary is factually inconsistent w.r.t the document. Please follow the following to correct it. First, identify the parts in the summary which are factually inconsistent w.r.t the document. Then, find the correct information in the document that should replace the factually inconsistent parts in the summary. Finally, edit the summary by incorporating the correct information from the document, resulting in a precise and factually accurate summary. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Output only the corrected summary and nothing more.""",
        "prompt 10": """Your assignment involves receiving a document and its summary from me. The summary currently exhibits factual discrepancies with respect to the document, indicating the presence of one or more facts that lack verification within the document. Your duty is to produce a revised version of the summary that maintains factual consistency. Strive to make only essential changes, ensuring the corrected summary closely mirrors the original while rectifying factual inaccuracies. Note that accurate facts should not be altered, even if rephrased or expressed more generally than in the document. Output only the corrected summary and nothing more.""",
        "prompt 11": """Your assignment involves receiving a document and its summary from me. The summary currently exhibits factual discrepancies with respect to the document, indicating the presence of one or more facts that lack verification within the document. Your duty is to produce a revised version of the summary that maintains factual consistency. Strive to make only essential changes, ensuring the corrected summary closely mirrors the original while rectifying factual inaccuracies. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Note that accurate facts should not be altered, even if rephrased or expressed more generally than in the document. Output only the corrected summary and nothing more.""",
        "prompt 12": """Your assignment involves receiving a document and its summary from me. The summary currently exhibits factual discrepancies with respect to the document, indicating the presence of one or more facts that lack verification within the document. Your duty is to produce a revised version of the summary that maintains factual consistency. Strive to make only essential changes, ensuring the corrected summary closely mirrors the original while rectifying factual inaccuracies. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Note that accurate facts should not be altered, even if rephrased or expressed more generally than in the document. Output only the corrected summary and nothing more.""",
        "prompt 13": """There are factual discrepancies between the document and the following summary. Your responsibility is to rectify these inconsistencies with minimal alterations to the summary.Output only the corrected summary and nothing more.""",
        "prompt 14": """There are factual discrepancies between the document and the following summary. Your responsibility is to rectify these inconsistencies with minimal alterations to the summary. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Output only the corrected summary and nothing more.""",
        "prompt 15": """There are factual discrepancies between the document and the following summary. Your responsibility is to rectify these inconsistencies with minimal alterations to the summary. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Output only the corrected summary and nothing more."""}
    args = parseargs_llms()
    df = pd.read_csv(args.input_path, index_col=0)
    df = df[df['pre_revision_seahorse_score'] < 0.5]
    texts = df['text'].tolist()
    summaries = df['generated_summary'].tolist()
    original_dir_path = args.dir_path
    for prompt_id, prompt in prompts.items():
        df_copy = df.copy(deep=True)
        df_copy['prompt'] = prompt
        args.revision_prompt = prompt
        output_path = os.path.join(original_dir_path, prompt_id + '.csv')
        revision_model = chose_revision_model(args)
        revised_summaries, errors = [], []
        for text, summary in tqdm(zip(texts, summaries)):
            revised_summary, error = revision_model.revise_single(text=text, summary=summary)
            revised_summaries.append(revised_summary)
            errors.append(error)
            # time.sleep(2)
        df_copy['revised_summary'] = revised_summaries
        df_copy['error'] = errors
        df_copy.to_csv(output_path)


def parseargs_create_summaries():
    args = argparse.ArgumentParser()
    args.add_argument('--model_checkpoint', type=str)
    args.add_argument('--output_path', type=str)
    args.add_argument('--num_of_samples', type=int, default=30)
    return args.parse_args()


def create_summaries_from_baseline():
    args = parseargs_create_summaries()
    device = "cuda:0"
    model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    dataset = split_xsum_dataset(split='factuality_test',
                                 path_to_documents_for_summarization_indices="experiments/data/datasets_splits/xsum_summarization_20000_revision_10000_seed_42.json",
                                 num_of_documents_for_summarization=20000,
                                 num_of_documents_for_revision=10000,
                                 seed=42)
    np.random.seed(42)
    selected = [int(x) for x in list(np.random.choice(len(dataset), args.num_of_samples, replace=False))]
    texts = [dataset[i]['text'] for i in selected]
    indices = selected
    original_summaries = [dataset[i]['summary'] for i in selected]
    generated_summaries = t5_summarize(texts=texts, model=model, tokenizer=tokenizer, prompt='summarize: ',
                                       device=device, batch_size=16, max_generation_length=128, beam_size=4,
                                       early_stopping=True, length_penalty=0.6,
                                       max_encoding_length=2048)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                         tokenizer_name='google/seahorse-xxl-q4',
                                         device='auto', batch_size=1, torch_dtype=torch.float16,
                                         max_length=2048, return_none=True)
    scores = factuality_metric.score(summaries=generated_summaries, texts=texts)
    pd.DataFrame.from_dict({'indices': indices, 'text': texts, 'original_summary': original_summaries,
                            'generated_summary': generated_summaries, 'pre_revision_seahorse_score': scores}).to_csv(
        args.output_path + '.csv')


def analysis():
    data_dir = "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/prompt_check"
    files = os.listdir(data_dir)
    files = [file for file in files if file.startswith('prompt')]
    factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                         tokenizer_name='google/seahorse-xxl-q4',
                                         device='auto', batch_size=1, torch_dtype=torch.float16,
                                         max_length=2048, return_none=True)
    rouge_metric = evaluate.load('rouge')
    for file in files:
        print("processing", file)
        file = os.path.join(data_dir, file)
        df = pd.read_csv(file, index_col=0)
        df = df[df['revised_summary'].notna()]
        print("length of df", len(df))
        df['post_revision_seahorse_score'] = \
            factuality_metric.score(summaries=df['revised_summary'].tolist(), texts=df['text'].tolist())
        pre_revision_fragments = Fragments()
        results = pre_revision_fragments.score(metrics=['density', 'coverage'],
                                               summaries=df['generated_summary'].tolist(), texts=df['text'].tolist())
        for key in results:
            df['pre_revision_' + key] = results[key]
        post_revision_fragments = Fragments()
        results = post_revision_fragments.score(metrics=['coverage', 'density'],
                                                summaries=df['revised_summary'].tolist(),
                                                texts=df['text'].tolist())
        for key in results:
            df['post_revision_' + key] = results[key]
        df['pre_revision_length'] = [len(word_tokenize(summary)) for summary in df['generated_summary'].tolist()]
        df['post_revision_length'] = [len(word_tokenize(summary)) for summary in df['revised_summary'].tolist()]
        df['base_to_original_rougeL'] = \
            rouge_metric.compute(predictions=df['generated_summary'].tolist(),
                                 references=df['original_summary'].tolist(),
                                 use_aggregator=False)['rougeL']
        df['revised_to_original_rougeL'] = \
            rouge_metric.compute(predictions=df['revised_summary'].tolist(), references=df['original_summary'].tolist(),
                                 use_aggregator=False)['rougeL']
        df['revised_to_base_rougeL'] = \
            rouge_metric.compute(predictions=df['revised_summary'].tolist(),
                                 references=df['generated_summary'].tolist(),
                                 use_aggregator=False)['rougeL']
        df.to_csv(file)


def read_results():
    data_dir = "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/prompt_check"
    files = os.listdir(data_dir)
    files = [file for file in files if file.startswith('prompt')]
    files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0].split(' ')[1]))
    indices = pd.read_csv(os.path.join(data_dir, files[0]), index_col=0)['indices'].tolist()
    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file), index_col=0)
        df.drop(
            columns=['text', 'original_summary', 'generated_summary', 'revised_summary', 'error', 'indices', 'prompt'],
            inplace=True)
        for col in df.columns:
            print(f"The {col} for prompt {file.split('/')[-1]} is: ", np.mean(df[col]))
        print("----------------------------------------------------------------------")
    # for i in indices:
    #     print("text number ", i)
    #     df = pd.read_csv(os.path.join(data_dir, files[0]), index_col=0)
    #     df = df[df['indices'] == i]
    #     print(df['text'].tolist()[0])
    #     print(df['generated_summary'].tolist()[0])
    #     print(df['pre_revision_density'].tolist()[0])
    #     print()
    #     for file in files:
    #         df = pd.read_csv(os.path.join(data_dir, file), index_col=0)
    #         df = df[df['indices'] == i]
    #         print()
    #         print(df['revised_summary'].tolist()[0])
    #         print(df['post_revision_density'].tolist()[0])
    #     print("----------------------------------------------------------------------")
    #


def check_chosen_prompt():
    args = parseargs_llms()
    with open(args.revision_prompt_path, 'r') as f:
        prompt = f.read()
    if args.examples_path is not None:
        with open(args.examples_path, 'r') as f:
            examples = json.load(f)
            chosen_examples = []
            for example_key in list(examples.keys())[:args.examples_num]:
                chosen_examples.append(examples[example_key])
            prompt = add_examples_to_prompt(prompt, chosen_examples,connector="here are a few examples:")

    df = pd.read_csv(args.input_path, index_col=0)
    df = df[df['pre_revision_seahorse_score'] < 0.5]
    print(len(df))
    texts = df['text'].tolist()
    summaries = df['generated_summary'].tolist()
    original_dir_path = args.dir_path
    # prompt = prompts[prompt_id]
    df_copy = df.copy(deep=True)
    df_copy['prompt'] = prompt
    args.revision_prompt = prompt
    output_path = args.output_path
    df_copy.to_csv(output_path)
    #output_path = os.path.join(original_dir_path, 'full_' + prompt_id + '.csv')
    revision_model = chose_revision_model(args)
    revised_summaries, errors = [], []
    for text, summary in tqdm(zip(texts, summaries)):
        revised_summary, error = revision_model.revise_single(text=text, summary=summary)
        revised_summaries.append(revised_summary)
        errors.append(error)
        # time.sleep(2)
    df_copy['revised_summary'] = revised_summaries
    df_copy['error'] = errors
    df_copy.to_csv(output_path)


def file_analysis():
    file = "experiments/revision/data/prompt_check/prompt_3_with_4_examples.csv"
    factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                         tokenizer_name='google/seahorse-xxl-q4',
                                         device='auto', batch_size=1, torch_dtype=torch.float16,
                                         max_length=2048, return_none=True)
    rouge_metric = evaluate.load('rouge')
    print("processing", file)
    df = pd.read_csv(file, index_col=0)
    df = df[df['revised_summary'].notna()]
    print("length of df", len(df))
    df['post_revision_seahorse_score'] = \
        factuality_metric.score(summaries=df['revised_summary'].tolist(), texts=df['text'].tolist())
    pre_revision_fragments = Fragments()
    results = pre_revision_fragments.score(metrics=['density', 'coverage'],
                                           summaries=df['generated_summary'].tolist(), texts=df['text'].tolist())
    for key in results:
        df['pre_revision_' + key] = results[key]
    post_revision_fragments = Fragments()
    results = post_revision_fragments.score(metrics=['coverage', 'density'],
                                            summaries=df['revised_summary'].tolist(),
                                            texts=df['text'].tolist())
    for key in results:
        df['post_revision_' + key] = results[key]
    df['pre_revision_length'] = [len(word_tokenize(summary)) for summary in df['generated_summary'].tolist()]
    df['post_revision_length'] = [len(word_tokenize(summary)) for summary in df['revised_summary'].tolist()]
    df['base_to_original_rougeL'] = \
        rouge_metric.compute(predictions=df['generated_summary'].tolist(),
                             references=df['original_summary'].tolist(),
                             use_aggregator=False)['rougeL']
    df['revised_to_original_rougeL'] = \
        rouge_metric.compute(predictions=df['revised_summary'].tolist(), references=df['original_summary'].tolist(),
                             use_aggregator=False)['rougeL']
    df['revised_to_base_rougeL'] = \
        rouge_metric.compute(predictions=df['revised_summary'].tolist(),
                             references=df['generated_summary'].tolist(),
                             use_aggregator=False)['rougeL']
    df.to_csv(file)


def look_at_new_prompt():
    new_prompt_df = pd.read_csv('experiments/revision/data/prompt_check/full_prompt_3.csv', index_col=0)
    new_prompt_df = new_prompt_df[new_prompt_df['revised_summary'].notna()].reset_index(drop=True)
    new_prompt_df['density_diff'] = new_prompt_df['post_revision_density'] - new_prompt_df['pre_revision_density']
    show = {'below': [], '0': [],
            'up to 0.5': [], 'up to 1': [],
            'above': []}
    for i, x in new_prompt_df.iterrows():
        if x['density_diff'] < 0:
            show['below'].append(i)
        elif x['density_diff'] == 0:
            show['0'].append(i)
        elif x['density_diff'] <= 0.5:
            show['up to 0.5'].append(i)
        elif x['density_diff'] <= 1:
            show['up to 1'].append(i)
        elif x['density_diff'] > 1:
            show['above'].append(i)
    for key in show:
        print(key)
        df = new_prompt_df.iloc[show[key]]
        print(df.describe().to_markdown())
        # for i in range(min(5,len(df))):
        #     for col in df.columns:
        #         print(col, df[col].tolist()[i])
        # for i, x in df.iterrows():
        #     for col in df.columns:
        #         print(col, x[col])
        # print("---------------------------------------------------------")

    # import matplotlib.pyplot as plt
    # plt.hist(new_prompt_df['density_diff'].tolist(), bins=100)
    # plt.show()


def show_prompt_results():
    original_df = pd.read_csv("experiments/revision/data/prompt_check/base_summaries/base_summaries_200.csv",
                              index_col=0)
    chosen_indices = original_df['indices'].tolist()
    original_df.drop(columns=['text', 'original_summary', 'generated_summary'], inplace=True)
    original_prompt_df = pd.read_csv('experiments/revision/data/prompt_check/full_prompt_1.csv', index_col=0)
    chosen_indices = set(chosen_indices).intersection(set(original_prompt_df['indices'].tolist()))
    new_prompt_df = pd.read_csv('experiments/revision/data/prompt_check/full_prompt_3.csv', index_col=0)
    chosen_indices = set(chosen_indices).intersection(set(new_prompt_df['indices'].tolist()))
    original_df = original_df[original_df['indices'].isin(chosen_indices)].sort_values(by=['indices'])
    original_prompt_df = original_prompt_df[original_prompt_df['indices'].isin(chosen_indices)].sort_values(
        by=['indices'])
    new_prompt_df = new_prompt_df[new_prompt_df['indices'].isin(chosen_indices)].sort_values(by=['indices'])
    need_to_drop = ['text', 'original_summary', 'generated_summary', 'revised_summary', 'error', 'prompt',
                    'pre_revision_density', 'pre_revision_coverage', 'pre_revision_length',
                    'pre_revision_seahorse_score',
                    'base_to_original_rougeL', 'indices']
    original_prompt_df.drop(columns=need_to_drop, inplace=True)
    new_prompt_df.drop(columns=need_to_drop, inplace=True)
    results = {'original': {}, 'old_prompt': {}, 'new_prompt': {}}
    for col in original_df.columns:
        if 'density' in col:
            results['original']['density'] = original_df[col].tolist()
        elif 'coverage' in col:
            results['original']['coverage'] = original_df[col].tolist()
        elif 'length' in col:
            results['original']['length'] = original_df[col].tolist()
        elif 'seahorse' in col:
            results['original']['seahorse'] = original_df[col].tolist()
        elif 'to_original_rougeL' in col:
            results['original']['to_original_xsum_summary_rougeL'] = original_df[col].tolist()
    results['original']['to_model_generated_summary_rougeL'] = [1 for i in range(len(results['original']['density']))]
    for col in original_prompt_df.columns:
        if 'density' in col:
            results['old_prompt']['density'] = original_prompt_df[col].tolist()
            results['new_prompt']['density'] = new_prompt_df[col].tolist()
        elif 'coverage' in col:
            results['old_prompt']['coverage'] = original_prompt_df[col].tolist()
            results['new_prompt']['coverage'] = new_prompt_df[col].tolist()
        elif 'length' in col:
            results['old_prompt']['length'] = original_prompt_df[col].tolist()
            results['new_prompt']['length'] = new_prompt_df[col].tolist()
        elif 'seahorse' in col:
            results['old_prompt']['seahorse'] = original_prompt_df[col].tolist()
            results['new_prompt']['seahorse'] = new_prompt_df[col].tolist()
        elif 'to_original_rougeL' in col:
            results['old_prompt']['to_original_xsum_summary_rougeL'] = original_prompt_df[col].tolist()
            results['new_prompt']['to_original_xsum_summary_rougeL'] = new_prompt_df[col].tolist()
        elif "to_base_rougeL" in col:
            results['old_prompt']['to_model_generated_summary_rougeL'] = original_prompt_df[col].tolist()
            results['new_prompt']['to_model_generated_summary_rougeL'] = new_prompt_df[col].tolist()
    x = pd.DataFrame(results)
    cols = ['seahorse', 'to_original_xsum_summary_rougeL', 'to_model_generated_summary_rougeL', 'density', 'coverage',
            'length']
    x = x.transpose()[cols].transpose()
    import matplotlib.pyplot as plt
    # plt.scatter(x['old_prompt']['density'],x['new_prompt']['density'])
    # plt.plot([0,20],[0,20],c = 'r')
    # plt.show()
    # plt.hist(x['old_prompt']['density'],bins=20,label='old_prompt',alpha = 0.7)
    # plt.hist(x['new_prompt']['density'],bins=20,label='new_prompt',alpha = 0.7)
    # plt.legend()
    # plt.show()
    # plt.hist(np.array(x['old_prompt']['density'])-np.array(x['new_prompt']['density']),bins=100)
    # plt.xlim(-5,5)
    # plt.show()
    # plt.hist(np.array(x['original']['density'])-np.array(x['new_prompt']['density']),bins=100,label="original_minus_new",alpha = 0.7)
    # plt.hist(np.array(x['original']['density'])-np.array(x['old_prompt']['density']),bins=100,label="original_minus_old",alpha = 0.7)
    # plt.xlim(-5,5)
    # plt.legend()
    # plt.show()
    # new_results = {'original': {}, 'old_prompt': {}, 'new_prompt': {}}
    # for i in np.linspace(0, 100, 21):
    #     percentile = np.percentile(x['original']['density'], i)
    #     below_percentile = [value for value in x['original']['density'] if value <= percentile]
    #     new_results['original'][i] = np.mean(below_percentile)
    #     print(f"The mean of the examples below {i} percentile of the original prompt is ", np.mean(below_percentile))
    #     percentile = np.percentile(x['new_prompt']['density'], i)
    #     below_percentile = [value for value in x['new_prompt']['density'] if value <= percentile]
    #     new_results['new_prompt'][i] = np.mean(below_percentile)
    #     print(f"The mean of the examples below {i} percentile of the new prompt is ", np.mean(below_percentile))
    #     percentile = np.percentile(x['old_prompt']['density'], i)
    #     below_percentile = [value for value in x['old_prompt']['density'] if value <= percentile]
    #     new_results['old_prompt'][i] = np.mean(below_percentile)
    #     print(f"The mean of the examples below {i} percentile of the old prompt is ", np.mean(below_percentile))
    # df = pd.DataFrame(new_results)
    # df.index.name = 'percentile'
    # print(df.round(3))
    return x


def get_examples():
    new_prompt_df = pd.read_csv('experiments/revision/data/prompt_check/full_prompt_3.csv', index_col=0)
    new_prompt_df['density_diff'] = new_prompt_df['post_revision_density'] - new_prompt_df['pre_revision_density']
    new_prompt_df = new_prompt_df[new_prompt_df['revised_summary'].notna()].reset_index(drop=True)
    new_prompt_df['seahorse_diff'] = new_prompt_df['post_revision_seahorse_score'] - new_prompt_df[
        'pre_revision_seahorse_score']
    new_prompt_df = new_prompt_df[new_prompt_df['seahorse_diff'] > 0.5]
    new_prompt_df = new_prompt_df[new_prompt_df['density_diff'] <= 0.5]
    new_prompt_df = new_prompt_df[new_prompt_df['revised_to_base_rougeL'] > 0.6]
    texts = new_prompt_df['text'].tolist()
    revised_summaries = new_prompt_df['revised_summary'].tolist()
    model_summaries = new_prompt_df['generated_summary'].tolist()
    pre_revision_density = new_prompt_df['pre_revision_density'].tolist()
    post_revision_density = new_prompt_df['post_revision_density'].tolist()
    pre_revision_seahorse_score = new_prompt_df['pre_revision_seahorse_score'].tolist()
    post_revision_seahorse_score = new_prompt_df['post_revision_seahorse_score'].tolist()
    revised_to_base_rougeL = new_prompt_df['revised_to_base_rougeL'].tolist()
    for i in range(len(texts)):
        print("text number ", i)
        print("text", texts[i])
        print("pre revision density", pre_revision_density[i])
        print("post revision density", post_revision_density[i])
        print("pre revision seahorse score", pre_revision_seahorse_score[i])
        print("post revision seahorse score", post_revision_seahorse_score[i])
        print("revised to base rougeL", revised_to_base_rougeL[i])
        print("model summary", model_summaries[i])
        print("revised summary", revised_summaries[i])
        print("---------------------------------------------------------")


def temp():
    df = show_prompt_results()
    density_df = pd.DataFrame({'old_prompt': df['old_prompt']['density'],
                               'new_prompt': df['new_prompt']['density'],
                               'original': df['original']['density']})
    # density_df = density_df[density_df['old_prompt'] < 5]
    density_df = density_df[density_df['new_prompt'] < 5]
    print(len(density_df))
    plt.scatter(density_df['original'], density_df['new_prompt'])
    plt.plot([0, 5], [0, 5], c='r')
    plt.xlabel("original density")
    plt.ylabel("new prompt density")
    plt.show()


def main():


# create_summaries_from_baseline()
#
# check_all_prompts()
# analysis()
# read_results()
# file_analysis()
    #check_chosen_prompt()
    file_analysis()
# show_prompt_results()
# temp()
# get_examples()

if __name__ == "__main__":
    main()

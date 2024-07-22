import gc
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

import json
import argparse
import os
import pandas as pd
import evaluate
import torch
from experiments.scoring import score
from experiments.data.datasets_splits import split_xsum_dataset
from general.t5_trainer import t5_summarize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from general.fragments_metrics import Fragments
from nltk.tokenize import word_tokenize


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-revision_data_file', type=str)
    parser.add_argument('-revision_summaries', type=int, default=10000)
    parser.add_argument('-summarization_documents', type=int, default=20000)
    parser.add_argument('-base_models_dir', type=str, default="experiments/baseline_model/checkpoints")
    parser.add_argument('-max_generation_length', type=int, default=128)
    parser.add_argument('-num_beams', type=int, default=4)
    parser.add_argument('-length_penalty', type=float, default=0.6)
    parser.add_argument('-max_encoding_length', type=int, default=2048)
    parser.add_argument('-base_model_checkpoint', type=str)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-revision_data_dir', type=str, default="experiments/revision/data")
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-dataset_split', type=str)
    args = parser.parse_args()
    return args


def get_data(args):
    if not os.path.exists(args.revision_data_dir):
        os.makedirs(args.revision_data_dir)
    if not os.path.exists(os.path.join(args.revision_data_dir, args.revision_data_file + '.csv')):
        with open(os.path.join(args.revision_data_dir, "args.json"), 'w') as file:
            json.dump(args.__dict__, file)
        dataset = split_xsum_dataset(split=args.dataset_split,
                                     path_to_documents_for_summarization_indices=f"experiments/data/datasets_splits/xsum_summarization_{args.summarization_documents}_revision_{args.revision_summaries}_seed_{args.seed}.json",
                                     num_of_documents_for_summarization=args.summarization_documents,
                                     num_of_documents_for_revision=args.revision_summaries,
                                     seed=args.seed)
        texts = [dataset[i]['text'] for i in range(len(dataset))]
        original_summaries = [dataset[i]['summary'] for i in range(len(dataset))]
        summarization_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(args.base_models_dir, args.base_model_checkpoint)).to(args.device)
        tokenizer = T5Tokenizer.from_pretrained(os.path.join(args.base_models_dir, args.base_model_checkpoint))
        model_summaries = t5_summarize(texts=texts, model=summarization_model, tokenizer=tokenizer,
                                       prompt="summarize: ",
                                       device=args.device, max_generation_length=args.max_generation_length,
                                       batch_size=args.batch_size,
                                       beam_size=args.num_beams,
                                       early_stopping=True,
                                       length_penalty=args.length_penalty, max_encoding_length=args.max_encoding_length)
        del summarization_model
        gc.collect()
        torch.cuda.empty_cache()
        if args.dataset_split == 'validation_model' or args.dataset_split == 'factuality_test':
            df = pd.DataFrame.from_dict({'text': texts, 'model_summary': model_summaries, 'indices': range(len(texts)),
                                         'original_summary': original_summaries})
        else:
            df = pd.DataFrame.from_dict({'text': texts, 'model_summary': model_summaries, 'indices': dataset.indices,
                                         'original_summary': original_summaries})
        df.to_csv(os.path.join(args.revision_data_dir, args.revision_data_file + '.csv'))
    else:
        df = pd.read_csv(os.path.join(args.revision_data_dir, args.revision_data_file + '.csv'), index_col=0)
    df = df[df['text'].notnull()]
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    original_summaries = df['original_summary'].tolist()
    factuality_scores = score(texts=texts, summaries=summaries, metrics=['trueteacher'])
    df['model_summary_trueteacher'] = factuality_scores['trueteacher']
    # df['model_summary_seahorse'] = factuality_scores['seahorse']
    rouge_metric = evaluate.load('rouge')
    rouge_scores = rouge_metric.compute(predictions=summaries, references=original_summaries)['rougeL']
    df['rougeL_base_to_original'] = rouge_scores
    fragments_metric = Fragments()
    fragments_scores = fragments_metric.score(metrics=['density', 'coverage'], texts=texts, summaries=summaries)
    df['model_summary_density'] = fragments_scores['density']
    df['model_summary_coverage'] = fragments_scores['coverage']
    df['model_summary_length'] = [len(word_tokenize(summary)) for summary in summaries]
    df['text_length'] = [len(word_tokenize(text)) for text in texts]
    df.to_csv(os.path.join(args.revision_data_dir, args.revision_data_file + '.csv'))


def main():
    args = parseargs()
    get_data(args)



if __name__ == "__main__":
    main()

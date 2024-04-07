import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from data.cost_etimation import Cost_estimator
from experiments.data.datasets_splits import split_xsum_dataset
import argparse
from revision_utils import add_examples_to_prompt


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarization_documents', type=int, default=20000)
    parser.add_argument('--revision_documents', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompt_path', type=str, default=None)
    parser.add_argument('--examples_path', type=str, default=None)
    parser.add_argument('--examples_num', type=int, default=1)
    parser.add_argument('--scored_outputs_path', type=str, default=None)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    thresholds = []
    costs = []
    input_tokens = []
    output_tokens = []
    num_of_documents = []
    with open(args.scored_outputs_path + '/args.json', 'r') as file:
        generation_args = json.load(file)
        checkpoint = generation_args['base_models_dir'] + '/' + generation_args['base_model_checkpoint']
        test_df = pd.read_csv(checkpoint + '/model_outputs.csv', index_col=0)
        density = test_df['density'].mean()
        coverage = test_df['coverage'].mean()
        trueteacher_score = test_df['trueteacher'].mean()
        rougeL = test_df['rougeL'].mean()
    for threshold in np.linspace(0, 1, 11):
        args.threshold = threshold
        estimator = Cost_estimator(model='gpt-4-turbo', input_price=0.01, output_price=0.03)
        if args.scored_outputs_path is not None:
            df = pd.read_csv(args.scored_outputs_path + '/base_model_outputs.csv', index_col=0)
            df = df.dropna()
            df = df[df['factuality_score'] < args.threshold]
            texts = df['text'].tolist()
            summaries = df['model_summary'].tolist()
        else:
            dataset = split_xsum_dataset(split='revision_documents',
                                         path_to_documents_for_summarization_indices=f"experiments/data/datasets_splits/xsum_summarization_{args.summarization_documents}_revision_{args.revision_documents}_seed_{args.seed}.json",
                                         num_of_documents_for_summarization=args.summarization_documents,
                                         num_of_documents_for_revision=args.revision_documents,
                                         seed=args.seed)
            texts = [dataset[i]['text'] for i in range(len(dataset))]
            summaries = [dataset[i]['summary'] for i in range(len(dataset))]
        if args.prompt_path is not None:
            with open(args.prompt_path, 'r') as file:
                prompt = file.read()
            if args.examples_path is not None:
                chosen_examples = []
                with open(args.examples_path, 'r') as file:
                    examples = json.load(file)
                for key in list(examples.keys())[:args.examples_num]:
                    chosen_examples += [examples[key]]
                prompt = add_examples_to_prompt(prompt, chosen_examples, connector="here are some examples:")
        else:
            raise ValueError("prompt path is None")

        cost, threshold_input_tokens, threshold_output_tokens = estimator.estimate_for_revision(prompt=prompt,
                                                                                                texts=texts,
                                                                                                summaries=summaries)
        possible_overhead_addition = 10 * len(texts) * 0.03 / 1000
        cost += possible_overhead_addition
        threshold_output_tokens += 10 * len(texts)
        thresholds.append(threshold)
        costs.append(cost)
        input_tokens.append(threshold_input_tokens)
        output_tokens.append(threshold_output_tokens)
        num_of_documents.append(len(texts))
    df = pd.DataFrame.from_dict(
        {'threshold': thresholds, 'num_of_documents': num_of_documents, 'cost': costs, 'input_tokens': input_tokens,
         'output_tokens': output_tokens})
    print()
    print()
    print("30000 documents outside of training set")
    print("Stats on test set: Density = ", np.round(density,2), "Coverage = ", np.round(coverage*100,2), "Trueteacher = ", np.round(trueteacher_score*100,2),
          "RougeL = ", np.round(rougeL*100,2))
    df = df.round(2)
    df['input_tokens'] = df['input_tokens'].astype(str)
    df['output_tokens'] = df['output_tokens'].astype(str)
    markdown = df.to_markdown()
    print(markdown)


if __name__ == '__main__':
    main()

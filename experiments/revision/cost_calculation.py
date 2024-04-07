import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

import pandas as pd
from data.cost_etimation import Cost_estimator
import argparse
import tiktoken


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default=None)
    args = parser.parse_args()
    return args


def transform_to_input(prompt, texts, summaries):
    inputs = []
    for i in range(len(texts)):
        model_input = prompt
        model_input += 'Document: ' + '\n' + texts[i] + '\n'
        model_input += "Summary: " + '\n' + summaries[i] + '\n'
        inputs.append(model_input)
    return inputs


def main():
    args = parser_args()
    with open(args.prompt_path, 'r') as file:
        prompt = file.read()
    df = pd.read_csv(
        'experiments/revision/data/base_model_50000_documents/prompt_3/base_model_outputs_below_0.5_1000_revised_scored.csv',
        index_col=0)
    df = df.iloc[30:129]
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    model_inputs = transform_to_input(prompt, texts, summaries)
    revised_summaries_full_text = df['revised_summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    encoder = tiktoken.encoding_for_model('gpt-4')
    y = [len(encoder.encode(x)) for x in revised_summaries_full_text]
    print(np.mean(y))
    cost_estimator = Cost_estimator(model='gpt-4-turbo', input_price=0.01, output_price=0.03)
    cost, input, output = cost_estimator.calculate_cost(input_texts=model_inputs, output_texts=revised_summaries_full_text)
    print(cost )
    cost,input,output = cost_estimator.calculate_cost(input_texts=model_inputs, output_texts=revised_summaries)
    print(cost)


if __name__ == "__main__":
    main()

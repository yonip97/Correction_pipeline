import json

import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')


def add_length(df):
    from nltk.tokenize import word_tokenize
    df['length'] = [len(word_tokenize(summary)) for summary in df['summary']]
    return df


def main():
    df = pd.read_csv('cliff/outputs/pegasus_syslowcon_results.csv', index_col=0)
    df = add_length(df)
    print(df.columns)
    for col in df.columns:
        if 'text' in col or 'summary' in col:
            continue
        temp_df = df[df[col].notnull()]
        print(len(temp_df))
        print(f"{col} mean: ", np.mean(temp_df[col].tolist()))
        print(f"{col} median: ", np.median(temp_df[col].tolist()))
    return df

def compute_extractiveness_across_outputs():
    from general.fragments_metrics import Fragments
    path = "/data/home/yehonatan-pe/Correction_pipeline/cliff/cliff_summ/data"
    with open("/data/home/yehonatan-pe/Correction_pipeline/cliff/cliff_summ/data/xsum_raw/test.source") as file:
        texts = [line.strip() for line in file]
    for model in ['bart', 'pegasus']:
        output_path = path + f"/{model}/extractiveness.json"
        with open(output_path, 'w') as file:
            json.dump({}, file)
        new_path = path + f"/{model}"
        all_files = os.listdir(new_path)

        # Filter only the text files
        text_files = [file for file in all_files if file.endswith(".txt")]
        for file in text_files:
            with open(output_path, 'r') as f:
                extractiveness = json.load(f)
            extractiveness[file] = {}
            with open(os.path.join(new_path, file), 'r') as f:
                summaries = [line.strip() for line in f]
            fragments = Fragments()
            results = fragments.score(metrics=['density', 'compression', 'coverage'], texts=texts, summaries=summaries)
            for key in results.keys():
                if key not in extractiveness.keys():
                    extractiveness[file][key] = []
                extractiveness[file][key] += results[key]
            with open(output_path, 'w') as f:
                json.dump(extractiveness, f)


def check_extractiveness_across_outputs():
    from general.fragments_metrics import Fragments
    df = None
    import evaluate
    rouge_metric = evaluate.load('rouge')
    path = "/data/home/yehonatan-pe/Correction_pipeline/cliff/cliff_summ/data"
    with open("/data/home/yehonatan-pe/Correction_pipeline/cliff/cliff_summ/data/xsum_raw/test.source") as file:
        texts = [line.strip() for line in file.readlines()]
    with open("/data/home/yehonatan-pe/Correction_pipeline/cliff/cliff_summ/data/xsum_raw/test.target") as file:
        original_summaries = [line.strip() for line in file.readlines()]
    for model in ['bart', 'pegasus']:
        output_path = path + f"/{model}/extractiveness.json"
        new_path = path + f"/{model}"
        all_files = os.listdir(new_path)
        print(model)
        # Filter only the text files
        text_files = [file for file in all_files if file.endswith(".txt")]
        for file in text_files:
            print(file)
            with open(output_path, 'r') as f:
                extractiveness = json.load(f)
            with open(os.path.join(new_path, file), 'r') as f:
                summaries = f.readlines()
            res = rouge_metric.compute(predictions=summaries, references=original_summaries, use_aggregator=True)
            print(res)
            for key in extractiveness[file].keys():
                print(key, np.mean(extractiveness[file][key]))
            if model=='bart' and 'sys' in file:
                df = pd.DataFrame.from_dict({'text': texts, 'summary': summaries,
                                             'density': extractiveness[file]['density'],
                                             'compression': extractiveness[file]['compression'],
                                             'coverage': extractiveness[file]['coverage']})
            print("------------------------------------------------------------")
            print()
    return df

def check():
    from datasets import load_dataset
    dataset = load_dataset('xsum', split='test')
    texts = [dataset[i]['document'] for i in range(len(dataset))]
    summaries = [dataset[i]['summary'] for i in range(len(dataset))]
    from general.fragments_metrics import Fragments
    fragments = Fragments()
    fragments_scores = fragments.score(texts=texts, summaries=summaries, metrics=['density', 'compression', 'coverage'])
    for key in fragments_scores.keys():
        print(key, np.mean(fragments_scores[key]))



if __name__ == '__main__':
    df = main()
    #df2 = check_extractiveness_across_outputs()
    c =1
    #compute_extractiveness_across_outputs()
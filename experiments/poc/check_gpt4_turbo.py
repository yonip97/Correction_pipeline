import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from tqdm import tqdm
import pandas as pd
import time
from general.revision_pipeline import chose_revision_model
import argparse
import datetime
from data.factuality_datasets import chose_dataset

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset')
    parser.add_argument('-output_path')
    parser.add_argument('-revision_model_name', type=str, default='mock')
    parser.add_argument('-revision_prompt', type=str,default="""I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific  compared to the document, you should not change it. Output only the corrected summary and nothing more.""")

    parser.add_argument('-API_KEY_revision_model', type=str, default=None)
    parser.add_argument('-contingency_file_dir', type=str, default='contingency_tables')

    args = parser.parse_args()
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f'run_{formatted_datetime}'
    args.dir_path = args.contingency_file_dir + '/' + run_name
    os.makedirs(args.dir_path)
    return args


def main():
    args = parseargs()
    dataset = chose_dataset(args.dataset)
    revision_model = chose_revision_model(args)
    texts = [dataset[i]['text'] for i in range(len(dataset))]
    summaries = [dataset[i]['summary'] for i in range(len(dataset))]
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    revised_summaries,errors = [],[]
    for text, summary in tqdm(zip(texts, summaries)):
        revised_summary, error = revision_model.revise_single(text=text, summary=summary)
        revised_summaries.append(revised_summary)
        errors.append(error)
        time.sleep(2)

    pd.DataFrame(data={'text': texts, 'summary': summaries, 'revised_summary': revised_summaries, 'label': labels,
                           'error': errors}).to_csv(args.output_path + '.csv', index=False)
    return revised_summaries, errors


main()

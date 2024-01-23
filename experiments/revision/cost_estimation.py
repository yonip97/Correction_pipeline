import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from data.cost_etimation import Cost_estimator
from experiments.data.datasets_splits import split_xsum_dataset
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summarization_documents', type=int, default=20000)
    parser.add_argument('--revision_summaries', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main():
    args = parser_args()
    estimator = Cost_estimator(model='gpt-4-turbo', input_price=0.01, output_price=0.03)
    dataset = split_xsum_dataset(split='revision_documents',
                                 path_to_documents_for_summarization_indices=f"experiments/data/datasets_splits/xsum_summarization_{args.summarization_documents}_revision_{args.revision_summaries}_seed_{args.seed}.json",
                                 num_of_documents_for_summarization=args.summarization_documents,
                                 num_of_documents_for_revision=args.revision_summaries,
                                 seed=args.seed)
    prompt = """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific  compared to the document, you should not change it. Output only the corrected summary and nothing more."""
    texts = [dataset[i]['text'] for i in range(len(dataset))]
    summaries = [dataset[i]['summary'] for i in range(len(dataset))]
    estimation = estimator.estimate_for_revision(prompt=prompt, texts=texts, summaries=summaries)
    print(estimation)


if __name__ == '__main__':
    main()

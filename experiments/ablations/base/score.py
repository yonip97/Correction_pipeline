import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

from experiments.scoring import score
from experiments.data.datasets_splits import split_xsum_dataset


def score_xsum_test():
    dataset_xsum = split_xsum_dataset(split='factuality_test',
                                      path_to_documents_for_summarization_indices=
                                      "experiments/data/datasets_splits/xsum_summarization_20000_revision_10000_seed_42.json",
                                      num_of_documents_for_summarization=20000, num_of_documents_for_revision=10000,
                                      seed=42)
    texts = [dataset_xsum[i]['text'] for i in range(len(dataset_xsum))]
    summaries = [dataset_xsum[i]['summary'] for i in range(len(dataset_xsum))]
    factuality_results = score(texts=texts, summaries=summaries, metrics=['trueteacher'])
    from general.fragments_metrics import Fragments
    fragments = Fragments()
    fragments_scores = fragments.score(texts=texts, summaries=summaries, metrics=['density', 'compression', 'coverage'])
    import pandas as pd
    from nltk.tokenize import word_tokenize
    lengths = [len(word_tokenize(summary)) for summary in summaries]
    df = pd.DataFrame.from_dict({'text': texts, 'summary': summaries, 'true_teacher': factuality_results['trueteacher'], 'density': fragments_scores['density'],
                                 'coverage': fragments_scores['coverage'], 'length': lengths})
    df.to_csv('experiments/ablations/base/xsum_test_scores.csv', index=False)

if __name__ == "__main__":
    score_xsum_test()

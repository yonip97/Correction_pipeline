import os
import json
import sys

import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')


def worst_density(density, num=20):
    worst_indexes = np.argsort(density)[-num:]
    return worst_indexes


def random_summaries(indexes, num=80):
    np.random.seed(42)
    return np.random.choice(indexes, num, replace=False)


def main():
    from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
    xsum_test_set = split_xsum_dataset(split='factuality_test',
                                       path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                       num_of_documents_for_summarization=20000,
                                       seed=42)
    from general.fragments_metrics import Fragments
    texts = [xsum_test_set[i]['text'] for i in range(len(xsum_test_set))]
    original_summaries = [xsum_test_set[i]['summary'] for i in range(len(xsum_test_set))]
    data_dir = 'experiments/xsum_4_sets_experiment'
    file = "revision_by_flan_large_nli_scored.json"
    with open(os.path.join(data_dir, file), 'r') as file:
        results = json.load(file)
    baseline_summaries = results['baseline']['summaries']
    xtra_results = Fragments().score(metrics=['density', 'coverage'], texts=texts,summaries=baseline_summaries)
    old_density = xtra_results['density']
    results = results['diff_and_rouge_threshold']
    density = results['density']
    summaries = results['summaries']
    worst_indexes = worst_density(density, num=20)
    indexes = np.arange(len(summaries))
    indexes = np.delete(indexes, worst_indexes)
    random_indexes = random_summaries(indexes, num=80)
    for index in worst_indexes:
        print('Text:')
        print(texts[index])
        print()
        print('Original Summary:')
        print(original_summaries[index])
        print()
        print('Baseline Summary:')
        print(baseline_summaries[index])
        print('New Summary:')
        print(summaries[index])
        print()
        print(density[index])
        print(results['to_original_rougeL'][index])
        print(results['to_baseline_rougeL'][index])
        print('________________________________________________________________________________________________')
    print("Random Summaries:")
    for index in random_indexes:
        print('Text:')
        print(texts[index])
        print()
        print('Original Summary:')
        print(original_summaries[index])
        print()
        print('Baseline Summary:')
        print(baseline_summaries[index])
        print('New Summary:')
        print(summaries[index])
        print()
        print("new density: " ,density[index])
        print("old density: ", old_density[index])
        print("rouge L to original summary: ", results['to_original_rougeL'][index])
        print("rouge L to baseline summary:" ,results['to_baseline_rougeL'][index])
        print('________________________________________________________________________________________________')

main()

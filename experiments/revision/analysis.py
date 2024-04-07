import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from experiments.data.datasets_splits import split_xsum_dataset
import numpy as np
import matplotlib.pyplot as plt


def remove_top_and_bottom(lst, x_percent):
    # Calculate the number of elements to remove from the top and bottom
    num_to_remove_top = int(len(lst) * (x_percent / 100.0 / 2))
    num_to_remove_bottom = int(len(lst) * (x_percent / 100.0 / 2))

    # Sort the list
    sorted_lst = sorted(lst)

    # Remove elements from the top and bottom
    trimmed_lst = sorted_lst[num_to_remove_top:-num_to_remove_bottom]

    return trimmed_lst


def print_every_xth_percentile(lst, x):
    # Sort the list
    sorted_lst = sorted(lst)

    # Calculate and print the element at every 10th percentile
    for i in range(0, 101, x):
        percentile_value = np.percentile(sorted_lst, i)
        print(f"{i}th percentile: {percentile_value}")


def create_pre_post_dict(df):
    cols = ['pre_revision_seahorse', 'pre_revision_trueteacher', 'pre_revision_density', 'pre_revision_coverage',
            'pre_revision_length', 'post_revision_seahorse', 'post_revision_trueteacher', 'post_revision_density',
            'post_revision_coverage', 'post_revision_length', 'rougeL_base_to_original', 'rougeL_revised_to_original',
            'rougeL_revised_to_base']

    results = {}
    for col in cols:
        if 'density' not in col and 'length' not in col:
            df[col] *= 100
        if 'pre_revision' in col:
            new_col = col.replace('pre_revision_', '')
            if new_col not in results:
                results[new_col] = {}
            results[new_col]['pre'] = df[col].tolist()
        elif 'post_revision' in col:
            new_col = col.replace('post_revision_', '')
            if new_col not in results:
                results[new_col] = {}
            results[new_col]['post'] = df[col].tolist()
        elif 'rougeL_revised_to_original' in col:
            if 'rougeL' not in results:
                results['rougeL'] = {}
            results['rougeL']['post'] = df[col].tolist()
        elif 'rougeL_base_to_original' in col:
            if 'rougeL' not in results:
                results['rougeL'] = {}
            results['rougeL']['pre'] = df[col].tolist()
    return results


def plot_differences(results, trim=0):
    for key in results:
        differences = [results[key]['post'][i] - results[key]['pre'][i] for i in range(len(results[key]['pre']))]
        differences_trimmed = remove_top_and_bottom(differences, trim)
        plt.hist(differences_trimmed, bins=20)
        plt.title(f"Difference in {key} between post and pre revision")
        plt.show()


def create_table(results):
    for key in results:
        results[key]['pre'] = np.nanmean(results[key]['pre'])
        results[key]['post'] = np.nanmean(results[key]['post'])
    print(pd.DataFrame.from_dict(results).round(2).T.to_markdown())


def sample(df, num_of_samples):
    subdf = df.sample(n=num_of_samples,random_state=42)
    for i in range(num_of_samples):
        print(subdf['text'].iloc[i])
        print()
        print(subdf['model_summary'].iloc[i])
        print(subdf['revised_summary'].iloc[i])
        print(f"Density score before revision: {subdf['pre_revision_density'].iloc[i]}")
        print(f"Density score after revision: {subdf['post_revision_density'].iloc[i]}")
        print(f"Pre revision factuality score: {subdf['pre_revision_seahorse'].iloc[i]}")
        print(f"Post revision factuality score: {subdf['post_revision_seahorse'].iloc[i]}")
        print(f"RougeL score between original and base model summary: {subdf['rougeL_base_to_original'].iloc[i]}")
        print(f"RougeL score between original and revised summary: {subdf['rougeL_revised_to_original'].iloc[i]}")
        print(f"RougeL score between base model and revised summary: {subdf['rougeL_revised_to_base'].iloc[i]}")
        print("--------------------------------------------------------------------------------------------------")
        print("\n")



def main():
    df = pd.read_csv(
        "experiments/revision/data/base_model_50000_documents/base_model_outputs_below_0.5_1000_revised_scored.csv",
        index_col=0)
    df.rename(columns={'pre_revision_factuality_scores': 'pre_revision_seahorse',
                       'post_revision_factuality_score': 'post_revision_seahorse'}, inplace=True)
    #print(sum([x.endswith('.') for x in df['revised_summary'].tolist()]))
    sample(df, 30)
    # results = create_pre_post_dict(df)
    # create_table(results)
    # print(f"For the revision of {len(df)} examples, the following results were obtained:")
    # print(
    #     f"The rougeL between the original model summary to the revised summary has a mean of {df['rougeL_revised_to_base'].mean().round(2)} and a median of {np.round(df['rougeL_revised_to_base'].median(), 2)}")
    # properly_revised = df[df['post_revision_seahorse'] > 50]
    # factuality_decreased = df[df['post_revision_seahorse'] < df['pre_revision_seahorse']]
    # print(
    #     f"Out of {len(df)} examples, {len(properly_revised)} were revised to be factuality consistent and {len(factuality_decreased)} were revised to have a lower factuality score than the original model summary")
    #

if __name__ == "__main__":
    main()

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))


def get_better_summaries_data():
    df = pd.read_csv(
        "experiments/ablations/better_summaries/data/prompts/4/results/6/results.csv",
        index_col=0)
    return df


def get_refinement_data():
    test_df = pd.read_csv(
        "experiments/revision/data/base_model_50000_documents/test/base_model_summaries.csv",
        index_col=0)
    df = pd.read_csv(
        "experiments/refinement/data/test/filter_revised_filter_unrevised_upsample_revised/2/test_results.csv",
        index_col=0)
    df.rename(columns={"rougeL_to_original": "rougeL_post_revision_to_original",
                       'rougeL_to_base': 'rougeL_post_revision_to_base',
                       "model_summary_density": 'post_revision_summary_density',
                       'model_summary_coverage': 'post_revision_summary_coverage',
                       'model_summary_trueteacher': 'post_revision_trueteacher',
                       'model_summary_length': 'post_revision_summary_length',
                       'summary': 'post_revision_summary'}, inplace=True)

    for col in df.columns:
        test_df[col] = df[col]
    print(test_df.columns)
    return test_df
    # df = pd.read_csv(
    #     "experiments/refinement/data/test/filter_revised_filter_unrevised_upsample_revised/2/test_results.csv",index_col=0)
    # print(df.columns)


def check_distribution_of_factual_consistency_compared_to_rouge(df, rouge_col, trueteacher_col):
    rouge_bins = np.linspace(0, 1, 11)
    mean_trueteacher = []
    mean_factual = []
    for i in range(10):
        temp_df = df[(df[rouge_col] >= rouge_bins[i]) & (df[rouge_col] < rouge_bins[i + 1])]
        print(len(temp_df))
        mean_trueteacher.append(temp_df[trueteacher_col].mean())
        mean_factual.append((temp_df[trueteacher_col] > 0.5).mean())
    plt.plot(rouge_bins[:-1], mean_trueteacher)
    plt.show()
    plt.plot(rouge_bins[:-1], mean_factual)
    plt.show()


def check_distribution_of_factual_consistency_compared_to_density(df, density_col, trueteacher_col):
    density_bins = np.linspace(0.5, 4, 36)
    mean_trueteacher = []
    mean_factual = []
    for i in range(35):
        temp_df = df[(df[density_col] >= density_bins[i]) & (df[density_col] < density_bins[i + 1])]
        print(len(temp_df))
        mean_trueteacher.append(temp_df[trueteacher_col].mean())
        mean_factual.append((temp_df[trueteacher_col] > 0.5).mean())
    plt.plot(density_bins[:-1], mean_trueteacher)
    plt.show()
    plt.plot(density_bins[:-1], mean_factual)
    plt.show()


def check_distribution_of_factual_consistency_compared_to_coverage(df, coverage_col, trueteacher_col):
    consistency_bins = np.linspace(0, 1, 21)
    mean_trueteacher = []
    mean_factual = []
    for i in range(20):
        temp_df = df[(df[coverage_col] >= consistency_bins[i]) & (df[coverage_col] < consistency_bins[i + 1])]
        print(len(temp_df))
        mean_trueteacher.append(temp_df[trueteacher_col].mean())
        mean_factual.append((temp_df[trueteacher_col] > 0.5).mean())
    plt.plot(consistency_bins[:-1], mean_trueteacher)
    plt.show()


def manual_compare_not_factual(df_refinement, df_better_summaries, num_samples=50):
    not_factual_df_refinement = df_refinement[df_refinement['model_summary_trueteacher'] < 0.5]
    not_factual_df_better_summaries = df_better_summaries[df_better_summaries['model_summary_trueteacher'] < 0.5]
    for i in range(num_samples):
        text1 = not_factual_df_refinement['text'].iloc[i]
        text2 = not_factual_df_better_summaries['text'].iloc[i]
        if text1 == text2:
            print(f"Indices: {not_factual_df_refinement['indices'].iloc[i]}")
            print()
            print(text1)
            print()
            print("Model summary:")
            print(not_factual_df_refinement['model_summary'].iloc[i])
            print()
            print()
            print("Better summaries:")
            print(not_factual_df_better_summaries['model_summary_llm'].iloc[i])
            print("Rouge: ", not_factual_df_better_summaries['rougeL_llm_to_base'].iloc[i])
            print()
            print("Refinement:")
            print(not_factual_df_refinement['post_revision_summary'].iloc[i])
            print("Rouge: ", not_factual_df_refinement['rougeL_post_revision_to_base'].iloc[i])
            print()
            print('----------------------------------------------------------------------------------------------------')


def manual_compare_factual(df_refinement, df_better_summaries, num_samples=50):
    factual_df_refinement = df_refinement[df_refinement['model_summary_trueteacher'] >= 0.5]
    factual_df_better_summaries = df_better_summaries[df_better_summaries['model_summary_trueteacher'] >=0.5]
    for i in range(num_samples):
        text1 = factual_df_refinement['text'].iloc[i]
        text2 = factual_df_better_summaries['text'].iloc[i]
        if text1 == text2:
            print(f"Indices: {factual_df_refinement['indices'].iloc[i]}")
            print(text1)
            print()
            print("Model summary:")
            print(factual_df_refinement['model_summary'].iloc[i])
            print()
            print()
            print("Better summaries:")
            print(factual_df_better_summaries['model_summary_llm'].iloc[i])
            print()
            print("Refinement:")
            print(factual_df_refinement['post_revision_summary'].iloc[i])
            print('----------------------------------------------------------------------------------------------------')

def main():
    better_summaries_df = get_better_summaries_data()
    refinement_df = get_refinement_data()
    print(len(refinement_df))
    print(len(better_summaries_df))
    # check_distribution_of_factual_consistency_compared_to_rouge(refinement_df, 'rougeL_post_revision_to_base',
    #                                                             'post_revision_trueteacher')
    # check_distribution_of_factual_consistency_compared_to_rouge(better_summaries_df, 'rougeL_llm_to_base',
    #                                                             'model_summary_llm_trueteacher')
    #
    # check_distribution_of_factual_consistency_compared_to_density(refinement_df, 'post_revision_summary_density',
    #                                                               'post_revision_trueteacher')
    # check_distribution_of_factual_consistency_compared_to_density(better_summaries_df, 'model_summary_llm_density',
    #                                                               'model_summary_llm_trueteacher')
    # check_distribution_of_factual_consistency_compared_to_coverage(refinement_df, 'post_revision_summary_coverage',
    #                                                                'post_revision_trueteacher')
    # check_distribution_of_factual_consistency_compared_to_coverage(better_summaries_df, 'model_summary_llm_coverage',
    #                                                                  'model_summary_llm_trueteacher')
    manual_compare_not_factual(df_refinement=refinement_df, df_better_summaries=better_summaries_df, num_samples=50)
    manual_compare_factual(df_refinement=refinement_df, df_better_summaries=better_summaries_df, num_samples=50)


if __name__ == '__main__':
    main()

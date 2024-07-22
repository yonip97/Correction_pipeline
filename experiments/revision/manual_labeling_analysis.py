import numpy as np
import pandas as pd
import os
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


def display_csv_row_by_row(df):
    # Load the CSV file
    relevant_columns = ['text', 'model_summary', 'revised_summary', 'indices']
    df = df[relevant_columns]
    for i in range(len(df)):
        # Clear the terminal screen
        # os.system('clear')  # or 'cls' for Windows

        # Get the current row data
        row_data = df.iloc[i]

        # Display the row data
        for column, value in row_data.items():
            print(f"\n{column}: \n{value}")

        # Wait for the user to press Enter to display the next row
        input("Press Enter to see the next row...")

    print("No more rows to display.")


def get_stats(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            print(df[col].describe())
    print("---------------------------------------------")


def word_level_edit_distance(sentence1, sentence2):
    words1 = word_tokenize(sentence1)
    words2 = word_tokenize(sentence2)

    m = len(words1)
    n = len(words2)

    # Create a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Compute word-level edit distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # Deletion
                                   dp[i][j - 1],  # Insertion
                                   dp[i - 1][j - 1])  # Substitution

    return dp[m][n]


def main():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/base_model_50000_documents/cot_prompts/original/base_model_outputs_below_0.5_text_length_above_65_10000_samples_revised_scored.csv",
        index_col=0)
    df['diff_seahorse'] = df['revised_summary_seahorse'] - df['model_summary_seahorse']
    plt.scatter(df['model_summary_seahorse'], df['diff_seahorse'])
    plt.show()
    df['edit_distance'] = df.apply(lambda x: word_level_edit_distance(x['model_summary'], x['revised_summary']), axis=1)
    #df['edit_distance'] = df.apply(lambda x: x['edit_distance'] / len(word_tokenize(x['revised_summary'])), axis=1)
    df['edit_distance'].hist(bins=60,range = (0,60))
    plt.show()
    from scipy.stats import pearsonr
    print(pearsonr(df['edit_distance'], df['rougeL_revised_to_base']))
    plt.scatter(df['edit_distance'], df['rougeL_revised_to_base'])
    plt.show()

    df = df.iloc[100:202]
    manual_df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/100_summaries_labeling.csv")
    manual_df = manual_df.merge(df, left_on='Text index', right_on='indices')

    get_stats(manual_df[manual_df['Factuality'] == 1])
    get_stats(manual_df[manual_df['Factuality'] == 0])
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(manual_df['Factuality'], manual_df['revised_summary_trueteacher'] >= 0.5))
    print(confusion_matrix(manual_df['Factuality'], manual_df['revised_summary_seahorse'] >= 0.5))
    print(manual_df[manual_df['Extractivness'] == 1]['revised_summary_density'].mean())
    print(manual_df[manual_df['Extractivness'] == 1]['model_summary_density'].mean())
    print(manual_df[manual_df['Extractivness'] == 0]['revised_summary_density'].mean())
    print(manual_df[manual_df['Extractivness'] == 0]['model_summary_density'].mean())
    print("---------------------------------------------")
    print(manual_df[manual_df['Extractivness'] == 1]['revised_summary_coverage'].mean())
    print(manual_df[manual_df['Extractivness'] == 1]['model_summary_coverage'].mean())
    print(manual_df[manual_df['Extractivness'] == 0]['revised_summary_coverage'].mean())
    print(manual_df[manual_df['Extractivness'] == 0]['model_summary_coverage'].mean())
    print("---------------------------------------------")
    print(manual_df[manual_df['Similarity'] == 1]['rougeL_revised_to_base'].mean())
    print(manual_df[manual_df['Similarity'] == 0]['rougeL_revised_to_base'].mean())
    print("---------------------------------------------")
    print(manual_df[manual_df['Revision successful'] == 1]['model_summary_trueteacher'].mean())
    print(manual_df[manual_df['Revision successful'] == 0]['model_summary_trueteacher'].mean())
    print("-----------------------------------------------")
    filtered_df = manual_df[(manual_df['revised_summary_seahorse'] - manual_df['model_summary_seahorse'] > 0.2) &
                            (manual_df['revised_summary_density'] - manual_df['model_summary_density'] <= 1.5) & (
                                    manual_df['rougeL_revised_to_base'] > 0.5)]
    print(len(filtered_df))
    best = manual_df[
        (manual_df['Factuality'] == 1) & (manual_df['Similarity'] == 1) & (manual_df['Additional data'] == 0) & (
                manual_df['Revision successful'] == 1)]
    print(sum(best['Text index'].isin(filtered_df['Text index'])), len(best))
    passable = manual_df[manual_df['Revision successful'] == 1]
    print(sum(passable['Text index'].isin(filtered_df['Text index'])), len(passable))
    best_mean_geo = [0, 0, 0, None]
    best_mean_arti = [0, 0, 0, None]
    ratio_data_to_noise = sum(passable['Text index'].isin(manual_df['Text index'])) / len(manual_df)
    ratio_data_to_available_data = sum(passable['Text index'].isin(manual_df['Text index'])) / len(passable)
    ratio_data_to_perfect = sum(best['Text index'].isin(manual_df['Text index'])) / len(best)
    print(ratio_data_to_noise, ratio_data_to_available_data, ratio_data_to_perfect)
    for threshold_factuality in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for threshold_density in [0, 0.5, 1, 1.5, 2, 3]:
            for threshold_rouge in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                new_filtered_df = manual_df[(manual_df['revised_summary_seahorse'] - manual_df[
                    'model_summary_seahorse'] > threshold_factuality) &
                                            (manual_df['revised_summary_density'] - manual_df[
                                                'model_summary_density'] <= threshold_density) & (
                                                    manual_df['rougeL_revised_to_base'] > threshold_rouge)]
                ratio_data_to_noise = sum(passable['Text index'].isin(new_filtered_df['Text index'])) / len(
                    new_filtered_df)
                ratio_data_to_available_data = sum(passable['Text index'].isin(new_filtered_df['Text index'])) / len(
                    passable)
                ratio_data_to_perfect = sum(best['Text index'].isin(new_filtered_df['Text index'])) / len(best)
                if (ratio_data_to_noise + ratio_data_to_available_data) / 2 > (
                        best_mean_arti[0] + best_mean_arti[1]) / 2:
                    best_mean_arti[0] = ratio_data_to_noise
                    best_mean_arti[1] = ratio_data_to_available_data
                    best_mean_arti[2] = ratio_data_to_perfect
                    best_mean_arti[3] = (threshold_factuality, threshold_density, threshold_rouge)
                if (ratio_data_to_noise * ratio_data_to_available_data) ** 0.5 > (
                        best_mean_geo[0] * best_mean_geo[1]) ** 0.5:
                    best_mean_geo[0] = ratio_data_to_noise
                    best_mean_geo[1] = ratio_data_to_available_data
                    best_mean_geo[2] = ratio_data_to_perfect
                    best_mean_geo[3] = (threshold_factuality, threshold_density, threshold_rouge)

    print(best_mean_arti)
    print(best_mean_geo)

    # display_csv_row_by_row(df)


if __name__ == "__main__":
    main()

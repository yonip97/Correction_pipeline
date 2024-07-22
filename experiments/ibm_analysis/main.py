import pandas as pd
from nltk.tokenize import word_tokenize
import scipy.stats as stats


def get_cot_prompt_examples():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/base_model_50000_documents/cot_prompts/original"
    df = pd.read_csv(path + '/base_model_outputs_below_0.5_text_length_above_65_10000_samples_revised_scored.csv',
                     index_col=0)
    print(df.columns)
    for i in range(100, 200):
        print(df['indices'].tolist()[i])
        print()
        print(df['text'].tolist()[i])
        print()
        print(df['model_summary'].tolist()[i])
        print()
        print(df['revised_summary_full_text'].tolist()[i])
        print()
        print("density: ", df['revised_summary_density'].tolist()[i])
        print("seahorse: ", df['revised_summary_seahorse'].tolist()[i])
        print("rougeL: ", df['rougeL_revised_to_base'].tolist()[i])
        print("--------------------------------------------------------------------------")


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


def edit_distance():
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/base_model_50000_documents/cot_prompts/original"
    cot_df = pd.read_csv(path + '/base_model_outputs_below_0.5_text_length_above_65_10000_samples_revised_scored.csv',
                         index_col=0)
    path = "/data/home/yehonatan-pe/Correction_pipeline/experiments/ablations/better_summaries/data/prompts/4"
    ablation_df = pd.read_csv(
        path + '/base_model_outputs_below_0.5_text_length_above_65_10000_samples_summrized_by_chatgpt_scored_post_processed.csv',
        index_col=0)
    common_indices = set(cot_df['indices'].tolist()).intersection(set(ablation_df['indices'].tolist()))
    cot_df = cot_df[cot_df['indices'].isin(common_indices)]
    ablation_df = ablation_df[ablation_df['indices'].isin(common_indices)]
    cot_df['edit_distance'] = cot_df.apply(lambda x: word_level_edit_distance(x['model_summary'], x['revised_summary']),
                                           axis=1)
    ablation_df['edit_distance'] = ablation_df.apply(
        lambda x: word_level_edit_distance(x['model_summary'], x['model_summary_llm']), axis=1)
    print(cot_df['edit_distance'].mean(), ablation_df['edit_distance'].mean())
    print(cot_df['revised_summary_seahorse'].mean(), ablation_df['model_summary_llm_seahorse'].mean())
    print(cot_df['revised_summary_density'].mean(), ablation_df['model_summary_llm_density'].mean())
    print(cot_df['rougeL_revised_to_base'].mean(), ablation_df['rougeL_llm_to_base'].mean())
    print(stats.ttest_rel(cot_df['edit_distance'], ablation_df['edit_distance']))
    print(stats.ttest_rel(cot_df['revised_summary_seahorse'], ablation_df['model_summary_llm_seahorse']))
    print(stats.ttest_rel(cot_df['revised_summary_density'], ablation_df['model_summary_llm_density']))
    print(stats.ttest_rel(cot_df['rougeL_revised_to_base'], ablation_df['rougeL_llm_to_base']))

if __name__ == '__main__':
    #get_cot_prompt_examples()
    edit_distance()

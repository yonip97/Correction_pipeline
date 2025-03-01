import pandas as pd
import ast
from collections import Counter


# This is the analysis of the work before the llms did anything
def how_many_filtered_during_initial_annotation():
    #This function checks how many descriptions and summaries were removed during the manual annotation
    df = pd.read_csv("data/all_finalized_data/Initial_dataset_after_manual_processing.csv")
    filtered = pd.read_csv("data/all_finalized_data/Filtered_manual_dataset.csv")
    data_to_extract_end_of_annotation = pd.read_csv(
        "data/all_finalized_data/Factually_inconsistent_samples_final_annotation.csv")
    final_annotated_text = data_to_extract_end_of_annotation['text'].tolist()[-1]
    filtered_index = filtered[filtered['text'] == final_annotated_text].index[0]
    filtered = filtered.loc[:filtered_index]
    df_index = df[df['text'] == final_annotated_text].index[0]
    df = df.loc[:df_index]
    print("Before filtering we had", len(df), "annotations and ", len(df['model_summary'].unique()), "summaries")
    print(f"Amount of annotations filtered: {len(df) - len(filtered)}")
    removed = df[~df['text'].isin(filtered['text'])]
    print("Amount of annotation removed with all summary", len(removed))
    print("Amount of summaries removed", len(df['model_summary'].unique()) - len(filtered['model_summary'].unique()))
    print("Amount of annotation removed without removing summaries ", len(df) - len(filtered) - len(removed))
    print("After filtering we have", len(filtered), "annotations and ", len(filtered['model_summary'].unique()),
          "summaries")
    print()
    print("-------------------------------------------------------------------------------------------------")


def filtered_original_improved():
    #This function checks how many descriptions were replaced by a better version during the manual annotation
    # It also gives out reasons for the replacement
    filtered = pd.read_csv("data/all_finalized_data/Filtered_manual_dataset.csv")
    data_to_extract_end_of_annotation = pd.read_csv(
        "data/all_finalized_data/Factually_inconsistent_samples_final_annotation.csv")
    final_annotated_text = data_to_extract_end_of_annotation['text'].tolist()[-1]
    filtered_index = filtered[filtered['text'] == final_annotated_text].index[0]
    filtered = filtered.loc[:filtered_index]
    alternative = filtered[filtered['alternative explanation'].notnull()]
    reasons_to_improve = alternative['What is the problem'].tolist()
    reasons_to_improve = [x.split(',') for x in reasons_to_improve]
    reasons_to_improve_filtered = []
    for i in range(len(reasons_to_improve)):
        for j in range(len(reasons_to_improve[i])):
            if 'explanation' in reasons_to_improve[i][j]:
                reasons_to_improve_filtered.append(reasons_to_improve[i][j].strip().replace('?', ''))
    reasons_counter = Counter(reasons_to_improve_filtered)
    for key, value in reasons_counter.items():
        print(f"Because {key} we replaced {value} descriptions")
    print(f"Amount of descriptions that were changed: {len(alternative)}")
    print(f"Amount of summaries that were changed: {len(alternative['model_summary'].unique())}")
    print()
    print("-------------------------------------------------------------------------------------------------")


def analysis_of_originally_inconsistent_summaries():
    # This function checks which of the manual descriptions did not appear in the final version,
    # Meaning they were replaced by an llm explanation which was better,
    # And how many new descriptions were added to the dataset using the llm.
    df = pd.read_csv("data/all_finalized_data/Factually_inconsistent_samples_final_annotation.csv")
    df['maybe'] = df['maybe'].fillna('[]')
    df_removed = df[(df['remove'] == 1) | (df['maybe'] != '[]')]
    df_not_removed = df[(df['remove'] == 0) & (df['maybe'] == '[]')]
    descriptions = df_not_removed['explanations'].tolist()
    df2 = pd.read_csv(
        "data/all_finalized_data/Final_manual_dataset.csv")
    df2 = (df2.groupby(['text', 'model_summary'], sort=False)['explanation'].apply(list)).reset_index()

    df2 = df2[df2['text'].isin(df_not_removed['text'])]
    explanations = df2['explanation'].tolist()
    how_many_human_explanations_were_replaced_for_llm = 0
    for i in range(len(descriptions)):
        curr_description = descriptions[i]
        curr_description = eval(curr_description)
        for j in range(len(explanations[i])):
            if explanations[i][j] not in curr_description:
                how_many_human_explanations_were_replaced_for_llm += 1
    amount_of_new_descriptions = sum([len(eval(x)) for x in descriptions])
    amount_of_old_descriptions = sum([len(x) for x in explanations][:len(descriptions)])
    print(f"Amount of summaries that were removed: {len(df_removed)}")
    print(f"Amount of descriptions that were replaced: {how_many_human_explanations_were_replaced_for_llm}")
    print(f"Amount of new descriptions: {amount_of_new_descriptions - amount_of_old_descriptions}")
    print(f"Amount of old descriptions: {amount_of_old_descriptions}")
    print()
    print("-------------------------------------------------------------------------------------------------")


def analysis_of_originally_consistent_summaries():
    # This function checks how many new descriptions were added to the dataset
    df = pd.read_csv("data/all_finalized_data/Factually_consistent_samples_final_annotation.csv")
    df['descriptions'] = df['explanations'].apply(lambda x: ast.literal_eval(x))
    df['maybe'] = df['maybe'].fillna('[]')
    df_removed = df[(df['remove'] == 1) | (df['maybe'] != '[]')]
    df_not_removed = df[(df['remove'] == 0) & (df['maybe'] == '[]')]
    print("The amount of summaries that were removed: ", len(df_removed))
    print("The amount of summaries that were not removed: ", len(df_not_removed))
    descriptions = df_not_removed['descriptions'].tolist()
    descriptions_len = [len(x) for x in descriptions]
    print(f"There were {sum([1 for x in descriptions_len if x == 0])} summaries which were consistent")
    print(f"There were {sum([1 for x in descriptions_len if x != 0])} summaries which were inconsistent")
    print(f"There were {sum(descriptions_len)} descriptions in the consistent summaries")
    print()
    print("-------------------------------------------------------------------------------------------------")


def main():
    print("Analysis of the dataset, from raw to before the llm annotation")
    how_many_filtered_during_initial_annotation()
    filtered_original_improved()
    # We had 1174 summaries, we removed 74 summaries with 115 annotations, and 28 additional annotations during the manual proccessing of the data
    # In addition, we replaced the 269 explanations with a better version (they were part of a 209 summaries)
    # Of the 269 explanation, we replaced 55 because they had no explanation for that specific factual inconsistency,
    # made 151 explanation clear because the original was not and 63 explanation were imported
    print(
        "Analysis of the dataset, from before the llm annotation to after the llm annotation, on inconsistent summaries")
    analysis_of_originally_inconsistent_summaries()
    # We had 1100 summaries, we removed 105 summaries. We also replaced 154 human generated annotations with llm generated annotations
    # In addition, we added 356 annotations, which the llm identified as inconsistent.
    # In total, we had 1633 descriptions, now we have 1989 descriptions, 21% boost
    print(
        "Analysis of the dataset, from before the llm annotation to after the llm annotation, on consistent summaries")
    analysis_of_originally_consistent_summaries()
    # We had 500 summaries, we removed 98 summaries.
    # Out of 402 summaries, 274 were found to be consistent and 128 were found to be inconsistent, with 148 inconsistencies in total
    # We can see that the amount of actual inconsistencies in the consistent vs inconsistent is vastly different.
    # 2 per summary vs 1.156 per summary


if __name__ == '__main__':
    main()

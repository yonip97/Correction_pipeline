import os

import pandas as pd

import textwrap
import argparse
import ast
import re
import json
from inference_utils import transform_to_enumerated_descriptions
import random
from  collections import Counter

def wrap_text(text, width=120):
    """Wrap text to a specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def show_comparison_interface(texts, summaries, llm_explanations, human_explanations, output_file):
    # write a documentation for this function
    """
    texts: list of strings, the input texts
    summaries: list of strings, the model summaries
    llm_explanations: list of strings, the LLM explanations
    human_explanations: list of strings, the human explanations
    output_file: str, the path to the output file
    the function will show the comparison interface for the annotation of the dataset. There will be the text,the summary,
    the human annotations and the llm outputs.
    The outputs of the llm will be labeled if they already exist, and if not ,if they are correct or not.
     The interface will ask questions, and record the answers in the output file
    """

    # Load existing results if any
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)
    else:
        results = []

    # Determine the starting index
    start_index = len(results)
    texts = texts[start_index:]
    summaries = summaries[start_index:]
    llm_explanations = llm_explanations[start_index:]
    human_explanations = human_explanations[start_index:]

    for i, (text, summary, llm_output, human_output) in enumerate(
            zip(texts, summaries, llm_explanations, human_explanations), start=start_index
    ):
        print("=" * 50)
        print(f"Entry {i + 1}:")
        print("Text:")
        print(wrap_text(text))
        print("\nSummary:")
        print(summary)
        print("\nLLM Explanation:")
        print(llm_output)
        print("\nHuman Explanation:")
        print(human_output)
        print("=" * 50)

        # Questions for manual input
        both_identified = input("1. What facts did the LLM and the human both identify? (Separate with commas): ")
        correct_llm = input("2. What facts did the LLM identify which are correct? (Separate with commas): ")
        wrong_llm = input("3. What facts did the LLM identify which are wrong? (Separate with commas): ")
        maybe_llm = input("4. What facts did the LLM identify which are maybe correct? (Separate with commas): ")
        notes = input("5. Do you have any notes? (Optional, press Enter to skip): ")

        # Validate revisit input
        while True:
            revisit = input("6. Should you come back to this entry? (1 for yes, 0 for no): ").strip()
            if revisit in ["1", "0"]:
                revisit = int(revisit)
                break
            else:
                print("Invalid input. Please enter 1 for yes or 0 for no.")

        # Store the answers
        result = {
            "entry": i + 1,
            "both_identified": [fact.strip() for fact in both_identified.split(",") if fact.strip()],
            "correct_llm": [fact.strip() for fact in correct_llm.split(",") if fact.strip()],
            "wrong_llm": [fact.strip() for fact in wrong_llm.split(",") if fact.strip()],
            "maybe_llm": [fact.strip() for fact in maybe_llm.split(",") if fact.strip()],
            "notes": notes,
            "revisit": revisit,
        }
        results.append(result)

        # Save results after each entry to ensure progress is not lost
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        # Option to stop annotation
        stop = input("Do you want to stop the annotation? (yes to stop, no to continue): ").strip().lower()
        if stop == "yes":
            print("Annotation stopped. You can resume later.")
            break


def parse_args_for_initial_annotation():
    args = argparse.ArgumentParser()
    args.add_argument('-output_path')
    args.add_argument("-llm_output_path", type=str)
    args.add_argument("-data_path", type=str)
    args = args.parse_args()
    return args


def annotate():
    """
    Annotate the dataset with the LLM outputs and the human explanations
    """
    args = parse_args_for_initial_annotation()
    df_llm = pd.read_csv(args.llm_output_path)
    llm_outputs = df_llm['output'].tolist()
    texts = df_llm['text'].tolist()
    summaries = df_llm['model_summary'].tolist()
    if args.data_path is not None:
        df_human = pd.read_csv(args.data_path)
        human_annotation = df_human['explanation'].tolist()
    else:
        human_annotation = [None] * len(texts)
    show_comparison_interface(texts, summaries, llm_outputs, human_annotation, args.output_path)


def args_for_final_annotation():
    args = argparse.ArgumentParser()
    args.add_argument('-output_path')
    args.add_argument("-llm_output_path", type=str)
    args.add_argument("-data_path", type=str)
    args.add_argument("-annotation_path", type=str)
    args = args.parse_args()
    return args


def create_final_version_inconsistent():
    """
    Create the final version of the dataset with inconsistent samples.
    Parse the raw annotations, and the notes, and then add the correct llm outputs which were not found by humans into the dataset.
    In addition, add needed correction (rewrite a human description, rewrite llm description, and so on)
    """
    args = args_for_final_annotation()
    df = pd.read_csv(
        args.data_path)
    with open(
            args.annotation_path,
            "r") as f:
        results = json.load(f)
    grouped_df = (
        df.groupby(["text", "model_summary"], sort=False)["explanation"]
        .apply(list)
        .reset_index()
    )
    full_llm_outputs = pd.read_csv(
        args.llm_output_path,
        index_col=0).reset_index(drop=True)
    texts = grouped_df["text"].tolist()
    summaries = grouped_df["model_summary"].tolist()
    explanations = grouped_df["explanation"].tolist()
    if os.path.exists(args.output_path):
        final_df = pd.read_csv(args.output_path)
        final_dataset = final_df.to_dict("records")
    else:
        final_dataset = []
    for i in range(len(final_dataset), len(results)):
        text = texts[i]
        summary = summaries[i]
        explanation = explanations[i]
        maybe = []
        result = results[i]
        added_explanations = []
        print(full_llm_outputs.loc[i, "output"])
        print("------------------------------------------------------------")
        print(result)
        print("------------------------------------------------------------")
        print(explanation)
        print("------------------------------------------------------------")
        should_skip = input("Do you want to skip or remove this entry? (skip , remove  or no): ").strip().lower()
        while should_skip not in ["skip", "remove", "no"]:
            should_skip = input("Invalid input. Please enter skip, remove or no: ").strip().lower()
        if should_skip == "remove":
            final_dataset.append(
                {
                    "text": text,
                    "model_summary": summary,
                    "explanations": explanation,
                    "maybe": maybe,
                    "remove": 1
                }
            )
            final_df = pd.DataFrame(final_dataset)
            final_df.to_csv(args.output_path, index=False)
            continue
        elif should_skip == "skip":
            final_dataset.append(
                {
                    "text": text,
                    "model_summary": summary,
                    "explanations": explanation,
                    "maybe": maybe,
                    "remove": 0
                }
            )
            final_df = pd.DataFrame(final_dataset)
            final_df.to_csv(args.output_path, index=False)
            continue
        change_all_explanations = input("Do you want to change all explanations? (yes or no): ").strip().lower()
        if change_all_explanations == "yes":
            explanation = []
        while True:
            are_there_more_explanations = input(
                "Are there more explanations for this entry? (yes or no): ").strip().lower()
            if are_there_more_explanations == "yes":
                new_explanation = input("Enter the new explanation: ")
                added_explanations.append(new_explanation)
            else:
                break
        while True:
            are_there_maybes = input(
                "Are there more maybe correct explanations for this entry? (yes or no): ").strip().lower()
            if are_there_maybes == "yes":
                new_explanation = input("Enter the new maybe correct explanation: ")
                maybe.append(new_explanation)
            else:
                break
        explanation += added_explanations
        final_dataset.append(
            {
                "text": text,
                "model_summary": summary,
                "explanations": explanation,
                "maybe": maybe,
                "remove": 0
            }
        )
        final_df = pd.DataFrame(final_dataset)
        final_df.to_csv(args.output_path, index=False)


def create_final_version_consistent():
    """
    Create the final version of the dataset with consistent samples.
    Parse the raw annotations, and the notes, and then add the correct llm outputs which were not found by humans into the dataset.
    In addition, add needed correction (rewrite llm description, and so on)
    """
    args = args_for_final_annotation()
    df = pd.read_csv(
        args.data_path)
    with open(
            args.annotation_path,
            "r") as f:
        results = json.load(f)
    grouped_df = (
        df.groupby(["text", "model_summary"], sort=False)["explanation"]
        .apply(list)
        .reset_index()
    )
    full_llm_outputs = pd.read_csv(
        args.llm_output_path,
        index_col=0).reset_index(drop=True)
    texts = grouped_df["text"].tolist()
    summaries = grouped_df["model_summary"].tolist()
    if os.path.exists(args.output_path):
        final_df = pd.read_csv(args.output_path)
        final_dataset = final_df.to_dict("records")
    else:
        final_dataset = []
    for i in range(len(final_dataset), len(results)):
        text = texts[i]
        summary = summaries[i]
        explanation = []
        maybe = []
        result = results[i]
        added_explanations = []
        print(full_llm_outputs.loc[i, "output"])
        print("------------------------------------------------------------")
        print(result)
        print("------------------------------------------------------------")
        print(explanation)
        print("------------------------------------------------------------")
        should_skip = input("Do you want to skip or remove this entry? (skip , remove  or no): ").strip().lower()
        while should_skip not in ["skip", "remove", "no"]:
            should_skip = input("Invalid input. Please enter skip, remove or no: ").strip().lower()
        if should_skip == "remove":
            final_dataset.append(
                {
                    "text": text,
                    "model_summary": summary,
                    "explanations": explanation,
                    "maybe": maybe,
                    "remove": 1
                }
            )
            final_df = pd.DataFrame(final_dataset)
            final_df.to_csv(args.output_path, index=False)
            continue
        elif should_skip == "skip":
            final_dataset.append(
                {
                    "text": text,
                    "model_summary": summary,
                    "explanations": explanation,
                    "maybe": maybe,
                    "remove": 0
                }
            )
            final_df = pd.DataFrame(final_dataset)
            final_df.to_csv(args.output_path, index=False)
            continue
        while True:
            are_there_more_explanations = input(
                "Are there more explanations for this entry? (yes or no): ").strip().lower()
            if are_there_more_explanations == "yes":
                new_explanation = input("Enter the new explanation: ")
                added_explanations.append(new_explanation)
            else:
                break
        while True:
            are_there_maybes = input(
                "Are there more maybe correct explanations for this entry? (yes or no): ").strip().lower()
            if are_there_maybes == "yes":
                new_explanation = input("Enter the new maybe correct explanation: ")
                maybe.append(new_explanation)
            else:
                break
        explanation += added_explanations
        final_dataset.append(
            {
                "text": text,
                "model_summary": summary,
                "explanations": explanation,
                "maybe": maybe,
                "remove": 0
            }
        )
        final_df = pd.DataFrame(final_dataset)
        final_df.to_csv(args.output_path, index=False)


def reduce_to_description(output):
    outputs = output.split("Fact:")[1:]
    final = []
    for sample in outputs:
        sample = sample.split("Description:")[1].split('\n\n')[0].replace("```", "").strip()
        final.append(sample)
    return final


def unite():
    # Takes the inconsistent and consistent final versions and unites them,
    # while removing all those needed to be removed, and also those with maybes
    # It also groups together the samples used for few shot, for development and for test.
    df_consistent = pd.read_csv(
        "data/all_finalized_data/post_llm_annotation/Factually_consistent_samples_final_annotation.csv")
    df_consistent.rename(columns={'explanations': "descriptions"}, inplace=True)
    df_inconsistent = pd.read_csv(
        "data/all_finalized_data/post_llm_annotation/Factually_inconsistent_samples_final_annotation.csv")
    df_inconsistent.rename(columns={'explanations': "descriptions"}, inplace=True)
    df_inconsistent['set'] = 'test'
    df_consistent['set'] = 'test'
    df_inconsistent.loc[:49, 'set'] = 'dev'
    df_inconsistent.loc[206, 'set'] = 'few_shot'
    df_inconsistent.loc[256, 'set'] = 'few_shot'
    df_consistent.loc[500:, 'set'] = 'dev'
    df_consistent.loc[530, "set"] = "few_shot"
    df_inconsistent['maybe'] = df_inconsistent['maybe'].apply(lambda x: ast.literal_eval(x))
    df_consistent['maybe'] = df_consistent['maybe'].apply(lambda x: ast.literal_eval(x))
    df_inconsistent = df_inconsistent[df_inconsistent['remove'] == 0]
    df_consistent = df_consistent[df_consistent['remove'] == 0]
    df_inconsistent = df_inconsistent[df_inconsistent['maybe'].apply(len) == 0]
    df_consistent = df_consistent[df_consistent['maybe'].apply(len) == 0]
    df_inconsistent['original_label'] = 'inconsistent'
    df_consistent['original_label'] = 'consistent'
    df = pd.concat([df_consistent, df_inconsistent])
    df = df.drop(columns=['remove', 'maybe'])
    df.reset_index(inplace=True, drop=True)
    df.to_csv("data/all_finalized_data/final/final_dataset.csv", index=False)
    dev = df[df['set'] == 'dev']
    dev.reset_index(inplace=True, drop=True)
    test = df[df['set'] == 'test']
    test.reset_index(inplace=True, drop=True)
    few_shot = df[df['set'] == 'few_shot']
    few_shot.reset_index(drop=True, inplace=True)
    dev.to_csv("data/all_finalized_data/final/final_dataset_dev.csv", index=False)
    test.to_csv("data/all_finalized_data/final/final_dataset_test.csv", index=False)
    few_shot.to_csv("data/all_finalized_data/final/final_dataset_few_shot.csv")


def annotation_of_dataset_augmentation(path_original_data, llm_outputs, output_file):
    """
    The function goes over the development set outputs for an llm outputs, and marks all the additional inconsistencies
    that the model found, which are not in the dataset.
    This function is different from the other annotation function has it records what are the inconsistencies that the model found,
    not just the identifier, so we can see the intersection between different models.
    """
    df_original_dataset = pd.read_csv(path_original_data)
    df_original_dataset = (df_original_dataset.groupby(["text", "model_summary"], sort=False)["explanation"]
                           .apply(list).reset_index())
    df_llm = pd.read_csv(llm_outputs)
    llm_outputs = df_llm['output'].tolist()
    descriptions = df_original_dataset["explanation"].tolist()
    texts = df_original_dataset["text"].tolist()
    summaries = df_original_dataset["model_summary"].tolist()
    results = []
    for i in range(len(texts)):
        print("=" * 50)
        print(f"Entry {i + 1}:")
        print("Text:")
        print(wrap_text(texts[i]))
        print("\nSummary:")
        print(summaries[i])
        print("\nLLM descriptions:")
        print(llm_outputs[i])
        print("\nHuman descriptions:")
        print(descriptions[i])
        print("=" * 50)
        llm_augmentation_correct = input(
            "What did the model identify which is not in the dataset?, please separate by commas: ")
        results.append(llm_augmentation_correct)
        df = pd.DataFrame(
            {"text": texts[:len(results)], "model_summary": summaries[:len(results)], "Correct_llm_answers": results})
        df.to_csv(output_file, index=False)


def split_labeled_list(text):
    # Regex to match labels (e.g., "A.", "B.", etc.) and the corresponding text
    pattern = r'([A-Z]\.)\s*(.*?)\s*(?=[A-Z]\.|$)'
    matches = re.findall(pattern, text)

    # Convert to dictionary or list format
    result = [item.strip() for label, item in matches]
    result = [item for item in result if "Description:" in item]
    result = [item.split('Description:')[1].replace('```', "").strip() for item in result]

    return result


def transform(description):
    description = description.replace('\n', " ")
    description = split_labeled_list(description)
    return description


def stratified_sample(data, sample_size=100, label_key='original_label', seed=42):
    random.seed(seed)  # Fix the seed for reproducibility

    # Count occurrences of each label
    label_counts = Counter(d[label_key] for d in data)

    # Compute how many samples to take from each label group
    total_samples = sum(label_counts.values())
    label_sample_sizes = {
        label: round((count / total_samples) * sample_size)
        for label, count in label_counts.items()
    }

    # Ensure the total is exactly 100 by adjusting the largest group
    total_assigned = sum(label_sample_sizes.values())
    if total_assigned != sample_size:
        max_label = max(label_sample_sizes, key=lambda k: label_sample_sizes[k])
        label_sample_sizes[max_label] += (sample_size - total_assigned)

    # Group data by label
    grouped_data = {label: [] for label in label_counts}
    for item in data:
        grouped_data[item[label_key]].append(item)

    # Sample from each group
    sampled_data = []
    for label, count in label_sample_sizes.items():
        sampled_data.extend(random.sample(grouped_data[label], min(count, len(grouped_data[label]))))

    return sampled_data
def prepare_for_human_verification():
    rel_cols = ['text', 'model_summary', 'human_description', 'llm_output', 'entry', "original_label"]
    df_human_inconsistent = pd.read_csv(
        "data/all_finalized_data/manual_annotation/Manual_dataset_list_format_with_sets.csv")

    df_human_inconsistent['human_description'] = ["\n\n".join(eval(x)) for x in
                                                  df_human_inconsistent['explanation'].tolist()]
    df_llm_inconsistent = pd.read_csv(
        "data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_all_processed_dataset.csv")

    df_human_consistent = pd.read_csv("data/all_finalized_data/raw_data/initial_data_consistent.csv")
    df_llm_consistent = pd.read_csv(
        "data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_factually_consistent_summaries.csv")
    df_human_consistent['human_description'] = ""
    llm_inconsistent_outputs = df_llm_inconsistent['output'].tolist()
    llm_inconsistent_outputs = [transform(x) for x in llm_inconsistent_outputs]
    llm_inconsistent_outputs = transform_to_enumerated_descriptions(llm_inconsistent_outputs)
    df_human_inconsistent['llm_output'] = llm_inconsistent_outputs
    df_human_inconsistent['entry'] = [i + 1 for i in range(len(df_human_inconsistent))]
    df_human_inconsistent['original_label'] = 'inconsistent'
    df_human_inconsistent = df_human_inconsistent[rel_cols]

    llm_consistent_outputs = df_llm_consistent['output'].tolist()
    llm_consistent_outputs = [transform(x) for x in llm_consistent_outputs]
    llm_consistent_outputs = transform_to_enumerated_descriptions(llm_consistent_outputs)
    df_human_consistent['llm_output'] = llm_consistent_outputs
    df_human_consistent['original_label'] = 'consistent'
    df_human_consistent['entry'] = [i + 1 for i in range(len(df_human_consistent))]
    df_human_consistent = df_human_consistent[rel_cols]
    all_data = pd.concat([df_human_inconsistent, df_human_consistent])
    df_final_test = pd.read_csv("data/all_finalized_data/final/final_dataset_test.csv")
    all_data = all_data[(all_data['text'].isin(df_final_test['text'].tolist())) & (
        all_data['model_summary'].isin(df_final_test['model_summary'].tolist()))]

    records = all_data.to_dict("records")
    with open("data/all_finalized_data/human_annotation_of_llm_outputs/data_for_human_verification.json", "w") as f:
        json.dump(records, f)
    sample = stratified_sample(records)
    with open("data/all_finalized_data/human_annotation_of_llm_outputs/sample_for_human_verification.json", "w") as f:
        json.dump(sample, f)





    # This dict is good, it contains all the text, summaries and descriptions by human and llm outputs. it also has entry number.
    # Need to upload this, sample 10 things, and even randomly mark them. And then when i take it from the label studio, ican see if it matches in the number of annoations and so on.
    # with open("data/all_finalized_data/human_annotation_of_llm_outputs/data_for_human_verification.json", "w") as f:
    #     json.dump(records, f)

    # df_human.rename(columns={'explanation': 'human_description'}, inplace=True)
    # df_human['llm_description'] = llms_outputs
    # # df_human.to_csv("data/all_finalized_data/human_annotation_of_llm_outputs/dummy_inconsistent.csv",index=False)
    # x = json.loads(df_human.to_json(orient="records"))
    # for i, y in enumerate(x):
    #     y['enrty'] = i
    # print(y.keys())
    # x = [{"data":y} for y in x]
    # with open("data/all_finalized_data/human_annotation_of_llm_outputs/dummy_inconsistent.json",'w') as f:
    #     json.dump(x,f)


# def split_labeled_list(text):
#     # Regex to match labels (e.g., "A.", "B.", etc.) and the corresponding text
#     pattern = r'([A-Z]\.)\s*(.*?)\s*(?=[A-Z]\.|$)'
#     matches = re.findall(pattern, text)
#
#     # Convert to dictionary or list format
#     result = [item.strip() for label, item in matches]
#     result = [item for item in result if "Description:" in item]
#     result = [item.split('Description:')[1].replace('```', "").strip() for item in result]
#
#     return result
#
#
# def transform(description):
#     description = description.replace('\n', " ")
#     description = split_labeled_list(description)
#     return description


# def should_be_added_to_human_verification():
#     # Some samples are removed, some samples have repeat of the same information twice, and some include irrelevant information
#     # The removed samples should not be annotated, and the remaining samples should not be annotated because it increases the task complexity and the guidelines complexity with no benefit.
#     # It is also very little data (30 removed, 50 with some noise). Therefore, we will not give our annotators to annotate them.
#     # Should I put in removed samples because of maybe?
#     df_inconsistent = pd.read_csv(
#         "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_all_processed_dataset.csv")
#     df_consistent = pd.read_csv(
#         "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_factually_consistent_summaries.csv")
#     possible_ids_inconsistent = pd.read_csv(
#         "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/post_llm_annotation/Factually_inconsistent_samples_final_annotation.csv")[
#         'remove'].tolist()
#     possible_ids_inconsistent = [i for i in range(len(possible_ids_inconsistent)) if possible_ids_inconsistent[i] == 0]
#     possible_ids_consistent = pd.read_csv(
#         "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/post_llm_annotation/Factually_consistent_samples_final_annotation.csv")[
#         'remove'].tolist()
#     possible_ids_consistent = [i for i in range(len(possible_ids_consistent)) if possible_ids_consistent[i] == 0]
#     inconsistent_outputs = df_inconsistent['output'].tolist()
#     inconsistent_outputs = [transform(x) for x in inconsistent_outputs]
#     consistent_outputs = df_consistent['output'].tolist()
#     consistent_outputs = [transform(x) for x in consistent_outputs]
#     with open(
#             "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/human_annotation_of_llm_outputs/annotation_of_factually_inconsistent_explanations.json",
#             "r") as f:
#         inconsistent = json.load(f)
#     with open(
#             "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/human_annotation_of_llm_outputs/annotation_of_factually_consistent_explanations.json",
#             "r") as f:
#         consistent = json.load(f)
#     for i in range(len(inconsistent)):
#         sample = inconsistent[i]
#         output = inconsistent_outputs[i]
#         num_of_annotated = len(sample['both_identified']) + len(sample['correct_llm']) + len(sample['wrong_llm']) + len(
#             sample['maybe_llm'])
#         if len(output) != num_of_annotated or i not in possible_ids_inconsistent:
#             sample['for_human_verification'] = "No"
#         elif len(sample['maybe_llm']) > 0:
#             sample['for_human_verification'] = "Maybe"
#         else:
#             sample['for_human_verification'] = "Yes"
#     for i in range(len(consistent)):
#         sample = consistent[i]
#         output = consistent_outputs[i]
#         num_of_annotated = len(sample['both_identified']) + len(sample['correct_llm']) + len(sample['wrong_llm']) + len(
#             sample['maybe_llm'])
#         if len(output) != num_of_annotated or i not in possible_ids_consistent:
#             print("hey")
#             sample['for_human_verification'] = "No"
#         elif len(sample['maybe_llm']) > 0:
#             print("ok")
#             sample['for_human_verification'] = "Maybe"
#         else:
#             sample['for_human_verification'] = "Yes"


def main():
    # annotate()
    # create_final_version_consistent()
    # create_description_format()
    # check_how_many_samples_were_modified()
    # unite()
    # parse_inconsistent_to_separate_llm_and_human()
    # annotation_of_dataset_augmentation("data/all_finalized_data/Final_manual_dataset.csv","data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_llama_3.1_405_dev.csv","data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_llama_3.1_405_dev_annotated.csv")
    prepare_for_human_verification()


if __name__ == "__main__":
    main()

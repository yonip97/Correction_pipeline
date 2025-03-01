
import json
import os

import pandas as pd

import textwrap
import argparse
import ast
def wrap_text(text, width=120):
    """Wrap text to a specified width."""
    return '\n'.join(textwrap.wrap(text, width=width))


def show_comparison_interface(texts, summaries, llm_explanations, human_explanations, output_file):
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
    args = parse_args_for_initial_annotation()
    df_llm = pd.read_csv(args.llm_output_path)
    llm_outputs = df_llm['output'].tolist()
    texts = df_llm['text'].tolist()
    summaries = df_llm['model_summary'].tolist()
    if args.data_path is not None:
        df_human = pd.read_csv(args.data_path)
        human_annotation = df_human['explanation'].tolist()
    else:
        human_annotation = [None]*len(texts)
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
        final_dataset =[]
    for i in range(len(final_dataset),len(results)):
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
                    "maybe":maybe,
                    "remove":1
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
                    "maybe":maybe,
                    "remove":0
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
                "remove":0
            }
        )
        final_df = pd.DataFrame(final_dataset)
        final_df.to_csv(args.output_path, index=False)

def create_final_version_consistent():
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
    for i in range(len(final_dataset),len(results)):
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
                    "maybe":maybe,
                    "remove":1
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
                    "maybe":maybe,
                    "remove":0
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
                "remove":0
            }
        )
        final_df = pd.DataFrame(final_dataset)
        final_df.to_csv(args.output_path, index=False)
def num_to_uppercase_letter(num):
    if 0 <= num <= 25:
        return chr(num + ord('A'))
    else:
        raise ValueError("Number must be between 0 and 25")



def reduce_to_description(output):
    outputs = output.split("Fact:")[1:]
    final = []
    for sample in outputs:
        sample = sample.split("Description:")[1].split('\n\n')[0].replace("```","").strip()
        final.append(sample)
    return final

# def check_how_many_samples_were_modified():
#     # This shows how many samples were modified, it separates llm facts that were added,
#     # facts which were missing and added, and facts which were improved
#
#     final_data_consistent = pd.read_csv("data/factually_consistent_samples_final.csv")
#     consistent_texts = final_data_consistent['text'].tolist()
#     final_data_consistent = final_data_consistent[final_data_consistent["remove"] == 0]
#     import ast
#     final_data_consistent['maybe'] = final_data_consistent['maybe'].apply(lambda x: ast.literal_eval(x))
#     final_data_inconsistent = pd.read_csv("data/factually_inconsistent_samples_final.csv")
#     inconsistent_texts = final_data_inconsistent['text'].tolist()
#     final_data_inconsistent = final_data_inconsistent[final_data_inconsistent["remove"] == 0]
#     final_data_inconsistent['maybe'] = final_data_inconsistent['maybe'].apply(lambda x: '[]' if pd.isna(x) else x)
#     final_data_inconsistent['maybe'] = final_data_inconsistent['maybe'].apply(lambda x: ast.literal_eval(x))
#     final_data_inconsistent =  final_data_inconsistent[final_data_inconsistent['maybe'].apply(len) == 0]
#     final_data_consistent = final_data_consistent[final_data_consistent['maybe'].apply(len) == 0]
#     original_dataset_explanations = pd.read_csv("data/Dataset_construction_initial_data_for_annotation_from_drive.csv")
#     original_dataset_explanations = original_dataset_explanations[original_dataset_explanations['text'].isin(inconsistent_texts)]
#     llm_inconsistent_output = pd.read_csv("data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_all_processed_dataset.csv")
#     llm_inconsistent_output = llm_inconsistent_output[llm_inconsistent_output['text'].isin(inconsistent_texts)]
#     llm_consistent_output = pd.read_csv("data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_factually_consistent_summaries.csv")
#     llm_consistent_output = llm_consistent_output[llm_consistent_output['text'].isin(consistent_texts)]
#     original_dataset = original_dataset_explanations['explanation human annotation'].tolist()
#     adjusted = original_dataset_explanations['alternative explanation'].tolist()
#     original_dataset = [str(x).strip() for x in original_dataset ]
#     llm_inconsistent_output = llm_inconsistent_output['output'].tolist()
#     llm_inconsistent_output = [x.strip() for x in llm_inconsistent_output]
#     llm_inconsistent_output= [reduce_to_description(x) for x in llm_inconsistent_output]
#     llm_inconsistent_output = [x for sublist in llm_inconsistent_output for x in sublist]
#     llm_consistent_output = llm_consistent_output['output'].tolist()
#     llm_consistent_output = [x.strip() for x in llm_consistent_output]
#     llm_consistent_output = [reduce_to_description(x) for x in llm_consistent_output]
#     llm_consistent_output = [x for sublist in llm_consistent_output for x in sublist]
#     counter = 0
#     other = 0
#     total = []
#     import evaluate
#     rouge = evaluate.load('rouge')
#     for x in final_data_consistent['explanations']:
#         x = ast.literal_eval(x)
#         total.append(len(x))
#         for sample in x:
#             sample = sample.strip()
#             if sample in llm_consistent_output:
#                 counter += 1
#             else:
#                 other += 1
#                 scores = rouge.compute(predictions=[sample]*len(llm_consistent_output), references=llm_consistent_output,rouge_types =['rougeL'], use_aggregator=False)
#                 if max(scores['rougeL']) > 0.9:
#                     index = scores['rougeL'].index(max(scores['rougeL']))
#                     print(max(scores['rougeL']))
#                     print(llm_consistent_output[index])
#                     print(sample)
#                     print()
#                     print("-------------------------------------------------")
#     import matplotlib.pyplot as plt
#     plt.hist(total, bins=20)
#     plt.show()
#     print("For the consistent summaries:")
#     print("Amount of samples:",len(final_data_consistent))
#     print("Amount of factually consistent summaries:",sum([1 for x in final_data_consistent['explanations'] if len(ast.literal_eval(x)) == 0]))
#     print("Amount of factually inconsistent summaries:",sum([1 for x in final_data_consistent['explanations'] if len(ast.literal_eval(x)) > 0]))
#     print("Amount of factuality mistakes:",sum(total))
#     print("Amount of factuality mistakes that were adjusted:",other)
#     counter_original = 0
#     counter_llm = 0
#     other = 0
#     adjusted_before = 0
#     total = []
#     for x in final_data_inconsistent['explanations']:
#         x = ast.literal_eval(x)
#         total.append(len(x))
#         for sample in x:
#             sample = sample.strip()
#             if sample in llm_inconsistent_output:
#                 counter_llm += 1
#             elif sample in original_dataset:
#                 counter_original += 1
#             elif sample in adjusted:
#                 adjusted_before +=1
#             else:
#                 other += 1
#     import matplotlib.pyplot as plt
#     plt.hist(total, bins=20)
#     plt.show()
#     print("For the inconsistent summaries:")
#     print("Amount of samples:",len(final_data_inconsistent))
#     print("Amount of factuality mistakes:",sum(total))
#     print("Amount of factuality mistakes that were in the original dataset:",counter_original)
#     print("Amount of factuality mistakes that were in the llm output:",counter_llm)
#     print("Amount of factuality mistakes that were adjusted before:",adjusted_before)
#     print("Amount of factuality mistakes that were adjusted:",other)

# def parse_inconsistent_to_separate_llm_and_human():
#     df = pd.read_csv("data/factually_inconsistent_samples_final.csv")
#     df['maybe'] = df['maybe'].apply(lambda x: ast.literal_eval(x))
#     df = df[df['maybe'].apply(len) == 0]
#     df = df[df['remove'] == 0]
#     original_df = pd.read_csv("data/Dataset_construction_initial_data_for_annotation_from_drive.csv")
#     original_df_explanations = [str(x).strip() for x in original_df['explanation human annotation'].tolist()]
#     altered_explanations = [str(x).strip() for x in original_df['alternative explanation'].tolist()]
#     llm_output = pd.read_csv("data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_all_processed_dataset.csv")
#     llm_explanations = llm_output['output'].tolist()
#     llm_explanations = [reduce_to_description(x) for x in llm_explanations]
#     llm_explanations = [x.strip() for sublist in llm_explanations for x in sublist]
#     in_llm = []
#     in_original = []
#     in_altered = []
#     other = []
#     import evaluate
#     overall_llm_samples=[]
#     overall_dataset_samples=[]
#     rouge = evaluate.load('rouge')
#     for x in df['explanations']:
#         x = ast.literal_eval(x)
#         llm_samples=[]
#         dataset_samples=[]
#         for sample in x:
#             sample = sample.strip()
#             if sample in llm_explanations:
#                 in_llm.append(sample)
#                 llm_samples.append(sample)
#             elif sample in original_df_explanations:
#                 in_original.append(sample)
#                 dataset_samples.append(sample)
#             elif sample in altered_explanations:
#                 in_altered.append(sample)
#                 dataset_samples.append(sample)
#             else:
#                 llm_scores = rouge.compute(predictions=[sample]*len(llm_explanations), references=llm_explanations,rouge_types =['rougeL'], use_aggregator=False)
#                 dataset_scores = rouge.compute(predictions=[sample]*len(original_df_explanations+altered_explanations), references=original_df_explanations+altered_explanations,rouge_types =['rougeL'], use_aggregator=False)
#                 if max(llm_scores['rougeL']) > max(dataset_scores['rougeL']):
#                     in_llm.append(sample)
#                     llm_samples.append(sample)
#                 else:
#                     in_original.append(sample)
#                     dataset_samples.append(sample)
#
#         overall_llm_samples.append(llm_samples)
#         overall_dataset_samples.append(dataset_samples)
#     df['llm_samples'] = overall_llm_samples
#     df['dataset_samples'] = overall_dataset_samples
#     df.to_csv("data/factually_inconsistent_samples_final_separated.csv", index=False)
#
#     print("Amount of samples in llm:",len(in_llm))
#     print("Amount of samples in original dataset:",len(in_original))
#     print("Amount of samples in altered dataset:",len(in_altered))
#     print("Amount of samples in other:",len(other))

def unite():
    # Takes the inconsistent and consistent final versions and unites them,
    # while removing all those needed to be removed, and also those with maybes
    df_consistent = pd.read_csv("data/all_finalized_data/post_llm_annotation/Factually_consistent_samples_final_annotation.csv")
    df_consistent.rename(columns={'explanations': "descriptions"}, inplace=True)
    df_inconsistent = pd.read_csv("data/all_finalized_data/post_llm_annotation/Factually_inconsistent_samples_final_annotation.csv")
    df_inconsistent.rename(columns={'explanations': "descriptions"}, inplace=True)
    df_inconsistent['set'] = 'test'
    df_consistent['set'] = 'test'
    df_inconsistent.loc[:49,'set'] = 'dev'
    df_inconsistent.loc[206,'set'] = 'few_shot'
    df_inconsistent.loc[256,'set'] = 'few_shot'
    df_consistent.loc[500:,'set'] = 'dev'
    df_consistent.loc[530,"set"] ="few_shot"
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
    df.reset_index(inplace=True,drop=True)
    df.to_csv("data/all_finalized_data/final/final_dataset.csv", index=False)
    dev = df[df['set'] == 'dev']
    dev.reset_index(inplace=True,drop=True)
    test = df[df['set'] == 'test']
    test.reset_index(inplace=True,drop=True)
    few_shot = df[df['set'] == 'few_shot']
    few_shot.reset_index(drop=True, inplace=True)
    dev.to_csv("data/all_finalized_data/final/final_dataset_dev.csv", index=False)
    test.to_csv("data/all_finalized_data/final/final_dataset_test.csv", index=False)
    few_shot.to_csv("data/all_finalized_data/final/final_dataset_few_shot.csv")
def annotation_of_dataset_augmentation(path_original_data,llm_outputs,output_file):
    df_original_dataset = pd.read_csv(path_original_data)
    df_original_dataset = (df_original_dataset.groupby(["text", "model_summary"], sort=False)["explanation"]
        .apply(list).reset_index())
    df_llm = pd.read_csv(llm_outputs)
    llm_outputs = df_llm['output'].tolist()
    descriptions = df_original_dataset["explanation"].tolist()
    texts = df_original_dataset["text"].tolist()
    summaries = df_original_dataset["model_summary"].tolist()
    results =[]
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
        llm_augmentation_correct = input("What did the model identify which is not in the dataset?, please separate by commas: ")
        results.append(llm_augmentation_correct)
        df = pd.DataFrame({"text":texts[:len(results)],"model_summary":summaries[:len(results)],"Correct_llm_answers":results})
        df.to_csv(output_file,index=False)



import re
import json

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

def should_be_added_to_human_verification():
    #Some samples are removed, some samples have repeat of the same information twice, and some include irrelevant information
    #The removed samples should not be annotated, and the remaining samples should not be annotated because it increases the task complexity and the guidelines complexity with no benefit.
    #It is also very little data (30 removed, 50 with some noise). Therefore, we will not give our annotators to annotate them.
    # Should I put in removed samples because of maybe?
    df_inconsistent = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_all_processed_dataset.csv")
    df_consistent = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_gpt_4o_factually_consistent_summaries.csv")
    possible_ids_inconsistent = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/post_llm_annotation/Factually_inconsistent_samples_final_annotation.csv")['remove'].tolist()
    possible_ids_inconsistent = [i for i in range(len(possible_ids_inconsistent)) if possible_ids_inconsistent[i] == 0]
    possible_ids_consistent = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/post_llm_annotation/Factually_consistent_samples_final_annotation.csv")['remove'].tolist()
    possible_ids_consistent = [i for i in range(len(possible_ids_consistent)) if possible_ids_consistent[i] == 0]
    inconsistent_outputs = df_inconsistent['output'].tolist()
    inconsistent_outputs = [transform(x) for x in inconsistent_outputs]
    consistent_outputs = df_consistent['output'].tolist()
    consistent_outputs = [transform(x) for x in consistent_outputs]


    import json
    with open("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/human_annotation_of_llm_outputs/annotation_of_factually_inconsistent_explanations.json","r") as f:
        inconsistent = json.load(f)
    with open("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/all_finalized_data/human_annotation_of_llm_outputs/annotation_of_factually_consistent_explanations.json","r") as f:
        consistent = json.load(f)
    for i in range(len(inconsistent)):
        sample = inconsistent[i]
        output = inconsistent_outputs[i]
        num_of_annotated = len(sample['both_identified']) + len(sample['correct_llm']) + len(sample['wrong_llm'])+len(sample['maybe_llm'])
        if len(output) != num_of_annotated or i not in possible_ids_inconsistent:
            sample['for_human_verification'] = "No"
        elif len(sample['maybe_llm']) > 0:
            sample['for_human_verification'] = "Maybe"
        else:
            sample['for_human_verification'] = "Yes"
    for i in range(len(consistent)):
        sample = consistent[i]
        output = consistent_outputs[i]
        num_of_annotated = len(sample['both_identified']) + len(sample['correct_llm']) + len(sample['wrong_llm'])+len(sample['maybe_llm'])
        if len(output) != num_of_annotated or i not in possible_ids_consistent:
            sample['for_human_verification'] = "No"
        elif len(sample['maybe_llm']) > 0:
            sample['for_human_verification'] = "Maybe"
        else:
            sample['for_human_verification'] = "Yes"


def main():
    #annotate()
    #create_final_version_consistent()
    #create_description_format()
    #check_how_many_samples_were_modified()
    unite()
    #parse_inconsistent_to_separate_llm_and_human()
    #annotation_of_dataset_augmentation("data/all_finalized_data/Final_manual_dataset.csv","data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_llama_3.1_405_dev.csv","data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6/results_llama_3.1_405_dev_annotated.csv")


if __name__ == "__main__":
    main()

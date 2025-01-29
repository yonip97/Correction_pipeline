import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def proccess_llm_output(output):
    output = output.split('{')[-1].split('}')[0]
    output = "{" + output + "}"
    output = ast.literal_eval(output)
    return output


def num_to_uppercase_letter(num):
    if 0 <= num <= 25:
        return chr(num + ord('A'))
    else:
        raise ValueError("Number must be between 0 and 25")


def create_precision_recall_dicts(llm_outputs, human_explanations_list):
    precision_dicts = [proccess_llm_output(x) for x in llm_outputs]
    human_explanations_list = [ast.literal_eval(x) for x in human_explanations_list]
    recall_dicts = [{num_to_uppercase_letter(key): [] for key in range(len(human_explanations_list[i]))} for i in
                    range(len(human_explanations_list))]
    for i in range(len(precision_dicts)):
        for key, value_list in precision_dicts[i].items():
            for value in value_list:
                recall_dicts[i][value].append(key)
    return precision_dicts, recall_dicts

def transform(dicts):
    for i in range(len(dicts)):
        for key, value_list in dicts[i].items():
            dicts[i][key] = 1 if len(value_list) > 0 else 0
    return dicts

def calculate_precision_recall_f1(precision_dicts, recall_dicts):
    precision = []
    recall = []
    for i in range(len(precision_dicts)):
        precision.append([])
        for key, value_list in precision_dicts[i].items():
            precision[i].append(1 if len(value_list) > 0 else 0)
    for i in range(len(recall_dicts)):
        recall.append([])
        for key, value_list in recall_dicts[i].items():
            recall[i].append(1 if len(value_list) > 0 else 0)
    precision_summary_wise = [sum(x) / len(x) if len(x) > 0 else 0 for x in precision]

    recall_summary_wise = [sum(x) / len(x) for x in recall]
    f1_summary_wise = [2 * (precision_summary_wise[i] * recall_summary_wise[i]) / (
            precision_summary_wise[i] + recall_summary_wise[i]) if  precision_summary_wise[i] + recall_summary_wise[i] !=0 else 0  for i in range(len(precision_summary_wise))]
    precision_summary_wise = sum(precision_summary_wise) / len(precision_summary_wise)
    recall_summary_wise = sum(recall_summary_wise) / len(recall_summary_wise)
    f1_summary_wise = sum(f1_summary_wise) / len(f1_summary_wise)
    precision_overall = sum([sum(x) for x in precision]) / sum([len(x) for x in precision])
    recall_overall = sum([sum(x) for x in recall]) / sum([len(x) for x in recall])
    f1_overall = 2 * (precision_overall * recall_overall) / (precision_overall + recall_overall)
    # print(f"Precision summary wise: {precision_summary_wise}")
    # print(f"Recall summary wise: {recall_summary_wise}")
    # print(f"F1 summary wise: {f1_summary_wise}")
    # print(f"Precision overall: {precision_overall}")
    # print(f"Recall overall: {recall_overall}")
    # print(f"F1 overall: {f1_overall}")
    results = {'precision_summary_wise': precision_summary_wise, 'recall_summary_wise': recall_summary_wise,
               'f1_summary_wise': f1_summary_wise, 'precision_overall': precision_overall, 'recall_overall': recall_overall,
               'f1_overall': f1_overall}
    return results




def plot_key_value_bars(gt,predicted,names,title):
    """
    Plots a bar chart for a dictionary where each key maps to a list of two values.

    Args:
        data_dict (dict): A dictionary with 6 keys, each containing a list of 2 values.
                          The values represent "predicted" and "ground truth".
    """



    # Set the position of the bars on the x-axis
    x = np.arange(len(predicted))

    # Bar width
    bar_width = 0.35

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - bar_width / 2, predicted, width=bar_width, label='Predicted', color='blue')
    ax.bar(x + bar_width / 2, gt, width=bar_width, label='Ground Truth', color='orange')

    # Add labels, title, and legend
    ax.set_xlabel('Keys')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_xticklabels(names, rotation=45)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def analysis():
    df_human_raw = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/final_dataset_50_samples.csv", index_col=0)

    df_llm = pd.read_csv(
        "data/llm_as_a_judge/detection/evaluate_detection_in_enumerated_descriptions/precision/prompt4/zero_shot/gpt-4o/results_gpt_4o_prompt7.csv",
        index_col=0)
    gt = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/ground truth - Zero shot prompt 7.csv")
    gt_precision = gt['human annotation precision '].tolist()
    gt_recall = gt['human annotation recall'].tolist()
    gt_precision = [ast.literal_eval(x) for x in gt_precision]
    gt_recall = [ast.literal_eval(x) for x in gt_recall]
    print("Zero shot")
    precision_dicts, recall_dicts = create_precision_recall_dicts(df_llm["output"].tolist(), df_human_raw["explanations"].tolist())
    est_results = calculate_precision_recall_f1(precision_dicts, recall_dicts)
    gt_results = calculate_precision_recall_f1(gt_precision, gt_recall)
    for key in est_results:
        print(f"Absolute difference in {key}: {est_results[key] - gt_results[key]}")
    predicted = np.array(list(est_results.values()))
    gt = np.array(list(gt_results.values()))
    plot_key_value_bars(gt,predicted,list(est_results.keys()) ,"Zero Shot")
    df_llm =pd.read_csv("data/llm_as_a_judge/detection/evaluate_detection_in_enumerated_descriptions/precision/prompt4/few_shot/gpt-4o/results_gpt_4o_prompt4.csv")
    gt = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/ground truth - few shot descriptions prompt 4.csv")
    gt_precision = gt['human annotation precision '].tolist()
    gt_recall = gt['human annotation recall'].tolist()
    gt_precision = [ast.literal_eval(x) for x in gt_precision]
    gt_recall = [ast.literal_eval(x) for x in gt_recall]
    print("Few shot")
    precision_dicts, recall_dicts = create_precision_recall_dicts(df_llm["output"].tolist(), df_human_raw["explanations"].tolist())
    est_results = calculate_precision_recall_f1(precision_dicts, recall_dicts)
    gt_results = calculate_precision_recall_f1(gt_precision, gt_recall)
    for key in est_results:
        print(f"Absolute difference in {key}: {est_results[key] - gt_results[key]}")
    predicted = np.array(list(est_results.values()))
    gt = np.array(list(gt_results.values()))
    plot_key_value_bars(gt,predicted,list(est_results.keys()),"Few Shot" )

    df_llm =pd.read_csv("data/llm_as_a_judge/detection/evaluate_detection_in_enumerated_descriptions/precision/prompt4/cot/gpt-4o/results_gpt_4o_prompt8.csv")
    gt = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/ground truth - cot prompt8.csv")
    gt_precision = gt['human annotation precision'].tolist()
    gt_recall = gt['human annotation recall'].tolist()
    gt_precision = [ast.literal_eval(x) for x in gt_precision]
    gt_recall = [ast.literal_eval(x) for x in gt_recall]
    print("Cot")
    precision_dicts, recall_dicts = create_precision_recall_dicts(df_llm["output"].tolist(), df_human_raw["explanations"].tolist())
    est_results = calculate_precision_recall_f1(precision_dicts, recall_dicts)
    gt_results = calculate_precision_recall_f1(gt_precision, gt_recall)
    for key in est_results:
        print(f"Absolute difference in {key}: {est_results[key] - gt_results[key]}")
    predicted = np.array(list(est_results.values()))
    gt = np.array(list(gt_results.values()))
    plot_key_value_bars(gt,predicted,list(est_results.keys()) ,"Chain of thought")
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
            precision_summary_wise[i] + recall_summary_wise[i]) if precision_summary_wise[i] + recall_summary_wise[
        i] != 0 else 0 for i in range(len(precision_summary_wise))]
    precision_summary_wise = sum(precision_summary_wise) / len(precision_summary_wise)
    recall_summary_wise = sum(recall_summary_wise) / len(recall_summary_wise)
    f1_summary_wise = sum(f1_summary_wise) / len(f1_summary_wise)
    precision_overall = sum([sum(x) for x in precision]) / sum([len(x) for x in precision])
    recall_overall = sum([sum(x) for x in recall]) / sum([len(x) for x in recall])
    f1_overall = 2 * (precision_overall * recall_overall) / (precision_overall + recall_overall)
    results = {'precision_summary_wise': precision_summary_wise, 'recall_summary_wise': recall_summary_wise,
               'f1_summary_wise': f1_summary_wise, 'precision_overall': precision_overall,
               'recall_overall': recall_overall,
               'f1_overall': f1_overall}
    return results


def plot_key_value_bars(gt, predicted, names, title):
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
    precision_dicts, recall_dicts = create_precision_recall_dicts(df_llm["output"].tolist(),
                                                                  df_human_raw["explanations"].tolist())
    est_results = calculate_precision_recall_f1(precision_dicts, recall_dicts)
    gt_results = calculate_precision_recall_f1(gt_precision, gt_recall)
    for key in est_results:
        print(f"Absolute difference in {key}: {est_results[key] - gt_results[key]}")
    predicted = np.array(list(est_results.values()))
    gt = np.array(list(gt_results.values()))
    plot_key_value_bars(gt, predicted, list(est_results.keys()), "Zero Shot")
    df_llm = pd.read_csv(
        "data/llm_as_a_judge/detection/evaluate_detection_in_enumerated_descriptions/precision/prompt4/few_shot/gpt-4o/results_gpt_4o_prompt4.csv")
    gt = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/ground truth - few shot descriptions prompt 4.csv")
    gt_precision = gt['human annotation precision '].tolist()
    gt_recall = gt['human annotation recall'].tolist()
    gt_precision = [ast.literal_eval(x) for x in gt_precision]
    gt_recall = [ast.literal_eval(x) for x in gt_recall]
    print("Few shot")
    precision_dicts, recall_dicts = create_precision_recall_dicts(df_llm["output"].tolist(),
                                                                  df_human_raw["explanations"].tolist())
    est_results = calculate_precision_recall_f1(precision_dicts, recall_dicts)
    gt_results = calculate_precision_recall_f1(gt_precision, gt_recall)
    for key in est_results:
        print(f"Absolute difference in {key}: {est_results[key] - gt_results[key]}")
    predicted = np.array(list(est_results.values()))
    gt = np.array(list(gt_results.values()))
    plot_key_value_bars(gt, predicted, list(est_results.keys()), "Few Shot")

    df_llm = pd.read_csv(
        "data/llm_as_a_judge/detection/evaluate_detection_in_enumerated_descriptions/precision/prompt4/cot/gpt-4o/results_gpt_4o_prompt8.csv")
    gt = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/ground truth - cot prompt8.csv")
    gt_precision = gt['human annotation precision'].tolist()
    gt_recall = gt['human annotation recall'].tolist()
    gt_precision = [ast.literal_eval(x) for x in gt_precision]
    gt_recall = [ast.literal_eval(x) for x in gt_recall]
    print("Cot")
    precision_dicts, recall_dicts = create_precision_recall_dicts(df_llm["output"].tolist(),
                                                                  df_human_raw["explanations"].tolist())
    est_results = calculate_precision_recall_f1(precision_dicts, recall_dicts)
    gt_results = calculate_precision_recall_f1(gt_precision, gt_recall)
    for key in est_results:
        print(f"Absolute difference in {key}: {est_results[key] - gt_results[key]}")
    predicted = np.array(list(est_results.values()))
    gt = np.array(list(gt_results.values()))
    plot_key_value_bars(gt, predicted, list(est_results.keys()), "Chain of thought")


def llm_output_to_metrics():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/final_prompts_to_be_tested/zero_shot/prompt1/results_gpt_4_dev_evaluated.csv")
    judge_outputs = df['output'].tolist()
    raw = pd.read_csv("data/final_dataset.csv")
    raw = raw[raw['set'] == 'dev']
    descriptions = raw['explanations'].tolist()
    # descriptions = [ast.literal_eval(x) for x in descriptions]
    precision_dicts, recall_dicts = create_precision_recall_dicts(judge_outputs, descriptions)
    est_results = calculate_precision_recall_f1(precision_dicts, recall_dicts)
    print(est_results)


def llm_judgment_to_precision_recall_matching(judgement_outputs, raw_descriptions):
    # This functions takes the llm judgment, which is for every key of adescription in the evaluated output , we have the matched keys in the ground truth
    # From this we can get the precision (which is the mathcing itslef, every key which has a match in the gt is true positive, those who dont are false positives
    # For the recall we flip the matching dict. Every key becomes value and vise versa. So now we have all the keys in the gt which were matched to the predictions.
    # We add all missing gt keys using the ground truth explanations.
    precision_dicts = [proccess_llm_output(x) for x in judgement_outputs]
    recall_dicts = [{num_to_uppercase_letter(key): [] for key in range(len(raw_descriptions[i]))} for i in
                    range(len(raw_descriptions))]
    #Sometimes, the llm as a judge will make up some keys which are not in the ground truth.
    #If such keys exist in any sample, the sample will be discarded.
    discard = []
    for i in range(len(precision_dicts)):
        for key, value_list in precision_dicts[i].items():
            if any([x not in recall_dicts[i].keys() for x in value_list]):
                discard.append(i)
                break
            for value in value_list:
                recall_dicts[i][value].append(key)
    precision_dicts = [precision_dicts[i] for i in range(len(precision_dicts)) if i not in discard]
    recall_dicts = [recall_dicts[i] for i in range(len(recall_dicts)) if i not in discard]

    for x in precision_dicts:
        for key in x:
            x[key] = 1 if len(x[key]) > 0 else 0
    for x in recall_dicts:
        for key in x:
            x[key] = 1 if len(x[key]) > 0 else 0
    return precision_dicts, recall_dicts,discard


def compute_metrics(precision_dicts, recall_dicts):
    precision_dicts_when_gt_has_samples = [precision_dicts[i] for i in range(len(recall_dicts)) if
                                           len(recall_dicts[i]) != 0]
    recall_dicts_when_gt_has_samples = [recall_dicts[i] for i in range(len(recall_dicts)) if len(recall_dicts[i]) != 0]
    metrics = compute_metrics_for_samples_with_relevant_instances(precision_dicts_when_gt_has_samples,
                                                                  recall_dicts_when_gt_has_samples)
    precision_dicts_when_gt_does_not_have_samples = [precision_dicts[i] for i in range(len(recall_dicts)) if
                                                     len(recall_dicts[i]) == 0]
    proportion_of_summaries_with_wrong_predictions = len(
        [x for x in precision_dicts_when_gt_does_not_have_samples if len(x.keys()) > 0]) / len(
        precision_dicts_when_gt_does_not_have_samples) if len(precision_dicts_when_gt_does_not_have_samples) > 0 else 0
    metrics['proportion_of_consistent_summaries_with_wrong_predictions'] = proportion_of_summaries_with_wrong_predictions
    metrics['proportion_of_consistent_summaries_with_right_predictions'] = 1 - proportion_of_summaries_with_wrong_predictions
    wrong_predictions_per_consistent_summary = sum(
        [len(x.keys()) for x in precision_dicts_when_gt_does_not_have_samples]) / len(
        precision_dicts_when_gt_does_not_have_samples) if len(precision_dicts_when_gt_does_not_have_samples) > 0 else 0
    metrics['wrong_predictions_per_consistent_summary'] = wrong_predictions_per_consistent_summary
    return metrics



#
def compute_metrics_for_samples_with_relevant_instances(precision_dicts, recall_dicts):
    summary_wise_precision = [sum(x.values()) / len(x.keys()) if len(x.keys()) > 0 else 0 for x in precision_dicts]
    summary_wise_recall = [sum(x.values()) / len(x.keys()) for x in recall_dicts]
    summary_wise_f1 = [2 * x * y / (x + y) if x + y > 0 else 0 for x, y in
                       zip(summary_wise_precision, summary_wise_recall)]
    summary_wise_precision_mean = sum(summary_wise_precision) / len(summary_wise_precision)
    summary_wise_recall_mean = sum(summary_wise_recall) / len(summary_wise_recall)
    summary_wise_f1_mean = sum(summary_wise_f1) / len(summary_wise_f1)
    overall_precision = sum([sum(x.values()) for x in precision_dicts]) / sum([len(x.keys()) for x in precision_dicts])
    overall_recall = sum([sum(x.values()) for x in recall_dicts]) / sum([len(x.keys()) for x in recall_dicts])
    overall_f1 = 2 * overall_recall * overall_precision / (overall_recall + overall_precision)
    metrics = {'summary_prec': summary_wise_precision_mean, "summary_rec": summary_wise_recall_mean,
               "summary_f1": summary_wise_f1_mean,
               "overall_prec": overall_precision, "overall_rec": overall_recall, "overall_f1": overall_f1}
    return metrics

# llm_output_to_metrics()
# df = pd.read_csv(
#     "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/llm_inference/fine_grain_classification/final_prompts_to_be_tested/zero_shot/prompt1/results_gpt_4_dev_evaluated.csv")
# judge_outputs = df['output'].tolist()
# raw = pd.read_csv("data/final_dataset.csv")
# raw = raw[raw['set'] == 'dev']
# descriptions = raw['explanations'].tolist()
# descriptions = [ast.literal_eval(x) for x in descriptions]
# x = llm_judgment_to_precision_recall_matching(judge_outputs, descriptions)
# print(compute_metrics(x[0], x[1]))
# c = 1

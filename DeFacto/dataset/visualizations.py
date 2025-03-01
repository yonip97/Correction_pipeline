import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.patches import PathPatch
import numpy as np
import matplotlib.patches as mpatches  # For manual legend handles
import matplotlib.colors as mcolors
def plot_sorted_dict_of_dicts(data, model_name):
    """
    Takes a dictionary of dictionaries with numerical values and plots a bar chart.
    The inner dictionary keys are sorted alphabetically, and the outer dictionary keys
    determine the order of rows in the DataFrame.

    :param data: Dictionary of dictionaries where each inner dictionary has the same keys with numerical values.
    """

    # Sort the outer dictionary by keys

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(data, orient="index").T
    from tabulate import tabulate
    print(model_name)
    print(tabulate(df.T*100, headers='keys', tablefmt='psql'))

    # Plot the bar chart
    df.plot(kind="bar", figsize=(10, 6))

    # Customize plot
    plt.ylabel("Values")
    plt.title(model_name)
    plt.legend()
    plt.ylim(0, 1)
    plt.xticks(rotation=0)

    # Show plot
    plt.show()


def compare_results_per_model(path,set_name,prompt_says_inconsistent = False):
    by_model_results = {}
    for model in os.listdir(path):
        if model not in by_model_results:
            by_model_results[model] = {}
        new_path = os.path.join(path, model)
        for prompt in os.listdir(new_path):
            final_path = os.path.join(new_path, prompt)
            if prompt_says_inconsistent and "maybe_consistent" in prompt:
                continue
            if not prompt_says_inconsistent and "maybe_consistent" not in prompt:
                continue
            if os.path.exists(os.path.join(final_path, set_name, 'results.json')):
                with open(os.path.join(final_path, set_name, 'results.json'), 'r') as f:
                    data = json.load(f)
                    discarded_samples = data['discarded_samples']
                    df = pd.read_csv(os.path.join(final_path, set_name, 'results.csv'))
                    x = df.loc[discarded_samples]
                    print("number of discarded samples: ", len(discarded_samples),"out of",len(df))
                    judgments = x['judgment_outputs'].tolist()
                    y = ['{' not in judgment for judgment in judgments]
                    print("number of discarded samples because no json: ", sum(y))
                    if len(discarded_samples)- sum(y) > 2:
                        print("problem")
                    print(final_path)
                    by_model_results[model][prompt] = data
    #Remove empty dicts:
    by_model_results = {k: v for k, v in by_model_results.items() if v}
    relevant_metrics = ["overall_f1","overall_rec","overall_prec"]

    for model in by_model_results:
        data = {key: {metric: by_model_results[model][key][metric] for metric in relevant_metrics}
                for key in by_model_results[model]}
        sorted_outer_keys = sorted(data.keys())

        # Extract and sort inner dictionary keys alphabetically
        sorted_data = {
            "_".join(outer_key.split('_')[:-1]): {key: data[outer_key][key] for key in sorted(data[outer_key].keys())}
            for
            outer_key in
            sorted_outer_keys}
        plot_sorted_dict_of_dicts(sorted_data, model)


def compare_results_per_prompt(path,set_name,only_prompt_type=False,prompt_says_inconsistent = False):
    by_prompt_results = {}
    for model in os.listdir(path):
        new_path = os.path.join(path, model)
        for original_prompt in os.listdir(new_path):
            if prompt_says_inconsistent and "maybe_consistent" in original_prompt:
                continue
            if not prompt_says_inconsistent and "maybe_consistent" not in original_prompt:
                continue
            if only_prompt_type:
                prompt = "_".join(original_prompt.split('_')[:-3])
            else:
                prompt = "_".join(original_prompt.split('_')[:-2])
            final_path = os.path.join(new_path, original_prompt)
            if os.path.exists(os.path.join(final_path, set_name, 'results.json')):
                with open(os.path.join(final_path, set_name, 'results.json'), 'r') as f:
                    data = json.load(f)
                    if prompt not in by_prompt_results:
                        by_prompt_results[prompt] = {}
                        by_prompt_results[prompt][model] = {}
                    by_prompt_results[prompt][model] = data
    relevant_metrics = ["overall_prec","overall_rec", "overall_f1"]
    for prompt in by_prompt_results:
        data = {
            key: {metric: by_prompt_results[prompt][key][metric] for metric in relevant_metrics}
            for key in by_prompt_results[prompt]}
        sorted_outer_keys = sorted(data.keys())

        # Extract and sort inner dictionary keys alphabetically
        sorted_data = {
            outer_key: {key: data[outer_key][key] for key in sorted(data[outer_key].keys())}
            for
            outer_key in
            sorted_outer_keys}
        plot_sorted_dict_of_dicts(sorted_data, prompt)


def process_answers(answers):
    final_answers = []
    for answer in answers:
        if answer != ' ':
            final_answers += answer.split(',')
    final_answers = [x.strip() for x in final_answers]
    return final_answers


def get_results(path_gpt, path_claude, path_gemini, path_llama, original_dataset_path):
    df_gpt = pd.read_csv(path_gpt)
    df_claude = pd.read_csv(path_claude)
    df_gemini = pd.read_csv(path_gemini)
    df_llama = pd.read_csv(path_llama)
    dataset_df = pd.read_csv(original_dataset_path)
    dataset_df = (dataset_df.groupby(['text', 'model_summary'], sort=False)['explanation'].apply(list)).reset_index()
    dataset_df = dataset_df[dataset_df['model_summary'].isin(df_llama['model_summary'].tolist())]

    base_dataset_descriptions_amount = sum([len(x) for x in dataset_df['explanation'].tolist()])
    print("gpt")
    gpt_answers = process_answers(df_gpt['Correct_llm_answers'].tolist())
    print("claude")
    claude_answers = process_answers(df_claude['Correct_llm_answers'].tolist())
    print("gemini")
    gemini_answers = process_answers(df_gemini['Correct_llm_answers'].tolist())
    print("llama")
    llama_answers = process_answers(df_llama['Correct_llm_answers'].tolist())
    return gpt_answers, claude_answers, gemini_answers, llama_answers, base_dataset_descriptions_amount


def create_venn_diagram_no_zeros(path_to_dev_annotation):
    results = pd.read_csv(path_to_dev_annotation)
    gpt = []
    gemini = []
    claude = []
    for i in range(50):
        temp = results[results['Index'] == i]
        answers = temp['Answer'].tolist()
        if answers[0] != ' ':
            gpt += answers[0].split(',')
        if answers[1] != ' ':
            gemini += answers[1].split(',')
        if answers[2] != ' ':
            claude += answers[2].split(',')
    gpt = [x.strip() for x in gpt]
    gemini = [x.strip() for x in gemini]
    claude = [x.strip() for x in claude]

    venn = venn3([set(gpt), set(gemini), set(claude)], ('gpt', 'gemini', 'claude'))

    # Customize the diagram (optional)
    for subset in venn.subset_labels:
        if subset and subset.get_text() != "0":  # Only modify non-None labels
            subset.set_fontsize(10)
        else:
            subset.set_text('')
    # 4 is gpt and claude

    patch = venn.patches[4]
    path = patch.get_path()
    transform = patch.get_transform()

    # Remove the original patch
    patch.remove()

    # Create a new custom patch (e.g., a Circle)
    new_patch = PathPatch(path, transform=transform, facecolor=venn.patches[0].get_facecolor(), alpha=0.4,
                          edgecolor="none")

    # Add the new patch to the plot
    plt.gca().add_patch(new_patch)
    another_patch = PathPatch(venn.patches[4].get_path(), transform=venn.patches[3].get_transform(),
                              facecolor=venn.patches[3].get_facecolor(), alpha=0.1, edgecolor="none")
    plt.gca().add_patch(another_patch)
    # 5 is gemini and claude
    patch = venn.patches[5]
    path = patch.get_path()
    transform = patch.get_transform()

    # Remove the original patch
    patch.remove()

    # Create a new custom patch (e.g., a Circle)
    new_patch = PathPatch(path, transform=transform, facecolor=venn.patches[1].get_facecolor(), alpha=0.4,
                          edgecolor="none")

    # Add the new patch to the plot
    plt.gca().add_patch(new_patch)
    another_patch = PathPatch(venn.patches[5].get_path(), transform=venn.patches[3].get_transform(),
                              facecolor=venn.patches[3].get_facecolor(), alpha=0.1, edgecolor="none")
    plt.gca().add_patch(another_patch)
    plt.savefig("figs/venn.png")

    # Display the plot
    plt.show()
    all = sorted(set(gpt).union(set(gemini)).union(set(claude)))
    print(len(all))
    print(sorted(all))


def amount_of_unique_descriptions_per_amount_of_models(path_gpt, path_gemini, path_claude, path_llama,
                                                       original_dataset_path):
    gpt, claude, gemini, llama, base_dataset_descriptions_amount = get_results(path_gpt, path_claude, path_gemini,
                                                                               path_llama, original_dataset_path)
    sets = {
        'gpt': set(gpt),
        'gemini': set(gemini),
        'claude': set(claude),
        'llama': set(llama)
    }
    names = list(sets.keys())
    # Write a code which gets the biggest set, then find the union of 2 sets which results in the biggest set
    # Then find the union of the 3 sets which results in the biggest set
    # Then find the union of all 4 sets
    biggest_2_union = None
    len_biggest_2_union = 0
    biggest_3_union = None
    len_biggest_3_union = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            curr_union = sets[names[i]].union(sets[names[j]])
            if len(curr_union) > len_biggest_2_union:
                biggest_2_union = curr_union
                len_biggest_2_union = len(curr_union)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            for k in range(j + 1, len(names)):
                curr_union = sets[names[i]].union(sets[names[j]]).union(sets[names[k]])
                if len(curr_union) > len_biggest_3_union:
                    biggest_3_union = curr_union
                    len_biggest_3_union = len(curr_union)
    all_4_union = set(gpt).union(set(gemini)).union(set(claude)).union(set(llama))
    plt.bar(['Best model', 'Best 2 union', 'Best 3 union', 'All 4 union'],
            [len(sets[names[0]]) / base_dataset_descriptions_amount,
             len_biggest_2_union / base_dataset_descriptions_amount,
             len_biggest_3_union / base_dataset_descriptions_amount,
             len(all_4_union) / base_dataset_descriptions_amount])
    plt.title("Improvement in Coverage")
    plt.xlabel("Model combinations")
    plt.ylabel("Percentage Improvement in Samples Found")
    plt.savefig("figs/Improvement in Coverage.png")

    plt.show()
def amount_of_inconsistencies_in_each_summary():
    df = pd.read_csv("data/all_finalized_data/final/final_dataset.csv")
    descriptions = df['descriptions'].tolist()
    descriptions = [eval(x) for x in descriptions]
    descriptions_len = [len(x) for x in descriptions]
    from collections import Counter
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    counter = Counter(descriptions_len)
    counter = {k: v / sum(counter.values()) for k, v in counter.items()}

    plt.bar(counter.keys(), counter.values())
    plt.title("Inconsistencies per Summary")
    plt.xlabel("Amount of Inconsistencies")
    plt.ylabel("Percentage of Summaries")

    # Format y-axis as percentage with integer values
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
    plt.tight_layout()
    plt.savefig("figs/Inconsistencies per Summary.png")

    plt.show()


def get_files(gt_dir):
    models = []
    files_per_model = {}
    for subdir in os.listdir(gt_dir):
        models.append(subdir)
        subdir_path = os.path.join(gt_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if subdir not in files_per_model:
                    files_per_model[subdir] = []
                files_per_model[subdir].append(file.split('.')[0])
    return models, files_per_model

def gt_preprocess(gt_dir,pred_dir,model,prompt):
    from results_parser import transform_matching_to_recall_precision, transform_precision_recall, impute_dicts, \
        llm_judgment_to_precision_recall
    df_gt = pd.read_csv(f"{gt_dir}/{model}/{prompt}.csv")
    df_llm = pd.read_csv(
        f"{pred_dir}/{model}/{prompt}_consistent_gpt-4o-2024-11-20/dev/llm_judgment_results.csv")
    raw_gt_dicts = df_gt['gt'].tolist()
    gt_dicts = [eval(x) for x in raw_gt_dicts]
    gt_raw_descriptions = df_gt['raw_descriptions'].tolist()
    gt_raw_descriptions = [eval(x) for x in gt_raw_descriptions]
    precision_dicts, recall_dicts, _ = transform_matching_to_recall_precision(gt_dicts,
                                                                                    gt_raw_descriptions,
                                                                                    [])
    gt_precision_dicts, gt_recall_dicts = transform_precision_recall(precision_dicts, recall_dicts)
    gt_precision_dicts, gt_recall_dicts = impute_dicts(gt_precision_dicts, gt_recall_dicts, [])
    judgement = df_llm['judgment_output'].tolist()
    pred_precision_dicts, pred_recall_dicts, discard = llm_judgment_to_precision_recall(judgement,gt_raw_descriptions)
    pred_precision_dicts, pred_recall_dicts = impute_dicts(pred_precision_dicts, pred_recall_dicts, discard)
    return gt_precision_dicts, gt_recall_dicts, pred_precision_dicts, pred_recall_dicts,discard


def compute(gt_dicts,pred_dicts,llm_no_judgment):
    total = sum([len(gt_dicts[i]) for i in range(len(gt_dicts)) if i not in llm_no_judgment])
    real_tp = sum([sum(gt_dicts[i].values()) for i in range(len(gt_dicts)) if i not in llm_no_judgment])
    pred_tp = sum([sum(pred_dicts[i].values()) for i in range(len(pred_dicts)) if i not in llm_no_judgment])
    return real_tp / total, pred_tp / total

def gt_to_pred_llm_judgment(gt_dir, pred_dir):
    models, files_per_model = get_files(gt_dir)
    results_per_model = {}
    results_per_prompt = {}
    for model in models:
        results_per_model[model] = {"precision_gt": [], "recall_gt": [], "precision_pred": [], "recall_pred": []}
        for prompt in files_per_model[model]:
            gt_precision_dicts, gt_recall_dicts, pred_precision_dicts, pred_recall_dicts,llm_no_judgment = gt_preprocess(gt_dir,pred_dir,model,prompt)
            prompt = "_".join(prompt.split('_')[:-2])
            if prompt not in results_per_prompt:
                results_per_prompt[prompt] = {"precision_gt": [], "recall_gt": [], "precision_pred": [], "recall_pred": []}
            gt_precision, pred_precision = compute(gt_precision_dicts, pred_precision_dicts, llm_no_judgment)
            results_per_model[model]['precision_gt'].append(gt_precision)
            results_per_model[model]['precision_pred'].append(pred_precision)
            results_per_prompt[prompt]['precision_gt'].append(gt_precision)
            results_per_prompt[prompt]['precision_pred'].append(pred_precision)
            gt_recall, pred_recall = compute(gt_recall_dicts, pred_recall_dicts, llm_no_judgment)
            results_per_model[model]['recall_gt'].append(gt_recall)
            results_per_model[model]['recall_pred'].append(pred_recall)
            results_per_prompt[prompt]['recall_gt'].append(gt_recall)
            results_per_prompt[prompt]['recall_pred'].append(pred_recall)
    for model in results_per_model:
        gt_recall = results_per_model[model]['recall_gt']
        gt_precision = results_per_model[model]['precision_gt']
        gt_f1 = [2 * gt_recall[i] * gt_precision[i] / (gt_recall[i] + gt_precision[i]) for i in range(len(gt_recall))]
        pred_recall = results_per_model[model]['recall_pred']
        pred_precision = results_per_model[model]['precision_pred']
        pred_f1 = [2 * pred_recall[i] * pred_precision[i] / (pred_recall[i] + pred_precision[i]) for i in
                   range(len(pred_recall))]
        results_per_model[model]['f1_gt'] = gt_f1
        results_per_model[model]['f1_pred'] = pred_f1
    for prompt in results_per_prompt:
        gt_recall = results_per_prompt[prompt]['recall_gt']
        gt_precision = results_per_prompt[prompt]['precision_gt']
        gt_f1 = [2 * gt_recall[i] * gt_precision[i] / (gt_recall[i] + gt_precision[i]) for i in range(len(gt_recall))]
        pred_recall = results_per_prompt[prompt]['recall_pred']
        pred_precision = results_per_prompt[prompt]['precision_pred']
        pred_f1 = [2 * pred_recall[i] * pred_precision[i] / (pred_recall[i] + pred_precision[i]) for i in
                   range(len(pred_recall))]
        results_per_prompt[prompt]['f1_gt'] = gt_f1
        results_per_prompt[prompt]['f1_pred'] = pred_f1

    mean_results_per_model = {}



    colors = {"recall": "tomato", "precision": "royalblue", "f1": "green"}
    metric_colors_plot = {
        "Predicted Recall": colors["recall"],
        "GT Recall": colors["recall"],
        "Predicted Precision": colors["precision"],
        "GT Precision": colors["precision"],
        "Predicted F1": colors["f1"],
        "GT F1": colors["f1"]
    }
    graph_model_names = {"gpt-4o-2024-11-20": "GPT 4o", "gemini-1.5-pro": "Gemini 1.5 pro",
                            "claude-3-5-sonnet-20241022": "Claude sonnet 3.5", "llama3.1-405b": "Llama 3.1 405B"}
    for key in results_per_model:
        new_key = graph_model_names[key]
        mean_results_per_model[new_key] = {}
        for key2 in results_per_model[key]:
            mean_results_per_model[new_key][key2] = sum(results_per_model[key][key2]) / len(results_per_model[key][key2])
    plot_recall_precision(mean_results_per_model,metric_colors_plot,"Predicted vs Ground Truth automatic judgment per model")
    # legend_patches = [mpatches.Patch(color=mcolors.to_rgba(color, 0.7), label="Predicted_"+ metric)for metric, color in colors.items()]
    # legend_patches += [mpatches.Patch(color=mcolors.to_rgba(color, 1.0), label="GT_"+ metric)for metric, color in colors.items()]
    # fig.legend(handles=legend_patches, loc="upper right", ncol=2, fontsize=12)
    # plt.tight_layout(rect=[0, 0, 1, 1])
    # plt.savefig("figs/Predicted vs Ground Truth automatic judgment per model.png")
    # plt.show()


    graph_prompt_names={"few_shot":"Few shot","zero_shot": "Zero shot","cot":"CoT"}
    mean_results_per_prompt = {}
    for key in results_per_prompt:
        new_key = graph_prompt_names[key]
        mean_results_per_prompt[new_key] = {}
        for key2 in results_per_prompt[key]:
            mean_results_per_prompt[new_key][key2] = sum(results_per_prompt[key][key2]) / len(results_per_prompt[key][key2])
    plot_recall_precision(mean_results_per_prompt, metric_colors_plot,
                          "Predicted vs Ground Truth automatic judgment per prompt")
    #
    # fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    #
    # plot_recall_precision(axes,mean_results_per_prompt,"Prompts",metric_colors_plot)
    # fig.legend(handles=legend_patches, loc="upper right", ncol=2, fontsize=12)
    # #fig.suptitle("Predicted vs Ground Truth" , fontsize=28)
    # plt.tight_layout(rect=[0, 0, 1, 0.9])
    # plt.savefig("figs/Predicted vs Ground Truth automatic judgment per prompt.png")
    # plt.show()

def plot_recall_precision(data,metric_colors,save_path):
    # Labels for bars

    # Set bar positions
    x = np.arange(len(data))
    width = 0.1  # Width of bars

    metric_values = {
        "Predicted Recall": [values['recall_pred'] for values in data.values()],
        "Predicted Precision": [values['precision_pred'] for values in data.values()],
        "Predicted F1": [values['f1_pred'] for values in data.values()],
        "GT Recall": [values['recall_gt'] for values in data.values()],
        "GT Precision": [values['precision_gt'] for values in data.values()],
        "GT F1": [values['f1_gt'] for values in data.values()],
    }
    for i, metric in enumerate(metric_colors.keys()):
        plt.bar(x + (i - 2.5) * width, metric_values[metric], width,
               color=metric_colors[metric], alpha=0.7 if "Predicted" in metric else 1.0,label = metric)

    plt.xticks(np.arange(len(data)), data.keys(),fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=10,loc="upper right",ncol=3)
    plt.tight_layout()
    plt.savefig(f"figs/{save_path}.png")
    plt.show()


def main():
    amount_of_inconsistencies_in_each_summary()
    #compare_results_per_model('data/results',"dev")
    #compare_results_per_prompt('data/results',"dev",only_prompt_type=True,prompt_says_inconsistent = False)
    #main_annotation_path = "data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6"
    # amount_of_unique_descriptions_per_amount_of_models(os.path.join(main_annotation_path, "results_gpt-4o_dev_annotated.csv"),
    #                                                    os.path.join(main_annotation_path,
    #                                                                 "results_gemini-1.5-pro_dev_annotated.csv"),
    #                                                    os.path.join(main_annotation_path,
    #                                                                 "results_claude-sonnet-3.5_dev_annotated.csv"),
    #                                                    os.path.join(main_annotation_path,
    #                                                                 "results_llama_3.1_405_dev_annotated.csv"),
    #                                                    "data/all_finalized_data/Final_manual_dataset.csv")
    #gt_to_pred_llm_judgment("data/gt_for_judgment","data/results")


if __name__ == "__main__":
    main()

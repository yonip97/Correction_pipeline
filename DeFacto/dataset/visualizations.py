import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib.patches import PathPatch
import numpy as np
import matplotlib.ticker as mticker
from sklearn.metrics import precision_score, recall_score, f1_score


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
    print(tabulate(df.T * 100, headers='keys', tablefmt='psql'))

    # Plot the bar chart
    df.plot(kind="bar", figsize=(10, 6))

    # Customize plot
    plt.ylabel("Values")
    plt.title(model_name)
    plt.legend()
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Show plot
    plt.show()


def compare_results_per_model(path, set_name, prompt_says_inconsistent=False):
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
                    print("number of discarded samples: ", len(discarded_samples), "out of", len(df))
                    judgments = x['judgment_outputs'].tolist()
                    y = ['{' not in judgment for judgment in judgments]
                    print("number of discarded samples because no json: ", sum(y))
                    if len(discarded_samples) - sum(y) > 2:
                        print("problem")
                    print(final_path)
                    by_model_results[model][prompt] = data
    # Remove empty dicts:
    by_model_results = {k: v for k, v in by_model_results.items() if v}
    relevant_metrics = ["overall_f1", "overall_rec", "overall_prec"]

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


def compare_results_per_prompt(path, set_name, only_prompt_type=False, prompt_says_inconsistent=False,
                               best_prompt_variation=False):
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
    if best_prompt_variation:
        final = {}
        for prompt in by_prompt_results:
            new_prompt = "_".join(prompt.split('_')[:-1])
            if new_prompt not in final:
                final[new_prompt] = {}
            for model in by_prompt_results[prompt]:
                if model not in final[new_prompt]:
                    final[new_prompt][model] = by_prompt_results[prompt][model]
                else:
                    if final[new_prompt][model]['overall_f1'] < by_prompt_results[prompt][model]['overall_f1']:
                        final[new_prompt][model] = by_prompt_results[prompt][model]
        by_prompt_results = final
    relevant_metrics = ["overall_prec", "overall_rec", "overall_f1"]
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
        'GPT-4o': set(gpt),
        'Gemini-1.5-pro': set(gemini),
        'Claude-sonnet-3.5': set(claude),
        'Llama-3.1-405B': set(llama)
    }
    names = list(sets.keys())
    # Write a code which gets the biggest set, then find the union of 2 sets which results in the biggest set
    # Then find the union of the 3 sets which results in the biggest set
    # Then find the union of all 4 sets
    len_biggest_2_union = 0
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
    plt.figure(figsize=(8, 6))
    plt.bar(["Original", '+Best model', '+Best 2 union', '+Best 3 union', '+All 4 union'],
            [1, 1 + len(sets[names[0]]) / base_dataset_descriptions_amount,
             1 + len_biggest_2_union / base_dataset_descriptions_amount,
             1 + len_biggest_3_union / base_dataset_descriptions_amount,
             1 + len(all_4_union) / base_dataset_descriptions_amount], width=0.7)
    plt.title("Improvement in Coverage")
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.ylabel("Percentage of Improvement")
    plt.tight_layout()
    plt.savefig("figs/Improvement in Coverage.png")

    plt.show()
    model_order = ["Original", 'Claude-sonnet-3.5', 'Gemini-1.5-pro', 'Llama-3.1-405B', 'GPT-4o']
    # plt.bar(["Original", '+GPT-4O', '+Claude-sonnet-3.5', '+Gemini-1.5-pro', '+llama-3.1-405B'],
    #         [1, 1 + len(sets[names[0]]) / base_dataset_descriptions_amount,
    #             1 + len(sets[names[1]]) / base_dataset_descriptions_amount,
    #             1 + len(sets[names[2]]) / base_dataset_descriptions_amount,
    #             1 + len(sets[names[3]]) / base_dataset_descriptions_amount], width=0.7)
    plt.bar(model_order, [1] + [1 + len(sets[model_order[i]]) / base_dataset_descriptions_amount for i in
                                range(1, len(model_order))])
    plt.xticks(fontsize=12, rotation=45)
    # plt.title("Coverage of each model")
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(1))
    plt.ylabel("Percentage of Improvement", fontsize=12)
    plt.tight_layout()
    plt.savefig("figs/Coverage of each model.png")
    plt.show()


def amount_of_inconsistencies_in_each_summary():
    df = pd.read_csv("data/all_finalized_data/final/final_dataset_dev.csv")
    df2 = pd.read_csv("data/all_finalized_data/final/final_dataset_test.csv")
    df = pd.concat([df, df2])
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


def gt_preprocess(gt_dir, pred_dir, model, prompt):
    from results_parser import transform_matching_to_recall_precision, transform_precision_recall, impute_dicts, \
        llm_judgment_to_precision_recall,llm_judgment_one_to_one_to_precision_recall
    df_gt = pd.read_csv(f"{gt_dir}/{model}/{prompt}.csv")
    # df_llm = pd.read_csv(
    #     f"{pred_dir}/{model}/{prompt}_consistent_gpt-4o-2024-11-20/dev/llm_judgment_results.csv")
    df_llm = pd.read_csv(f"{pred_dir}/{model}/{prompt.replace('_maybe','')}_judgment_results.csv")
    df_llm = df_llm[:60]
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
    pred_precision_dicts, pred_recall_dicts, discard = llm_judgment_one_to_one_to_precision_recall(judgement, gt_raw_descriptions)
    pred_precision_dicts, pred_recall_dicts = impute_dicts(pred_precision_dicts, pred_recall_dicts, discard)
    return gt_precision_dicts, gt_recall_dicts, pred_precision_dicts, pred_recall_dicts, discard


def compute(gt_dicts, pred_dicts, llm_no_judgment):
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
            gt_precision_dicts, gt_recall_dicts, pred_precision_dicts, pred_recall_dicts, llm_no_judgment = gt_preprocess(
                gt_dir, pred_dir, model, prompt)
            prompt = "_".join(prompt.split('_')[:-2])
            if prompt not in results_per_prompt:
                results_per_prompt[prompt] = {"precision_gt": [], "recall_gt": [], "precision_pred": [],
                                              "recall_pred": []}
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
            mean_results_per_model[new_key][key2] = sum(results_per_model[key][key2]) / len(
                results_per_model[key][key2])
    dict_order = ["Claude sonnet 3.5", "GPT 4o", "Gemini 1.5 pro", "Llama 3.1 405B"]
    mean_results_per_model = {k: mean_results_per_model[k] for k in dict_order}
    plot_recall_precision(mean_results_per_model, metric_colors_plot,
                          "Predicted vs Ground Truth automatic judgment per model")

    graph_prompt_names = {"few_shot": "Few shot", "zero_shot": "Zero shot", "cot": "CoT"}
    mean_results_per_prompt = {}
    for key in results_per_prompt:
        new_key = graph_prompt_names[key]
        mean_results_per_prompt[new_key] = {}
        for key2 in results_per_prompt[key]:
            mean_results_per_prompt[new_key][key2] = sum(results_per_prompt[key][key2]) / len(
                results_per_prompt[key][key2])
    prompts_order = ["Zero shot", "Few shot", "CoT"]
    mean_results_per_prompt = {k: mean_results_per_prompt[k] for k in prompts_order}
    plot_recall_precision(mean_results_per_prompt, metric_colors_plot,
                          "Predicted vs Ground Truth automatic judgment per prompt")


def how_many_mistakes_total(gt_dir, pred_dir):
    models, files_per_model = get_files(gt_dir)
    models = ["gemini-1.5-pro"]
    results = {}
    for model in models:
        print(model)
        results[model] = {}
        for prompt in files_per_model[model]:
            print(prompt)
            gt_precision_dicts, gt_recall_dicts, pred_precision_dicts, pred_recall_dicts, llm_no_judgment = gt_preprocess(
                gt_dir, pred_dir, model, prompt)
            prompt = "_".join(prompt.split('_')[:-2])
            results[model][prompt] = {"precision": {"total": 0, "mistakes": 0}, "recall": {"total": 0, "mistakes": 0}}
            for i, (pred_entry, gt_entry) in enumerate(zip(pred_precision_dicts, gt_precision_dicts)):
                for key in gt_entry:
                    if key not in pred_entry or pred_entry[key] != gt_entry[key]:
                        results[model][prompt]["precision"]["mistakes"] += 1
                    results[model][prompt]["precision"]["total"] += 1
                if pred_entry is not None:
                    for key in pred_entry:
                        if key not in gt_entry:
                            results[model][prompt]["precision"]["mistakes"] += 1
            for pred_entry, gt_entry in zip(pred_recall_dicts, gt_recall_dicts):
                for key in gt_entry:
                    if key not in pred_entry or pred_entry[key] != gt_entry[key]:
                        results[model][prompt]["recall"]["mistakes"] += 1
                    results[model][prompt]["recall"]["total"] += 1
                if pred_entry is not None:
                    for key in pred_entry:
                        if key not in gt_entry:
                            print(key)

                            results[model][prompt]["recall"]["mistakes"] += 1

    print("Total")
    final = {"precision": {"total": 0, "mistakes": 0}, "recall": {"total": 0, "mistakes": 0}}
    for model in results:
        print(model)
        for prompt in results[model]:
            final["precision"]["total"] += results[model][prompt]["precision"]["total"]
            final["precision"]["mistakes"] += results[model][prompt]["precision"]["mistakes"]
            final["recall"]["total"] += results[model][prompt]["recall"]["total"]
            final["recall"]["mistakes"] += results[model][prompt]["recall"]["mistakes"]
    print(results)
    print(final)


def plot_recall_precision(data, metric_colors, save_path):
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
                color=metric_colors[metric], alpha=0.7 if "Predicted" in metric else 1.0, label=metric)

    plt.xticks(np.arange(len(data)), data.keys(), fontsize=12)
    plt.ylim(0, 1)
    plt.legend(fontsize=10, loc="upper right", ncol=3)
    plt.tight_layout()
    plt.savefig(f"figs/{save_path}.png")
    plt.show()


def collate_files(path):
    matching_paths = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            # Check if all target keywords are in the full path
            if "maybe" in dirpath and "test" == dirpath.split('/')[-1]:
                if "results.csv" == filename:
                    matching_paths.append(full_path)
    return matching_paths


def collect_binary_results(paths):
    results = {}
    for path in paths:
        model = path.split('/')[2]
        prompt = " ".join(path.split('/')[3].split('_')[:-4])
        if model not in results:
            results[model] = {}
        results[model][prompt] = {}
        df = pd.read_csv(path)
        df.dropna(subset=['recall'], inplace=True)
        binary_gt = [1 if x != "[]" else 0 for x in df['raw_descriptions'].tolist()]
        binary_predicted = [1 if len(eval(x)) != 0 else 0 for x in df['precision'].tolist()]
        results[model][prompt]['precision'] = precision_score(binary_gt, binary_predicted)
        results[model][prompt]['recall'] = recall_score(binary_gt, binary_predicted)
        results[model][prompt]['f1'] = f1_score(binary_gt, binary_predicted)
    return results


def collect_fine_grain_results(paths):
    results = {}
    for path in paths:
        model = path.split('/')[2]
        prompt = " ".join(path.split('/')[3].split('_')[:-4])
        if model not in results:
            results[model] = {}
        results[model][prompt] = {}
        df = pd.read_csv(path)
        df.dropna(subset=['recall'], inplace=True)
        recall = df['recall'].tolist()
        recall = [eval(x) for x in recall]
        precision = df['precision'].tolist()
        precision = [eval(x) for x in precision]
        results[model][prompt]['precision'] = sum([sum(x.values()) for x in precision]) / sum(
            [len(x) for x in precision])
        results[model][prompt]['recall'] = sum([sum(x.values()) for x in recall]) / sum(
            [len(x) for x in recall])
        results[model][prompt]['f1'] = 2 * results[model][prompt]['precision'] * \
                                       results[model][prompt]['recall'] / ( \
                                                   results[model][prompt]['precision'] +
                                                   results[model][prompt]['recall'])
    return results


def turn_to_binary(results_path, prompt_type):
    matching_paths = collate_files(results_path)
    binary_results = collect_binary_results(matching_paths)
    fine_grain_results = collect_fine_grain_results(matching_paths)

    fine_grain_results = {k: v[prompt_type] for k, v in fine_grain_results.items()}
    binary_results = {k: v[prompt_type] for k, v in binary_results.items()}
    models = list(fine_grain_results.keys())
    metrics = ['f1', 'precision', 'recall']
    spacing = 0.75
    x = np.arange(len(metrics)) * spacing
    bar_width = 0.3

    # Create 4 subplots in one row
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    models_mapping = {"gpt-4o-2024-11-20": "GPT 4o", "gemini-1.5-pro": "Gemini 1.5 pro",
                      "claude-3-5-sonnet-20241022": "Claude sonnet 3.5", "llama3.1-405b": "Llama 3.1 405B"}
    # Colors
    colors = ['#003366', 'skyblue']

    for i, model in enumerate(models):
        j = i // 2
        i = i % 2
        ax = axes[i][j]

        binary_vals = [binary_results[model][m] for m in metrics]
        fine_vals = [fine_grain_results[model][m] for m in metrics]

        # Plot bars
        ax.bar(x - bar_width / 2, binary_vals, bar_width, label='Binary', color=colors[0])
        ax.bar(x + bar_width / 2, fine_vals, bar_width, label='Fine-Grained', color=colors[1])

        # Set labels and title
        for j in range(len(metrics)):
            b = binary_vals[j]
            f = fine_vals[j]
            percent = 100 * (f - b) / b
            text = f'{percent:+.0f}%'  # e.g., '+15%', '-32%'
            max_height = max(b, f)
            ax.text(x[j]+bar_width / 2, max_height + 0.03, text, ha='center', va='bottom', fontsize=18)

            # Labels
        ax.set_title(models_mapping[model], fontsize=24)
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=20)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis='y', labelsize=16)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.annotate('', xy=(x[-1] + bar_width * 1.2, 0), xytext=(x[0] - bar_width, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

        # Y-axis arrow
        ax.annotate('', xy=(-bar_width, 1.1), xytext=(-bar_width, 0),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    # # Add legend to the first subplot
    fig.legend(['Binary', 'Fine-Grained'], loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=2, frameon=False,
               fontsize=24, columnspacing=5)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.9, wspace=0.1, hspace=0.5)

    # plt.tight_layout()
    plt.savefig("figs/binary_to_fine_comparison.png")
    plt.show()


def dict_to_multicolumn_table(data: dict) -> pd.DataFrame:
    """
    Converts a nested dictionary into a pandas DataFrame
    with multi-level columns (main key -> sub-metrics).

    Args:
        data (dict): The nested dictionary of format
            {key: {subkey: {metric: value}}}

    Returns:
        pd.DataFrame: Formatted table.
    """
    # Infer the ordering
    keys = list(data.keys())
    first_key = keys[0]
    subkeys = list(data[first_key].keys())
    metrics = list(data[first_key][subkeys[0]].keys())

    # Build column MultiIndex
    main_cols = []
    for key in keys:
        for metric in metrics:
            main_cols.append((key, metric))
    multi_index = pd.MultiIndex.from_tuples(main_cols)

    # Build the rows
    rows = []
    for subkey in subkeys:
        row = []
        for key in keys:
            metric_values = data[key][subkey]
            row.extend([metric_values[metric] for metric in metrics])
        rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows, index=subkeys, columns=multi_index)
    return df


def creat_tt(results, chosen, new_split):
    df_tt = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_trueteacher.csv")
    df_tt = df_tt[df_tt["scores"] <= 0.95]
    for model in results:
        if new_split:
            path = f"data/results_new_split_judge_1_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        else:
            path = f"data/results_new_split_judge_1_multiple_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(os.path.join(path, "results.csv"))
            df.dropna(subset=['recall'], inplace=True)
            total_gt_samples = sum([len(eval(x)) for x in df['raw_descriptions'].tolist()])
            df = df[(df['text'].isin(df_tt['text'].tolist())) & (
                df['model_summary'].isin(df_tt['model_summary'].tolist()))]
            recall = [eval(x) for x in df['recall'].tolist()]
            recall = sum([sum(x.values()) for x in recall]) / total_gt_samples
            precision = [eval(x) for x in df['precision'].tolist()]
            precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
            f1 = 2 * recall * precision / (recall + precision)
            results[model][f"TT+cot"] = {}
            results[model][f"TT+cot"]['f1'] = round(f1 * 100, 1)
            results[model][f"TT+cot"]['precision'] = round(precision * 100, 1)
            results[model][f"TT+cot"]['recall'] = round(recall * 100, 1)
        except:
            results[model][f"TT+cot"] = {}
            results[model][f"TT+cot"]['f1'] = 0
            results[model][f"TT+cot"]['precision'] = 0
            results[model][f"TT+cot"]['recall'] = 0
    return results
def creat_tt_with_inconsistent_option(results, chosen, new_split):
    df_tt = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_trueteacher.csv")
    df_tt = df_tt[df_tt["scores"] <= 0.95]
    for model in results:
        if new_split:
            path = f"data/results_new_split_judge_1_to_1/{model}/cot_{chosen[model]}_maybe_consistent_gpt-4o-2024-11-20/test"
        else:
            path = f"data/results_new_split_judge_1_multiple_to_1/{model}/cot_{chosen[model]}_maybe_consistent_gpt-4o-2024-11-20/test"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(os.path.join(path, "results.csv"))
            df.dropna(subset=['recall'], inplace=True)
            total_gt_samples = sum([len(eval(x)) for x in df['raw_descriptions'].tolist()])
            df = df[(df['text'].isin(df_tt['text'].tolist())) & (
                df['model_summary'].isin(df_tt['model_summary'].tolist()))]
            recall = [eval(x) for x in df['recall'].tolist()]
            recall = sum([sum(x.values()) for x in recall]) / total_gt_samples
            precision = [eval(x) for x in df['precision'].tolist()]
            precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
            f1 = 2 * recall * precision / (recall + precision)
            results[model][f"TT+cot_not_consistent"] = {}
            results[model][f"TT+cot_not_consistent"]['f1'] = round(f1 * 100, 1)
            results[model][f"TT+cot_not_consistent"]['precision'] = round(precision * 100, 1)
            results[model][f"TT+cot_not_consistent"]['recall'] = round(recall * 100, 1)
        except:
            results[model][f"TT+cot_not_consistent"] = {}
            results[model][f"TT+cot_not_consistent"]['f1'] = 0
            results[model][f"TT+cot_not_consistent"]['precision'] = 0
            results[model][f"TT+cot_not_consistent"]['recall'] = 0
    return results

def create_seahorse(results, chosen, new_split):
    df_seahorse = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_seahorse.csv")
    df_seahorse = df_seahorse[df_seahorse["scores"] <= 0.9]
    for model in results:
        if new_split:
            path = f"data/results_new_split_judge_1_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        else:
            path = f"data/results_new_split_judge_1_multiple_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(os.path.join(path, "results.csv"))
            df.dropna(subset=['recall'], inplace=True)
            total_gt_samples = sum([len(eval(x)) for x in df['raw_descriptions'].tolist()])
            df = df[(df['text'].isin(df_seahorse['text'].tolist())) & (
                df['model_summary'].isin(df_seahorse['model_summary'].tolist()))]
            recall = [eval(x) for x in df['recall'].tolist()]
            recall = sum([sum(x.values()) for x in recall]) / total_gt_samples
            precision = [eval(x) for x in df['precision'].tolist()]
            precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
            f1 = 2 * recall * precision / (recall + precision)
            results[model][f"Seahorse+cot"] = {}
            results[model][f"Seahorse+cot"]['f1'] = round(f1 * 100, 1)
            results[model][f"Seahorse+cot"]['precision'] = round(precision * 100, 1)
            results[model][f"Seahorse+cot"]['recall'] = round(recall * 100, 1)
        except:
            results[model][f"Seahorse+cot"] = {}
            results[model][f"Seahorse+cot"]['f1'] = 0
            results[model][f"Seahorse+cot"]['precision'] = 0
            results[model][f"Seahorse+cot"]['recall'] = 0
    return results


def model_based_classifier(results, chosen, new_split, classifier_model):
    if new_split:
        classifier_path = f"data/results_new_split_judge_1_to_1/{classifier_model}/cot_{chosen[classifier_model]}_maybe_consistent_gpt-4o-2024-11-20/test"
    else:
        classifier_path = f"data/results_first_split/{classifier_model}/cot_{chosen[classifier_model]}_gpt-4o-2024-11-20/test"
    df_classifier = pd.read_csv(os.path.join(classifier_path, "results.csv"))
    df_classifier["precision"].fillna("{}", inplace=True)
    df_classifier = df_classifier[[len(eval(x)) != 0 for x in df_classifier['precision'].tolist()]]
    for model in results:
        if new_split:
            path = f"data/results_new_split_judge_1_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        else:
            path = f"data/results_first_split/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(os.path.join(path, "results.csv"))
            df.dropna(subset=['recall'], inplace=True)
            total_gt_samples = sum([len(eval(x)) for x in df['raw_descriptions'].tolist()])
            df = df[(df['text'].isin(df_classifier['text'].tolist())) & (
                df['model_summary'].isin(df_classifier['model_summary'].tolist()))]
            recall = [eval(x) for x in df['recall'].tolist()]
            recall = sum([sum(x.values()) for x in recall]) / total_gt_samples
            precision = [eval(x) for x in df['precision'].tolist()]
            precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
            f1 = 2 * recall * precision / (recall + precision)
            results[model][f"cot+{classifier_model}"] = {}
            results[model][f"cot+{classifier_model}"]['f1'] = round(f1 * 100, 1)
            results[model][f"cot+{classifier_model}"]['precision'] = round(precision * 100, 1)
            results[model][f"cot+{classifier_model}"]['recall'] = round(recall * 100, 1)
        except:
            results[model][f"cot+{classifier_model}"] = {}
            results[model][f"cot+{classifier_model}"]['f1'] = 0
            results[model][f"cot+{classifier_model}"]['precision'] = 0
            results[model][f"cot+{classifier_model}"]['recall'] = 0
    return results


def create_oracle(results, chosen, new_split):
    for model in results:
        if new_split:
            path = f"data/results_new_split_judge_1_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        else:
            path = f"data/results_new_split_judge_1_multiple_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(os.path.join(path, "results.csv"))
            df.dropna(subset=['recall'], inplace=True)
            df = df[[len(eval(x)) != 0 for x in df['raw_descriptions']]]
            recall = [eval(x) for x in df['recall'].tolist()]
            recall = sum([sum(x.values()) for x in recall]) / sum([len(x) for x in recall])
            precision = [eval(x) for x in df['precision'].tolist()]
            precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
            f1 = 2 * recall * precision / (recall + precision)
            results[model][f"Oracle+cot"] = {}
            results[model][f"Oracle+cot"]['f1'] = round(f1 * 100, 1)
            results[model][f"Oracle+cot"]['precision'] = round(precision * 100, 1)
            results[model][f"Oracle+cot"]['recall'] = round(recall * 100, 1)
        except:
            results[model][f"Oracle+cot"] = {}
            results[model][f"Oracle+cot"]['f1'] = 0
            results[model][f"Oracle+cot"]['precision'] = 0
            results[model][f"Oracle+cot"]['recall'] = 0
    return results


def always_inconsistent(results, chosen, new_split):
    for model in results:
        if new_split:
            path = f"data/results_new_split_judge_1_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        else:
            path = f"data/results_first_split/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(os.path.join(path, "results.csv"))
            df.dropna(subset=['recall'], inplace=True)
            recall = [eval(x) for x in df['recall'].tolist()]
            recall = sum([sum(x.values()) for x in recall]) / sum([len(x) for x in recall])
            precision = [eval(x) for x in df['precision'].tolist()]
            precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
            f1 = 2 * recall * precision / (recall + precision)
            results[model][f"always_inconsistent"] = {}
            results[model][f"always_inconsistent"]['f1'] = round(f1 * 100, 1)
            results[model][f"always_inconsistent"]['precision'] = round(precision * 100, 1)
            results[model][f"always_inconsistent"]['recall'] = round(recall * 100, 1)
        except:
            results[model][f"always_inconsistent"] = {}
            results[model][f"always_inconsistent"]['f1'] = 0
            results[model][f"always_inconsistent"]['precision'] = 0
            results[model][f"always_inconsistent"]['recall'] = 0
    return results


def simple_ensemble(results, chosen, new_split):
    df_seahorse = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_seahorse.csv")
    df_tt = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_trueteacher.csv")
    df_tt['scores'] = (df_seahorse['scores'] + df_tt['scores']) / 2
    df_tt = df_tt[df_tt["scores"] <= 0.91]
    for model in results:
        if new_split:
            path = f"data/results_new_split_judge_1_to_1/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        else:
            path = f"data/results_first_split/{model}/cot_{chosen[model]}_gpt-4o-2024-11-20/test"
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(os.path.join(path, "results.csv"))
            df.dropna(subset=['recall'], inplace=True)
            total_gt_samples = sum([len(eval(x)) for x in df['raw_descriptions'].tolist()])
            df = df[(df['text'].isin(df_tt['text'].tolist())) & (
                df['model_summary'].isin(df_tt['model_summary'].tolist()))]
            recall = [eval(x) for x in df['recall'].tolist()]
            recall = sum([sum(x.values()) for x in recall]) / total_gt_samples
            precision = [eval(x) for x in df['precision'].tolist()]
            precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
            f1 = 2 * recall * precision / (recall + precision)
            results[model][f"ensemble+cot"] = {}
            results[model][f"ensemble+cot"]['f1'] = round(f1 * 100, 1)
            results[model][f"ensemble+cot"]['precision'] = round(precision * 100, 1)
            results[model][f"ensemble+cot"]['recall'] = round(recall * 100, 1)
        except:
            results[model][f"ensemble+cot"] = {}
            results[model][f"ensemble+cot"]['f1'] = 0
            results[model][f"ensemble+cot"]['precision'] = 0
            results[model][f"ensemble+cot"]['recall'] = 0
    return results

def not_chosen_prompts(results, chosen, new_split):
    models = ["gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", "llama3.1-405b"]
    prompts = ["zero_shot", "few_shot", "cot"]
    for model in models:
        for prompt in prompts:
            results[model][prompt+" not chosen"] = {}
            for i in [1, 2]:
                try:
                    if new_split:
                        path = f"data/results_new_split_judge_1_to_1/{model}/{prompt}_{i}_maybe_consistent_gpt-4o-2024-11-20/test_not_chosen"
                    else:
                        path = f"data/results_first_split/{model}/{prompt}_{i}_maybe_consistent_gpt-4o-2024-11-20/test_not_chosen"
                    with open(os.path.join(path, "results.json"), 'r') as f:
                        data = json.load(f)
                        results[model][prompt+" not chosen"]['f1'] = round(data['overall_f1'] * 100, 1)
                        results[model][prompt+" not chosen"]['precision'] = round(data['overall_prec'] * 100, 1)
                        results[model][prompt+" not chosen"]['recall'] = round(data['overall_rec'] * 100, 1)
                    break
                except:
                    results[model][prompt+" not chosen"]['f1'] = 0
                    results[model][prompt+" not chosen"]['precision'] = 0
                    results[model][prompt+" not chosen"]['recall'] = 0
    return results
def not_chosen_prompts_with_tt(results, chosen, new_split):
    df_tt = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_trueteacher.csv")
    df_tt = df_tt[df_tt["scores"] <= 0.95]
    for i in [1,2]:
        for model in results:
            if new_split:
                path = f"data/results_new_split_judge_1_to_1/{model}/cot_{i}_gpt-4o-2024-11-20/test_not_chosen"
            else:
                path = f"data/results_first_split/{model}/cot_{i}_gpt-4o-2024-11-20/test_not_chosen"
            try:
                df = pd.read_csv(os.path.join(path, "results.csv"))
                df.dropna(subset=['recall'], inplace=True)
                total_gt_samples = sum([len(eval(x)) for x in df['raw_descriptions'].tolist()])
                df = df[(df['text'].isin(df_tt['text'].tolist())) & (
                    df['model_summary'].isin(df_tt['model_summary'].tolist()))]
                recall = [eval(x) for x in df['recall'].tolist()]
                recall = sum([sum(x.values()) for x in recall]) / total_gt_samples
                precision = [eval(x) for x in df['precision'].tolist()]
                precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
                f1 = 2 * recall * precision / (recall + precision)
                results[model][f"TT+cot not chosen"] = {}
                results[model][f"TT+cot not chosen"]['f1'] = round(f1 * 100, 1)
                results[model][f"TT+cot not chosen"]['precision'] = round(precision * 100, 1)
                results[model][f"TT+cot not chosen"]['recall'] = round(recall * 100, 1)
            except:
                results[model][f"TT+cot not chosen"] = {}
                results[model][f"TT+cot not chosen"]['f1'] = 0
                results[model][f"TT+cot not chosen"]['precision'] = 0
                results[model][f"TT+cot not chosen"]['recall'] = 0
    return results
def not_chosen_prompts_oracle(results, chosen, new_split):
    for i in [1,2]:
        for model in results:
            if new_split:
                path = f"data/results_new_split_judge_1_to_1_judge_1_to_1/{model}/cot_{i}_gpt-4o-2024-11-20/test_not_chosen"
            else:
                path = f"data/results_first_split/{model}/cot_{i}_gpt-4o-2024-11-20/test_not_chosen"
            try:
                df = pd.read_csv(os.path.join(path, "results.csv"))
                df.dropna(subset=['recall'], inplace=True)
                df = df[[len(eval(x)) != 0 for x in df['raw_descriptions']]]
                recall = [eval(x) for x in df['recall'].tolist()]
                recall = sum([sum(x.values()) for x in recall]) / sum([len(x) for x in recall])
                precision = [eval(x) for x in df['precision'].tolist()]
                precision = sum([sum(x.values()) for x in precision]) / sum([len(x) for x in precision])
                f1 = 2 * recall * precision / (recall + precision)
                results[model][f"Oracle+cot not chosen"] = {}
                results[model][f"Oracle+cot not chosen"]['f1'] = round(f1 * 100, 1)
                results[model][f"Oracle+cot not chosen"]['precision'] = round(precision * 100, 1)
                results[model][f"Oracle+cot not chosen"]['recall'] = round(recall * 100, 1)
            except:
                results[model][f"Oracle+cot not chosen"] = {}
                results[model][f"Oracle+cot not chosen"]['f1'] = 0
                results[model][f"Oracle+cot not chosen"]['precision'] = 0
                results[model][f"Oracle+cot not chosen"]['recall'] = 0
    return results

def creat_result_table(new_split=True):
    models = ["gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", "llama3.1-405b"]
    prompts = ["zero_shot", "few_shot", "cot"]
    chosen = {}
    results = {}
    for model in models:
        results[model] = {}
        for prompt in prompts:
            results[model][prompt] = {}
            for i in [1, 2]:
                try:
                    if new_split:
                        if prompt == "factscore":
                            path = f"data/results_new_split_judge_1_to_1/{model}/{prompt}/test"
                        else:
                            path = f"data/results_new_split_judge_1_to_1/{model}/{prompt}_{i}_maybe_consistent_gpt-4o-2024-11-20/test"
                    else:
                        path = f"data/results_new_split_judge_1_multiple_to_1/{model}/{prompt}_{i}_maybe_consistent_gpt-4o-2024-11-20/test"
                    with open(os.path.join(path, "results.json"), 'r') as f:
                        if prompt == "cot":
                            chosen[model] = i
                        data = json.load(f)
                        results[model][prompt]['f1'] = round(data['overall_f1'] * 100, 1)
                        results[model][prompt]['precision'] = round(data['overall_prec'] * 100, 1)
                        results[model][prompt]['recall'] = round(data['overall_rec'] * 100, 1)
                    break
                except:
                    pass
    # for model in models:
    #     results = model_based_classifier(results, chosen, new_split, model)
    results = creat_tt(results, chosen, new_split)
    #results = create_seahorse(results, chosen, new_split)
    results = create_oracle(results, chosen, new_split)
    #results = always_inconsistent(results, chosen, new_split)
    #results = simple_ensemble(results, chosen, new_split)
   # results = not_chosen_prompts(results, chosen, new_split)
   # results = not_chosen_prompts_with_tt(results, chosen, new_split)
   # results = not_chosen_prompts_oracle(results, chosen, new_split)
  #  results = creat_tt_with_inconsistent_option(results, chosen, new_split)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    df = dict_to_multicolumn_table(results)
    print(df)
    return df


def classifier_performance():
    models = ["gpt-4o-2024-11-20", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", "llama3.1-405b"]
    for model in models:
        try:
            path = f"data/results_new_split/{model}/cot_1_maybe_consistent_gpt-4o-2024-11-20/dev/results.csv"
            results_df = pd.read_csv(path)
            results_df["precision"].fillna("{}", inplace=True)
            ground_truth = [len(eval(x)) != 0 for x in results_df['raw_descriptions'].tolist()]
            predictions = [len(eval(x)) != 0 for x in results_df['precision'].tolist()]
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(ground_truth, predictions)
            print(f"Accuracy for {model}: {accuracy}")
        except:
            pass
    seahorse = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_seahorse.csv")
    tt = pd.read_csv("data/llm_inference/summary_classification/fine_tuned_models/results_trueteacher.csv")
    df_dev = pd.read_csv("data/all_finalized_data/final_new_split/final_dataset_dev.csv")
    # seahorse_dev = seahorse[(seahorse["text"].isin(df_dev["text"].tolist())) & (
    #     seahorse["model_summary"].isin(df_dev["model_summary"].tolist()))]
    seahorse_dev = df_dev.merge(seahorse, on=["text", "model_summary"], how="inner")
    gt = [len(eval(x)) != 0 for x in seahorse_dev['descriptions'].tolist()]
    seahorse_predictions = [x <= 0.9 for x in seahorse_dev['scores'].tolist()]
    # tt_dev = tt[
    #     (tt["text"].isin(df_dev["text"].tolist())) & (tt["model_summary"].isin(df_dev["model_summary"].tolist()))]
    tt_dev = df_dev.merge(tt, on=["text", "model_summary"], how="inner")
    tt_predictions = [x <= 0.95 for x in tt_dev['scores'].tolist()]
    ensemble_scores = (seahorse_dev['scores'] + tt_dev['scores']) / 2
    ensemble_predictions = [x <= 0.91 for x in ensemble_scores.tolist()]
    from sklearn.metrics import accuracy_score,roc_auc_score
    accuracy = accuracy_score(gt, seahorse_predictions)
    roc_auc = roc_auc_score(gt, seahorse_predictions)
    print(f"Accuracy for seahorse: {accuracy} AUC: {roc_auc}")
    accuracy = accuracy_score(gt, tt_predictions)
    roc_auc = roc_auc_score(gt, tt_predictions)
    print(f"Accuracy for tt: {accuracy} AUC: {roc_auc}")
    accuracy = accuracy_score(gt, ensemble_predictions)
    roc_auc = roc_auc_score(gt, ensemble_predictions)
    print(f"Accuracy for ensemble: {accuracy} AUC: {roc_auc}")
    import matplotlib.pyplot as plt

    plt.hist(seahorse_dev['scores'], bins=20, alpha=0.5, label='Seahorse')
    plt.hist(tt_dev['scores'], bins=20, alpha=0.5, label='TT')
    #plt.hist(ensemble_scores, bins=50, alpha=0.5, label='Ensemble')
    plt.legend()
    plt.show()


def main():
    # amount_of_inconsistencies_in_each_summary()
    # compare_results_per_model('data/results',"dev")
    # compare_results_per_prompt('data/results',"dev",only_prompt_type=True,prompt_says_inconsistent = False,best_prompt_variation = True)
    # main_annotation_path = "data/llm_inference/fine_grain_classification/spans_and_explanations/prompt6"
    # amount_of_unique_descriptions_per_amount_of_models(os.path.join(main_annotation_path, "results_gpt-4o_dev_annotated.csv"),
    #                                                    os.path.join(main_annotation_path,
    #                                                                 "results_gemini-1.5-pro_dev_annotated.csv"),
    #                                                    os.path.join(main_annotation_path,
    #                                                                 "results_claude-sonnet-3.5_dev_annotated.csv"),
    #                                                    os.path.join(main_annotation_path,
    #                                                                 "results_llama_3.1_405_dev_annotated.csv"),
    #                                                    "data/all_finalized_data/manual_annotation/Final_manual_dataset.csv")
    # gt_to_pred_llm_judgment("data/gt_for_judgment","data/results")
   # how_many_mistakes_total("data/gt_for_judgment","data/new_judge_testing/prompt5")
    turn_to_binary("data/results_new_split_judge_1_to_1", "cot")
    #df_new = creat_result_table(new_split=True)




if __name__ == "__main__":
    main()

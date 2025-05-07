import pandas as pd
from tqdm import tqdm
from inference_utils import chose_model, transform_to_enumerated_descriptions
import concurrent.futures
import string
import re
import argparse
import os
from results_parser import llm_judgment_to_precision_recall, compute_only_overall_metrics
import json


def merge_errors(flatten_errors, lengths):
    result = []
    index = 0
    for length in lengths:
        sublist = flatten_errors[index:index + length]
        index += length
        if all(item is None for item in sublist):
            result.append(None)
        else:
            joined = " ".join("None" if item is None else item for item in sublist)
            result.append(joined)
    return result


def create_dir_run(split, directory, inference_model, judgment_model, args):
    classify_prompt_number = args.classification_prompt_path.split("/")[-1].split(".")[0].replace("prompt", "")
    directory += f"classify_{classify_prompt_number}"
    if not os.path.isdir(directory):
        os.makedirs(directory)
        #raise ValueError(f"Directory {directory} does not exist.")
    directory = os.path.join(directory, inference_model)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if not os.path.isdir(os.path.join(directory, judgment_model)):
        os.makedirs(os.path.join(directory, judgment_model))
    final_path = os.path.join(directory, judgment_model, split)
    if os.path.isdir(final_path):
        raise ValueError(f"Directory {final_path} already exists.")
    else:
        os.makedirs(final_path)
        return final_path


def extract_identifier_list(text):
    # TODO: This is a patch, maybe need a more permanent solution, maybe change prompt
    text = text.replace("*", "")
    pattern = r'([A-Z]\.\s(?:(?!\n\n).)*?(?:\n|$))'
    list_items = re.findall(pattern, text, re.DOTALL)
    full_list = "\n".join(list_items)
    return full_list


def extract_bullet_list(text):
    pattern = r'([*-] .*(?:\n[*-] .*)*)'
    match = re.search(pattern, text)
    return match.group(1)


def preprocess_atomic_facts(atomic_facts):
    possible_list_bullets = ["-", "*"]
    for bullet in possible_list_bullets:
        atomic_facts = [x.replace(bullet, "-") for x in atomic_facts]
    list_starts = [x.find('-') for x in atomic_facts]
    atomic_facts = [x[list_starts[i]:] for i, x in enumerate(atomic_facts)]
    atomic_facts = [x.replace('-', "") for x in atomic_facts]
    atomic_facts = [x.split('\n') for x in atomic_facts]
    return atomic_facts


def call_model_parallel(model, text_inputs, max_workers=10, max_new_tokens=2000):
    inputs = [None] * len(text_inputs)
    outputs = [None] * len(text_inputs)
    errors = [None] * len(text_inputs)
    prices = [None] * len(text_inputs)

    def call_model(index, input_text):
        return index, model.call(input_text, max_new_tokens=max_new_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i in range(len(text_inputs)):
            inputs[i] = text_inputs[i]  # Save input immediately at correct index
            futures[executor.submit(call_model, i, text_inputs[i])] = i  # Store future by index

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(text_inputs)):
            idx = futures[future]  # Retrieve the index
            try:
                _, (output, error, price) = future.result()
            except Exception as e:
                output, error, price = None, str(e), 0  # Handle exceptions safely

            outputs[idx] = output
            errors[idx] = error
            prices[idx] = price

    return inputs, outputs, errors, prices


def remove_prefix(text):
    return re.sub(r'^\s*(\d+\.\s*|[-,*]\s*)', '', text)


def process_with_explanation(initial_model_outputs):
    inconsistencies = []
    for i in range(len(initial_model_outputs)):
        summary_inconsistencies = []
        for j in range(len(initial_model_outputs[i])):
            fact_classification = initial_model_outputs[i][j]
            import re
            yes_or_no = re.split(r'[ \n]+', fact_classification)[0]
            yes_or_no = yes_or_no.translate(str.maketrans('', '', string.punctuation)).strip().lower()
            if "no" == yes_or_no:
                fact_classification = fact_classification.split(' ', 1)[1].strip()
                if len(fact_classification.split('\n')) > 1:
                    fact_classification = fact_classification.split('\n')
                else:
                    fact_classification = [fact_classification]
                fact_classification = [remove_prefix(x) for x in fact_classification]
                fact_classification = [x.strip() for x in fact_classification]
                summary_inconsistencies += fact_classification
        inconsistencies.append(summary_inconsistencies)
    return inconsistencies


def break_to_atomic_facts(model, prompt, summaries):
    inputs = []
    for summary in summaries:
        summary_input = prompt + ":" + summary
        inputs.append(summary_input)
    inputs, atomic_facts, errors, prices = call_model_parallel(model, inputs)
    return inputs, atomic_facts, errors, prices


def compare_facts_to_text(model, atomic_facts, texts, prompt):
    atomic_facts = [extract_bullet_list(x) for x in atomic_facts]
    atomic_facts = preprocess_atomic_facts(atomic_facts)
    inputs = []
    for i in range(len(texts)):
        summary_atomic_facts = atomic_facts[i]
        text = texts[i]
        summary_inputs = []
        for atomic_fact in summary_atomic_facts:
            prompt_for_fact = prompt + "\n" + "Text:\n" + text + "\n" + "Atomic fact:\n" + atomic_fact
            summary_inputs.append(prompt_for_fact)
        inputs.append(summary_inputs)
    lengths = [len(x) for x in inputs]
    flattened_inputs = [item for sublist in inputs for item in sublist]
    _, summary_model_responses, errors, prices = call_model_parallel(model, flattened_inputs)
    outputs = [summary_model_responses[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]
    prices = [sum(prices[sum(lengths[:i]):sum(lengths[:i + 1])]) for i in range(len(lengths))]
    errors = merge_errors(errors, lengths)
    return outputs, errors, prices


def split_and_merge(model, summaries, raw_inconsistencies, prompt):
    inconsistencies = process_with_explanation(raw_inconsistencies)
    selected_inputs = []
    mapping = {}
    for i, summary_inconsistencies in enumerate(inconsistencies):
        if len(summary_inconsistencies) == 0:
            continue
        summary_inconsistencies = "\n".join(summary_inconsistencies)
        prompt_for_fact = prompt + "\n" + "Summary:\n" + summaries[i] + "\nInconsistencies:\n" + summary_inconsistencies
        selected_inputs.append(prompt_for_fact)
        mapping[i] = len(selected_inputs) - 1
    selected_inputs, processed_inconsistencies, output_errors, output_prices = call_model_parallel(model,
                                                                                                   selected_inputs)
    outputs = ["" for _ in range(len(inconsistencies))]
    prices = [0 for _ in range(len(inconsistencies))]
    inputs = ["" for _ in range(len(inconsistencies))]
    errors = [None for _ in range(len(inconsistencies))]
    for source_index, target_index in mapping.items():
        inputs[source_index] = selected_inputs[target_index]
        prices[source_index] = output_prices[target_index]
        errors[source_index] = output_errors[target_index]
        outputs[source_index] = processed_inconsistencies[target_index]
    return inputs, outputs, errors, prices


def judgment(model, prompt, past_text_prompt, summaries, gt_descriptions, predicted_descriptions, max_tokens):
    formatted_descriptions = transform_to_enumerated_descriptions(gt_descriptions)
    inconsistencies = [extract_identifier_list(x) for x in predicted_descriptions]
    inputs = []
    for i in range(len(summaries)):
        summary = summaries[i]
        golden_descriptions = formatted_descriptions[i]
        predicted_descriptions = inconsistencies[i]
        llm_input = (
            f"{prompt}\n\nSummary:\n{summary}\n"
            f"Gold label:\n{golden_descriptions}\n"
            f"Predicted Output:\n{predicted_descriptions}\n"
            f"{past_text_prompt}"
        )
        inputs.append(llm_input)
    inputs, outputs, errors, prices = call_model_parallel(model, inputs, max_workers=10, max_new_tokens=max_tokens)
    return inputs, outputs, errors, prices

def merge(model, prompt, past_text_prompt, summaries, predicted_descriptions, max_tokens):
    formatted_predicted_descriptions= transform_to_enumerated_descriptions(predicted_descriptions)
    inputs = []
    for i in range(len(summaries)):
        summary = summaries[i]
        predicted_summary_inconsistencies = formatted_predicted_descriptions[i]
        llm_input = (
            f"{prompt}\n\nSummary:\n{summary}\n"
            f"Predicted Inconsistencies:\n{predicted_summary_inconsistencies}\n"
            f"{past_text_prompt}"
        )
        inputs.append(llm_input)
    inputs, outputs, errors, prices = call_model_parallel(model, inputs, max_workers=10, max_new_tokens=max_tokens)
    return inputs, outputs, errors, prices


def compute(gt_descriptions, judgments):
    precision_dicts, recall_dicts, discarded_samples = llm_judgment_to_precision_recall(judgments,
                                                                                        gt_descriptions)
    metrics = compute_only_overall_metrics(precision_dicts, recall_dicts)
    print(f"Precision: {metrics['overall_prec']}")
    print(f"Recall: {metrics['overall_rec']}")
    print(f"F1: {metrics['overall_f1']}")
    print("Amount of predicted inconsistencies: ", sum([len(x) for x in precision_dicts]))
    return metrics


def parse_args():
    args = argparse.ArgumentParser(description="Process some integers.")
    args.add_argument('-data_path', type=str)
    args.add_argument('-split', type=str)
    args.add_argument('-model_tested', type=str)
    args.add_argument('-model_judge', type=str)
    args.add_argument('-atomic_facts_file', type=str)
    args.add_argument('-classification_file', type=str)
    args.add_argument('-judgment_raw_file', type=str)
    args.add_argument('-split_and_merge_file', type=str)
    args.add_argument('-judgment_file', type=str)
    args.add_argument('-split_and_merge_judgment_file', type=str)
    args.add_argument('-break_to_atomic_facts_prompt_path', type=str)
    args.add_argument('-classification_prompt_path', type=str)
    args.add_argument('-split_and_merge_prompt_path', type=str)
    args.add_argument('-judgment_prompt_path', type=str)
    args.add_argument('-judgment_past_text_prompt_path', type=str)
    args.add_argument('-output_dir', type=str)
    args.add_argument('-max_new_tokens', type=int, default=2000)
    args.add_argument('-llamaapi', action='store_true')
    args.add_argument('-break_to_facts', action='store_true')
    args.add_argument('-classify', action='store_true')
    args.add_argument('-judge_raw', action='store_true')
    args.add_argument('-split_and_merge', action='store_true')
    args.add_argument('-judge', action='store_true')
    args.add_argument('-compute', action='store_true')
    args.add_argument('-new_run', action='store_true')
    args = args.parse_args()
    if args.split is None:
        raise ValueError("Split must be provided")
    if args.new_run:
        # Revise
        args.output_dir = create_dir_run(args.split, args.output_dir, args.model_tested, args.model_judge, args)

    if args.break_to_atomic_facts_prompt_path is not None:
        with open(args.break_to_atomic_facts_prompt_path, 'r', encoding='windows-1252') as file:
            args.break_to_atomic_facts_prompt = file.read()
            args.break_to_atomic_facts_prompt = args.break_to_atomic_facts_prompt.strip()
    else:
        args.break_to_atomic_facts_prompt = ''
    if args.classification_prompt_path is not None:
        with open(args.classification_prompt_path, 'r', encoding='windows-1252') as file:
            args.classification_prompt = file.read()
            args.classification_prompt = args.classification_prompt.strip()
    else:
        args.classification_prompt = ''
    if args.split_and_merge_prompt_path is not None:
        with open(args.split_and_merge_prompt_path, 'r', encoding='windows-1252') as file:
            args.split_and_merge_prompt = file.read()
            args.split_and_merge_prompt = args.split_and_merge_prompt.strip()
    else:
        args.split_and_merge_prompt = ''
    if args.judgment_prompt_path is not None:
        with open(args.judgment_prompt_path, 'r', encoding='windows-1252') as file:
            args.judgment_prompt = file.read()
            args.judgment_prompt = args.judgment_prompt.strip()
    else:
        args.judgment_prompt = ''
    if args.judgment_past_text_prompt_path is not None:
        with open(args.judgment_past_text_prompt_path, 'r', encoding='windows-1252') as file:
            args.judgment_past_text_prompt = file.read()
            args.judgment_past_text_prompt = args.judgment_past_text_prompt.strip()
    else:
        args.judgment_past_text_prompt = ''
    if args.output_dir is not None:
        args.inference_temp_save_dir = os.path.join(args.output_dir, "inference_temp")
        args.judgment_temp_save_dir = os.path.join(args.output_dir, "judgment_temp")
        args.atomic_facts_file = os.path.join(args.output_dir, args.atomic_facts_file)
        args.classification_file = os.path.join(args.output_dir, args.classification_file)
        args.judgment_raw_file = os.path.join(args.output_dir, args.judgment_raw_file)
        args.split_and_merge_file = os.path.join(args.output_dir, args.split_and_merge_file)
        args.judgment_file = os.path.join(args.output_dir, args.judgment_file)
    return args


def get_data(args):
    df = pd.read_csv(args.data_path + ".csv")
    summaries = df['model_summary'].tolist()
    texts = df['text'].tolist()
    raw_descriptions = df['descriptions'].tolist()
    raw_descriptions = [eval(x) for x in raw_descriptions]
    if "atomic_facts" in df.columns:
        atomic_facts = df['atomic_facts'].tolist()
    else:
        atomic_facts = None
    if "raw_predicted_inconsistencies" in df.columns:
        raw_predicted_inconsistencies = df['raw_predicted_inconsistencies'].tolist()
        raw_predicted_inconsistencies = [eval(x) for x in raw_predicted_inconsistencies]
    else:
        raw_predicted_inconsistencies = None
    if "processed_predicted_inconsistencies" in df.columns:
        predicted_inconsistencies = df['processed_predicted_inconsistencies'].fillna("").tolist()
    else:
        predicted_inconsistencies = None
    if "judgment" in df.columns:
        judgments = df['judgment'].tolist()
    else:
        judgments = None
    if "judgment_raw" in df.columns:
        judgments_raw = df['judgment_raw'].tolist()
    else:
        judgments_raw = None
    return summaries, texts, raw_descriptions, atomic_facts, raw_predicted_inconsistencies, predicted_inconsistencies, judgments, judgments_raw


def main():
    args = parse_args()
    summaries, texts, raw_descriptions, atomic_facts, raw_predicted_inconsistencies, processed_predicted_inconsistencies, judgments, judgments_raw = get_data(
        args)
    tested_model = chose_model(model_name=args.model_tested, temp_save_dir=args.inference_temp_save_dir,
                               llamaapi=args.llamaapi, azure=False, dtype=None, device_map=None, pipline=False)
    judgment_model = chose_model(model_name=args.model_judge, temp_save_dir=args.judgment_temp_save_dir,
                                 llamaapi=False, azure=False, dtype=None, device_map=None, pipline=False)
    if args.break_to_facts:
        atomic_facts_inputs, atomic_facts, errors, prices = break_to_atomic_facts(tested_model,
                                                                                  args.break_to_atomic_facts_prompt,
                                                                                  summaries)
        if args.atomic_facts_file is not None:
            df = pd.DataFrame({'text': texts, 'model_summary': summaries, "descriptions": raw_descriptions,
                               "input": atomic_facts_inputs, "atomic_facts": atomic_facts, "error": errors,
                               "price": prices})
            df.to_csv(args.atomic_facts_file + ".csv", index=False)
    if args.classify:
        raw_predicted_inconsistencies, errors, prices = compare_facts_to_text(tested_model, atomic_facts, texts,
                                                                              args.classification_prompt)
        if args.classification_file is not None:
            df = pd.DataFrame(
                {'text': texts, 'model_summary': summaries, "descriptions": raw_descriptions,
                 "atomic_facts": atomic_facts,
                 "raw_predicted_inconsistencies": raw_predicted_inconsistencies, "error": errors, "price": prices})
            df.to_csv(args.classification_file + ".csv", index=False)
    if args.judge_raw:
        inconsistencies = process_with_explanation(raw_predicted_inconsistencies)
        inconsistencies = transform_to_enumerated_descriptions(inconsistencies)
        inputs, judgments_raw, errors, prices = judgment(judgment_model, args.judgment_prompt,
                                                         args.judgment_past_text_prompt,
                                                         summaries, raw_descriptions, inconsistencies,
                                                         args.max_new_tokens)
        if args.judgment_raw_file is not None:
            df = pd.DataFrame({'text': texts, 'model_summary': summaries, "descriptions": raw_descriptions,
                               "atomic_facts": atomic_facts,
                               "raw_predicted_inconsistencies": inconsistencies,
                               "input": inputs, "judgment_raw": judgments_raw, "error": errors, "price": prices})
            df.to_csv(args.judgment_raw_file + ".csv", index=False)
    if args.compute and judgments_raw is not None:
        metrics = compute(raw_descriptions, judgments_raw)
        results_json = os.path.join(args.output_dir, "raw_results.json")
        with open(results_json, 'w') as f:
            json.dump(metrics, f)

    if args.split_and_merge:
        inputs, processed_predicted_inconsistencies, errors, prices = split_and_merge(tested_model, summaries,
                                                                                      raw_predicted_inconsistencies,
                                                                                      args.split_and_merge_prompt)
        if args.split_and_merge_file is not None:
            df = pd.DataFrame(
                {'text': texts, 'model_summary': summaries, "descriptions": raw_descriptions,
                 "atomic_facts": atomic_facts,
                 "raw_predicted_inconsistencies": raw_predicted_inconsistencies,
                 "input": inputs, "processed_predicted_inconsistencies": processed_predicted_inconsistencies,
                 "error": errors, "price": prices})
            df.to_csv(args.split_and_merge_file + ".csv", index=False)
    if args.judge:
        inputs, judgments, errors, prices = judgment(judgment_model, args.judgment_prompt,
                                                     args.judgment_past_text_prompt,
                                                     summaries, raw_descriptions, processed_predicted_inconsistencies,
                                                     args.max_new_tokens)
        if args.judgment_file is not None:
            df = pd.DataFrame({'text': texts, 'model_summary': summaries, "descriptions": raw_descriptions,
                               "atomic_facts": atomic_facts,
                               "raw_predicted_inconsistencies": raw_predicted_inconsistencies,
                               "processed_predicted_inconsistencies": processed_predicted_inconsistencies,
                               "input": inputs, "judgment": judgments, "error": errors, "price": prices})
            df.to_csv(args.judgment_file + ".csv", index=False)

    if args.compute and judgments is not None:
        metrics = compute(raw_descriptions, judgments)
        results_json = os.path.join(args.output_dir, "results.json")
        with open(results_json, 'w') as f:
            json.dump(metrics, f)


def compute_inference_cost():
    for model in ["gpt-4o-2024-11-20", "gemini-1.5-pro", "claude-3-5-sonnet-20241022", "llama3.1-405b"]:
        main_dir = f"data/factscore/results/{model}/gpt-4o-2024-11-20/dev"
        df_1 = pd.read_csv(os.path.join(main_dir, "atomic_facts.csv"))
        df_2 = pd.read_csv(os.path.join(main_dir, "classification_and_descriptions.csv"))
        df_3 = pd.read_csv(os.path.join(main_dir, "processed_descriptions.csv"))
        df_4 = pd.read_csv(os.path.join(main_dir, "unprocessed_results.csv"))
        df_5 = pd.read_csv(os.path.join(main_dir, "raw_descriptions_judgments.csv"))
        test_model_price = df_1['price'].sum() + df_2['price'].sum() + df_3['price'].sum()
        judge_price = df_4['price'].sum() + df_5['price'].sum()
        print(f"Model: {model}")
        print(f"Test model price: {test_model_price}")
        print(f"Judge model price: {judge_price}")


def compare_to_others():
    models = ["gpt-4o-2024-11-20", "gemini-1.5-pro", "claude-3-5-sonnet-20241022", "llama3.1-405b"]
    matching_paths = []
    path = "data/results"
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            # Check if all target keywords are in the full path
            if "maybe" in dirpath and "dev" == dirpath.split('/')[-1] and dirpath.split('/')[-3] in models:
                if "results.json" == filename:
                    matching_paths.append(full_path)
    results = {}
    for path in matching_paths:
        model = path.split('/')[-4]
        prompt = path.split('/')[-3].split('_')[:-3]
        prompt = "_".join(prompt)
        if model not in results:
            results[model] = {}
        if prompt not in results[model]:
            results[model][prompt] = {}
        print(model, prompt)
        with open(path, 'r') as f:
            data = json.load(f)
        results[model][prompt] = data
    main_dir = "data/factscore/results"
    for model in results:
        path = os.path.join(main_dir, model, "gpt-4o-2024-11-20", "dev")
        with open(path + "/results.json", 'r') as f:
            data = json.load(f)
        results[model]['factsocre'] = data
    for model in results:
        prompts_f1 = [(k, round(v['overall_f1'] * 100, 2)) for k, v in results[model].items()]
        prompts_f1 = sorted(prompts_f1, key=lambda x: x[1], reverse=True)
        prompts_recall = [(k, round(v['overall_rec'] * 100, 2)) for k, v in results[model].items()]
        prompts_recall = sorted(prompts_recall, key=lambda x: x[1], reverse=True)
        prompts_prec = [(k, round(v['overall_prec'] * 100, 2)) for k, v in results[model].items()]
        prompts_prec = sorted(prompts_prec, key=lambda x: x[1], reverse=True)
        print(f"Model: {model}")
        print("F1: ", prompts_f1)
        print("Recall: ", prompts_recall)
        print("Precision: ", prompts_prec)


def compare():
    raw_df = pd.read_csv(
        "data/factscore/results/gemini-1.5-pro/gpt-4o-2024-11-20/dev/raw_descriptions_judgments.csv")
    raw_judgments = raw_df['judgment_raw'].tolist()
    processed_df = pd.read_csv(
        "data/factscore/results/gemini-1.5-pro//gpt-4o-2024-11-20/dev/unprocessed_results.csv")
    judgments = processed_df['judgment'].tolist()
    descriptions = processed_df['descriptions'].tolist()
    descriptions = [eval(x) for x in descriptions]
    _, recall_dicts, _ = llm_judgment_to_precision_recall(judgments, descriptions)
    _, raw_recall_dicts, _ = llm_judgment_to_precision_recall(raw_judgments, descriptions)
    for i in range(len(recall_dicts)):
        raw_sample = raw_recall_dicts[i]
        sample = recall_dicts[i]
        for key in sample:
            if sample[key] != raw_sample[key]:
                print(f"Sample {i}: {key} - Raw: {raw_sample[key]} - Processed: {sample[key]}")


def proccess_llm_output(output):
    import ast
    output = output.split('{')[-1].split('}')[0]
    output = "{" + output + "}"
    output = ast.literal_eval(output)
    return output


def check_judge():
    from collections import Counter
    df = pd.read_csv(
        "data/factscore/results_split_and_merge_7/gemini-1.5-pro/gpt-4o-2024-11-20/dev/raw_descriptions_judgments.csv")
    judgments = df['judgment_raw'].tolist()
    descriptions = df['descriptions'].tolist()
    descriptions = [eval(x) for x in descriptions]
    judgments = [proccess_llm_output(x) for x in judgments]
    for x in judgments:
        for key in x:
            if len(x[key]) > 1:
                for j in range(1, len(x[key])):
                    print("need split")
        x_values = x.values()
        x_values = [item for sublist in x_values for item in sublist]
        counter = Counter(x_values)
        for key in counter:
            if counter[key] > 1:
                for j in range(1, counter[key]):
                    print("need merge")


def compare_before_and_after_merge():
    df = pd.read_csv(
        "data/factscore/results_split_and_merge_7/gemini-1.5-pro/gpt-4o-2024-11-20/dev/classification_and_descriptions.csv")
    initial_models_responses = df['raw_predicted_inconsistencies'].tolist()
    initial_models_responses = [eval(x) for x in initial_models_responses]
    inconsistencies = process_with_explanation(initial_models_responses)
    df_merged = pd.read_csv(
        "data/factscore/results_split_and_merge_7/gemini-1.5-pro/gpt-4o-2024-11-20/dev/processed_descriptions.csv")
    merged_inconsistencies = df_merged['processed_predicted_inconsistencies'].fillna('').tolist()
    merged_inconsistencies = [extract_identifier_list(x) for x in merged_inconsistencies]
    merged_inconsistencies = [x.split('\n\n') for x in merged_inconsistencies]
    merged_inconsistencies = [[x for x in y if x != ""] for y in merged_inconsistencies]
    # for i in range(len(merged_inconsistencies)):
    #     print("Original:")
    #     print("\n".join(inconsistencies[i]))
    #     print("Merged:")
    #     print("\n".join(merged_inconsistencies[i]))
    print("Amount of inconsistencies before merge: ", sum([len(x) for x in inconsistencies]))
    print("Amount of inconsistencies after merge: ", sum([len(x) for x in merged_inconsistencies]))


def split_and_merge_testing():
    probs = [16, 17, 22]
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/factscore/testing/gpt-4o-2024-11-20/gpt-4o-2024-11-20/dev/processed_descriptions.csv")
    predictions = df["processed_predicted_inconsistencies"].tolist()
    raw_predictions = df["raw_predicted_inconsistencies"].tolist()
    raw_predictions = [eval(x) for x in raw_predictions]
    raw_predictions = process_with_explanation(raw_predictions)
    summaries = df["model_summary"].tolist()
    for i in probs:
        print("Original:")
        print("\n".join(raw_predictions[i]))
        print("Merged:")
        print(predictions[i])
    model = chose_model(model_name="gpt-4o-2024-11-20", temp_save_dir=None, llamaapi=False, azure=False, dtype=None,
                        device_map=None, pipline=False)
    with open("data/factscore/prompts/split_and_merge/prompt5.txt") as f:
        prompt = f.read()
    prompt = prompt.strip()
    inputs = [prompt + "\n" + "Summary:\n" + summaries[i] + "\nInconsistencies:\n" + "\n".join(raw_predictions[i]) for i
              in probs]
    for input in inputs:
        x, _, _ = model.call(input, max_new_tokens=2000)
        print(x)


if __name__ == "__main__":
    # check_judge()
    # compute_inference_cost()
    # compare_to_others()
    # compare_before_and_after_merge()
    # check_judge()
    main()

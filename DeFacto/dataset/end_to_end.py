import argparse
import ast
from inference_utils import chose_model, transform_to_enumerated_descriptions

from results_parser import llm_judgment_to_precision_recall, compute_metrics,compute_only_overall_metrics,impute_dicts
import pandas as pd
from tqdm import tqdm
import json
import os
import concurrent.futures


def create_dir_run(split, directory, inference_model, judgment_model, inference_prompt_path):
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    directory = os.path.join(directory, inference_model)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    prompt_type = inference_prompt_path.split('/')[-3]
    prompt_number = inference_prompt_path.split('/')[-2].replace('prompt', '')
    final_dir_name = f"{prompt_type}_{prompt_number}_{judgment_model}"
    if not os.path.isdir(os.path.join(directory, final_dir_name)):
        os.makedirs(os.path.join(directory, final_dir_name))
    final_path = os.path.join(directory, final_dir_name, split)
    if os.path.isdir(final_path):
        raise ValueError(f"Directory {final_path} already exists.")
    else:
        os.makedirs(final_path)
        return final_path


def check_args_the_same(args):
    with open(os.path.join(args.output_dir, f"args.json"), "r") as json_file:
        old_args = json.load(json_file)
    for key, value in vars(args).items():
        if key in ['new_run', 'infer', 'judge', 'compute', "data_path", "final_results_file", "judgment_file",
                   "llm_inference_file", 'azure_judgment', 'azure_inference']:
            continue
        if key not in old_args:
            print(f"Key {key} not in old args")
            continue
        if old_args[key] != value:
            if isinstance(value, str) and (old_args[key] in value or value in old_args[key]):
                continue
            else:
                print(f"Key {key} is different")
                return False
    return True


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-data_path', type=str)
    args.add_argument('-model_tested', type=str)
    args.add_argument('-model_judge', type=str)
    args.add_argument('-final_results_file', type=str)
    args.add_argument('-llm_inference_file', type=str)
    args.add_argument('-judgment_file', type=str)
    args.add_argument('-inference_prompt_path', type=str)
    args.add_argument('-inference_past_text_prompt_path', type=str)
    args.add_argument('-inference_delimiter', type=str)
    args.add_argument('-judgment_prompt_path', type=str)
    args.add_argument('-judgment_past_text_prompt_path', type=str)
    args.add_argument('-output_dir', type=str)
    args.add_argument('-new_run', action='store_true')
    args.add_argument('-max_new_tokens', type=int, default=2000)
    args.add_argument('-device_map', type=str)
    args.add_argument('-dtype', type=str)
    args.add_argument('-llamaapi_inference', action='store_true')
    args.add_argument('-llamaapi_judgment', action='store_true')
    args.add_argument('-infer', action='store_true')
    args.add_argument('-judge', action='store_true')
    args.add_argument('-compute', action='store_true')
    args.add_argument('-split', type=str)
    args.add_argument('-rerun_partial_inference', action='store_true')
    args.add_argument('-rerun_partial_judgment', action='store_true')
    args.add_argument('-azure_inference', action='store_true')
    args.add_argument('-azure_judgment', action='store_true')
    args = args.parse_args()
    if args.inference_delimiter is not None:
        args.inference_delimiter = args.inference_delimiter.replace("_", " ")
    if args.split is None:
        raise ValueError("Split must be provided")
    if args.new_run:
        args.output_dir = create_dir_run(args.split, args.output_dir, args.model_tested, args.model_judge,
                                         args.inference_prompt_path)
    if args.inference_prompt_path is not None:
        with open(args.inference_prompt_path, 'r', encoding='windows-1252') as file:
            args.inference_prompt = file.read()
            args.inference_prompt = args.inference_prompt.strip()
    else:
        args.inference_prompt = ''
    if args.inference_past_text_prompt_path is not None:
        with open(args.inference_past_text_prompt_path, 'r', encoding='windows-1252') as file:
            args.inference_past_text_prompt = file.read()
            args.inference_past_text_prompt = args.inference_past_text_prompt.strip()
    else:
        args.inference_past_text_prompt = ''
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
    return args


def get_data(args):
    df = pd.read_csv(args.data_path)
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    raw_descriptions = df['descriptions'].tolist()
    raw_descriptions = [ast.literal_eval(x) for x in raw_descriptions]
    if 'detection_output' in df.columns:
        outputs = df['detection_output'].tolist()
        if args.inference_delimiter is not None:
            outputs = [x.split(args.inference_delimiter)[-1].strip() for x in outputs]
    else:
        outputs = None
    if 'judgment_output' in df.columns:
        judgment_outputs = df['judgment_output'].tolist()
    else:
        judgment_outputs = None
    return texts, summaries, raw_descriptions, outputs, judgment_outputs


# def call_llm_for_inference(texts, summaries, model, prompt, past_text_prompt, max_new_tokens):
#     outputs = []
#     errors = []
#     prices = []
#     inputs = []
#     for text, summary in tqdm(zip(texts, summaries)):
#         input = prompt + '\n\n' 'Text: \n' + text + '\n' + 'Summary: \n' + summary + '\n' + past_text_prompt + '\n'
#         inputs.append(input)
#         output, error, price = model.call(input, max_new_tokens)
#         outputs.append(output)
#         errors.append(error)
#         prices.append(price)
#     return inputs, outputs, errors, prices



def parallel_call_llm_for_inference(texts, summaries, model, prompt, past_text_prompt, max_new_tokens, max_workers=10):
    inputs = [None] * len(texts)  # Ensure correct indexing
    outputs = [None] * len(texts)
    errors = [None] * len(texts)
    prices = [None] * len(texts)

    def call_model(index, input_text):
        return index, model.call(input_text, max_new_tokens=max_new_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for idx, (text, summary) in enumerate(zip(texts, summaries)):
            llm_input = f"{prompt}\n\nText:\n{text}\nSummary:\n{summary}\n{past_text_prompt}\n"
            inputs[idx] = llm_input  # Save input immediately at correct index
            futures[executor.submit(call_model, idx, llm_input)] = idx  # Store future by index

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(texts)):
            idx = futures[future]  # Retrieve the index
            try:
                _, (output, error, price) = future.result()
            except Exception as e:
                output, error, price = None, str(e), 0  # Handle exceptions safely

            outputs[idx] = output
            errors[idx] = error
            prices[idx] = price

    return inputs, outputs, errors, prices




def llm_inference(args, texts, summaries, raw_descriptions):
    model = chose_model(args.model_tested, args.inference_temp_save_dir, args.llamaapi_inference, args.azure_inference)
    # inputs, outputs, errors, prices = call_llm_for_inference(texts, summaries, model, args.inference_prompt,
    #                                                          args.inference_past_text_prompt, args.max_new_tokens)
    inputs, outputs, errors, prices = parallel_call_llm_for_inference(texts, summaries, model, args.inference_prompt,
                                                                      args.inference_past_text_prompt,
                                                                      args.max_new_tokens)

    if args.llm_inference_file is not None:
        df = pd.DataFrame(
            {'text': texts, 'model_summary': summaries, 'input': inputs, 'detection_output': outputs,
             "descriptions": raw_descriptions, 'error': errors, 'price': prices})
        output_dir = args.output_dir
        current_path = os.path.join(output_dir, args.llm_inference_file + '.csv')
        df.to_csv(current_path)
    if args.inference_delimiter is not None:
        outputs = [x.split(args.inference_delimiter)[-1].strip() for x in outputs]
    return outputs


# def call_llm_for_judgment(summaries, formatted_descriptions, llm_outputs, model, prompt, past_text_prompt,
#                           max_new_tokens):
#     outputs = []
#     errors = []
#     prices = []
#     inputs = []
#     for human_description, llm_description, summary in tqdm(
#             zip(formatted_descriptions, llm_outputs, summaries)):
#         input = prompt + '\n\n' "Summary:\n" + summary + '\n' + "Gold label: \n" + human_description + '\n' + 'Predicted Output: \n' + llm_description + '\n' + past_text_prompt
#         inputs.append(input)
#         output, error, price = model.call(input, max_new_tokens)
#         outputs.append(output)
#         errors.append(error)
#         prices.append(price)
#     return outputs, errors, prices, inputs



def parallel_call_llm_for_judgment(summaries, formatted_descriptions, llm_outputs, model, prompt, past_text_prompt,
                                   max_new_tokens, max_workers=10):
    inputs = []  # Save inputs immediately
    outputs = [None] * len(summaries)  # Pre-allocate space for correct ordering
    errors = [None] * len(summaries)
    prices = [None] * len(summaries)

    def call_model(index, input_text):
        return index, model.call(input_text, max_new_tokens=max_new_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for idx, (human_description, llm_description, summary) in enumerate(zip(formatted_descriptions, llm_outputs, summaries)):
            llm_input = (
                f"{prompt}\n\nSummary:\n{summary}\n"
                f"Gold label:\n{human_description}\n"
                f"Predicted Output:\n{llm_description}\n"
                f"{past_text_prompt}"
            )
            inputs.append(llm_input)  # Save input immediately
            futures[executor.submit(call_model, idx, llm_input)] = idx  # Track by index

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(summaries)):
            idx = futures[future]  # Retrieve the index
            try:
                _, (output, error, price) = future.result()
            except Exception as e:
                output, error, price = None, str(e), 0  # Handle errors safely

            outputs[idx] = output
            errors[idx] = error
            prices[idx] = price

    return inputs, outputs, errors, prices





def judge(args, raw_descriptions, llm_outputs, texts, summaries):
    formatted_descriptions = transform_to_enumerated_descriptions(raw_descriptions)
    model = chose_model(args.model_judge, args.judgment_temp_save_dir, args.llamaapi_judgment, args.azure_judgment)
    # outputs, errors, prices, inputs = call_llm_for_judgment(summaries, formatted_descriptions, llm_outputs, model,
    #                                                         args.judgment_prompt,
    #                                                         args.judgment_past_text_prompt, args.max_new_tokens)
    inputs, outputs, errors, prices = parallel_call_llm_for_judgment(summaries, formatted_descriptions, llm_outputs, model,
                                                            args.judgment_prompt,
                                                            args.judgment_past_text_prompt, args.max_new_tokens)
    if args.judgment_file is not None:
        df = pd.DataFrame(
            {'descriptions': raw_descriptions, 'input': inputs, 'detection_output': llm_outputs, 'text': texts,
             'model_summary': summaries,
             'judgment_output': outputs, 'error': errors,
             'price': prices})
        output_dir = args.output_dir
        current_path = os.path.join(output_dir, args.judgment_file + '.csv')
        df.to_csv(current_path)
    return outputs





def rerun_and_edit_inference(args):
    # This function need to load a file, where some of the calls incurred errors or the model did not output anything, and run them in the model again
    # After running them, edit the data by replacing the old inputs, outputs, prices and errors with new ones, only for the samples with errors
    df = pd.read_csv(args.data_path, index_col=0)
    inputs = df[(df['error'].notnull()) | (df['detection_output'].isna())]['input'].tolist()
    model = chose_model(args.model_tested, None, args.llamaapi_inference, args.azure_inference)
    outputs, errors, prices = [], [], []
    print("Rerunning inference for samples with errors.There are ", len(inputs), " samples with errors")
    for input in tqdm(inputs):
        output, error, price = model.call(input, args.max_new_tokens)
        outputs.append(output)
        errors.append(error)
        prices.append(price)
    for i, input in enumerate(inputs):
        df.loc[df['input'] == input, 'detection_output'] = outputs[i]
        df.loc[df['input'] == input, 'error'] = errors[i]
        df.loc[df['input'] == input, 'price'] = prices[i]
    df.to_csv(args.data_path)


def rerun_and_edit_judgment(args):
    df = pd.read_csv(args.data_path, index_col=0)
    inputs = df[(df['error'].notnull()) | (df['judgment_output'].isna())]['input'].tolist()
    model = chose_model(args.model_judge, None, args.llamaapi_judgment, args.azure_judgment)
    outputs, errors, prices = [], [], []
    print("Rerunning judgment for samples with errors.There are ", len(inputs), " samples with errors")
    for input in tqdm(inputs):
        output, error, price = model.call(input, args.max_new_tokens)
        outputs.append(output)
        errors.append(error)
        prices.append(price)
    for i, input in enumerate(inputs):
        df.loc[df['input'] == input, 'judgment_output'] = outputs[i]
        df.loc[df['input'] == input, 'error'] = errors[i]
        df.loc[df['input'] == input, 'price'] = prices[i]
    df.to_csv(args.data_path)


def evaluate():
    args = parse_args()
    if args.rerun_partial_inference:
        rerun_and_edit_inference(args)
        return
    if args.rerun_partial_judgment:
        rerun_and_edit_judgment(args)
        return
    if os.path.exists(os.path.join(args.output_dir, "args.json")):
        if not check_args_the_same(args):
            raise ValueError("The arguments are different from the previous run")
    else:
        with open(os.path.join(args.output_dir, "args.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    texts, summaries, raw_descriptions, outputs, judgment_outputs = get_data(args)
    if args.infer:
        if outputs is not None:
            raise ValueError("No need to reinfer")
        outputs = llm_inference(args, texts, summaries, raw_descriptions)
    if args.judge:
        if judgment_outputs is not None:
            raise ValueError("No need to rejudge")
        judgment_outputs = judge(args, raw_descriptions, outputs, texts, summaries)
    if args.compute:
        precision_dicts, recall_dicts, discarded_samples = llm_judgment_to_precision_recall(judgment_outputs,
                                                                                            raw_descriptions)
        #metrics = compute_metrics(precision_dicts, recall_dicts)
        metrics = compute_only_overall_metrics(precision_dicts, recall_dicts)
        metrics['number_of_discarded_samples'] = len(discarded_samples)
        metrics['discarded_samples'] = discarded_samples
        precision_dicts, recall_dicts = impute_dicts(precision_dicts, recall_dicts, discarded_samples)
        final_df = pd.DataFrame(
            {'text': texts, "model_summary": summaries, "raw_descriptions": raw_descriptions, "llm_outputs": outputs,
             "judgment_outputs": judgment_outputs, "precision": precision_dicts, "recall": recall_dicts})
        output_dir = args.output_dir
        curr_path = os.path.join(output_dir, args.final_results_file)
        final_df.to_csv(curr_path + '.csv')
        with open(curr_path + ".json", "w") as json_file:
            json.dump(metrics, json_file, indent=4)


def main():
    evaluate()


if __name__ == "__main__":
    main()

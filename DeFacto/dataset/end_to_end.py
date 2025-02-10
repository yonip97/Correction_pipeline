import argparse
import ast
from inference_utils import chose_model, transform_to_enumerated_descriptions


from results_parser import llm_judgment_to_precision_recall_matching, compute_metrics
import pandas as pd
from tqdm import tqdm
import json
import os


def create_dir_run(directory, inference_model, judgement_model, inference_prompt_path):
    if not os.path.isdir(directory):
           raise ValueError(f"Directory {directory} does not exist.")

    prompt_type = inference_prompt_path.split('/')[-3]
    prompt_number = inference_prompt_path.split('/')[-2].replace('prompt','')
    final_dir_name = f"{inference_model}_{prompt_type}_{prompt_number}_{judgement_model}"
    if os.path.isdir(os.path.join(directory, final_dir_name)):
        raise ValueError(f"Directory {final_dir_name} already exists.")
    else:
        os.makedirs(os.path.join(directory, final_dir_name))
        return os.path.join(directory, final_dir_name)


def check_args_the_same(args):
    with open(os.path.join(args.output_dir, "args.json"), "r") as json_file:
        old_args = json.load(json_file)
    for key, value in vars(args).items():
        if key in ['new_run','infer','judge']:
            continue
        elif key == 'data_path':
            continue
        if key not in old_args:
            return False
        if old_args[key] != value:
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
    args.add_argument('-judgement_prompt_path', type=str)
    args.add_argument('-judgement_past_text_prompt_path', type=str)
    args.add_argument('-output_dir', type=str)
    args.add_argument('-new_run', action='store_true')
    args.add_argument('-max_new_tokens', type=int, default=1000)
    args.add_argument('-device_map', type=str)
    args.add_argument('-dtype', type=str)
    args.add_argument('-llamaapi_inference', action='store_true')
    args.add_argument('-llamaapi_judgement', action='store_true')
    args.add_argument('-infer', action='store_true')
    args.add_argument('-judge', action='store_true')
    args = args.parse_args()
    if args.new_run:
        args.output_dir = create_dir_run(args.output_dir, args.model_tested, args.model_judge, args.inference_prompt_path)
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
    if args.judgement_prompt_path is not None:
        with open(args.judgement_prompt_path, 'r', encoding='windows-1252') as file:
            args.judgement_prompt = file.read()
            args.judgement_prompt = args.judgement_prompt.strip()
    else:
        args.judgement_prompt = ''
    if args.judgement_past_text_prompt_path is not None:
        with open(args.judgement_past_text_prompt_path, 'r', encoding='windows-1252') as file:
            args.judgement_past_text_prompt = file.read()
            args.judgement_past_text_prompt = args.judgement_past_text_prompt.strip()
    else:
        args.judgement_past_text_prompt = ''
    args.inference_temp_save_dir = os.path.join(args.output_dir, "inference_temp")
    args.judgement_temp_save_dir = os.path.join(args.output_dir, "judgement_temp")
    return args


def get_data(args):
    df = pd.read_csv(args.data_path)
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    raw_descriptions = df['descriptions'].tolist()
    raw_descriptions = [ast.literal_eval(x) for x in raw_descriptions]
    if 'detection_output' in df.columns:
        outputs = df['detection_output'].tolist()
    else:
        outputs = None
    if 'judgement_output' in df.columns:
        judgement_outputs = df['judgement_output'].tolist()
    else:
        judgement_outputs = None
    return texts, summaries, raw_descriptions, outputs, judgement_outputs


def llm_inference(args, texts, summaries):
    model = chose_model(args.model_tested, args.inference_temp_save_dir, args.llamaapi_inference)
    outputs = []
    errors = []
    prices = []
    inputs = []
    prompt = args.inference_prompt
    past_text_prompt = args.inference_past_text_prompt
    for text, summary in tqdm(zip(texts, summaries)):
        input = prompt + '\n\n' 'Text: \n' + text + '\n' + 'Summary: \n' + summary + '\n' + past_text_prompt + '\n'
        inputs.append(input)
        output, error, price = model.call(input, args.max_new_tokens)
        outputs.append(output)
        errors.append(error)
        prices.append(price)
    df = pd.DataFrame({'text': texts, 'model_summary': summaries, 'input': inputs, 'detection_output': outputs, 'error': errors,
                       'price': prices})
    if args.llm_inference_file is not None:
        output_dir = args.output_dir
        current_path = os.path.join(output_dir, args.llm_inference_file + '.csv')
        df.to_csv(current_path)
    return outputs


def judge(args, raw_descriptions, llm_outputs, texts, summaries):
    formatted_descriptions = transform_to_enumerated_descriptions(raw_descriptions)
    model = chose_model(args.model_judge, args.judgement_temp_save_dir, args.llamaapi_judgement)
    outputs = []
    errors = []
    prices = []
    inputs = []
    prompt = args.judgement_prompt
    past_text_prompt = args.judgement_past_text_prompt
    for human_description, llm_description, text, summary in tqdm(zip(formatted_descriptions, llm_outputs, texts, summaries)):
        input = prompt + '\n\n' "Summary:\n" + summary + '\n' + "Gold label: \n" + human_description + '\n' + 'Predicted Output: \n' + llm_description + '\n' + past_text_prompt
        inputs.append(input)
        output, error, price = model.call(input, args.max_new_tokens)
        outputs.append(output)
        errors.append(error)
        prices.append(price)
    df = pd.DataFrame(
        {'descriptions': raw_descriptions, 'input': inputs, 'llm_outputs_judged': llm_outputs, 'text': texts,
         'model_summary': summaries,
         'judgement_output': outputs, 'error': errors,
         'price': prices})
    if args.judgment_file is not None:
        output_dir = args.output_dir
        current_path = os.path.join(output_dir, args.judgment_file + '.csv')
        df.to_csv(current_path)
    return outputs

def impute_dicts(precision_dicts,recall_dicts,discarded_samples):
    for i in discarded_samples:
        precision_dicts = precision_dicts[:i] + [None] + precision_dicts[i:]
        recall_dicts = recall_dicts[:i] + [None] + recall_dicts[i:]
    return precision_dicts,recall_dicts

def evaluate():
    args = parse_args()
    if os.path.exists(os.path.join(args.output_dir, "args.json")):
        if not check_args_the_same(args):
            raise ValueError("The arguments are different from the previous run")
    else:
        with open(os.path.join(args.output_dir, "args.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    texts, summaries, raw_descriptions, outputs, judgement_outputs = get_data(args)
    if args.infer:
        if outputs is not None:
            raise ValueError("No need to reinfer")
        outputs = llm_inference(args, texts, summaries)
    if args.judge:
        if judgement_outputs is not None:
            raise ValueError("No need to rejudge")
        judgement_outputs = judge(args, raw_descriptions, outputs, texts, summaries)
    precision_dicts, recall_dicts,discarded_samples = llm_judgment_to_precision_recall_matching(judgement_outputs, raw_descriptions)
    metrics = compute_metrics(precision_dicts, recall_dicts)
    metrics['number_of_discarded_samples'] = len(discarded_samples)
    metrics['discarded_samples'] = discarded_samples
    precision_dicts,recall_dicts = impute_dicts(precision_dicts,recall_dicts,discarded_samples)
    final_df = pd.DataFrame(
        {'text': texts, "model_summary": summaries, "raw_descriptions": raw_descriptions, "llm_outputs": outputs,
         "judgment_outputs": judgement_outputs, "precision": precision_dicts, "recall": recall_dicts})
    output_dir = args.output_dir
    curr_path = os.path.join(output_dir, args.final_results_file)
    final_df.to_csv(curr_path+ '.csv')
    with open(curr_path + ".json", "w") as json_file:
        json.dump(metrics, json_file, indent=4)


def main():
    evaluate()


if __name__ == "__main__":
    main()

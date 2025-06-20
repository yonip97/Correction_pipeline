import argparse
import pandas as pd
from tqdm import tqdm
from constants import MODEL_PRICE_MAP as model_price_map, DTYPE_MAP as dtype_map
import os
from inference_utils import chose_model
import concurrent.futures


def create_dir_run(split, directory, inference_model, inference_prompt_path):
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    directory = os.path.join(directory, inference_model)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    prompt_type = inference_prompt_path.split('/')[-3]
    prompt_number = inference_prompt_path.split('/')[-2].replace('prompt', '')
    prompt_name = f"{prompt_type}_{prompt_number}"
    if not os.path.isdir(os.path.join(directory, prompt_name)):
        os.makedirs(os.path.join(directory, prompt_name))
    final_path = os.path.join(directory, prompt_name, split)
    if os.path.isdir(final_path):
        raise ValueError(f"Directory {final_path} already exists.")
    else:
        os.makedirs(final_path)
        return final_path


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    args.add_argument('-prompt_path', type=str)
    args.add_argument('-past_text_prompt_path', type=str)
    args.add_argument('-max_new_tokens', type=int, default=1000)
    args.add_argument('-device_map', type=str)
    args.add_argument('-dtype', type=str)
    args.add_argument('-data_path', type=str)
    args.add_argument('-num_of_samples', type=int)
    args.add_argument('-api_key', type=str)
    args.add_argument('-llamaapi', action='store_true')
    args.add_argument('-azure', action='store_true')
    args.add_argument('-sample_id', type=int)
    args.add_argument('-output_dir', type=str)
    args.add_argument('-split', type=str)
    args.add_argument('-parallel', action='store_true')
    args.add_argument("-batch_size", type=int, default=1)
    args.add_argument("-seed", type=int, default=42)
    args.add_argument("-sample", action='store_true')
    args = args.parse_args()
    if args.model in model_price_map:
        args.input_price = model_price_map[args.model]['input']
        args.output_price = model_price_map[args.model]['output']
    else:
        args.input_price = 0
        args.output_price = 0
    if args.dtype in dtype_map:
        args.torch_dtype = dtype_map[args.dtype]
    if args.prompt_path is not None:
        with open(args.prompt_path, 'r', encoding='windows-1252') as file:
            args.prompt = file.read()
            args.prompt = args.prompt.strip()
    else:
        args.prompt = ''
    if args.past_text_prompt_path is not None:
        with open(args.past_text_prompt_path, 'r', encoding='windows-1252') as file:
            args.past_text_prompt = file.read()
            args.past_text_prompt = args.past_text_prompt.strip()
    else:
        args.past_text_prompt = ''
    args.output_dir = create_dir_run(args.split, args.output_dir, args.model, args.prompt_path)
    args.output_path = os.path.join(args.output_dir, 'results.csv')
    if args.output_path is not None:
        args.temp_save_dir = os.path.dirname(args.output_path)
    else:
        args.temp_save_dir = None
    return args


def get_data(args):
    df = pd.read_csv(args.data_path)
    df = df[['text', 'model_summary', "descriptions"]]
    df.drop_duplicates(inplace=True)
    if args.num_of_samples is not None:
        if args.sample is not None:
            df = df.sample(n=args.num_of_samples, random_state=args.seed)
        else:
            df = df[:args.num_of_samples]
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    descriptions = df['descriptions'].tolist()
    descriptions = [eval(x) for x in descriptions]
    return texts, summaries, descriptions


def get_sample(args):
    texts, summaries, descriptions = get_data(args)
    text = texts[args.sample_id]
    summary = summaries[args.sample_id]
    description = descriptions[args.sample_id]
    return text, summary, description


def call(args, model, texts, summaries):
    outputs = []
    errors = []
    prices = []
    inputs = []
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt
    for text, summary in tqdm(zip(texts, summaries)):
        input = prompt + '\n\n' 'Text: \n' + text + '\n' + 'Summary: \n' + summary + '\n' + past_text_prompt + '\n'
        inputs.append(input)
        output, error, price = model.call(input, args.max_new_tokens)
        outputs.append(output)
        errors.append(error)
        prices.append(price)
    return inputs, outputs, errors, prices


def call_batches(args, model, contents):
    outputs = []
    errors = []
    prices = []
    inputs = []
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt
    batch_inputs = []
    batch_counter = 0
    for content in tqdm(contents):
        input = f"{prompt}\n\n{content}\n{past_text_prompt}\n"
        batch_inputs.append(input)
        inputs.append(input)
        batch_counter += 1
        if batch_counter == args.batch_size:
            batch_outputs, batch_errors, batch_prices = model.call(batch_inputs, args.max_new_tokens)
            for output, error, price in zip(batch_outputs, batch_errors, batch_prices):
                outputs.append(output)
                errors.append(error)
                prices.append(price)
            batch_inputs = []
            batch_counter = 0
    if len(batch_inputs) > 0:
        batch_outputs, batch_errors, batch_prices = model.call(batch_inputs, args.max_new_tokens)
        for output, error, price in zip(batch_outputs, batch_errors, batch_prices):
            outputs.append(output)
            errors.append(error)
            prices.append(price)
    return inputs, outputs, errors, prices


def call_parallel(args, model, contents, max_workers=10):
    inputs = [None] * len(contents)  # Ensure correct indexing
    outputs = [None] * len(contents)
    errors = [None] * len(contents)
    prices = [None] * len(contents)
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt

    def call_model(index, input_text):
        return index, model.call(input_text, max_new_tokens=args.max_new_tokens)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for idx, content in enumerate(contents):
            llm_input = f"{prompt}\n\n{content}\n{past_text_prompt}\n"
            inputs[idx] = llm_input  # Save input immediately at correct index
            futures[executor.submit(call_model, idx, llm_input)] = idx  # Store future by index

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(contents)):
            idx = futures[future]  # Retrieve the index
            try:
                _, (output, error, price) = future.result()
            except Exception as e:
                output, error, price = None, str(e), 0  # Handle exceptions safely

            outputs[idx] = output
            errors[idx] = error
            prices[idx] = price

    return inputs, outputs, errors, prices


def create_content_for_detection(texts, summaries):
    contents = []
    for text, summary in zip(texts, summaries):
        contents.append(f"Text: \n{text}\nSummary: \n{summary}")
    return contents


def inference_for_factual_inconsistencies_detection():
    args = parse_args()
    model = chose_model(args.model, args.temp_save_dir, args.llamaapi, azure=args.azure, dtype=args.dtype,
                        device_map=args.device_map, pipline=False)
    texts, summaries, _ = get_data(args)
    contents = create_content_for_detection(texts, summaries)
    if args.parallel:
        inputs, outputs, errors, prices = call_parallel(args, model, contents)
    else:
        inputs, outputs, errors, prices = call_batches(args, model, contents)

    df = pd.DataFrame({'text': texts, 'model_summary': summaries, 'input': inputs, 'output': outputs, 'error': errors,
                       'price': prices})
    df.to_csv(args.output_path)


def send_sample():
    args = parse_args()
    model = chose_model(args.model, args.temp_save_dir, args.llamaapi, azure=args.azure, dtype=None, device_map=None)
    text, summary, _ = get_sample(args)
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt
    input = prompt + '\n\n' 'Text: \n' + text + '\n' + 'Summary: \n' + summary + '\n' + past_text_prompt + '\n'
    output, error, price = model.call(input, args.max_new_tokens)
    print(output)
    print(price)


def create_content_for_classification(texts,summaries, descriptions):
    contents = []
    for text,summary, description in zip(texts,summaries, descriptions):
        content = f"Text:\n{text}\n Summary: \n{summary}\n Factual inconsistency description:\n {description}"
        contents.append(content)
    return contents


def inference_for_factual_inconsistencies_classification():
    args = parse_args()
    model = chose_model(args.model, args.temp_save_dir, args.llamaapi, azure=args.azure, dtype=args.dtype,
                        device_map=args.device_map, pipline=False)
    texts, summaries, descriptions = get_data(args)
    texts = [[text for i in range(len(description))] for text, description in zip(texts, descriptions)]
    texts = [item for sublist in texts for item in sublist]
    summaries = [[summary for i in range(len(description))] for summary, description in zip(summaries, descriptions)]
    summaries = [item for sublist in summaries for item in sublist]
    descriptions = [item for sublist in descriptions for item in sublist]
    contents = create_content_for_classification(texts,summaries, descriptions)
    if args.parallel:
        inputs, outputs, errors, prices = call_parallel(args, model, contents)
    else:
        inputs, outputs, errors, prices = call_batches(args, model, contents)

    df = pd.DataFrame(
        {'text': texts, 'model_summary': summaries, "factual_inconsistency_descriptions": descriptions, 'input': inputs,
         'output': outputs, 'error': errors,
         'price': prices})
    df.to_csv(args.output_path)


def send_together():
    args = parse_args()
    model = chose_model(args.model, args.temp_save_dir, args.llamaapi, azure=args.azure, dtype=args.dtype,
                        device_map=args.device_map, pipline=False)
    texts, summaries, descriptions = get_data(args)

    summaries = [[summary for i in range(len(description))] for summary, description in zip(summaries, descriptions)]
    summaries = [item for sublist in summaries for item in sublist]
    descriptions = [item for sublist in descriptions for item in sublist]
    contents = create_content_for_classification(summaries, descriptions)
    input = args.prompt + '\n\n'
    for i in range(len(contents)):
        input += f"Factual inconsistency {i}:\n" + contents[i] + "\n"
    input += args.past_text_prompt + '\n'
    output, error, price = model.call(input, args.max_new_tokens)
    print(output)


def send_sample_classification():
    args = parse_args()
    model = chose_model(args.model, args.temp_save_dir, args.llamaapi, azure=args.azure, dtype=None, device_map=None)
    df = pd.read_csv(args.data_path)
    texts = df['text'].tolist()
    raw_descriptions = df['descriptions'].tolist()
    raw_descriptions = [eval(x) for x in raw_descriptions]
    summaries = df['model_summary'].tolist()
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt
    input = prompt + '\n\n' "Text:\n" + texts[args.sample_id] + "\n" + 'Summary: \n' + summaries[
        args.sample_id] + '\n' + "Factual inconsistency:\n" + \
            raw_descriptions[args.sample_id][0] + '\n' + past_text_prompt + '\n'
    output, error, price = model.call(input, args.max_new_tokens)
    print(output)


if __name__ == '__main__':
    # main()
    # send_sample()
    # inference_for_factual_inconsistencies_classification()
    # send_together()
    #send_sample_classification()

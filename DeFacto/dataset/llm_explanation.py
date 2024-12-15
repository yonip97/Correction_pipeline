from call_llms import ModelCaller, OpenAICaller, AnthropicCaller, WatsonCaller, GeminiCaller
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from constants import MODEL_PRICE_MAP as model_price_map, DTYPE_MAP as dtype_map


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-output_path')
    args.add_argument('-model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
    args.add_argument('-prompt_path', type=str)
    args.add_argument('-past_text_prompt_path', type=str)
    args.add_argument('-max_new_tokens', type=int, default=1000)
    args.add_argument('-device_map', type=str)
    args.add_argument('-dtype', type=str)
    args.add_argument('-data_path', type=str)
    args.add_argument('-num_of_samples', type=int)
    args.add_argument('-api_key', type=str)
    args.add_argument('-watson', action='store_true')
    args.add_argument('-azure', action='store_true')
    args.add_argument('-temp_save_dir', type=str)
    args.add_argument('-iam_token', type=str)
    args.add_argument('-hf_token', type=str)
    args.add_argument('-project_id', type=str)
    args.add_argument('-sample_id', type=int)
    args = args.parse_args()
    if args.model in model_price_map:
        args.input_price = model_price_map[args.model]['input']
        args.output_price = model_price_map[args.model]['output']
    if args.dtype in dtype_map:
        args.torch_dtype = dtype_map[args.dtype]
    if args.prompt_path is not None:
        with open(args.prompt_path, 'r', encoding='windows-1252') as file:
            args.prompt = file.read()
            args.prompt = args.prompt.strip()
    else:
        args.prompt = ''
    if args.past_text_prompt_path is not None:
        with open(args.past_text_prompt_path, 'r') as file:
            args.past_text_prompt = file.read()
            args.past_text_prompt = args.past_text_prompt.strip()
    else:
        args.past_text_prompt = ''
    return args


def get_data(args):
    df = pd.read_csv(args.data_path)
    df = df[['text', 'model_summary']]
    df.drop_duplicates(inplace=True)
    if args.num_of_samples is not None:
        df = df[:args.num_of_samples]
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    return texts, summaries


def get_sample(args):
    texts, summaries = get_data(args)
    text = texts[args.sample_id]
    summary = summaries[args.sample_id]
    return text, summary


def chose_model(args):
    if 'gpt' in args.model:
        return OpenAICaller(model=args.model, azure=args.azure, temp_save_dir=args.temp_save_dir,
                            input_price=args.input_price, output_price=args.output_price)
    elif 'gemini' in args.model:
        return GeminiCaller(model=args.model, temp_save_dir=args.temp_save_dir,
                            input_price=args.input_price,
                            output_price=args.output_price)
    elif 'claude' in args.model:
        return AnthropicCaller(model=args.model, temp_save_dir=args.temp_save_dir,
                               input_price=args.input_price, output_price=args.output_price)
    elif args.watson:
        return WatsonCaller(model=args.model, temp_save_dir=args.temp_save_dir,
                            input_price=args.input_price, output_price=args.output_price)
    else:
        return ModelCaller(model_id=args.model, device_map=args.device_map,
                           torch_dtype=args.dtype)


def main():
    args = parse_args()
    model = chose_model(args)
    texts, summaries = get_data(args)
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
    df = pd.DataFrame({'text': texts, 'model_summary': summaries, 'input': inputs, 'output': outputs, 'error': errors,
                       'price': prices})
    df.to_csv(args.output_path)





def send_sample():
    args = parse_args()
    model = chose_model(args)
    text, summary = get_sample(args)
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt
    input = prompt + '\n' 'Text: \n' + text + '\n' + 'Summary: \n' + summary + '\n' + past_text_prompt + '\n'
    output, error, price = model.call(input, args.max_new_tokens)
    print(output)
    print(price)


if __name__ == '__main__':
    main()
    # send_sample()


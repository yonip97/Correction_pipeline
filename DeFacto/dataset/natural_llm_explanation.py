from call_llms import ModelCaller, OpenAICaller, AnthropicCaller, WatsonCaller, GeminiCaller
import torch
import argparse
import pandas as pd
from tqdm import tqdm


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
    args.add_argument('-num_of_samples', type=int, default=100)
    args.add_argument('-api_key', type=str)
    args.add_argument('-watson', action='store_true')
    args.add_argument('-azure', action='store_true')
    args.add_argument('-temp_save_dir', type=str)
    args.add_argument('-iam_token', type=str)
    args.add_argument('-hf_token', type=str)
    args.add_argument('-project_id', type=str)
    args = args.parse_args()
    dtype_map = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}
    model_price_map = {'gpt-4-o': {'input': 2.5, 'output': 10},
                       'claude-3-5-sonnet-20241022': {'input': 3, 'output': 15},
                       'gemini-1.5-pro': {'input': 1.5, 'output': 5},
                       "llama-3-1-70b-instruct": {'input': 1.8, 'output': 1.8},
                       'llama-3-405b-instruct': {'input': 5, 'output': 16}}
    if args.model in model_price_map:
        args.input_price = model_price_map[args.model]['input']
        args.output_price = model_price_map[args.model]['output']
    args.torch_dtype = dtype_map[args.dtype]
    if args.prompt_path is not None:
        with open(args.prompt_path, 'r') as file:
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
    df = df[:args.num_of_samples]
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    return texts, summaries


def chose_model(args):
    if 'gpt' in args.model:
        return OpenAICaller(model=args.model, api_key=args.api_key, azure=args.azure, temp_save_dir=args.temp_save_dir,
                            input_price=args.input_price, output_price=args.output_price)
    elif 'gemini' in args.model:
        return GeminiCaller(model=args.model, temp_save_dir=args.temp_save_dir, api_key=args.api_key,
                            input_price=args.input_price,
                            output_price=args.output_price)
    elif 'claude' in args.model:
        return AnthropicCaller(model=args.model, temp_save_dir=args.temp_save_dir, api_key=args.api_key,
                               input_price=args.input_price, output_price=args.output_price)
    elif args.watson:
        return WatsonCaller(model=args.model, temp_save_dir=args.temp_save_dir, iam_token=args.iam_token,
                            hf_token=args.hf_token, project_id=args.project_id,
                            input_price=args.input_price, output_price=args.output_price)
    else:
        return ModelCaller(model_id=args.model, device_map=args.device_map,
                           hf_token=args.hf_token, torch_dtype=args.dtype)


def main():
    args = parse_args()
    model = chose_model(args)
    # model = ModelCaller(model_id=args.model, device_map=args.device_map,
    #                     hf_token="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",
    #                     torch_dtype=args.dtype)
    texts, summaries = get_data(args)
    outputs = []
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt

    for text, summary in tqdm(zip(texts, summaries)):
        input = prompt + '\n' 'Text: ' + text + '\n' + 'Summary: ' + summary + '\n' + past_text_prompt + '\n'
        output, _, _ = model.call(input, args.max_new_tokens)
        outputs.append(output)
    df = pd.DataFrame({'text': texts, 'model_summary': summaries, 'output': outputs})
    df.to_csv(args.output_path)


if __name__ == '__main__':
    main()

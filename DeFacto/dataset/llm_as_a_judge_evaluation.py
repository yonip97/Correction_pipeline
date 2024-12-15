from call_llms import ModelCaller, OpenAICaller, AnthropicCaller, WatsonCaller, GeminiCaller
import argparse
import pandas as pd
from tqdm import tqdm
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
    args.add_argument('-llm_output_path', type=str)
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
    df = pd.read_csv(args.data_path, encoding='utf-8')
    df = df[['text', 'explanation', 'model_summary']]
    df.drop_duplicates(inplace=True)
    df2 = pd.read_csv(args.llm_output_path, encoding='utf-8')
    df2 = df2[['output', 'model_summary']]
    df2.dropna(inplace=True)
    df = df.merge(df2, on='model_summary', how='inner')

    if args.num_of_samples is not None:
        df = df[:args.num_of_samples]
    llm_outputs = df['output'].tolist()
    explanations = df['explanation'].tolist()
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    return llm_outputs, explanations, texts, summaries


def get_llm_outputs(args):
    df = pd.read_csv(args.llm_output_path)
    df = df[['output']]
    if args.num_of_samples is not None:
        df = df[:args.num_of_samples]
    outputs = df['output'].tolist()
    return outputs





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
    llm_outputs, explanations, texts, summaries = get_data(args)
    outputs = []
    errors = []
    prices = []
    inputs = []
    prompt = args.prompt
    past_text_prompt = args.past_text_prompt
    for human_exp, llm_exp, text, summary in tqdm(zip(explanations, llm_outputs, texts, summaries)):
        # input = prompt + '\n\n' + 'text:\n' + text + '\n' + 'summary:\n' + summary + '\n' + 'llm explanation: \n' + llm_exp + '\n' + 'human explanation: \n' + human_exp + '\n' + past_text_prompt + '\n'
        input = prompt + '\n\n' + 'llm explanation: \n' + llm_exp + '\n' + 'human explanation: \n' + human_exp + '\n' + past_text_prompt
        inputs.append(input)
        # input = prompt + '\n\n' 'explanation: \n' + human_exp + '\n' + past_text_prompt + '\n'
        output, error, price = model.call(input, args.max_new_tokens)
        outputs.append(output)
        errors.append(error)
        prices.append(price)
    df = pd.DataFrame(
        {'explanation': explanations, 'input': inputs, 'llm_output': llm_outputs, 'text': texts, 'summary': summaries,
         'output': outputs, 'error': errors,
         'price': prices})
    df.to_csv(args.output_path)


if __name__ == '__main__':
    main()

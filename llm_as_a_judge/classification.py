import os
import sys

os.chdir('../')
sys.path.append(os.getcwd())
import os
import pandas as pd
from tqdm import tqdm
from general.LLMS import LLM_as_a_judge
import argparse
from general.llama import LLama
import torch
from general.utils import iter_list


def parseargs_llms():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_path')
    parser.add_argument('-model_name', type=str, default='mock')
    parser.add_argument('-prompt_path', type=str)
    parser.add_argument('-past_text_prompt_path', type=str)
    parser.add_argument('-API_KEY_model', type=str, default=None)
    parser.add_argument('-temp_dir_path', type=str, default='contingency_tables')
    parser.add_argument('-data_dir', type=str, default="")
    parser.add_argument('-data_file', type=str)
    parser.add_argument('-azure', action='store_true')
    parser.add_argument('-groq', action='store_true')
    parser.add_argument('-device', type=str, default='auto')
    parser.add_argument('-max_encoding_length', type=int, default=4096)
    parser.add_argument('-max_new_tokens', type=int, default=1000)
    parser.add_argument('-top_p', type=float, default=0.9)
    parser.add_argument('-temperature', type=float, default=0.9)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-output', type=str, default='output')
    parser.add_argument('-resume', action='store_true')
    args = parser.parse_args()
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
    args.temp_dir_path = os.path.join(args.data_dir, args.temp_dir_path)
    if not os.path.exists(args.temp_dir_path):
        os.makedirs(args.temp_dir_path)
    model_prices = {'gpt-4-turbo': {'input': 0.01, 'output': 0.03}}
    if args.model_name in model_prices:
        args.input_price = model_prices[args.model_name]['input']
        args.output_price = model_prices[args.model_name]['output']
    return args


def get_data(args):
    df = pd.read_csv(os.path.join(args.data_dir, args.data_file), index_col=0)
    texts = df['text'].tolist()
    model_summaries = df['model_summary'].tolist()
    return texts, model_summaries


def main():
    args = parseargs_llms()
    all_texts, all_summaries = get_data(args)
    if 'gpt' in args.model_name:
        if args.resume:
            df = pd.read_csv(args.output_path + '.csv', index_col=0)
            outputs = df[args.output].tolist()
            errors = df['error'].tolist()
            prices = df['price'].tolist()
        else:
            outputs = []
            errors = []
            prices = []
        model = LLM_as_a_judge(temp_save_dir=args.temp_dir_path, prompt=args.prompt,
                               past_text_prompt=args.past_text_prompt,
                               model=args.model_name,
                               API_KEY=args.API_KEY_model, azure=args.azure, input_price=args.input_price,
                               output_price=args.output_price)
        for text, summary in tqdm(zip(all_texts[len(outputs):], all_summaries[len(outputs):])):
            output, error, price = model.call(text=text, summary=summary, max_length=args.max_new_tokens)
            outputs.append(output)
            errors.append(error)
            prices.append(price)
        pd.DataFrame.from_dict(
            {'text': all_texts, 'model_summary': all_summaries, args.output: outputs,
             'error': errors, 'price': prices}).to_csv(
            args.output_path + '.csv')
        print(f"Generation cost was {sum(prices)}")
    elif 'llama' in args.model_name:
        if args.resume:
            df = pd.read_csv(args.output_path + '.csv', index_col=0)
            outputs = df[args.output].tolist()
        else:
            outputs = []
        model = LLama(model_id=args.model_name, device=args.device, dtype=torch.bfloat16)
        generation_kwargs = {'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature,
                             'top_p': args.top_p}
        tokenizer_kwargs = {'truncation': True, 'padding': 'longest', 'max_length': args.max_encoding_length}
        role = "You are a factual consistency classifier."
        for indexes in tqdm(iter_list(list(range(len(all_texts))[len(outputs):]), args.batch_size)):
            batch = []
            for i in indexes:
                text = all_texts[i]
                summary = all_summaries[i]
                #input = args.prompt + '\n' + 'Premise: \n' + text + '\n' + 'hypothesis: \n' + summary + '\n' + args.past_text_prompt
                input = args.prompt + '\n' + 'Document: \n' + text + '\n' + 'Summary: \n' + summary + '\n' + args.past_text_prompt
                batch.append(input)
            batch_outputs = model.call(inputs=batch, generation_kwargs=generation_kwargs,
                                       tokenizer_kwargs=tokenizer_kwargs, score=False)
            outputs += batch_outputs
            pd.DataFrame.from_dict(
                {'text': all_texts[:len(outputs)], 'model_summary': all_summaries[:len(outputs)],
                 args.output: outputs}).to_csv(
                args.output_path + '.csv')
    else:
        raise ValueError("No such model exists!")


def analysis():
    import numpy as np
    from collections import Counter
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/llm_as_a_judge/data/classification/prompt1/results_llama_8B.csv",
        index_col=0)
    outputs = df['output'].tolist()
    answers = [x.split('assistant\n')[-1].strip() for x in outputs]
    #answers = [x.split('Answer:')[-1].strip().replace('.','') for x in answers]
    counter = Counter(answers)
    print(counter)
    map = {'Yes': 1, 'No': 0}
    answers = [map[x] if x in map else 2 for x in answers]
    print(Counter(answers))
    answers = np.array(answers)
    gt_df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/all_data.csv", index_col=0)
    gt_answers = gt_df['error_in_model_summary'].tolist()[:len(answers)]
    print(Counter(gt_answers))
    gt_answers = np.array(gt_answers)
    print(np.sum(answers == gt_answers) / len(answers))
    from sklearn.metrics import confusion_matrix, classification_report,balanced_accuracy_score
    print(confusion_matrix(gt_answers, answers))
    print(classification_report(gt_answers, answers))
    print(balanced_accuracy_score(gt_answers, answers))


if __name__ == "__main__":
    #main()
    analysis()
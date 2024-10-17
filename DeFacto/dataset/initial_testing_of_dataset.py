import pandas as pd
import google.generativeai as genai
import argparse
import time
from tqdm import tqdm
from google.api_core.exceptions import ResourceExhausted


def call_gemini(gen_config, prompt, document, summary, post_prompt):
    input = prompt + "Document:\n" + document + '\n' + "Summary:\n" + summary + "\n" + post_prompt
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(input, generation_config=gen_config)
    except ResourceExhausted as e:
        print(f"ResourceExhausted: {e}")
        return "ResourceExhausted"
    if response.candidates[0].finish_reason.name != 'STOP':
        return response.candidates[0].finish_reason.name
    return response.text


def test_data(df, prompt, post_prompt, gen_config):
    texts = df['text'].tolist()
    model_summaries = df['model_summary'].tolist()
    responses = []
    for text, summary in tqdm(zip(texts, model_summaries)):
        response = call_gemini(gen_config, prompt, text, summary, post_prompt)
        if response == "ResourceExhausted":
            print("ResourceExhausted, waiting 60 seconds")
            time.sleep(60)
            response = call_gemini(gen_config, prompt, text, summary, post_prompt)
            if response == "ResourceExhausted":
                print("ResourceExhausted, bailing out")
                return responses +[None]* (len(texts) - len(responses))
        responses.append(response)
        # time.sleep(4)
    return responses


def data_need_marking(df):
    df = df[df['What is the problem'] == 'marking']
    df_inputs = df[['text', 'model_summary']]
    df_inputs.drop_duplicates(inplace=True)
    return df_inputs


def data_splitting(df):
    df = df[df['What is the problem'] == 'splitting']
    df_inputs = df[['text', 'model_summary']]
    df_inputs.drop_duplicates(inplace=True)
    return df_inputs


def data_clear_or_explanation_faulty_or_classification_needed(df):
    df = df[(df['marking'] == 0) & (df['splitting'] == 0) & (df['marking(middle)'] == 0)]
    df_inputs = df[['text', 'model_summary']]
    df_inputs.drop_duplicates(inplace=True)
    return df_inputs


def get_data():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/Dataset construction - initial_data_for_annotation.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True)
    df['What is the problem'] = df['What is the problem'].astype(str)
    df = df[~df['What is the problem'].str.contains('rewrite', case=False, na=False)]
    df = df[~df['What is the problem'].str.contains('quality', case=False, na=False)]
    df = df[~df['What is the problem'].str.contains('no change', case=False, na=False)]
    df = df[~df['What is the problem'].str.contains('not English', case=False, na=False)]
    df = df[~(df['What is the problem'] == "????")]
    df['What is the problem'] = df['What is the problem'].str.replace('?', '', case=False, regex=False).str.strip()
    df['What is the problem'] = df['What is the problem'].str.replace('nan', 'clear', case=False,
                                                                      regex=False).str.strip()

    df_dummies = df['What is the problem'].str.get_dummies(sep=',')
    df = pd.concat([df, df_dummies], axis=1)
    return df


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default=None)
    parser.add_argument('--post_prompt_path', type=str, default=None)
    parser.add_argument('--max_output_tokens', type=int, default=200)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--API_KEY', type=str, default=None)
    args = parser.parse_args()
    if args.prompt_path is not None:
        with open(args.prompt_path, 'r') as file:
            args.prompt = file.read()
    else:
        args.prompt = ""
    if args.post_prompt_path is not None:
        with open(args.post_prompt_path, 'r') as file:
            args.post_prompt = file.read()
    else:
        args.post_prompt = ""
    return args


def main():
    args = get_args()
    genai.configure(api_key=args.API_KEY)
    df = get_data()
    df_need_marking = data_need_marking(df)
    print(len(df_need_marking))
    df_splitting = data_splitting(df)
    print(len(df_splitting))
    df_clear = data_clear_or_explanation_faulty_or_classification_needed(df)
    print(len(df_clear))
    prompt = args.prompt
    post_prompt = args.post_prompt
    gen_config = {"max_output_tokens": args.max_output_tokens}
    marking_responses = test_data(df_need_marking, prompt, post_prompt, gen_config)
    splitting_responses = test_data(df_splitting, prompt, post_prompt, gen_config)
    clear_responses = test_data(df_clear, prompt, post_prompt, gen_config)
    marking_output_path = args.output_path + "marking_responses.csv"
    splitting_output_path = args.output_path + "splitting_responses.csv"
    clear_output_path = args.output_path + "clear_responses.csv"


    df_need_marking['response'] = marking_responses
    df_splitting['response'] = splitting_responses
    df_clear['response'] = clear_responses
    df_need_marking.to_csv(marking_output_path)
    df_splitting.to_csv(splitting_output_path)
    df_clear.to_csv(clear_output_path)


if __name__ == "__main__":
    main()

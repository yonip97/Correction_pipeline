
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import google.generativeai as genai
import argparse
import time
from tqdm import tqdm
from google.api_core.exceptions import ResourceExhausted
import re
from fuzzywuzzy import fuzz
from scipy.optimize import linear_sum_assignment
import evaluate
import Levenshtein
Rouge = evaluate.load('rouge')

def call_gemini(gen_config, prompt, document, summary, post_prompt):
    input = prompt + "Document:\n" + document + '\n' + "Summary:\n" + summary + "\n" + post_prompt
    model = genai.GenerativeModel("gemini-1.5-flash")
    try:
        response = model.generate_content(input, generation_config=gen_config)
    except ResourceExhausted as e:
        print(f"ResourceExhausted: {e}")
        return "ResourceExhausted"
    if len(response.candidates) == 0:
        return response.prompt_feedback.block_reason.name
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
                return responses + [None] * (len(texts) - len(responses))
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
    grouped_df = df.groupby(['text', 'model_summary'])

    def contains_excluded_substring(group):
        exclude_values = ['marking','splitting']
        return not any(group['What is the problem'].apply(lambda val: any(excl in val for excl in exclude_values)))

    # Apply the filtering condition: check if none of the values contain the target_value
    filtered_samples = grouped_df.filter(contains_excluded_substring)
    #
    # df = df[(df['marking'] == 0) & (df['splitting'] == 0) & (df['marking(middle)'] == 0)]
    # df_inputs = df[['text', 'model_summary']]
    filtered_samples = filtered_samples[['text', 'model_summary']].drop_duplicates()
    return filtered_samples


def get_data():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/Dataset_construction_initial_data_for_annotation.csv")
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
    filter_ids_with_all_values(df, 'clear')
    return df


def filter_ids_with_all_values(df, target_value):
    # Group by the two ID-defining columns (e.g., 'text' and 'text2')
    grouped_df = df.groupby(['text', 'model_summary'])

    # Filter groups where all 'value' entries match the target_value
    filtered_df = grouped_df.filter(lambda group: (group['What is the problem'] == 'clear').all())

    # Return the filtered DataFrame with unique IDs
    return filtered_df[['text', 'model_summary']].drop_duplicates()

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
    df = df[['text','model_summary']].drop_duplicates()
    prompt = args.prompt
    post_prompt = args.post_prompt
    gen_config = {"max_output_tokens": args.max_output_tokens}
    responses = test_data(df,prompt,post_prompt,gen_config)
    df['response'] = responses
    df.to_csv(os.path.join(args.output_path,'all.csv'))
def exact_match_score(possible_answers, model_answers):
    total = len(model_answers)
    correct = sum([model_answer in possible_answers for model_answer in model_answers])
    return correct / total


def fuzzy_match_score(model_answers,possible_answers):
    total = len(model_answers)
    if total == 0:
        return 0
    scores = []

    for model_answer in model_answers:
        best_score = max([fuzz.ratio(model_answer, possible_answer) for possible_answer in possible_answers])
        scores.append(best_score)

    average_score = sum(scores) / total
    return average_score / 100


def create_distance_matrix(model_answers, ground_truth_answers):
    distance_matrix = np.zeros((len(model_answers), len(ground_truth_answers)))
    for i, model_answer in enumerate(model_answers):
        for j, ground_truth_answer in enumerate(ground_truth_answers):
            #distance_matrix[i, j] = Levenshtein.distance(model_answer, ground_truth_answer)
            distance_matrix[i, j] = 1 - Rouge.compute(predictions=[model_answer], references=[ground_truth_answer], use_aggregator=False)['rougeL'][0]


    return distance_matrix
def find_best_matches_hungarian(model_answers, ground_truth_answers):
    # Create the distance matrix
    distance_matrix = create_distance_matrix(model_answers, ground_truth_answers)
    # Use the Hungarian algorithm to find the optimal assignment
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    best_matches = []
    for model_idx, ground_truth_idx in zip(row_indices, col_indices):
        best_matches.append((model_answers[model_idx], ground_truth_answers[ground_truth_idx], distance_matrix[model_idx, ground_truth_idx]))
    return best_matches
def compute_rouge(model_answers, ground_truth_answers):
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=model_answers, references=ground_truth_answers, use_aggregator=False)
    return scores['rougeL']

def match_facts(model_predictions, ground_truth):
    all_scores = []
    for prediction, sample in zip(model_predictions,ground_truth):
        x = find_best_matches_hungarian(prediction,sample)
        print(x)
        #score = fuzzy_match_score(prediction,sample)
    #    score = exact_match_score(prediction,sample)
        #all_scores.append(score)
    # import matplotlib.pyplot as plt
    # plt.hist(all_scores)
    # plt.show()
    # print(np.mean(all_scores))


def percentage_below_each_value(numbers):
    total_samples = len(numbers)
    unique_values = sorted(set(numbers))  # Get sorted unique values from the list
    result = {}

    # Iterate over each unique value as the threshold
    for value in unique_values:
        # Count how many numbers are below the current value
        count = sum(1 for num in numbers if num < value)

        # Calculate the percentage
        percentage = (count / total_samples) * 100

        # Store the result in the dictionary
        result[value] = percentage

    return result


def look_at_samples():
    df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/mistake_detection/gemini-flash-1.5-pro/prompt2/marking_responses.csv",index_col=0)
    responses = df['response'].tolist()
    for i in range(len(responses)):
        print(responses[i])
        print('-------------------------------------------------------')



def read():
    temp_df = get_data()
    clear_data = data_clear_or_explanation_faulty_or_classification_needed(temp_df)
    df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/mistake_detection/gemini-flash-1.5-pro/prompt1/clear_responses.csv",index_col=0)
    df = df[(df['text'].isin(clear_data['text']))&(df['model_summary'].isin(clear_data['model_summary']))]
    responses = df['response'].tolist()
    facts = []
    explanations = []
    for response in responses:
        response_facts = re.findall(r'Fact:(.*?)Explanation:', response, re.DOTALL)
        response_facts = [x.strip() for x in response_facts]
        response_explanations = re.findall(r'Explanation:(.*?)(Fact:|$)', response, re.DOTALL)
        response_explanations = [x[0].strip() for x in response_explanations]
        facts.append(response_facts)
        explanations.append(response_explanations)
    texts = df['text'].tolist()
    model_summaries = df['model_summary'].tolist()
    df2 = get_data()
    df2 = df2[df2['text'].isin(texts) & df2['model_summary'].isin(model_summaries)]
    df2 = df2.dropna(subset=['mistake human annotation'])
    ground_truth = df2.groupby(['text', 'model_summary'],sort=False)['mistake human annotation'].apply(list).reset_index()['mistake human annotation'].tolist()
    match_facts(facts,ground_truth)

def removal_analysis():
    df = get_data()
    #df = df[['text','model_summary','revised_summary']].drop_duplicates()
    model_summaries = df['model_summary'].tolist()
    revised_summaries= df['revised_summary'].tolist()
    scores = Rouge.compute(predictions = model_summaries,references=revised_summaries,use_aggregator = False)['rougeL']
    prev = 0
    for i in np.linspace(0.05,1.05,21):
        ls = [j for j in range(len(scores)) if (scores[j]>=prev and scores[j]<i)]
        print(prev,i,len(ls))

        prev = i
        if i <0.4:
            for j in ls:
                print(model_summaries[j])
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/Dataset construction - initial_data_for_annotation.csv")
    df['What is the problem'] = df['What is the problem'].astype(str)
    df_rewrite = df[df['What is the problem'].str.contains('rewrite')]
    df_rewrite = df_rewrite[['text','model_summary','revised_summary']].drop_duplicates()
    model_summaries = df_rewrite['model_summary'].tolist()
    revised_summaries=df_rewrite['revised_summary'].tolist()
    scores = Rouge.compute(predictions = model_summaries,references=revised_summaries,use_aggregator = False)['rougeL']
    for i in range(len(scores)):
        if scores[i] > 0.5:
            print(model_summaries[i])


if __name__ == "__main__":
    main()
    #read()
    #look_at_samples()
    #removal_analysis(),
    get_data()

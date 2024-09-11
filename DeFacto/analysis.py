import os

import numpy as np
import pandas as pd
import json
import textwrap
import evaluate

os.chdir('/data/home/yehonatan-pe/Correction_pipeline')
import sys

sys.path.append(os.getcwd())
from collections import Counter
import matplotlib.pyplot as plt

def look_at_some(num_of_examples):
    dfs = []
    instructions = []
    intrinsic_errors = []
    extrinsic_errors = []
    for name in ['train', 'val', 'test']:
        df = pd.read_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}_scores.csv", index_col=0)
        with open(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}.jsonl") as f:
            data = [json.loads(line) for line in f]
            print(len(data))
            for x in data:
                instructions.append(x['feedback']['instruction'])
                intrinsic_errors.append(x['intrinsic_error'])
                extrinsic_errors.append(x['extrinsic_error'])
        dfs.append(df)
    import textwrap
    df = pd.concat(dfs)
    print(len(df))
    df['instruction'] = instructions
    df['intrinsic_error'] = intrinsic_errors
    df['extrinsic_error'] = extrinsic_errors
    df = df[df['error_in_model_summary'] == True]
    df.reset_index(drop=True, inplace=True)
    df = df[:num_of_examples]
    df.to_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{num_of_examples}_summaries_with_revision.csv")
    print("Stats for revised:")
    print(
        f"Model density mean {df['model_summary_density'].mean():.4f} median {df['model_summary_density'].median():.4f}")
    print(
        f"Model trueteacher mean {df['model_summary_trueteacher'].mean():.4f} median {df['model_summary_trueteacher'].median():.4f}")
    print(
        f"Model seahorse mean {df['model_summary_seahorse'].mean():.4f} median {df['model_summary_seahorse'].median():.4f}")
    print(
        f"Revised density mean {df['revised_summary_density'].mean():.4f} median {df['revised_summary_density'].median():.4f}")
    print(
        f"Revised trueteacher mean {df['revised_summary_trueteacher'].mean():.4f} median {df['revised_summary_trueteacher'].median():.4f}")
    print(
        f"Revised seahorse mean {df['revised_summary_seahorse'].mean():.4f} median {df['revised_summary_seahorse'].median():.4f}")
    print(
        f"rougeL similarity mean {df['revised_summary_rougeL_to_base'].mean():.4f} median {df['revised_summary_rougeL_to_base'].median():.4f}")

    for i in range(num_of_examples):
        print("Text:")
        print()
        text = textwrap.wrap(df.iloc[i]['text'], 200)
        text = "\n".join(text)
        print(text)
        print()
        print("Model summary:")
        print()
        print(df.iloc[i]['model_summary'])
        print()
        print("instructions: ")
        print()
        print(df.iloc[i]['instruction'])
        print()
        print("Revised summary:")
        print()
        print(df.iloc[i]['revised_summary'])
        print()
        print("---------------------------------------------------------------")


def analysis():
    dfs = []
    for name in ['train', 'val', 'test']:
        df = pd.read_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}_scores.csv", index_col=0)
        dfs.append(df)
    df = pd.concat(dfs)
    df_with_revised_summaries = df[df['error_in_model_summary'] == True]
    print("Stats for revised:")
    print(
        f"Model density mean {df_with_revised_summaries['model_summary_density'].mean():.4f} median {df_with_revised_summaries['model_summary_density'].median():.4f}")
    print(
        f"Model trueteacher mean {df_with_revised_summaries['model_summary_trueteacher'].mean():.4f} median {df_with_revised_summaries['model_summary_trueteacher'].median():.4f}")
    print(
        f"Model seahorse mean {df_with_revised_summaries['model_summary_seahorse'].mean():.4f} median {df_with_revised_summaries['model_summary_seahorse'].median():.4f}")
    print(
        f"Model length mean {df_with_revised_summaries['model_summary_length'].mean():.4f} median {df_with_revised_summaries['model_summary_length'].median():.4f}")
    print(
        f"Revised density mean {df_with_revised_summaries['revised_summary_density'].mean():.4f} median {df_with_revised_summaries['revised_summary_density'].median():.4f}")
    print(
        f"Revised trueteacher mean {df_with_revised_summaries['revised_summary_trueteacher'].mean():.4f} median {df_with_revised_summaries['revised_summary_trueteacher'].median():.4f}")
    print(
        f"Revised seahorse mean {df_with_revised_summaries['revised_summary_seahorse'].mean():.4f} median {df_with_revised_summaries['revised_summary_seahorse'].median():.4f}")
    print(
        f"rougeL similarity mean {df_with_revised_summaries['revised_summary_rougeL_to_base'].mean():.4f} median {df_with_revised_summaries['revised_summary_rougeL_to_base'].median():.4f}")
    print(
        f"Revised length mean {df_with_revised_summaries['revised_summary_length'].mean():.4f} median {df_with_revised_summaries['revised_summary_length'].median():.4f}")
    print("Stats not revised:")

    df_without_revised_summaries = df[df['error_in_model_summary'] == False]
    print(
        f"Model density mean {df_without_revised_summaries['model_summary_density'].mean():.4f} median {df_without_revised_summaries['model_summary_density'].median():.4f}")
    print(
        f"Model trueteacher mean {df_without_revised_summaries['model_summary_trueteacher'].mean():.4f} median {df_without_revised_summaries['model_summary_trueteacher'].median():.4f}")
    print(
        f"Model seahorse mean {df_without_revised_summaries['model_summary_seahorse'].mean():.4f} median {df_without_revised_summaries['model_summary_seahorse'].median():.4f}")
    print(
        f"Model length mean {df_without_revised_summaries['model_summary_length'].mean():.4f} median {df_without_revised_summaries['model_summary_length'].median():.4f}")


def join_model_and_data(model_file):
    data_df = pd.read_csv("DeFacto/data/summaries_with_errors_100.csv",
                          index_col=0).reset_index(drop=True)
    model_df = pd.read_csv(
        "DeFacto/data/revision_prompts/" + model_file + '.csv',
        index_col=0).reset_index(drop=True)
    texts = data_df['text'].tolist()
    texts2 = model_df['text'].tolist()
    common_texts = [x for x in texts if x in texts2]
    data_df = data_df[data_df['text'].isin(common_texts)].reset_index(drop=True)
    model_df = model_df[model_df['text'].isin(common_texts)].reset_index(drop=True)
    revised_cols_model = [col for col in model_df.columns if
                          'revised' in col or 'instruction' in col or 'explanation' in col]
    model_df = model_df[revised_cols_model]
    new_cols = {col: col + '_model' for col in model_df.columns}
    model_df.rename(columns=new_cols, inplace=True)
    # instructions = model_df['instruction_model'].tolist()
    # instructions = [x.lower().split('assistant\n')[-1].strip() for x in instructions]
    # revised_summaries = model_df['revised_summary_model'].tolist()
    # revised_summaries = [x.lower().split('assistant\n')[-1].strip().split('\n')[-1].strip() for x in revised_summaries]
    # # model_df['instruction_model'] = instructions
    # model_df['revised_summary_model'] = revised_summaries

    # full_revision_text = model_df['revised_summary_model'].tolist()
    # full_revision_text = [x.split('assistant\n')[-1].strip() for x in full_revision_text]
    # model_instructions = [x.lower().split("steps:")[-1].split('corrected:')[0].strip() for x in full_revision_text]
    # model_explanations = [x.lower().split("steps:")[0].strip() for x in full_revision_text]
    # revised_summaries = [x.lower().split("corrected:")[-1].strip() for x in full_revision_text]
    # model_df['instruction_model'] = model_instructions
    # model_df['explanation_model'] = model_explanations
    # model_df['revised_summary_model'] = revised_summaries

    df = pd.concat([data_df, model_df], axis=1)
    # print(
    #     f"Model density mean {df['model_summary_density'].mean():.4f} median {df['model_summary_density'].median():.4f}")
    # print(
    #     f"Model trueteacher mean {df['model_summary_trueteacher'].mean():.4f} median {df['model_summary_trueteacher'].median():.4f}")
    # print(
    #     f"Model seahorse mean {df['model_summary_seahorse'].mean():.4f} median {df['model_summary_seahorse'].median():.4f}")
    # print(f"Model length mean {df['model_summary_length'].mean():.4f} median {df['model_summary_length'].median():.4f}")
    # print(
    #     f"Revised density mean {df['revised_summary_density'].mean():.4f} median {df['revised_summary_density'].median():.4f}")
    # print(
    #     f"Revised trueteacher mean {df['revised_summary_trueteacher'].mean():.4f} median {df['revised_summary_trueteacher'].median():.4f}")
    # print(
    #     f"Revised seahorse mean {df['revised_summary_seahorse'].mean():.4f} median {df['revised_summary_seahorse'].median():.4f}")
    # print(
    #     f"rougeL similarity mean {df['revised_summary_rougeL_to_base'].mean():.4f} median {df['revised_summary_rougeL_to_base'].median():.4f}")
    # print(
    #     f"Revised length mean {df['revised_summary_length'].mean():.4f} median {df['revised_summary_length'].median():.4f}")
    # print("Model results:")
    # print(
    #     f"Revised density mean {df['revised_summary_density_model'].mean():.4f} median {df['revised_summary_density_model'].median():.4f}")
    # print(
    #     f"Revised trueteacher mean {df['revised_summary_trueteacher_model'].mean():.4f} median {df['revised_summary_trueteacher_model'].median():.4f}")
    # print(
    #     f"Revised seahorse mean {df['revised_summary_seahorse_model'].mean():.4f} median {df['revised_summary_seahorse_model'].median():.4f}")
    # print(
    #     f"rougeL similarity mean {df['rougeL_revised_to_base_model'].mean():.4f} median {df['rougeL_revised_to_base_model'].median():.4f}")
    # print(
    #     f"Revised length mean {df['revised_summary_length_model'].mean():.4f} median {df['revised_summary_length_model'].median():.4f}")
    return df


def compare(df):
    import evaluate
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=df['revised_summary'].str.lower(), references=df['revised_summary_model'].str.lower(),
                           use_aggregator=False)
    scores_2 = rouge.compute(predictions=df['model_summary'].str.lower(), references=df['revised_summary_model'].str.lower(),
                             use_aggregator=False)
    ones = [i for i, x in enumerate(scores['rougeL']) if x == 1]
    print(ones)
    import numpy as np
    print(np.argsort(scores['rougeL']))
    print(sorted(scores['rougeL']))
    print(np.mean(scores['rougeL']))
    print(np.mean(scores['rouge1']))
    print(np.mean(scores['rouge2']))
    print(np.mean(scores_2['rougeL']))
    c = 1
    for i in range(len(df)):
        print("index:", i)
        print()
        print("Text:")
        print()
        text = textwrap.wrap(df.iloc[i]['text'], 200)
        text = "\n".join(text)
        print(text)
        print()
        print("Model summary:")
        print()
        print(df.iloc[i]['model_summary'])
        print()
        print("Data explanation: ")
        print()
        print(df.iloc[i]['explanation'])
        print()
        print("Data instructions: ")
        print()
        print(df.iloc[i]['instruction'])
        print()
        print("Data Revised summary:")
        print()
        print(df.iloc[i]['revised_summary'])
        print()
        print()
        # print("Model explanation: ")
        # print()
        # print(df.iloc[i]['explanation_model'])
        # print()
        print("Model instructions: ")
        print()
        print(df.iloc[i]['instruction_model'])
        print()
        print("Model Revised summary:")
        print()
        print(df.iloc[i]['revised_summary_model'])
        print("RougeL score to base:", scores_2['rougeL'][i])
        print("RougeL score to human revision:", scores['rougeL'][i])
        print("---------------------------------------------------------------")


def post_process():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/revision_prompts/original_cot_prompt/summaries_with_errors_100_llama_3.1_70B.csv",
        index_col=0)
    revised_summary_full_text = df['revised_summary'].tolist()
    instructions = [x.lower().split("steps:")[-1].split('corrected:')[0].strip() for x in revised_summary_full_text]
    explanations = [x.lower().split("steps:")[0].strip() for x in revised_summary_full_text]
    revised_summaries = [x.lower().split("corrected:")[-1].strip() for x in revised_summary_full_text]
    df['instruction'] = instructions
    df['explanation'] = explanations
    df['revised_summary'] = revised_summaries
    return df


def compare_models(path1, path2):
    df = pd.read_csv(path1, index_col=0)
    df2 = pd.read_csv(path2, index_col=0)
    df3 = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/summaries_with_errors_100.csv",
                      index_col=0)
    common_texts = set(df['text'].tolist()).intersection(set(df2['text'].tolist()))
    df = df[df['text'].isin(common_texts)]
    df2 = df2[df2['text'].isin(common_texts)]
    df3 = df3[df3['text'].isin(common_texts)]
    df['explanation'] = [x.lower().split("steps:")[0].strip() for x in df['revised_summary_full_text']]
    df['instruction'] = [x.lower().split("steps:")[-1].split('corrected:')[0].strip() for x in
                         df['revised_summary_full_text']]
    df2['explanation'] = [x.lower().split("steps:")[0].strip() for x in df2['revised_summary_full_text']]
    df2['instruction'] = [x.lower().split("steps:")[-1].split('corrected:')[0].strip() for x in
                          df2['revised_summary_full_text']]
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=df['revised_summary'].tolist(), references=df2['revised_summary'].tolist())
    print(f"between revisions: {scores}")
    scores = rouge.compute(predictions=df['revised_summary'].tolist(), references=df3['revised_summary'].tolist())
    print(f"between data and model 1: {scores}")
    scores = rouge.compute(predictions=df2['revised_summary'].tolist(), references=df3['revised_summary'].tolist())
    print(f"between data and model 2: {scores}")
    scores = rouge.compute(predictions=df['revised_summary'].tolist(), references=df2['revised_summary'].tolist()
                           , use_aggregator=False)['rougeL']
    for i in range(len(df)):
        if scores[i] != 1.0:
            print("can not copy")
        print("index:", i)
        print()
        print("Text:")
        print()
        text = textwrap.wrap(df.iloc[i]['text'], 200)
        text = "\n".join(text)
        print(text)
        print()
        print("Model summary:")
        print()
        print(df.iloc[i]['model_summary'])
        print()
        print("Model explanation: ")
        print()
        print(df.iloc[i]['explanation'])
        print()
        print("Model instructions: ")
        print()
        print(df.iloc[i]['instruction'])
        print()
        print("Model Revised summary:")
        print()
        print(df.iloc[i]['revised_summary'])
        print()
        print("Model 2 explanation: ")
        print()
        print(df2.iloc[i]['explanation'])
        print()
        print("Model 2 instructions: ")
        print()
        print(df2.iloc[i]['instruction'])
        print()
        print("Model 2 Revised summary:")
        print()
        print(df2.iloc[i]['revised_summary'])
        print()
        print("Data explanation: ")
        print()
        print(df3.iloc[i]['explanation'])
        print()
        print("Data instructions: ")
        print()
        print(df3.iloc[i]['instruction'])
        print()
        print("Data Revised summary:")
        print()
        print(df3.iloc[i]['revised_summary'])
        # print(scores[i])
        print("---------------------------------------------------------------")


def conditional_split(text, acceptable_words):
    result = []
    sentence = ''

    words = text.split()
    i = 0
    while i < len(words):
        word = words[i]
        if '.' in word:
            # Extract the word after the dot
            if word.endswith('.'):
                next_word = words[i + 1] if i + 1 < len(words) else ''
            else:
                parts = word.split('.')
                next_word = parts[1] if len(parts) > 1 else ''
                word = parts[0] + '.'

            if next_word and next_word in acceptable_words:
                sentence += word
                result.append(sentence.strip())
                sentence = next_word + ' '
            else:
                sentence += word + ' '
        else:
            sentence += word + ' '
        i += 1

    if sentence:
        result.append(sentence.strip())

    return result


def auto_metrics():
    main_path = "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/revision_prompts"
    for dir in os.listdir(main_path):
        new_path = os.path.join(main_path, dir)
        print(new_path)
        for file in os.listdir(new_path):
            if "scored" in file and 'gpt' in file:
                df = pd.read_csv(os.path.join(new_path, file), index_col=0)
                for col in df.columns:
                    if 'revised' in col and df[col].dtype == 'float64':
                        print(f"{col} mean {df[col].mean():.4f} median {df[col].median():.4f}")
        print()
        print("---------------------------------------------------------------")


def split_mistakes_col(series):
    mistakes = series.str.split('[,.]').tolist()
    mistakes = [[x.strip().replace('?', "") for x in sublist] if sublist is not np.nan else [] for sublist in mistakes]
    return mistakes


def look_at_mistakes(path):
    df = pd.read_csv(path)
    df['type of information not fixed'] = df['type of information not fixed'].str.split('[,.]')
    all_mistakes_not_fixed = df[df['type of information not fixed'].notna()]['type of information not fixed'].tolist()
    all_mistakes_not_fixed = [x.strip().replace('?', "") for sublist in all_mistakes_not_fixed for x in sublist]
    df['type of information fixed'] = df['type of information fixed'].str.split('[,.]')
    all_mistakes_fixed = df[df['type of information fixed'].notna()]['type of information fixed'].tolist()
    all_mistakes_fixed = [x.strip().replace('?', "") for sublist in all_mistakes_fixed for x in sublist]
    from collections import Counter
    mistakes_not_fixed = Counter(all_mistakes_not_fixed)
    mistakes_fixed = Counter(all_mistakes_fixed)
    print(path)
    print("Mistakes not fixed:")
    unsupported_as_supported = df['unsupported as supported not fixed']
    supported_as_unsupported= df['supported as unsupported not fixed']
    edit_fail = df['edit faliure ']
    print(mistakes_not_fixed)
    print(sum(mistakes_not_fixed.values()))
    print(f"Unsupported as supported: {unsupported_as_supported.sum()}", f"Supported as unsupported: {supported_as_unsupported.sum()}, Edit fail: {edit_fail.sum()}")
    sorted_mistakes_not_fixed = dict(sorted(mistakes_not_fixed.items(), key=lambda item: item[1], reverse=True))
    normalized_values = [x / sum(sorted_mistakes_not_fixed.values()) for x in sorted_mistakes_not_fixed.values()]
    plt.bar(sorted_mistakes_not_fixed.keys(), normalized_values)
    plt.xticks(rotation=45)
    plt.title(path.split('/')[-1] +'_not_fixed')
    plt.show()
    print("Mistakes fixed:")
    print(mistakes_fixed)
    print(sum(mistakes_fixed.values()))
    sorted_mistakes_fixed = dict(sorted(mistakes_fixed.items(), key=lambda item: item[1], reverse=True))
    normalized_values = [x / sum(sorted_mistakes_fixed.values()) for x in sorted_mistakes_fixed.values()]
    plt.bar(sorted_mistakes_fixed.keys(), normalized_values)
    plt.xticks(rotation=45)
    plt.title(path.split('/')[-1] +'_fixed')
    plt.show()
    print("Total:")
    print(mistakes_not_fixed + mistakes_fixed)
    print(sum(mistakes_not_fixed.values()) + sum(mistakes_fixed.values()))
    print("---------------------------------------------------------------")


def compare_few_shot_to_original():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT generation.csv")
    df_with_examples = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT few shot.csv")
    df_with_examples['GPT given instructions'] = df_with_examples['GPT given instructions'].str.strip()
    df_with_examples = df_with_examples[
        df_with_examples['GPT given instructions'] != 'the summary is factually consistent with the document.']
    texts = df['Text '].tolist()
    texts2 = df_with_examples['Text '].tolist()
    common_texts = [x for x in texts if x in texts2]
    df = df[df['Text '].isin(common_texts)].reset_index(drop=True)
    df_with_examples = df_with_examples[df_with_examples['Text '].isin(common_texts)].reset_index(drop=True)
    df['type of information fixed'] = split_mistakes_col(df['type of information fixed'])
    df_with_examples['type of information fixed'] = split_mistakes_col(df_with_examples['type of information fixed'])
    df['type of information not fixed'] = split_mistakes_col(df['type of information not fixed'])
    df_with_examples['type of information not fixed'] = split_mistakes_col(
        df_with_examples['type of information not fixed'])
    print("For COT generation:")
    print("Fixed:")
    counter = Counter([x for sublist in df['type of information fixed'].tolist() for x in sublist])

    print(counter)
    print(sum(counter.values()))
    print("Not fixed:")
    counter = Counter([x for sublist in df['type of information not fixed'].tolist() for x in sublist])
    print(counter)
    print(sum(counter.values()))
    print("For COT few shot:")
    print("Fixed:")
    counter = Counter([x for sublist in df_with_examples['type of information fixed'].tolist() for x in sublist])
    print(counter)
    print(sum(counter.values()))
    print("Not fixed:")
    counter = Counter([x for sublist in df_with_examples['type of information not fixed'].tolist() for x in sublist])
    print(counter)
    print(sum(counter.values()))
    print("---------------------------------------------------------------")


def look_at_stats(path):
    df = pd.read_csv(path)
    cols = ['Factuality', 'extractiveness', 'similarity ', 'added information']
    df = df[cols]
    for col in cols:
        print(f"{col} mean {df[col].mean():.4f} ")
def instructions_quality(path):
    df = pd.read_csv(path)
    counter = Counter(df['Revision quality'].tolist())
    print(counter)
def cot_llama_vs_cot_gpt():
    df_cot = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT generation.csv")
    df_llama = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/llama-3.1-70B-instruct - Cot generation.csv")
    cols = ['unsupported as supported not fixed','unsupported as supported fixed']
    df_cot_mistakes_total = df_cot[cols].sum(axis = 1)
    df_llama_mistakes_total = df_llama[cols].sum(axis = 1)
    x = df_llama_mistakes_total == df_cot_mistakes_total
    print(x.sum())

    c = 1
def main():
    df = join_model_and_data("2_steps_generation_paper_prompts/summaries_with_errors_100_revised_with_instructions_gpt_4_scored")
    compare(df)
    # compare_few_shot_to_original()
    #cot_llama_vs_cot_gpt()
    # look_at_mistakes(
    #     "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT generation.csv")
    # look_at_mistakes(
    #     "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT few shot.csv")
    # look_at_mistakes(
    #     "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - 2 steps generation.csv")
    # look_at_mistakes("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/llama-3.1-70B-instruct - Cot generation.csv")
    # look_at_stats( "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT generation.csv")
    # look_at_stats( "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT few shot.csv")
    # look_at_stats( "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - 2 steps generation.csv")
    # look_at_some(10)
    # compare_models( "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/revision_prompts/original_cot_prompt/summaries_with_errors_100_revised_gpt_4_scored.csv",
    #     "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/revision_prompts/original_cot_prompt_with_few_shot_examples/summaries_with_errors_100_revised_gpt_4_scored.csv",
    #               )
    # instructions_quality("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT generation.csv")
    # instructions_quality("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - COT few shot.csv")
    # instructions_quality("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - 2 steps generation.csv")
    # instructions_quality("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/labeled/Different prompts labeling gpt 4 - Paper prompt with instructions.csv")

# auto_metrics()


if __name__ == "__main__":
    main()

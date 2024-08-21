import matplotlib.pyplot as plt
import pandas as pd
import json
import textwrap


def look_at_some(num_of_examples):
    dfs = []
    instructions = []
    intrinsic_errors =[]
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
    print(f"Model length mean {df_with_revised_summaries['model_summary_length'].mean():.4f} median {df_with_revised_summaries['model_summary_length'].median():.4f}")
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
def join_model_and_data():
    data_df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/summaries_with_errors_100.csv",
                          index_col=0).reset_index(drop=True)
    model_df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/revision_prompts/original_cot_prompt/summaries_with_errors_100_llama_3.1_70B.csv",
        index_col=0).reset_index(drop=True)
    revised_cols_model = [col for col in model_df.columns if 'revised' in col]
    model_df = model_df[revised_cols_model]
    new_cols = {col: col + '_model' for col in model_df.columns}
    model_df.rename(columns=new_cols, inplace=True)
    full_revision_text = model_df['revised_summary_model'].tolist()
    model_instructions = [x.lower().split("steps:")[-1].split('corrected:')[0].strip() for x in full_revision_text]
    model_explanations = [x.lower().split("steps:")[0].strip() for x in full_revision_text]
    revised_summaries = [x.lower().split("corrected:")[-1].strip() for x in full_revision_text]
    model_df['instruction_model'] = model_instructions
    model_df['explanation_model'] = model_explanations
    model_df['revised_summary_model'] = revised_summaries

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
        print("Model explanation: ")
        print()
        print(df.iloc[i]['explanation_model'])
        print()
        print("Model instructions: ")
        print()
        print(df.iloc[i]['instruction_model'])
        print()
        print("Model Revised summary:")
        print()
        print(df.iloc[i]['revised_summary_model'])
        print("---------------------------------------------------------------")
def post_process():
    df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/revision_prompts/original_cot_prompt/summaries_with_errors_100_llama_3.1_70B.csv",index_col=0)
    revised_summary_full_text = df['revised_summary'].tolist()
    instructions = [x.lower().split("steps:")[-1].split('corrected:')[0].strip() for x in revised_summary_full_text]
    explanations = [x.lower().split("steps:")[0].strip() for x in revised_summary_full_text]
    revised_summaries = [x.lower().split("corrected:")[-1].strip() for x in revised_summary_full_text]
    df['instruction'] = instructions
    df['explanation'] = explanations
    df['revised_summary'] = revised_summaries
    return df
def main():
    look_at_some(10)
    df = join_model_and_data()
    compare(df)

if __name__ == "__main__":
    main()




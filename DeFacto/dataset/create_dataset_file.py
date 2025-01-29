import pandas as pd


def annotated_dataset(path):
    df = pd.read_csv(path)
    return df


def format_of_fact_span_explanation(df):
    grouped = df.groupby(['text', 'model_summary'])
    samples = []
    for (text, summary), group in grouped:
        spans = group['minimal text span'].tolist()
        explanations = group['explanation'].tolist()
        final_sample = ""
        for span, explanation in zip(spans, explanations):
            final_sample += f"Fact: {span}\nExplanation: {explanation}\n"
        samples.append((text, summary, final_sample))
    original_df = df[['model_summary']].drop_duplicates(inplace=False)
    df = pd.DataFrame(samples, columns=['text', 'model_summary', 'explanation'])
    df = original_df.merge(df, on='model_summary', how='inner')
    return df


def format_of_only_explanation(df):
    grouped = df.groupby(['text', 'model_summary'])
    samples = []
    for (text, summary), group in grouped:
        spans = group['minimal text span'].tolist()
        explanations = group['explanation'].tolist()
        final_sample = ""
        for span, explanation in zip(spans, explanations):
            final_sample += f"Explanation: {explanation}\n"
        samples.append((text, summary, final_sample))
    original_df = df[['model_summary']].drop_duplicates(inplace=False)
    df = pd.DataFrame(samples, columns=['text', 'model_summary', 'explanation'])
    df = original_df.merge(df, on='model_summary', how='inner')
    return df
def num_to_uppercase_letter(num):
    if 0 <= num <= 25:
        return chr(num + ord('A'))
    else:
        raise ValueError("Number must be between 0 and 25")
def format_enumerated_descriptions(df):
    grouped = df.groupby(['text', 'model_summary'])
    samples = []
    for (text, summary), group in grouped:
        spans = group['minimal text span'].tolist()
        explanations = group['explanation'].tolist()
        final_sample = ""
        for i,(span, explanation) in enumerate(zip(spans, explanations)):
            final_sample += f"{num_to_uppercase_letter(i)}.\nDescription: {explanation}\n"
        samples.append((text, summary, final_sample))
    original_df = df[['model_summary']].drop_duplicates(inplace=False)
    df = pd.DataFrame(samples, columns=['text', 'model_summary', 'explanation'])
    df = original_df.merge(df, on='model_summary', how='inner')
    return df

def main():
    path = "/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/2_possible_annotation_almost_final_dataset.csv"
    df = annotated_dataset(path)
    df1 = format_of_fact_span_explanation(df.copy(deep=True))
    df1.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/2_possible_annotation_almost_final_dataset_fact_span_explanation_format.csv",encoding='utf-8')
    df2 = format_of_only_explanation(df.copy(deep=True))
    df2.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/2_possible_annotation_almost_final_dataset_only_explanation_format.csv",encoding='utf-8')
    df3 = format_enumerated_descriptions(df.copy(deep=True))
    df3.to_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/dataset/data/2_possible_annotation_almost_final_dataset_enumerated_descriptions_format.csv",encoding='utf-8')


if __name__ == "__main__":
    main()

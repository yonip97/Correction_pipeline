import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import T5Tokenizer
from nltk.tokenize import word_tokenize
def main():
    df = pd.read_csv(
        "/data/home/yehonatan-pe/Correction_pipeline/experiments/revision/data/base_model_50000_documents/cot_prompts/original/base_model_outputs_below_0.5_text_length_above_65_10000_samples_revised_scored.csv",
        index_col=0)
    df['diff_seahorse'] = df['revised_summary_seahorse'] - df['model_summary_seahorse']
    df['density_diff'] = df['revised_summary_density'] - df['model_summary_density']
    df = df[df['density_diff'] <= 1.5]
    df = df[df['diff_seahorse'] >= 0.2]
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-xl')
    df['model_summary_tokens'] = [len(tokenizer(x)['input_ids']) for x in df['model_summary']]
    df['revised_summary_tokens'] = [len(tokenizer(x)['input_ids']) for x in df['revised_summary']]
    df['diffs'] = df['revised_summary_tokens'] - df['model_summary_tokens']
    df['model_summary_length'] = [len(word_tokenize(x)) for x in df['model_summary']]
    df['revised_summary_length'] = [len(word_tokenize(x)) for x in df['revised_summary']]
    df['length_diffs'] = df['revised_summary_length'] - df['model_summary_length']
    #df = df[df['diffs'] < 10]
    df= df[df['diffs'] > -5]
    plt.hist(df['length_diffs'], bins=20, alpha=0.5, label='original')
    print(np.mean(df['length_diffs']))
    print(np.median(df['length_diffs']))
    plt.show()
    print(len(df))


if __name__ == '__main__':
    main()

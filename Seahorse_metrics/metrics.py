import os
import sys

import pandas as pd
#
# sys.path.append(os.path.dirname(os.getcwd()))
# os.chdir('../')
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from general.utils import iter_list
from general.metrics import add_similarity_metrics_scores
from data.factuality_datasets import TRUE_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from TrueTeacher.train import evaluate
from datasets import load_dataset
from sklearn.metrics import roc_auc_score


class Seahorse_metrics():
    def __init__(self, model_path, tokenizer_name, device='cpu', batch_size=8, max_length=2048,
                 torch_dtype=torch.float32, return_none=False):
        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.one_token = self.tokenizer('1').input_ids[0]
        if self.device == 'auto':
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path, device_map='auto',
                                                                     torch_dtype=torch_dtype).eval()
            self.input_device = 'cpu'
        else:
            self.input_device = self.device
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path,
                                                                     torch_dtype=torch_dtype).to(self.device).eval()
        self.return_none = return_none

    def score(self, texts, summaries):
        pairs = []
        results = []
        for summary, original_text in zip(summaries, texts):
            pairs.append(f'premise: {original_text} hypothesis: {summary}')
        with torch.no_grad():
            for batch in tqdm(iter_list(pairs, self.batch_size)):
                try:
                    model_input = self.tokenizer(batch, return_tensors='pt', truncation=True,
                                                 max_length=self.max_length,
                                                 padding=True).to(self.input_device)
                    decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id] * len(batch)]).reshape((-1, 1)).to(
                        self.input_device)
                    outputs = self.model(**model_input, decoder_input_ids=decoder_input_ids)
                    logits = torch.softmax(outputs.logits.float(), dim=-1)
                    batch_factuality_score = logits.detach().cpu()[:, 0, self.one_token]

                    results += batch_factuality_score.tolist()
                    torch.cuda.empty_cache()
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.return_none:
                        print("Out of memory. Trying to free up some GPU memory.")
                        # Free up memory (adjust as needed)
                        torch.cuda.empty_cache()
                        results += [None] * len(batch)
                    else:
                        raise e
        return results


def create_and_plot():
    model_path_prefix = 'google/seahorse-large-q'
    tokenizer_name_prefix = 'google/seahorse-large-q'
    df = pd.read_csv('data/poc_results_full_classification.csv', index_col=0)
    texts = df['document'].tolist()
    summaries = df['summary'].tolist()
    revised_summaries = df['revised_summary'].tolist()
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    for i, suffix in enumerate(['1', '2', '3', '4', '5', '6']):
        row = i // 2
        col = i % 2
        model_path = model_path_prefix + suffix
        tokenizer_name = tokenizer_name_prefix + suffix
        metric = Seahorse_metrics(model_path, tokenizer_name, device='cuda:1', batch_size=8, max_length=2048)
        scores_pre_revision = metric.score(texts, summaries)
        df[f'seahorse_scores_pre_revision_q{suffix}'] = scores_pre_revision
        scores_post_revision = metric.score(texts, revised_summaries)
        df[f'seahorse_scores_post_revision_q{suffix}'] = scores_post_revision
        axs[row, col].hist(scores_pre_revision, bins=20, alpha=0.5, label='pre revision')
        axs[row, col].hist(scores_post_revision, bins=20, alpha=0.5, label='post revision')
        axs[row, col].legend(loc='upper right')
        axs[row, col].title.set_text(f"Seahorse scores for q {suffix}")
        axs[row, col].set_xlim((0, 1))
    plt.tight_layout()
    plt.show()
    df.to_csv('data/seahorse/results.csv')
def plot(df):
    pre_revision_prefix = 'seahorse_scores_pre_revision_q'
    post_revision_prefix = 'seahorse_scores_post_revision_q'
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    for i, suffix in enumerate(['1', '2', '3', '4', '5', '6']):
        row = i // 2
        col = i % 2
        pre_revision_col = pre_revision_prefix + suffix
        post_revision_col = post_revision_prefix + suffix
        scores_pre_revision = df[pre_revision_col]
        scores_post_revision = df[post_revision_col]
        axs[row, col].hist(scores_pre_revision, bins=20, alpha=0.5, label='pre revision')
        axs[row, col].hist(scores_post_revision, bins=20, alpha=0.5, label='post revision')
        axs[row, col].legend(loc='upper right')
        axs[row, col].title.set_text(f"Seahorse scores for q {suffix}")
        axs[row, col].set_xlim((0, 1))
    plt.tight_layout()
    plt.show()

def analyse(df):
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    for i, suffix in enumerate(['1', '2', '3', '4', '5', '6']):
        row = i // 2
        col = i % 2
        scores_pre_revision = df[f'seahorse_scores_pre_revision_q{suffix}']
        scores_post_revision = df[f'seahorse_scores_post_revision_q{suffix}']
        axs[row, col].scatter(scores_pre_revision, scores_post_revision)
        axs[row, col].title.set_text(f"Seahorse scores for q {suffix}")
        axs[row, col].set_xlim((0, 1))
        axs[row, col].set_ylim((0, 1))
        axs[row, col].plot([0, 1], [0, 1], color='red')
        axs[row, col].set_xlabel('pre revision')
        axs[row, col].set_ylabel('post revision')
        print(
            f"For q{suffix} there are {sum([1 if pre < post else 0 for pre, post in zip(scores_pre_revision, scores_post_revision)])} examples where the score increased")
    plt.tight_layout()
    plt.show()


def check_correlations(df):
    corr = []
    cols_x = [col for col in df.columns if ('seahorse' in col) or ('rouge' in col)]
    cols_y = [col for col in df.columns if ('seahorse' in col) or ('rouge' in col)]
    plt.figure(figsize=(14, 14))
    for col in cols_x:
        col_corr = []
        for col2 in cols_y:
            print(col, col2)
            print(df[col].corr(df[col2]))
            print('-----------------------------------------')
            col_corr.append(round(df[col].corr(df[col2]), 2))
        corr.append(col_corr)
    import seaborn as sns
    sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=cols_x, yticklabels=cols_y)
    plt.title('Correlation Heatmap')
    plt.show()


def main():
    # metric = Seahorse_metrics('google/seahorse-large-q4', 'google/seahorse-large-q4', device='cuda:1', batch_size=8,
    #                           max_length=2048)
    # from data.factuality_datasets import TRUE_dataset
    # dataset = TRUE_dataset('data/true_data', ['summarization'])
    # texts = dataset.df['grounding'].tolist()
    # summaries = dataset.df['generated_text'].tolist()
    # labels = dataset.df['label'].tolist()
    # scores = metric.score(texts, summaries)
    # from sklearn.metrics import roc_auc_score
    # print(roc_auc_score(labels, scores))
    df = pd.read_csv('data/seahorse/poc_results.csv', index_col=0)
    plot(df)
    # df = add_similarity_metrics_scores(df, 'summary', 'revised_summary')
    # # analyse(df)
    # check_correlations(df)

#main()

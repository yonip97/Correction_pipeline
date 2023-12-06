import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
import matplotlib.pyplot as plt
import nltk
import numpy as np
import evaluate
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from data.factuality_datasets import BERTS2S_TConvS2S_xsum_trained_dataset
from tqdm import tqdm
from factCC.inference import Factcc_classifier
from Seahorse_metrics.metrics import Seahorse_metrics


def preprocess(document, summary_sents, metric):
    docs_sents = nltk.sent_tokenize(document)
    data = []
    for i in range(len(summary_sents)):
        relevant_passages = []
        scores = \
            metric.compute(predictions=[summary_sents[i]] * len(docs_sents), references=docs_sents,
                           use_aggregator=False)[
                'rougeLsum']
        rel_sents_indices = np.argsort(scores)[::-1][0:3]
        for ind in rel_sents_indices:
            start_index = max(0, ind - 2)
            end_index = min(len(docs_sents), ind + 3)
            relevant_passages.append(" ".join(docs_sents[start_index:end_index]))

        data.append(summary_sents[i] + ' <sep> ' + " ".join(summary_sents) + ' <sep> ' + " ".join(relevant_passages))
    return data


def predict(model, tokenizer, document, summary, metric, device):
    with torch.no_grad():
        summary_sents = nltk.sent_tokenize(summary)
        data = preprocess(document, summary_sents, metric)
        model_input = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        outputs = model.generate(**model_input, max_length=128)
        corrected_summary_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        corrected_summary = " ".join(corrected_summary_sents)
        return corrected_summary


def main():
    df = pd.read_csv('data/factedit/frank_cnn_model.csv',index_col=0)
    # device = 'cpu'
    # #checkpoint_path = 'correction_models/factedit/checkpoints/bart_base_factedit/checkpoint-15000'
    # checkpoint_path = 'correction_models/factedit/checkpoints/xsum_corr_model/best.ckpt'
    # x = torch.load(checkpoint_path)
    # model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    # state_dict = x['state_dict']
    # keys = list(state_dict.keys())
    # for key in keys:
    #     if "model" in key:
    #         new_key = key.replace("model.", "",1)
    #         state_dict[new_key] = state_dict.pop(key)
    # model.load_state_dict(state_dict)
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    # model.eval()
    # dataset = BERTS2S_TConvS2S_xsum_trained_dataset()
    # texts = [dataset[i]['text'] for i in range(len(dataset))]
    # summaries = [dataset[i]['summary'] for i in range(len(dataset))]
    # rouge = evaluate.load('rouge')
    # corrected_summaries = []
    # for i in tqdm(range(len(dataset))):
    #     item = dataset[i]
    #     document, summary = item['text'], item['summary']
    #     corrected_summary = predict(model, tokenizer, document, summary, metric=rouge, device=device)
    #     corrected_summaries.append(corrected_summary)
    #
    # # with open("data/factedit/BERTS2S_TConvS2S_xsum_trained_dataset/corrected_summaries.txt", "w") as file:
    # #     for string_item in corrected_summaries:
    # #         file.write(string_item + "\n")
    # model.to('cpu')
    #
    # del model
    #
    # torch.cuda.empty_cache()

    # corrected_summaries = []
    # with open("data/factedit/BERTS2S_TConvS2S_xsum_trained_dataset/corrected_summaries.txt", "r") as file:
    #     for line in file:
    #         corrected_summaries.append(line.strip())
    texts = df['source_articles'].tolist()
    summaries = df['old_summaries'].tolist()
    corrected_summaries = df['predictions'].tolist()
    factuality_metric_factcc = Factcc_classifier(checkpoint_path ='factCC/checkpoints/factcc-checkpoint',device='cuda:0')

    pre_revision_scores = factuality_metric_factcc.score(texts=texts, summaries=summaries)
    post_revision_scores = factuality_metric_factcc.score(texts=texts, summaries=corrected_summaries)
    #post_scores = factuality_metric.score(texts=texts, summaries=corrected_summaries)
    #print(f"The mean score after revision is {np.mean(post_scores):.4f}")
    #plt.hist(pre_scores, bins=20, alpha=0.5, label='pre revision')
    df['pre_revision_factcc_scores'] = pre_revision_scores
    df['post_revision_factcc_scores'] = post_revision_scores
    with open("correction_models/factedit/outputs/frank_results_factcc.txt",'w') as f:
        for model_name in df['models'].unique():
            df_model = df[df['models']==model_name]
            f.write(f"{model_name} results are:\n")
            f.write(f"pre revision scores {df_model['pre_revision_factcc_scores'].mean():.4f}\n")
            f.write(f"post revision scores {df_model['post_revision_factcc_scores'].mean():.4f}\n")
            plt.hist(df_model['pre_revision_factcc_scores'], bins=20, alpha=0.5, label='pre revision')
            plt.hist(df_model['post_revision_factcc_scores'], bins=20, alpha=0.5, label='post revision')

            plt.legend(loc='upper right')
            plt.xlim(0, 1)
            plt.savefig(f'correction_models/factedit/outputs/factuality_scores_factcc_their_checkpoint_{model_name}.png')
    factuality_metric_factcc.model.to('cpu')
    del factuality_metric_factcc
    factuality_metric_seahorse = Seahorse_metrics(model_path="google/seahorse-large-q4",
                                         tokenizer_name="google/seahorse-large-q4",
                                         device='cuda:0', batch_size=1, max_length=2048,
                                         return_none=True)
    pre_revision_scores = factuality_metric_seahorse.score(texts=texts, summaries=summaries)
    post_revision_scores = factuality_metric_seahorse.score(texts=texts, summaries=corrected_summaries)
    df['pre_revision_seahorse_scores'] = pre_revision_scores
    df['post_revision_seahorse_scores'] = post_revision_scores
    with open("correction_models/factedit/outputs/frank_results_seahorse_large.txt",'w') as f:
        for model_name in df['models'].unique():
            df_model = df[df['models'] == model_name]
            f.write(f"{model_name} results are:\n")
            f.write(f"pre revision scores {df_model['pre_revision_seahorse_scores'].mean():.4f}\n")
            f.write(f"post revision scores {df_model['post_revision_seahorse_scores'].mean():.4f}\n")
            plt.hist(df_model['pre_revision_seahorse_scores'], bins=20, alpha=0.5, label='pre revision')
            plt.hist(df_model['post_revision_seahorse_scores'], bins=20, alpha=0.5, label='post revision')

            plt.legend(loc='upper right')
            plt.xlim(0, 1)
            plt.savefig(
                f'correction_models/factedit/outputs/factuality_scores_seahorse_large_their_checkpoint_{model_name}.png')
    # with open("correction_models/factuality/outputs/results.txt",'w') as text_file:
    #     text_file.write(f"The mean score before revision is {np.mean(pre_scores):.4f}\n")
    #     text_file.write(f"The mean score after revision is {np.mean(post_scores):.4f}\n")
    # text_file.close()
    #


if __name__ == "__main__":
    main()

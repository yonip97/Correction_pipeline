import time
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
print(os.getcwd())

from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from general.t5_trainer import t5_revise
import gc
from Seahorse_metrics.metrics import Seahorse_metrics
import pandas as pd


def main():
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices="experiments/xsum_4_sets_experiment/datasets_splits/xsum_docs_for_summarization_20000_indices_seed_42.pkl",
                                      num_of_documents_for_summarization=20000, seed=42)
    texts = [xsum_dataset[i]['text'] for i in range(len(xsum_dataset))]
    summaries = [xsum_dataset[i]['summary'] for i in range(len(xsum_dataset))]
    df = pd.DataFrame.from_dict({'text': texts, 'summary': summaries})
    x = torch.load('experiments/xsum_4_sets_experiment/checkpoints/flan_t5_large_final_revision_model.pth')
    revision_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-large', device_map='auto',
                                                                torch_dtype=torch.float16)
    revision_model.load_state_dict(x['model'])
    tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-large')
    revisions = t5_revise(texts, summaries, revision_model, tokenizer, prompt='revise: ',
                          device='cuda',
                          batch_size=12,
                          generation_max_length=128, num_beams=4, early_stopping=True,
                          encoding_max_length=1024)
    df['revised_summaries'] = revisions
    del revision_model
    gc.collect()
    time.sleep(30)
    torch.cuda.empty_cache()
    factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                                         device='auto', batch_size=1, max_length=2048, torch_dtype=torch.float16,
                                         return_none=True)
    pre_revision_summaries_seahorse_scores = factuality_metric.score(summaries=summaries, texts=texts)
    post_revision_revised_summaries_seahorse_scores = factuality_metric.score(summaries=revisions, texts=texts)
    df['pre_revision_score_seahorse'] = pre_revision_summaries_seahorse_scores
    df['post_revision_score_seahorse'] = post_revision_revised_summaries_seahorse_scores
    df.to_csv(
        'experiments/ablations/revision_on_original/outputs/revision_results_original_flan_large_results.csv')


if __name__ == '__main__':
    main()

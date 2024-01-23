import gc
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')

import json
from datetime import datetime
import argparse
import os
import pandas as pd
from tqdm import tqdm
#from general.revision_pipeline import chose_revision_model
from Seahorse_metrics.metrics import Seahorse_metrics
import torch
from experiments.scoring import score
from experiments.data.datasets_splits import split_xsum_dataset
from general.t5_trainer import t5_summarize
from transformers import T5ForConditionalGeneration, T5Tokenizer


def parseargs_llms():
    parser = argparse.ArgumentParser()

    parser.add_argument('-output_path')
    parser.add_argument('-revision_model_name', type=str, default='mock')
    parser.add_argument('-revision_prompt', type=str,
                        default="""I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific  compared to the document, you should not change it. Output only the corrected summary and nothing more.""")

    parser.add_argument('-API_KEY_revision_model', type=str, default=None)
    parser.add_argument('-contingency_file_dir', type=str, default='contingency_tables')
    parser.add_argument('-revision_summaries', type=int, default=10000)
    parser.add_argument('-summarization_documents', type=int, default=20000)
    parser.add_argument('-base_models_dir', type=str, default="experiments/baseline_model/checkpoints")
    parser.add_argument('-max_generation_length', type=int, default=128)
    parser.add_argument('-num_beams', type=int, default=4)
    parser.add_argument('-length_penalty', type=float, default=0.6)
    parser.add_argument('-base_model_checkpoint', type=str)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-device', type=str, default='cpu')
    parser.add_argument('-revision_data_dir', type=str, default="experiments/revision/data")
    parser.add_argument('-batch_size', type=int, default=32)
    args = parser.parse_args()
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    run_name = f'run_{formatted_datetime}'
    args.dir_path = args.contingency_file_dir + '/' + run_name
    os.makedirs(args.dir_path)
    return args


def get_data(args):
    if not os.path.exists(args.revision_data_dir):
        os.makedirs(args.revision_data_dir)
    if not os.path.exists(os.path.join(args.revision_data_dir, "base_model_outputs.csv")):
        with open(os.path.join(args.revision_data_dir, "args.json"), 'w') as file:
            json.dump(args.__dict__, file)
        dataset = split_xsum_dataset(split='revision_documents',
                                     path_to_documents_for_summarization_indices=f"experiments/data/datasets_splits/xsum_summarization_{args.summarization_documents}_revision_{args.revision_summaries}_seed_{args.seed}.json",
                                     num_of_documents_for_summarization=args.summarization_documents,
                                     num_of_documents_for_revision=args.revision_summaries,
                                     seed=args.seed)
        texts = [dataset[i]['text'] for i in range(len(dataset))]
        summarization_model = T5ForConditionalGeneration.from_pretrained(
            os.path.join(args.base_models_dir, args.base_model_checkpoint)).to(args.device)
        tokenizer = T5Tokenizer.from_pretrained(os.path.join(args.base_models_dir, args.base_model_checkpoint))
        model_summaries = t5_summarize(texts=texts, model=summarization_model, tokenizer=tokenizer,
                                       prompt="summarize: ",
                                       device=args.device, max_generation_length=args.max_generation_length,
                                       batch_size=args.batch_size,
                                       beam_size=args.num_beams,
                                       early_stopping=True,
                                       length_penalty=args.length_penalty)
        del summarization_model
        gc.collect()
        torch.cuda.empty_cache()
        df = pd.DataFrame.from_dict({'text': texts, 'model_summary': model_summaries})
        df.to_csv(os.path.join(args.revision_data_dir, "base_model_outputs.csv"))
    else:
        df = pd.read_csv(os.path.join(args.revision_data_dir,"base_model_outputs.csv"), index_col=0)
    for col in df.columns:
        if 'factuality_score' == col:
            rel_df = df[(df['factuality_score'] < 0.5)]
            summaries = rel_df['model_summary'].tolist()
            texts = rel_df['text'].tolist()
            scores = rel_df['factuality_score'].tolist()
            return texts, summaries, scores
    texts = df['text'].tolist()
    summaries = df['model_summary'].tolist()
    factuality_scores = score(texts=texts, summaries=summaries, metrics=['seahorse'])['seahorse']
    df = pd.DataFrame({'factuality_score': factuality_scores,
                       'text': texts,
                       'model_summary': summaries})
    df.to_csv(os.path.join(args.revision_data_dir, "base_model_outputs.csv"))
    rel_df = df[(df['factuality_score'] < 0.5)]
    scores = rel_df['factuality_score'].tolist()
    summaries = rel_df['model_summary'].tolist()
    texts = rel_df['text'].tolist()
    return texts, summaries, scores


def llm_revision():
    args = parseargs_llms()
    texts, summaries, scores = get_data(args)
    revision_model = chose_revision_model(args)
    revised_summaries, errors = [], []
    for text, summary in tqdm(zip(texts, summaries)):
        revised_summary, error = revision_model.revise_single(text=text, summary=summary)
        revised_summaries.append(revised_summary)
        errors.append(error)
        # time.sleep(2)
    pd.DataFrame.from_dict(
        {'text': texts, 'model_summary': summaries, 'revised_summary': revised_summaries,
         'pre_revision_factuality_scores': scores, 'error': errors}).to_csv(
        args.output_path + '.csv')


def main():
    args = parseargs_llms()
    get_data(args)
if __name__=="__main__":
    main()
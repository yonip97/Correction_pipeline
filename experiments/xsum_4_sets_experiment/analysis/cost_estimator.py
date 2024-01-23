import pandas as pd

from data.cost_etimation import Cost_estimator
from experiments.xsum_4_sets_experiment.datasets_splits import split_xsum_dataset
from experiments.xsum_4_sets_experiment.datasets_splits import split_cnndm_dataset


# need to summarize and un TrueTeacher to check exacly how many to summarize

def estimate():
    # checkpoint_path ="/data/home/yehonatan-pe/Correction_pipeline/experiments/xsum_4_sets_experiment/checkpoints/t5_base_both_10_12_2023_08_54_06/checkpoint-115000"
    # model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    # tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    df = pd.read_csv('/data/home/yehonatan-pe/Correction_pipeline/experiments/xsum_4_sets_experiment//both_models_summaries.csv', index_col=0)
    estimator = Cost_estimator('gpt-4', 0.01, 0.03)
    path_to_documents_for_summarization_indices_xsum = "experiments/xsum_4_sets_experiment/datasets_split/xsum_docs_for_summarization_10000_indices_seed_42.pkl"
    path_to_documents_for_summarization_indices_cnndm = "experiments/xsum_4_sets_experiment/datasets_split/cnndm_docs_for_summarization_10000_indices_seed_42.pkl"
    xsum_dataset = split_xsum_dataset(split='documents_for_summarization',
                                      path_to_documents_for_summarization_indices=path_to_documents_for_summarization_indices_xsum,
                                      num_of_documents_for_summarization=10000, seed=42)
    cnndm_dataset = split_cnndm_dataset('documents_for_summarization',
                                        path_to_documents_for_summarization_indices_cnndm, 10000, 42)
    prompt = """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific  compared to the document, you should not change it. Output only the corrected summary and nothing more."""
    # for i in range(len(dataset_xsum)):
    #     print(i)
    #     text = dataset_xsum[i]['text']
    #     summary = dataset_xsum[i]['summary']
    #     texts.append(text)
    #     summaries.append(summary)
    xsum_indices_to_text = {xsum_dataset.indices[i]: xsum_dataset[i]['text'] for i in range(len(xsum_dataset))}
    cnndm_indices_to_text = {cnndm_dataset.indices[i]: cnndm_dataset[i]['text'] for i in range(len(cnndm_dataset))}

    df['text'] = df.apply(
            lambda x: xsum_indices_to_text[x['indices']] if x['dataset'] == 'xsum' else cnndm_indices_to_text[
                x['indices']], axis=1)

    rel_both = df[(df['factuality_score_seahorse_xxl'] < 0.5) & (df['factuality_score_true_teacher'] < 0.5)]
    rel_seahorse = df[(df['factuality_score_seahorse_xxl'] < 0.5) & (df['factuality_score_true_teacher'] >= 0.5)]
    rel_true_teacher = df[(df['factuality_score_seahorse_xxl'] >= 0.5) & (df['factuality_score_true_teacher'] < 0.5)]
    rel_either = df[(df['factuality_score_seahorse_xxl'] < 0.5) | (df['factuality_score_true_teacher'] < 0.5)]
    is_nan = df['factuality_score_seahorse_xxl'].isna()
    print("is_nan", sum(is_nan))
    is_nan = df['factuality_score_true_teacher'].isna()
    print("is_nan", sum(is_nan))
    print("rel_both", len(rel_both))
    print("rel_seahorse", len(rel_seahorse))
    print("rel_true_teacher", len(rel_true_teacher))
    print("rel_either", len(rel_either))
    rel_df_both = df[(df['factuality_score_seahorse_xxl'] < 0.5) & (df['factuality_score_true_teacher'] < 0.5)]
    xsum_df = rel_df_both[rel_df_both['dataset'] == 'xsum']
    xsum_texts = xsum_df['text'].tolist()
    xsum_summaries = xsum_df['model_summary'].tolist()
    estimation = estimator.estimate_for_revision(prompt=prompt, texts=xsum_texts, summaries=xsum_summaries)
    print("xsum estimation", estimation)
    cnndm_df = rel_df_both[rel_df_both['dataset'] == 'cnndm']
    cnndm_texts = cnndm_df['text'].tolist()
    cnndm_summaries = cnndm_df['model_summary'].tolist()
    estimation = estimator.estimate_for_revision(prompt=prompt, texts=cnndm_texts, summaries=cnndm_summaries)
    print("cnndm estimation", estimation)
    rel_df_just_seahorse = df[(df['factuality_score_seahorse_xxl'] < 0.5) & (df['factuality_score_true_teacher'] >= 0.5)]
    xsum_df = rel_df_just_seahorse[rel_df_just_seahorse['dataset'] == 'xsum']
    xsum_texts = xsum_df['text'].tolist()
    xsum_summaries = xsum_df['model_summary'].tolist()
    estimation = estimator.estimate_for_revision(prompt=prompt, texts=xsum_texts, summaries=xsum_summaries)
    print("xsum estimation", estimation)
    cnndm_df = rel_df_just_seahorse[rel_df_just_seahorse['dataset'] == 'cnndm']
    cnndm_texts = cnndm_df['text'].tolist()
    cnndm_summaries = cnndm_df['model_summary'].tolist()
    estimation = estimator.estimate_for_revision(prompt=prompt, texts=cnndm_texts, summaries=cnndm_summaries)
    print("cnndm estimation", estimation)
    rel_df_just_true_teacher = df[(df['factuality_score_seahorse_xxl'] >= 0.5) & (df['factuality_score_true_teacher'] < 0.5)]
    xsum_df = rel_df_just_true_teacher[rel_df_just_true_teacher['dataset'] == 'xsum']
    xsum_texts = xsum_df['text'].tolist()
    xsum_summaries = xsum_df['model_summary'].tolist()
    estimation = estimator.estimate_for_revision(prompt=prompt, texts=xsum_texts, summaries=xsum_summaries)
    print("xsum estimation", estimation)
    cnndm_df = rel_df_just_true_teacher[rel_df_just_true_teacher['dataset'] == 'cnndm']
    cnndm_texts = cnndm_df['text'].tolist()
    cnndm_summaries = cnndm_df['model_summary'].tolist()
    estimation = estimator.estimate_for_revision(prompt=prompt, texts=cnndm_texts, summaries=cnndm_summaries)
    print("cnndm estimation", estimation)

estimate()

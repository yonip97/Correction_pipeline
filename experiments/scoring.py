from Seahorse_metrics.metrics import Seahorse_metrics
from TrueTeacher.inference import TrueTeacher
import torch
import time
import gc


def score(texts, summaries, metrics):
    results = {}
    for metric in metrics:
        if 'seahorse' in metrics:
            factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                                 tokenizer_name='google/seahorse-xxl-q4',
                                                 device='auto', batch_size=1, torch_dtype=torch.float16,
                                                 max_length=2048, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['seahorse'] = scores
        elif 'teacher' in metric:
            factuality_metric = TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                                            tokenizer_name="google/t5_11b_trueteacher_and_anli",
                                            device='auto', batch_size=1, max_length=2048,
                                            torch_dtype=torch.float16, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['trueteacher'] = scores
        elif 'nli' in metric:
            from nli.nli_metric import NLI
            factuality_metric = NLI(batch_size=1, torch_dtype=torch.bfloat16, max_length=2048, device='auto',
                                    return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['nli'] = scores
        elif 'q_squared' in metric:
            from q_squared.inference import Q2_metric
            factuality_metric = Q2_metric(device='cuda:0', out_path=None, save=False, for_system=None)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['Q2'] = scores

            # from q_squared.run_nli import scores_with_nli, aggregate_per_response
            # from q_squared.prep_sys_experiment import cross_annotated_scores
            #
            # df = cross_annotated_scores(texts, summaries, out_path=None, save=False)
            # df = scores_with_nli(in_path=None, df=df)
            # df = aggregate_per_response(df=df, out_path=None, save=False)
            # results['Q2'] = df['Q2'].tolist()
            # if 'nli' in metric:
            #     factuality_metric = Q_squared_classifier(device_generation='cuda:0',device_answering='cuda:1', similarity_metric='nli', threshold=0.5, remove_personal=True)
            # else:
            #     factuality_metric = Q_squared_classifier(device_generation='cuda:0',device_answering='cuda:1', similarity_metric='f1', threshold=0.5, remove_personal=True)
            # scores = factuality_metric.score(texts=texts, summaries=summaries)
            # if 'nli' in metric:
            #     results['q_squared_nli'] = scores
            # else:
            #     results['q_squared_f1'] = scores
        del factuality_metric
        time.sleep(30)
        gc.collect()
        torch.cuda.empty_cache()
    return results

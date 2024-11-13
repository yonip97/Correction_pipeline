from Seahorse_metrics.metrics import Seahorse_metrics
from TrueTeacher.inference import TrueTeacher
import torch
import time
import gc


def score(texts, summaries, metrics, device='auto', torch_dtype=torch.float16, batch_size=1):
    results = {}
    gc.collect()
    torch.cuda.empty_cache()
    for metric in metrics:
        if 'seahorse' in metric:
            factuality_metric = Seahorse_metrics(model_path='google/seahorse-xxl-q4',
                                                 tokenizer_name='google/seahorse-xxl-q4',
                                                 device=device, batch_size=batch_size, torch_dtype=torch_dtype,
                                                 max_length=2048, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['seahorse'] = scores
        elif 'teacher' in metric:
            factuality_metric = TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                                            tokenizer_name="google/t5_11b_trueteacher_and_anli",
                                            device=device, batch_size=batch_size, max_length=2048,
                                            torch_dtype=torch_dtype, return_none=True)
            scores = factuality_metric.score(texts=texts, summaries=summaries)
            results['trueteacher'] = scores
        # elif 'nli' in metric:
        #     from nli.nli_metric import NLI
        #     factuality_metric = NLI(batch_size=1, torch_dtype=torch_dtype, max_length=2048, device=device,
        #                             return_none=True)
        #     scores = factuality_metric.score(texts=texts, summaries=summaries)
        #     results['nli'] = scores
        # elif 'q_squared' in metric:
        #     from q_squared.inference import Q2_metric
        #     factuality_metric = Q2_metric(device='cuda:0', out_path=None, save=False, for_system=None)
        #     scores = factuality_metric.score(texts=texts, summaries=summaries)
        #     results['Q2'] = scores
        del factuality_metric
        gc.collect()
        torch.cuda.empty_cache()
    return results

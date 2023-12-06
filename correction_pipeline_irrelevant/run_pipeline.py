import os
import sys

from torch.utils.data import DataLoader
from argparse import ArgumentParser
from correction_pipeline.pipeline import Correction_pipline
from correction_pipeline.question_generator import Question_generator_prompt_based, Question_generator_model_based
from correction_pipeline.question_answering import Question_answering_model_based, Question_answering_model_prompt_based
from correction_pipeline.filter_models import Base_filter, Strategy_based_filter_model
from correction_pipeline.disagreement_model import Disagreement_model_nli_based
from correction_pipeline.revision_models import Dummie_revision, LLM_prompt_based_revision_model
from correction_pipeline.llms import Cost_estimator
from correction_pipeline.utils import collate_fn
from data.factuality_datasets import TRUE_dataset, true_topics
from time import time

QG_MODELS = {'prompt_based': Question_generator_prompt_based, 'model_based': Question_generator_model_based}
QA_MODELS = {'prompt_based': Question_answering_model_prompt_based, 'model_based': Question_answering_model_based}
FILTER_MODELS = {'dummy': Base_filter, 'strategy_based': Strategy_based_filter_model}
DISAGREEMENT_MODELS = {'nli_based': Disagreement_model_nli_based}
REVISIONS_MODELS = {'dummy': Dummie_revision, 'prompt_based': LLM_prompt_based_revision_model}


def arguments_for_question_generation(args):
    args.add_argument('-qg_model', default='model_based', type=str)
    args.add_argument('-qg_max_length', default=512, type=int,
                      help="The max length of all the questions that will be generated in case of prompt model."
                           "In case of model based the length of one question")
    args.add_argument('-qg_prompt_path', default='', type=str)
    args.add_argument('-qg_method', default='greedy', type=str)
    args.add_argument('-qg_batch_size', default=16, type=int)
    args.add_argument('-qg_beam_size', default=5, type=int)
    args.add_argument('-qg_num_return', default=5, type=int)
    args.add_argument('-qg_top_k', default=50, type=int)
    args.add_argument('-qg_top_p', default=0.95, type=float)
    return args


def arguments_for_question_answering(args):
    args.add_argument('-qa_max_length', default=128, type=int)
    args.add_argument('-qa_model', default='model_based', type=str)
    args.add_argument('-qa_batch_size', default=16, type=int)
    args.add_argument('-qa_prompt_path', default='', type=str)
    # args.add_argument('-unanswerable_response', default='UNANSWERABLE', type=str)
    args.add_argument('-model_name', default='albert', type=str)
    return args


def arguments_for_filtering(args):
    args.add_argument('-filter_model', default='strategy_based', type=str)
    args.add_argument('-unanswerable', action='store_true')
    args.add_argument('-yes_or_no', action='store_true')
    args.add_argument('-multiple_repetitions', action='store_true')
    args.add_argument('-meaningless_answer', action='store_true')
    args.add_argument('-unanswerable_response', default='UNANSWERABLE', type=str)
    return args


def arguments_for_disagreement_model(args):
    args.add_argument('-disagreement_model', default='nli_based', type=str)
    args.add_argument('-model_type', default='hf', type=str)
    args.add_argument('-nli_confidence_cutoff', default=0.5, type=str)
    args.add_argument('-disagreement_batch_size', default=16, type=int)
    return args


def arguments_for_revision_model(args):
    args.add_argument('-revision_model', default='dummy', type=str)
    args.add_argument('-revision_prompt_path', default='', type=str)
    return args


def parser_args():
    args = ArgumentParser()
    args = arguments_for_question_generation(args)
    args = arguments_for_question_answering(args)
    args = arguments_for_filtering(args)
    args = arguments_for_disagreement_model(args)
    args = arguments_for_revision_model(args)
    args.add_argument('-LLM_model', default='gpt-3.5-turbo')
    args.add_argument('-api_key', default='', type=str)
    args.add_argument('-TRUE_dir_path', default='data', type=str)
    args.add_argument('-device', default='cuda', type=str)
    args = args.parse_args()
    return args


def create_correction_pipeline(args):
    qg_kwargs = {}
    qa_kwargs = {}
    revision_kwargs = {}
    qg_kwargs['max_length'] = args.qg_max_length
    if args.qg_model == 'model_based':
        qg_model = QG_MODELS[args.qg_model](method=args.qg_method, batch_size=args.qa_batch_size, device=args.device)
        qg_kwargs['beam_size'] = args.qg_beam_size
        qg_kwargs['num_return'] = args.qg_num_return
        qg_kwargs['top_k'] = args.qg_top_k
        qg_kwargs['top_p'] = args.qg_top_p
    elif args.qg_model == 'prompt_base':
        qg_model = QG_MODELS[args.qg_model](prompt_path=args.qg_prompt_path, model=args.LLM_model, API_KEY=args.api_key)
    else:
        raise ValueError("No such QA model exists!")
    if args.qa_model == 'model_based':
        qa_model = QA_MODELS[args.qa_model](model_name=args.model_name,
                                            batch_size=args.qa_batch_size, device=args.device)
    elif args.qa_model == 'prompt_based':
        qa_model = QA_MODELS[args.qa_model](prompt_path=args.qa_prompt_path, model=args.LLM_model, API_KEY=args.api_key)
        qa_kwargs['max_length'] = args.qa_max_length
    else:
        raise ValueError("No such QA model exist!")
    if args.filter_model == 'strategy_based':
        filter_model = FILTER_MODELS[args.filter_model](unanswerable_response=args.unanswerable_response,
                                                        unanswerable=args.unanswerable,
                                                        multiple_repetitions=args.multiple_repetitions,
                                                        yes_or_no_questions=args.yes_or_no,
                                                        meaningless_answer=args.meaningless_answer)
    elif args.filter_model == 'dummy':
        filter_model = FILTER_MODELS[args.filter_model]()
    else:
        raise ValueError("No such filter model exists!")
    if args.disagreement_model == 'nli_based':
        disagreement_model = DISAGREEMENT_MODELS[args.disagreement_model](model_type=args.model_type,
                                                                          confidence_cutoff=args.nli_confidence_cutoff,
                                                                          batch_size=args.disagreement_batch_size,
                                                                          device=args.device)
    else:
        raise ValueError("No such disagreement model exists!")
    if args.revision_model == 'dummy':
        revision_model = REVISIONS_MODELS[args.revision_model]()
    elif args.revison_model == 'prompt_based':
        revision_model = REVISIONS_MODELS[args.revison_model](prompt_path=args.revision_prompt_path,
                                                              model=args.LLM_model, API_KEY=args.api_key)
        revision_kwargs['max_length'] = args.revision_max_length
    else:
        raise ValueError("No such revision model exists!")
    pipeline = Correction_pipline(qg_model=qg_model, qa_model=qa_model, filter_model=filter_model,
                                  disagreement_model=disagreement_model,
                                  revision_model=revision_model)
    kwargs = {'qg_kwargs': qg_kwargs, 'qa_kwargs': qa_kwargs, 'revision_kwargs': revision_kwargs}
    return pipeline, kwargs


def main():
    args = parser_args()
    dataset = TRUE_dataset(args.TRUE_dir_path, ['summarization'])
    # datasets_names = true_topics(['summarization'])
    # dataset.filter_to_datasets(datasets_names)
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
    correction_pipeline, kwargs = create_correction_pipeline(args)
    start = time()
    inconsistent_stats = {'cost': 0, 'revisions': 0, 'amount': 0}
    consistent_stats = {'cost': 0, 'revisions': 0, 'amount': 0}
    prev_cost = 0
    prev_revisions = 0
    f = open('output.txt', 'w')
    for i, x in enumerate(dataloader):
        dataset, original_text, generated_text, label = x
        correction_pipeline.apply(original_text[0], generated_text[0], **kwargs)
        if label[0] == 1:
            consistent_stats['amount'] += 1
            consistent_stats['cost'] += correction_pipeline.revision_model.cost - prev_cost
            consistent_stats['revisions'] += correction_pipeline.revision_model.revisions_number - prev_revisions
        else:
            inconsistent_stats['amount'] += 1
            inconsistent_stats['cost'] += correction_pipeline.revision_model.cost - prev_cost
            inconsistent_stats['revisions'] += correction_pipeline.revision_model.revisions_number - prev_revisions
        prev_cost = correction_pipeline.revision_model.cost
        prev_revisions = correction_pipeline.revision_model.revisions_number

        if i % 100 == 0:
            f.write(f"Number of sample {i}\n")
            f.write(f"{i} examples took {time() - start} seconds\n")
            f.write(
                f"There were {consistent_stats['amount']} samples and {consistent_stats['revisions']} revisions done at {consistent_stats['cost']:.6f} dollars for consistent examples\n")
            f.write(
                f"There were {inconsistent_stats['amount']} samples and {inconsistent_stats['revisions']} revisions done at {inconsistent_stats['cost']:.6f} dollars for inconsistent examples\n")
            f.flush()
            print(f"Number of sample {i}")
            print(f"{i} examples took {time() - start} seconds")
            print(
                f"There were {consistent_stats['amount']} samples and {consistent_stats['revisions']} revisions done at {consistent_stats['cost']:.6f} dollars for consistent examples")
            print(
                f"There were {inconsistent_stats['amount']} samples and {inconsistent_stats['revisions']} revisions done at {inconsistent_stats['cost']:.6f} dollars for inconsistent examples")

    print(f"{len(dataset)} examples took {time() - start} seconds")
    print(
        f"There were {consistent_stats['amount']} samples and {consistent_stats['revisions']} revisions done at {consistent_stats['cost']:.6f} dollars for consistent examples")
    print(
        f"There were {inconsistent_stats['amount']} samples and {inconsistent_stats['revisions']} revisions done at {inconsistent_stats['cost']:.6f} dollars for inconsistent examples")
    f.write(f"{len(dataset)} examples took {time() - start} seconds\n")
    f.write(
        f"There were {consistent_stats['amount']} samples and {consistent_stats['revisions']} revisions done at {consistent_stats['cost']:.6f} dollars for consistent examples\n")
    f.write(
        f"There were {inconsistent_stats['amount']} samples and {inconsistent_stats['revisions']} revisions done at {inconsistent_stats['cost']:.6f} dollars for inconsistent examples\n")
    f.flush()
    f.close()


if __name__ == "__main__":
    main()

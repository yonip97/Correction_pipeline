import argparse
import time
import pandas as pd
from datetime import datetime
import os
from general.revision_pipeline import chose_revision_model
from tqdm import tqdm


def parseargs_llms():
    parser = argparse.ArgumentParser()
    parser.add_argument('-output_path')
    parser.add_argument('-revision_model_name', type=str, default='mock')
    parser.add_argument('-revision_prompt', type=str)
    parser.add_argument('-API_KEY_revision_model', type=str, default=None)
    parser.add_argument('-dir_path', type=str, default='experiments/revision/data/prompts_check')
    args = parser.parse_args()
    return args


def check_all_prompts():
    texts = []
    summaries = []
    prompts = {
        "prompt 1": """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific  compared to the document, you should not change it. Output only the corrected summary and nothing more.""",
        "prompt 2": """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific compared to the document, you should not change it. Output only the corrected summary and nothing more.""",
        "prompt 3": """I will provide you with a document and its summary. The summary is factually inconsistent w.r.t. the document, meaning that there are one or more facts that are not verifiable using the document. Your task is to provide a corrected version of the same summary which is factually consistent. The summary should be as close as possible to the original summary, with minimal changes, the only changes that you need to do are the ones that will convert it to factually consistent. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Note that if there is a fact that is correct but written in different words, or maybe generalized and less specific compared to the document, you should not change it. Output only the corrected summary and nothing more.""",
        "prompt 4": """The following summary is factually inconsistent w.r.t the document. Please make it factually consistent while making minimal changes to the summary. Output only the corrected summary and nothing more.""",
        "prompt 5": """The following summary is factually inconsistent w.r.t the document. Please make it factually consistent while making minimal changes to the summary. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Output only the corrected summary and nothing more.""",
        "prompt 6": """The following summary is factually inconsistent w.r.t the document. Please make it factually consistent while making minimal changes to the summary.  Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Output only the corrected summary and nothing more.""",
        "prompt 7": """The following summary is factually inconsistent w.r.t the document. Please follow the following to correct it. First, identify the parts in the summary which are factually inconsistent w.r.t the document. Then, find the correct information in the document that should replace the factually inconsistent parts in the summary. Finally, edit the summary by incorporating the correct information from the document, resulting in a precise and factually accurate summary. Output only the corrected summary and nothing more.""",
        "prompt 8": """The following summary is factually inconsistent w.r.t the document. Please follow the following to correct it. First, identify the parts in the summary which are factually inconsistent w.r.t the document. Then, find the correct information in the document that should replace the factually inconsistent parts in the summary. Finally, edit the summary by incorporating the correct information from the document, resulting in a precise and factually accurate summary. Focus on maintaining an abstractive approach and minimize direct copying from the source text.. Output only the corrected summary and nothing more.""",
        "prompt 9": """The following summary is factually inconsistent w.r.t the document. Please follow the following to correct it. First, identify the parts in the summary which are factually inconsistent w.r.t the document. Then, find the correct information in the document that should replace the factually inconsistent parts in the summary. Finally, edit the summary by incorporating the correct information from the document, resulting in a precise and factually accurate summary. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Output only the corrected summary and nothing more.""",
        "prompt 10": """Your assignment involves receiving a document and its summary from me. The summary currently exhibits factual discrepancies with respect to the document, indicating the presence of one or more facts that lack verification within the document. Your duty is to produce a revised version of the summary that maintains factual consistency. Strive to make only essential changes, ensuring the corrected summary closely mirrors the original while rectifying factual inaccuracies. Note that accurate facts should not be altered, even if rephrased or expressed more generally than in the document. Output only the corrected summary and nothing more.""",
        "prompt 11": """Your assignment involves receiving a document and its summary from me. The summary currently exhibits factual discrepancies with respect to the document, indicating the presence of one or more facts that lack verification within the document. Your duty is to produce a revised version of the summary that maintains factual consistency. Strive to make only essential changes, ensuring the corrected summary closely mirrors the original while rectifying factual inaccuracies. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Note that accurate facts should not be altered, even if rephrased or expressed more generally than in the document. Output only the corrected summary and nothing more.""",
        "prompt 12": """Your assignment involves receiving a document and its summary from me. The summary currently exhibits factual discrepancies with respect to the document, indicating the presence of one or more facts that lack verification within the document. Your duty is to produce a revised version of the summary that maintains factual consistency. Strive to make only essential changes, ensuring the corrected summary closely mirrors the original while rectifying factual inaccuracies. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Note that accurate facts should not be altered, even if rephrased or expressed more generally than in the document. Output only the corrected summary and nothing more.""",
        "prompt 13": """There are factual discrepancies between the document and the following summary. Your responsibility is to rectify these inconsistencies with minimal alterations to the summary.Output only the corrected summary and nothing more.""",
        "prompt 14": """There are factual discrepancies between the document and the following summary. Your responsibility is to rectify these inconsistencies with minimal alterations to the summary. Focus on maintaining an abstractive approach and minimize direct copying from the source text. Output only the corrected summary and nothing more.""",
        "prompt 15": """There are factual discrepancies between the document and the following summary. Your responsibility is to rectify these inconsistencies with minimal alterations to the summary. Try to use your own vocabulary while correcting the summary, instead of the document vocabulary. Output only the corrected summary and nothing more."""}
    args = parseargs_llms()
    original_dir_path = args.dir_path
    for prompt_id, prompt in prompts.items():
        args.prompt = prompt
        args.dir_path = os.path.join(original_dir_path, prompt_id)
        revision_model = chose_revision_model(args)
        revised_summaries, errors = [], []
        for text, summary in tqdm(zip(texts, summaries)):
            revised_summary, error = revision_model.revise_single(text=text, summary=summary)
            revised_summaries.append(revised_summary)
            errors.append(error)
            # time.sleep(2)
        pd.DataFrame.from_dict(
            {'text': texts, 'model_summary': summaries, 'revised_summary': revised_summaries, 'error': errors}).to_csv(
            args.output_path + '.csv')

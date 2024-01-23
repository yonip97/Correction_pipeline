# Copyright 2020 The Q2 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from collections import Counter
import numpy as np
import pandas as pd
import spacy
import torch.cuda
from tqdm import tqdm
# import sacrebleu
from bert_score import score

from q_squared.run_pipline import get_response_score
from q_squared.run_pipline import Pipeline
from allennlp.predictors.predictor import Predictor

nlp = spacy.load("en_core_web_sm")

NO_VALID_Q = -1
NO_ANS = '[CLS]'
NO_NLI = 'NO_NLI'
NO_Q = -1
ENTAILMENT_SCORE = 1
CONTRADICTION_SCORE = 0
NEUTRAL_SCORE = 0.5


def cross_annotated_scores(texts, summaries, out_path, save=False, for_system=None, device='cpu'):
    # df = pd.read_csv(in_path)
    if for_system is not None:
        logger = open(for_system, 'w')
    pipeline = Pipeline(device=device)
    counter = 0
    for sys_type in ['xsum']:
        f1_scores, questions, cands, answers = [], [], [], []
        # all_knowledge, responses, labels, ids = [], [], [], []
        all_knowledge, responses, ids = [], [], []
        # for idx, row in df.iterrows():
        for idx in tqdm(range(len(texts))):
            # response = row[sys_type + '_response']
            # knowledge = row['knowledge']
            response = summaries[idx]
            knowledge = texts[idx]
            try:
                # res, res_questions, res_cands, res_answers, res_scores \
                #     = get_response_score(response, knowledge, 'beam', single=True, remove_personal=True)
                # torch.cuda.empty_cache()
                res, res_questions, res_cands, res_answers, res_scores \
                    = pipeline.get_response_score(response, knowledge, 'beam', single=True, remove_personal=True)
                if res != NO_VALID_Q:
                    f1_scores.extend(res_scores)
                    questions.extend(res_questions)
                    cands.extend(res_cands)
                    answers.extend(res_answers)
                    all_knowledge.extend([knowledge] * len(res_questions))
                    responses.extend([response] * len(res_questions))
                    # labels.extend([row[sys_type + '_label']] * len(res_questions))
                    ids.extend([idx] * len(res_questions))

                else:
                    f1_scores.extend([NO_VALID_Q])
                    questions.extend(['NO_Q'])
                    cands.extend(['NO_Q'])
                    answers.extend(['NO_Q'])
                    all_knowledge.extend([knowledge])
                    responses.extend([response])
                    # labels.extend([row[sys_type + '_label']])
                    ids.extend([idx])
            except Exception as e:
                counter += 1
                print(counter)
                print(str(e))
                torch.cuda.empty_cache()
                if for_system is not None:
                    logger.write(str(e) + '\n')
                continue
        res_dict = {'id': ids, 'score': f1_scores, 'response': responses, 'cand': cands, 'question': questions,
                    'knowledge_ans': answers, 'knowledge': all_knowledge}
        res_df = pd.DataFrame(data=res_dict)
        if save:
            res_df.to_csv(out_path + '_' + sys_type + '.csv')
        return res_df

# if __name__ == '__main__':
# parser = argparse.ArgumentParser()
# parser.add_argument("--infile", type=str, required=True, help="Path to a csv file containing q^2 scores.")
# parser.add_argument("--outfile", type=str, required=True, help="Path to an output file")
# args = parser.parse_args()
#
# cross_annotated_scores(args.infile, args.outfile)

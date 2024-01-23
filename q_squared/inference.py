import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
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


class Q2_metric():
    def __init__(self, device, out_path, save=False, for_system=None):
        self.pipeline = Pipeline(device=device)
        self.out_path = out_path
        self.save = save
        self.logger = None
        if for_system is not None:
            self.logger = open(for_system, 'w')

        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
            predictor_name="textual_entailment")

    def score(self, texts, summaries):
        df = self.cross_annotated_scores(texts, summaries)
        df = self.scores_with_nli(df)
        df = self.aggregate_per_response(df)
        return df['Q2'].tolist()

    def cross_annotated_scores(self, texts, summaries):
        counter = 0
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
                    = self.pipeline.get_response_score(response, knowledge, 'beam', single=True, remove_personal=True)
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
                if self.logger is not None:
                    self.logger.write(str(e) + '\n')
                continue
        res_dict = {'id': ids, 'score': f1_scores, 'response': responses, 'cand': cands, 'question': questions,
                    'knowledge_ans': answers, 'knowledge': all_knowledge}
        res_df = pd.DataFrame(data=res_dict)
        if self.save:
            res_df.to_csv(self.out_path)
        return res_df

    def scores_with_nli(self, df):
        nli_scores = []
        f1_scores = []
        for _, row in tqdm(df.iterrows()):
            f1_score = row['score']

            evidence_answer = str(row['knowledge_ans'])

            nli_score = f1_score

            # Use NLI to determine answer similarity.
            # This is only applicable for responses that had at least one valid question generated

            if 0 <= f1_score < 1 and NO_ANS not in evidence_answer and evidence_answer != '' and evidence_answer != 'nan':
                f1_scores.append(f1_score)
                # If the score is 1, there is a full overlap between the
                # candidate and the predicted answer, so the score is 1
                # If there is no answer - can't run NLI, keep the original score (0)

                nli_label = self.get_nli_label(str(row['question']), str(row['cand']), evidence_answer)

                if nli_label == 'entailment':  # If entails, the score is 1
                    nli_score = ENTAILMENT_SCORE
                elif nli_label == 'contradiction':  # If contradicts, the score is 0
                    nli_score = CONTRADICTION_SCORE

            # Add fallback NLI to responses that are not covered by Q2 (no questions generated)
            elif f1_score == NO_Q:
                nli_fallback = self.get_e2e_nli_score(str(row['response']), str(row['knowledge']).lower())
                nli_score = nli_fallback
                f1_scores.append(nli_fallback)
            else:
                f1_scores.append(f1_score)

            nli_scores.append(nli_score)

        df['q2_score'] = nli_scores
        df['q2_no_nli'] = f1_scores
        return df

    def aggregate_per_response(self, df, for_systems_simulation=False):
        f1_scores_by_id = dict()
        nli_scores_by_id = dict()
        knowledge_by_id = dict()
        response_by_id = dict()
        label_by_id = dict()

        for _, row in df.iterrows():
            idx = row['id']
            f1_score = row['q2_no_nli']
            nli_score = row['q2_score']

            if idx in f1_scores_by_id:
                f1_scores_by_id[idx].append(f1_score)
                nli_scores_by_id[idx].append(nli_score)
            else:
                f1_scores_by_id[idx] = [f1_score]
                nli_scores_by_id[idx] = [nli_score]
                response_by_id[idx] = row['response']
                knowledge_by_id[idx] = row['knowledge']
                if for_systems_simulation:
                    label_by_id[idx] = row['label']

        mean_f1_scores = []
        mean_nli_scores = []
        responses = []
        knowledge = []
        labels = []

        for idx in f1_scores_by_id.keys():
            mean_f1_scores.append(np.mean(f1_scores_by_id[idx]))
            mean_nli_scores.append(np.mean(nli_scores_by_id[idx]))
            responses.append(response_by_id[idx])
            knowledge.append(knowledge_by_id[idx])
            if for_systems_simulation:
                labels.append(label_by_id[idx])

        print('Q2:', np.mean(mean_nli_scores))
        print('Q2, no nli:', np.mean(mean_f1_scores))
        data = {'id': list(f1_scores_by_id.keys()), 'response': responses, 'knowledge': knowledge,
                'Q2_no_nli': mean_f1_scores, 'Q2': mean_nli_scores}

        res_df = pd.DataFrame(data=data)
        if for_systems_simulation:
            res_df['label'] = labels
        return res_df

    def get_e2e_nli_score(self, response, knowledge):
        res = self.predictor.predict(
            premise=knowledge,
            hypothesis=response
        )

        nli_label = res['label']

        if nli_label == 'entailment':  # If entails, the score is 1
            return ENTAILMENT_SCORE
        elif nli_label == 'contradiction':  # If contradicts, the score is 0
            return CONTRADICTION_SCORE
        else:
            return NEUTRAL_SCORE

    def get_nli_label(self, question, cand, evidence_ans):
        premise = question + ' ' + evidence_ans + '.'
        hypothesis = question + ' ' + cand + '.'

        res = self.predictor.predict(
            premise=premise,
            hypothesis=hypothesis
        )

        return res['label']

import os

import math

from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from general.utils import clean_text
from general.metrics import word_wise_f1_score
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

INVALID_QUESTION = -1
NO_ANS = '[CLS]'
NO_VALID_QUESTIONS = 'NO_Q'


class Question_generation():
    def __init__(self, device='cpu'):
        self.qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        self.qg_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to(
            device)
        self.nlp = spacy.load("en_core_web_sm")
        self.device = device

    def get_answer_candidates(self, text):
        # This is different from the text. This you get all the nouns which are not entities.
        # The paper said they got all entities and nouns.
        doc = self.nlp(text)
        candidates = [ent.text.lower() for ent in list(doc.ents)]
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            found = False
            for cand in candidates:
                if chunk.text.lower() == cand.lower():
                    found = True
            if not found:
                candidates.append(chunk.text)
        candidates = [cand for cand in candidates if cand.lower() != 'i']
        return candidates

    def get_question_greedy(self, answers, context, max_length=128):
        with torch.no_grad():
            model_input = []
            for answer in answers:
                model_input.append("answer: %s  context: %s </s>" % (answer, context))
            features = self.qg_tokenizer(model_input, return_tensors='pt', padding=True)

            output = self.qg_model.generate(input_ids=features['input_ids'].to(self.device),
                                            attention_mask=features['attention_mask'].to(self.device),
                                            max_length=max_length)

            questions = self.qg_tokenizer.batch_decode(output, skip_special_tokens=True)
            questions = [q.replace("question: ", "", 1) for q in questions]
            return questions

    def get_questions_beam(self, answers, context, max_length=128, beam_size=5, num_return=5):
        with torch.no_grad():
            model_input = []
            for answer in answers:
                model_input.append("answer: %s  context: %s </s>" % (answer, context))
            features = self.qg_tokenizer(model_input, return_tensors='pt', padding=True)

            beam_outputs = self.qg_model.generate(input_ids=features['input_ids'].to(self.device),
                                                  attention_mask=features['attention_mask'].to(self.device),
                                                  max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                                  num_return_sequences=num_return, early_stopping=True)

            all_questions = self.qg_tokenizer.batch_decode(beam_outputs, skip_special_tokens=True)
            all_questions = [q.replace("question: ", "", 1) for q in all_questions]
            return all_questions

    def get_questions_sample(self, answers, context, max_length=128, top_k=50, top_p=0.95, num_return=5):
        with torch.no_grad():
            model_input = []
            for answer in answers:
                model_input.append("answer: %s  context: %s </s>" % (answer, context))
            features = self.qg_tokenizer(model_input, return_tensors='pt', padding=True)

            sampled_outputs = self.qg_model.generate(input_ids=features['input_ids'].to(self.device),
                                                     attention_mask=features['attention_mask'].to(self.device),
                                                     max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                                     num_return_sequences=num_return)
            all_questions = self.qg_tokenizer.batch_decode(sampled_outputs, skip_special_tokens=True)
            all_questions = [q.replace("question: ", "", 1) for q in all_questions]
            return all_questions

    def non_personal(self, question):
        question_tok = self.nlp(question)
        for tok in question_tok:
            if tok.dep_ == 'nsubj':
                if tok.text.lower() == 'i' or tok.text.lower() == 'you':
                    return False
            elif tok.dep_ == 'poss':
                if tok.text.lower() == 'my' or tok.text.lower() == 'your':
                    return False
        return True

    def filter_questions(self, exp_ans, pred_ans):
        if pred_ans == NO_ANS:
            return 'NO MATCH'
        if clean_text(exp_ans) != clean_text(pred_ans):
            return 'NO MATCH'
        return 'VALID'


class Question_answerer():
    def __init__(self, device='cpu'):
        self.qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2").to(
            device)
        self.device = device

    def get_answer(self, question, text):  # Code taken from https://huggingface.co/transformers/task_summary.html
        with torch.no_grad():
            inputs = self.qa_tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt",
                                                   max_length=512).to(self.device)
            input_ids = inputs["input_ids"].tolist()[0]

            answer_start_scores, answer_end_scores = self.qa_model(**inputs, return_dict=False)

            answer_start = torch.argmax(
                answer_start_scores
            )  # Get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(
                answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

            ans = self.qa_tokenizer.convert_tokens_to_string(
                self.qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
            return ans

    def get_answers(self, questions, text, batch_size):
        answers = []
        iterations = math.ceil(len(questions) / batch_size)
        with torch.no_grad():
            for i in range(iterations):
                batch_questions = questions[i * batch_size:(i + 1) * batch_size]
                batch_text = [text.lower()] * len(batch_questions)
                pairs = [(q, t) for q, t in zip(batch_questions, batch_text)]
                inputs = self.qa_tokenizer.batch_encode_plus(pairs, return_tensors="pt", add_special_tokens=True,
                                                             max_length=512, padding=True, truncation=True).to(
                    self.device)
                input_ids = inputs["input_ids"].tolist()

                answer_start_scores, answer_end_scores = self.qa_model(**inputs, return_dict=False)

                answer_start = torch.argmax(
                    answer_start_scores, dim=1
                )  # Get the most likely beginning of answer with the argmax of the score
                answer_end = torch.argmax(
                    answer_end_scores, dim=1) + 1  # Get the most likely end of answer with the argmax of the score
                for j in range(len(batch_questions)):
                    ans = self.qa_tokenizer.convert_tokens_to_string(
                        self.qa_tokenizer.convert_ids_to_tokens(input_ids[j][answer_start[j]:answer_end[j]]))
                    answers.append(ans)
        return answers


class Entailment_model():
    def __init__(self):
        path = os.getcwd()
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
            predictor_name="textual_entailment")
        os.chdir(path)
        self.NO_Q = -1
        self.ENTAILMENT_SCORE = 1
        self.CONTRADICTION_SCORE = 0
        self.NEUTRAL_SCORE = 0.5

    def get_e2e_nli_score(self, response, knowledge):
        res = self.predictor.predict(
            premise=knowledge,
            hypothesis=response
        )

        nli_label = res['label']

        if nli_label == 'entailment':  # If entails, the score is 1
            return self.ENTAILMENT_SCORE
        elif nli_label == 'contradiction':  # If contradicts, the score is 0
            return self.CONTRADICTION_SCORE
        else:
            return self.NEUTRAL_SCORE


class Q_squared_classifier():
    def __init__(self, device='cpu', similarity_metric='f1', threshold=0.5, remove_personal=True):
        self.qg = Question_generation(device=device)
        self.qa = Question_answerer(device=device)
        self.similarity_metric = similarity_metric
        if self.similarity_metric == 'nli':
            self.nli_model = Entailment_model()
        self.threshold = threshold
        self.remove_personal = remove_personal

    def classify(self, texts, summaries):
        scores = self.score(texts, summaries)
        return [1 if s > self.threshold else 0 for s in scores]
    def classify_single(self,text,summary):
        return self.score_single(text,summary) > self.threshold

    def score(self, texts, summaries):
        scores = []
        for i in range(len(summaries)):
            summary = summaries[i]
            text = texts[i]
            score = self.score_single(text=text, summary=summary)
            scores.append(score)
        return scores

    def score_single(self, text, summary):
        candidates = self.qg.get_answer_candidates(summary)
        questions = self.qg.get_question_greedy(candidates, summary)
        if self.remove_personal:
            idx = [i for i in range(len(questions)) if self.qg.non_personal(questions[i])]
            candidates = [candidates[i] for i in idx]
            questions = [questions[i] for i in idx]
        response_answers = self.qa.get_answers(questions, summary, batch_size=8)
        valid_questions_idx = [i for i, (cand, pred_ans) in enumerate(zip(candidates, response_answers)) if
                               self.qg.filter_questions(cand, pred_ans) == 'VALID']
        valid_questions = [questions[i] for i in valid_questions_idx]
        valid_candidates = [candidates[i] for i in valid_questions_idx]
        # valid_knowledge_answers = [self.qa.get_answer(q, knowledge) for q in valid_questions]
        valid_knowledge_answers = self.qa.get_answers(valid_questions, text, batch_size=8)
        if self.similarity_metric == 'nli':
            scores = [self.nli_model.get_e2e_nli_score(cand, knowledge_ans) if knowledge_ans != NO_ANS else 0 for
                      cand, knowledge_ans in zip(valid_candidates, valid_knowledge_answers)]
        elif self.similarity_metric == 'f1':
            scores = [word_wise_f1_score(cand, knowledge_ans) if knowledge_ans != NO_ANS else 0 for
                      cand, knowledge_ans in zip(valid_candidates, valid_knowledge_answers)]
        return np.mean(scores)

    # def questions_score(self, questions, candidates, response, knowledge):
    #     pred_answers_response = self.qa.get_answers(questions, response,)
    #     valid_questions_idx = [i for i, (pred_answer, cand) in enumerate(zip(pred_answers_response, candidates)) if
    #                            self.qg.filter_questions(cand, pred_answer) == 'VALID']
    #     filtered_questions = [questions[i] for i in valid_questions_idx]
    #     filtered_candidates = [candidates[i] for i in valid_questions_idx]
    #     # filtered_pred_answers_response = [pred_answers_response[i] for i in pred_answers_response]
    #     filtered_pred_answers_knowledge = self.qa.get_answers(questions, knowledge)
    #     scores = []
    #     answers = []
    #     # start = time.time()
    #     for cand, knowledge_answer in zip(filtered_candidates, filtered_pred_answers_knowledge):
    #         if knowledge_answer != NO_ANS:
    #             if self.similarity_metric == 'f1':
    #                 scores.append(word_wise_f1_score(cand, knowledge_answer))
    #                 answers.append(knowledge_answer)
    #                 # return f1_score(cand, knowledge_ans), knowledge_ans
    #             elif self.similarity_metric == 'nli':
    #                 scores.append(self.nli_model.get_e2e_nli_score(cand, knowledge_answer))
    #                 answers.append(knowledge_answer)
    #                 # return self.nli_model.get_e2e_nli_score(cand, knowledge_ans), knowledge_ans
    #         else:
    #             scores.append(0)
    #             answers.append(NO_ANS)
    #     # print(f"Just scoring time no filter {time.time()-start} seconds")
    #     return scores, answers, filtered_questions, filtered_candidates
    #
    # def single_question_score(self, question, cand, response, knowledge):
    #     pred_ans = self.qa.get_answer(question, response)
    #
    #     if self.qg.filter_questions(cand, pred_ans) == 'VALID':
    #         knowledge_ans = self.qa.get_answer(question, knowledge)
    #         if knowledge_ans != NO_ANS:
    #             if self.similarity_metric == 'f1':
    #                 return word_wise_f1_score(cand, knowledge_ans), knowledge_ans
    #             elif self.similarity_metric == 'nli':
    #                 return self.nli_model.get_e2e_nli_score(cand, knowledge_ans), knowledge_ans
    #         else:
    #             return 0, NO_ANS
    #     else:
    #         return INVALID_QUESTION, INVALID_QUESTION


def calc_scores(texts, summaries, device, similarity_metric='nli'):
    nli_q_scores = []
    f1_q_scores = []
    classifier = Q_squared_classifier(device=device, similarity_metric=similarity_metric)
    counter = 0
    for i in tqdm(range(len(texts))):
        text = texts[i]
        summary = summaries[i]
        nli_score, f1_score = \
            classifier.score([summary], [text])
        nli_q_scores.append(nli_score)
        f1_q_scores.append(f1_score)
        counter += 1
        if counter > 100:
            break

    # for text, summary in tqdm(zip(texts, summaries)):
    #     res = \
    #         classifier.classify(summary, text, remove_personal)
    #     q_scores.append(res)

    # valid_scores = [s for s in q_scores if s != -1]
    # valid_scores = []
    # for s in q_scores:
    #     if s == False:
    #         valid_scores.append(0)
    #     elif s == True:
    #         valid_scores.append(1)
    #     else:
    #         valid_scores.append(None)
    # print("total with at least 1 valid question:", len(valid_scores))
    # print("score:", np.mean(valid_scores))

    return nli_q_scores, f1_q_scores


def main():
    device = 'cuda'
    similarity_metric = 'nli'
    df = pd.read_csv('/data/home/yehonatan-pe/Correction_pipeline/data/true_data/summeval_download.csv')
    texts = df['grounding'].tolist()
    summaries = df['generated_text'].tolist()
    x = calc_scores(texts=texts, summaries=summaries, device=device,
                    similarity_metric=similarity_metric)
    print(np.mean(x))
    # x = TRUE_dataset('data')
    # x.filter_to_datasets(true_topics(['summarization']))
    # data = x.df
    # texts = data['grounding']
    # summaries = data['generated_text']
    # labels = data['label']

#
# if __name__ == '__main__':
#     main()

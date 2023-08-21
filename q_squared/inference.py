from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from utils import clean_text, f1_score
from allennlp.predictors.predictor import Predictor

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



class Entailment_model():
    def __init__(self):
        self.predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/snli_roberta-2020.06.09.tar.gz",
            predictor_name="textual_entailment")
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
    def __init__(self, device='cpu', similarity_metric='f1', threshold=0.5):
        self.qg = Question_generation(device=device)
        self.qa = Question_answerer(device=device)
        self.similarity_metric = similarity_metric
        if self.similarity_metric == 'nli':
            self.nli_model = Entailment_model()

    def get_response_score(self, response, knowledge, gen_method, single, remove_personal):
        f1 = 0
        num_questions = 0

        valid_questions = []
        valid_cands = []
        knowledge_answers = []
        scores = []
        candidates = self.qg.get_answer_candidates(response)
        # for cand in candidates:
        if gen_method == 'greedy':
            questions = self.qg.get_question_greedy(candidates, response)
        elif gen_method == 'beam':
            questions = self.qg.get_questions_beam(candidates, response)
        else:
            questions = self.qg.get_questions_sample(candidates, response)
        for question,cand in zip(questions,candidates):
            if not remove_personal or self.qg.non_personal(question):
                question_score, knowledge_ans = self.single_question_score(question, cand, response, knowledge)
                if question_score != INVALID_QUESTION:
                    num_questions += 1
                    f1 += question_score

                    valid_questions.append(question)
                    valid_cands.append(cand)
                    knowledge_answers.append(knowledge_ans)
                    scores.append(question_score)

                    if single:
                        break

        if len(valid_questions):
            avg_f1 = sum(scores) / len(valid_questions)
        else:
            avg_f1 = INVALID_QUESTION
        return avg_f1, valid_questions, valid_cands, knowledge_answers, scores

    def questions_score(self, questions, candidates, response, knowledge):
        pred_answers_response = self.qa.get_answers(questions, response)
        valid_questions_idx = [i for i, (pred_answer, cand) in enumerate(zip(pred_answers_response, candidates)) if
                               self.qg.filter_questions(cand, pred_answer) == 'VALID']
        filtered_questions = [questions[i] for i in valid_questions_idx]
        filtered_candidates = [candidates[i] for i in valid_questions_idx]
        # filtered_pred_answers_response = [pred_answers_response[i] for i in pred_answers_response]
        filtered_pred_answers_knowledge = self.qa.get_answers(questions, knowledge)
        scores = []
        answers = []
        #start = time.time()
        for cand, knowledge_answer in zip(filtered_candidates, filtered_pred_answers_knowledge):
            if knowledge_answer != NO_ANS:
                if self.similarity_metric == 'f1':
                    scores.append(f1_score(cand, knowledge_answer))
                    answers.append(knowledge_answer)
                    # return f1_score(cand, knowledge_ans), knowledge_ans
                elif self.similarity_metric == 'nli':
                    scores.append(self.nli_model.get_e2e_nli_score(cand, knowledge_answer))
                    answers.append(knowledge_answer)
                    # return self.nli_model.get_e2e_nli_score(cand, knowledge_ans), knowledge_ans
            else:
                scores.append(0)
                answers.append(NO_ANS)
        #print(f"Just scoring time no filter {time.time()-start} seconds")
        return scores, answers, filtered_questions, filtered_candidates

    def single_question_score(self, question, cand, response, knowledge):
        pred_ans = self.qa.get_answer(question, response)

        if self.qg.filter_questions(cand, pred_ans) == 'VALID':
            knowledge_ans = self.qa.get_answer(question, knowledge)
            if knowledge_ans != NO_ANS:
                if self.similarity_metric == 'f1':
                    return f1_score(cand, knowledge_ans), knowledge_ans
                elif self.similarity_metric == 'nli':
                    return self.nli_model.get_e2e_nli_score(cand, knowledge_ans), knowledge_ans
            else:
                return 0, NO_ANS
        else:
            return INVALID_QUESTION, INVALID_QUESTION


def calc_scores(texts,summaries, gen_method, single, remove_personal, device,similarity_metric = 'nli'):
    q_scores = []

    classifier = Q_squared_classifier(device=device,similarity_metric=similarity_metric)
    for text, summary in tqdm(zip(texts,summaries)):
        res, res_questions, res_cands, res_answers, res_scores = \
            classifier.get_response_score(summary, text, gen_method, single, remove_personal)

        q_scores.append(res)
    valid_scores = [s for s in q_scores if s != -1]
    print("total with at least 1 valid question:", len(valid_scores))
    print("score:", np.mean(valid_scores))

    return valid_scores


def main():

    device = 'cuda'
    similarity_metric = 'nli'
    df = pd.read_csv('/data/home/yehonatan-pe/Correction_pipeline/data/summeval_download.csv')
    texts = df['grounding'].tolist()
    summaries = df['generated_text'].tolist()
    x = calc_scores(texts=texts,summaries=summaries, gen_method='greedy',
                    single=True, remove_personal=True, device=device,similarity_metric=similarity_metric)
    print(np.mean(x))
    # x = TRUE_dataset('data')
    # x.filter_to_datasets(true_topics(['summarization']))
    # data = x.df
    # texts = data['grounding']
    # summaries = data['generated_text']
    # labels = data['label']


if __name__ == '__main__':
    main()

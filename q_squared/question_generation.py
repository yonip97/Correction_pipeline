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

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForQuestionAnswering

import spacy

# device = 'cuda:0'
# qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
# qg_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to(device)

nlp = spacy.load("en_core_web_sm")


class QGModel():
    def __init__(self, device):
        self.qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        self.qg_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to(
            device)
        self.device = device

    def get_answer_candidates(self, text):
        doc = nlp(text)
        candidates = [ent.text for ent in list(doc.ents)]
        noun_chunks = list(doc.noun_chunks)
        for chunk in noun_chunks:
            found = False
            for cand in candidates:
                if chunk.text.lower() == cand.lower():
                    found = True
            if not found:
                candidates.append(chunk.text)
        # candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
        candidates = [cand for cand in candidates if cand.lower() != 'i']
        return candidates

    def get_question_greedy_multi(self, answers, context, max_length=128):
        with torch.no_grad():
            input_texts = ["answer: %s  context: %s </s>" % (answer, context) for answer in answers]
            features = self.qg_tokenizer(input_texts, return_tensors='pt', padding=True)

            output = self.qg_model.generate(input_ids=features['input_ids'].to(self.device),
                                            attention_mask=features['attention_mask'].to(self.device),
                                            max_length=max_length).reshape((len(answers), -1))

            # question = qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
            questions = [self.qg_tokenizer.decode(output[i]).replace("question: ", "", 1) for i in range(len(answers))]
            return questions

    def get_questions_beam_multi(self, answers, context, max_length=128, beam_size=5, num_return=5):
        with torch.no_grad():
            all_questions = []
            input_texts = ["answer: %s  context: %s </s>" % (answer, context) for answer in answers]
            # input_text = "answer: %s  context: %s </s>" % (answer, context)
            features = self.qg_tokenizer(input_texts, return_tensors='pt', padding=True)

            beam_outputs = self.qg_model.generate(input_ids=features['input_ids'].to(self.device),
                                                  attention_mask=features['attention_mask'].to(self.device),
                                                  max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                                  num_return_sequences=num_return, early_stopping=True).reshape(
                (len(answers), num_return, -1))

            for beam_question_outputs in beam_outputs:
                all_questions.append([x.replace("question: ", "", 1) for x in
                                      self.qg_tokenizer.batch_decode(beam_question_outputs, skip_special_tokens=True)])
                # all_questions.append(
                #     qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))

            return all_questions

    def get_questions_sample_multi(self, answers, context, max_length=128, top_k=50, top_p=0.95, num_return=5):
        with torch.no_grad():
            all_questions = []
            input_texts = ["answer: %s  context: %s </s>" % (answer, context) for answer in answers]
            features = self.qg_tokenizer(input_texts, return_tensors='pt', padding=True)

            sampled_outputs = self.qg_model.generate(input_ids=features['input_ids'].to(self.device),
                                                     attention_mask=features['attention_mask'].to(self.device),
                                                     max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                                     num_return_sequences=num_return).reshape(
                (len(answers), num_return, -1))
            #
            # for sampled in sampled_outputs:
            #     all_questions.append(qg_tokenizer.decode(sampled, skip_special_tokens=True).replace("question: ", "", 1))
            for sampled_question_outputs in sampled_outputs:
                all_questions.append([x.replace("question: ", "", 1) for x in
                                      self.qg_tokenizer.batch_decode(sampled_question_outputs,
                                                                     skip_special_tokens=True)])
            return all_questions


def get_answer_candidates(text):
    doc = nlp(text)
    candidates = [ent.text for ent in list(doc.ents)]
    noun_chunks = list(doc.noun_chunks)
    for chunk in noun_chunks:
        found = False
        for cand in candidates:
            if chunk.text.lower() == cand.lower():
                found = True
        if not found:
            candidates.append(chunk.text)
    # candidates += [chunk.text for chunk in list(doc.noun_chunks) if chunk.text not in candidates]
    candidates = [cand for cand in candidates if cand.lower() != 'i']
    return candidates


# def get_answer_candidates(text):
#     doc = nlp(text)
#     candidates = [ent.text for ent in list(doc.ents)]
#     candidates_lower = [c.lower() for c in candidates]
#     noun_chunks = list(doc.noun_chunks)
#     candidates += [c.text for c in noun_chunks if c.text.lower() not in candidates_lower and c.text.lower() != 'i']
#     return candidates

def get_question_greedy_multi(answers, context, max_length=128):
    with torch.no_grad():
        input_texts = ["answer: %s  context: %s </s>" % (answer, context) for answer in answers]
        features = qg_tokenizer(input_texts, return_tensors='pt', padding=True)

        output = qg_model.generate(input_ids=features['input_ids'].to(device),
                                   attention_mask=features['attention_mask'].to(device),
                                   max_length=max_length).reshape((len(answers), -1))

        # question = qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
        questions = [qg_tokenizer.decode(output[i]).replace("question: ", "", 1) for i in range(len(answers))]
        return questions


def get_question_greedy(answer, context, max_length=128):
    with torch.no_grad():
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = qg_tokenizer([input_text], return_tensors='pt')

        output = qg_model.generate(input_ids=features['input_ids'].to(device),
                                   attention_mask=features['attention_mask'].to(device),
                                   max_length=max_length)

        question = qg_tokenizer.decode(output[0]).replace("question: ", "", 1)
        return question


def get_questions_beam_multi(answers, context, max_length=128, beam_size=5, num_return=5):
    with torch.no_grad():
        all_questions = []
        input_texts = ["answer: %s  context: %s </s>" % (answer, context) for answer in answers]
        # input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = qg_tokenizer(input_texts, return_tensors='pt', padding=True)

        beam_outputs = qg_model.generate(input_ids=features['input_ids'].to(device),
                                         attention_mask=features['attention_mask'].to(device),
                                         max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                         num_return_sequences=num_return, early_stopping=True).reshape(
            (len(answers), num_return, -1))

        for beam_question_outputs in beam_outputs:
            all_questions.append([x.replace("question: ", "", 1) for x in
                                  qg_tokenizer.batch_decode(beam_question_outputs, skip_special_tokens=True)])
            # all_questions.append(
            #     qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))

        return all_questions


def get_questions_beam(answer, context, max_length=128, beam_size=5, num_return=5):
    with torch.no_grad():
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = qg_tokenizer([input_text], return_tensors='pt')

        beam_outputs = qg_model.generate(input_ids=features['input_ids'].to(device),
                                         attention_mask=features['attention_mask'].to(device),
                                         max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                         num_return_sequences=num_return, early_stopping=True)

        for beam_output in beam_outputs:
            all_questions.append(
                qg_tokenizer.decode(beam_output, skip_special_tokens=True).replace("question: ", "", 1))

        return all_questions


def get_questions_sample_multi(answers, context, max_length=128, top_k=50, top_p=0.95, num_return=5):
    with torch.no_grad():
        all_questions = []
        input_texts = ["answer: %s  context: %s </s>" % (answer, context) for answer in answers]
        features = qg_tokenizer(input_texts, return_tensors='pt', padding=True)

        sampled_outputs = qg_model.generate(input_ids=features['input_ids'].to(device),
                                            attention_mask=features['attention_mask'].to(device),
                                            max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                            num_return_sequences=num_return).reshape((len(answers), num_return, -1))
        #
        # for sampled in sampled_outputs:
        #     all_questions.append(qg_tokenizer.decode(sampled, skip_special_tokens=True).replace("question: ", "", 1))
        for sampled_question_outputs in sampled_outputs:
            all_questions.append([x.replace("question: ", "", 1) for x in
                                  qg_tokenizer.batch_decode(sampled_question_outputs, skip_special_tokens=True)])
        return all_questions


def get_questions_sample(answer, context, max_length=128, top_k=50, top_p=0.95, num_return=5):
    with torch.no_grad():
        all_questions = []
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = qg_tokenizer([input_text], return_tensors='pt')

        sampled_outputs = qg_model.generate(input_ids=features['input_ids'].to(device),
                                            attention_mask=features['attention_mask'].to(device),
                                            max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                            num_return_sequences=num_return)

        for sampled in sampled_outputs:
            all_questions.append(qg_tokenizer.decode(sampled, skip_special_tokens=True).replace("question: ", "", 1))

        return all_questions

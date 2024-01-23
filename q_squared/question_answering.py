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
device = 'cuda:1'
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")

# qa_tokenizer = AutoTokenizer.from_pretrained('Primer/bart-squad2')
# qa_model = AutoModelForQuestionAnswering.from_pretrained('Primer/bart-squad2').to(device)
from transformers import pipeline


#
# model_name = "deepset/deberta-v3-base-squad2"
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device)
#

def get_answer(question, text):  # Code taken from https://huggingface.co/transformers/task_summary.html
    print("using old model")
    inputs = qa_tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]
    text_tokens = qa_tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = qa_model(**inputs, return_dict=False)
    answer_start = torch.argmax(
        answer_start_scores
    )  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    ans = qa_tokenizer.convert_tokens_to_string(qa_tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return ans


class QAModel():
    def __init__(self, device):
        self.model_name = "deepset/deberta-v3-base-squad2"
        self.nlp = pipeline('question-answering', model=self.model_name, tokenizer=self.model_name, device=device)

    def get_answer_new_model(self, question, text):
        with torch.no_grad():
            res = self.nlp(question=question, context=text, max_seq_len=2048, doc_stride=128)
            # print(res)
            return res['answer']


def get_answer_new_model(question, text):
    with torch.no_grad():
        res = nlp(question=question, context=text, max_seq_len=2048, doc_stride=128)
        # print(res)
        return res['answer']
    # b) Load model & tokenizer

    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_name = "ktrapeznikov/albert-xlarge-v2-squad-v2"
#
# nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
#
#
# def get_answer(question, text):
#     QA_input = {
#         'question': question,
#         'context': text
#     }
#     res = nlp(QA_input, handle_impossible_answer=True)
#
#     return res['answer']

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from correction_pipeline.llms import LLM_model
import openai
from correction_pipeline.utils import iter_list


class Question_answering_model_based():
    def __init__(self, model_name='albert', device='cpu', batch_size=16):
        self.model_name = model_name
        if model_name == 'albert':
            self.qa_tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2")
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/albert-xlarge-v2-squad-v2").to(
                device)
            self.max_length = 512
        elif model_name == 'longformer':
            self.qa_tokenizer = AutoTokenizer.from_pretrained("mrm8488/longformer-base-4096-finetuned-squadv2")
            self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
                "mrm8488/longformer-base-4096-finetuned-squadv2").to(device)
            self.max_length = 4096
        # what will the model return when it believes no answer exists
        self.unanswerable_response = 'UNANSWERABLE'
        # what the pipeline expects
        if model_name == 'albert':
            self.unanswerable_model_response = '[CLS]'
        elif model_name == 'longformer':
            self.unanswerable_model_response = 'UNANSWERABLE'
        self.device = device
        self.batch_size = batch_size

    def __call__(self, questions, text):
        answers = self.get_answers(questions, text)
        return answers

    def get_answers(self, questions, text):  # Code taken from https://huggingface.co/transformers/task_summary.html
        answers = []
        with torch.no_grad():
            if self.model_name == 'longformer':
                text += self.unanswerable_model_response
            for batch_questions in iter_list(questions, self.batch_size):

                texts = [text] * len(batch_questions)
                encoding = self.qa_tokenizer(batch_questions, texts, return_tensors="pt", padding=True,
                                             max_length=self.max_length, truncation=True).to(self.device)
                input_ids = encoding["input_ids"]

                # default is local attention everywhere
                # the forward method will automatically set global attention on question tokens
                attention_mask = encoding["attention_mask"]

                output = self.qa_model(input_ids, attention_mask=attention_mask)
                start_scores = output.start_logits
                end_scores = output.end_logits
                starts = torch.argmax(start_scores, dim=1)
                ends = torch.argmax(end_scores, dim=1)
                for i in range(len(batch_questions)):
                    answer = self.qa_tokenizer.decode(input_ids[i][starts[i]:ends[i] + 1])
                    if answer == self.unanswerable_model_response:
                        answer = self.unanswerable_response
                    answers.append(answer)
                # for i in range(len(batch_questions)):
                #     all_tokens = self.qa_tokenizer.convert_ids_to_tokens(input_ids[i].tolist())
                #
                #     answer_tokens = all_tokens[torch.argmax(start_scores):torch.argmax(end_scores) + 1]
                #     answer = self.qa_tokenizer.decode(self.qa_tokenizer.convert_tokens_to_ids(answer_tokens))
                #     answers.append(answer)
        return answers


class Question_answering_model_prompt_based(LLM_model):

    def __call__(self, questions, text, **kwargs):
        answers = []
        for q in questions:
            a = self.answer_question(q, text, **kwargs)
            answers.append(a)
        return answers

    def answer_question(self, question, text, **kwargs):
        model_input = self.create_model_input(text, question)
        answer = self.get_chatgpt_response(model_input, **kwargs)
        # answer = self.process_model_output(model_output)
        return answer

    def create_model_input(self, text, question):
        final_text = self.prompt + '\n'
        final_text += 'text: ' + text + '\n'
        final_text += 'question: ' + question + '\n'
        final_text += 'answer: '
        return final_text

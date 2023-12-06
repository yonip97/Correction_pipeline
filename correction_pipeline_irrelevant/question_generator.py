
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
import spacy
from correction_pipeline.llms import LLM_model
from correction_pipeline.utils import iter_list


class Question_generator_model_based():
    def __init__(self, method='greedy',batch_size = 16, device = 'cpu',**kwargs):
        self.qg_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
        self.qg_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap").to(device)
        self.nlp = spacy.load("en_core_web_sm")
        self.method = method
        self.device = device
        self.batch_size = batch_size

    def __call__(self, text, **kwargs):
        questions = []
        doc = self.nlp(text)
        nouns = [i for i in doc.noun_chunks]
        entities = [i for i in doc.ents]
        candidates = nouns
        candidates += entities
        for can_list in iter_list(candidates,self.batch_size):
            questions += self.get_questions(can_list, text, **kwargs)
        return questions

    def get_questions(self, answer, text, **kwargs):
        if self.method == 'greedy':
            return self.get_question_greedy(answer, text, **kwargs)
        elif self.method == 'beam':
            return self.get_questions_beam(answer, text, **kwargs)
        elif self.method == 'sample':
            return self.get_questions_sample(answer, text, **kwargs)

    def get_question_greedy(self, answers, text, max_length=128,**kwargs):
        with torch.no_grad():
            input_texts = []
            for answer in answers:
                input_text = "answer: %s  context: %s </s>" % (answer, text)
                input_texts.append(input_text)
            features = self.qg_tokenizer(input_texts, return_tensors='pt',padding=True).to(self.device)

            output = self.qg_model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'],
                                            max_length=max_length)

            questions = self.qg_tokenizer.batch_decode(output,skip_special_tokens=True)
            final_questions = []
            for q in questions:
                q = q.replace("question: ", "", 1).strip()
                final_questions.append(q)
        return final_questions

    def get_questions_beam(self, answers, text, max_length=128, beam_size=5, num_return=5,**kwargs):
        with torch.no_grad():
            all_questions = []
            input_texts = []
            for answer in answers:
                input_text = "answer: %s  context: %s </s>" % (answer, text)
                input_texts.append(input_text)
            features = self.qg_tokenizer(input_texts, return_tensors='pt',padding = True).to(self.device)
            beam_outputs = self.qg_model.generate(input_ids=features['input_ids'],
                                                  attention_mask=features['attention_mask'],
                                                  max_length=max_length, num_beams=beam_size, no_repeat_ngram_size=3,
                                                  num_return_sequences=num_return, early_stopping=True)
            beam_outputs = self.qg_tokenizer.batch_decode(beam_outputs,skip_special_tokens=True)
            for beam_output in beam_outputs:
                all_questions.append(beam_output.replace("question: ", "", 1).strip())
            return all_questions

    def get_questions_sample(self, answers, text, max_length=128, top_k=50, top_p=0.95, num_return=5,**kwargs):
        all_questions = []
        input_texts = []
        for answer in answers:
            input_text = "answer: %s  context: %s </s>" % (answer, text)
            input_texts.append(input_text)
        features = self.qg_tokenizer(input_texts, return_tensors='pt',padding = True)

        sampled_outputs = self.qg_model.generate(input_ids=features['input_ids'],
                                                 attention_mask=features['attention_mask'],
                                                 max_length=max_length, do_sample=True, top_k=top_k, top_p=top_p,
                                                 num_return_sequences=num_return)
        sampled_outputs = self.qg_tokenizer.batch_decode(sampled_outputs,skip_special_tokens=True)
        for sampled in sampled_outputs:
            all_questions.append(sampled.replace("question: ", "", 1).strip())
        return all_questions




class Question_generator_prompt_based(LLM_model):

    def __call__(self, text, **kwargs):
        model_input = self.create_model_input(text)
        model_output = self.get_chatgpt_response(model_input, **kwargs)
        questions = self.process_model_output(model_output)
        return questions

    def create_model_input(self, text):
        final_text = self.prompt + '\n'
        final_text += text
        return final_text


    def process_model_output(self, model_output):
        questions = model_output.split('\n')
        questions = [q.split('.')[1] for q in questions]
        return questions



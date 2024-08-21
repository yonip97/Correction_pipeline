import time

import openai
import tiktoken
import os
import csv
from general.utils import remove_punctuation

from openai import AzureOpenAI, OpenAI
from tqdm import tqdm
from groq import Groq


class LLM_model():
    def __init__(self, temp_save_dir, prompt, past_text_prompt='', model='gpt-3.5-turbo', API_KEY=None, azure=False,
                 input_price=0, output_price=0, groq=False,**kwargs):
        if prompt is None:
            raise ValueError("prompt can't be None")
        self.past_text_prompt = past_text_prompt
        self.prompt = prompt
        self.model = model
        print(f"model: {model}")
        if azure:
            self.client = AzureOpenAI(
                api_key=API_KEY,
                api_version='2023-09-01-preview',
                azure_endpoint='https://researchopenai2023eastus2.openai.azure.com/')
        if groq:
            self.client = Groq(api_key=API_KEY)
        else:
            self.client = OpenAI(api_key=API_KEY)
        if model == 'gpt-3.5-turbo':
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        elif model == 'gpt-4':
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-4')
        elif model == 'gpt-4-turbo':
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-4')
        elif 'llama-3.1' in model:
            from transformers import AutoTokenizer
            self.estimation_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-70B',token = "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")
        else:
            raise ValueError(f"model {model} not supported")
        self.open_ai_errors = 0
        self.other_errors = 0
        path_logger = temp_save_dir + '/' + 'logger.txt'
        self.logger = open(path_logger, 'w')
        self.input_price = input_price
        self.output_price = output_price

    def call_llm(self, text, max_length, **kwargs):
        input = self.prompt + '\n' + text + '\n' + self.past_text_prompt
        try:
            message = [{
                "role": "user",
                "content": input,
            }]
            if self.model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']:
                if self.model == 'gpt-3.5-turbo':
                    model = 'gpt-35-turbo'
                else:
                    model = self.model
                response = self.client.chat.completions.create(model=model,
                                                               messages=message,
                                                               temperature=0,
                                                               max_tokens=max_length, timeout=60, **kwargs)
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                price = input_tokens / 1000 * self.input_price + output_tokens / 1000 * self.output_price
                return response.choices[0].message.content, None, price
            if 'llama' in self.model:
                #message = {"role": "user", "content": input}
                response = self.client.chat.completions.create(model=self.model,
                                                               messages=message, max_tokens=max_length,
                                                                                            ** kwargs)
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                price = input_tokens / 1000 * self.input_price + output_tokens / 1000 * self.output_price
                return response.choices[0].message.content, None, price
            else:
                raise ValueError(f"model {self.model} not supported")
        except openai.OpenAIError as e:
            print(f"Error occurred: {e}")
            self.logger.write(f"Error occurred: {e}")
            self.open_ai_errors += 1
            return None, f"{e}"
        except Exception as e:
            self.other_errors += 1
            self.logger.write(f"Error occurred: {e}")
            print(f"Error in output occurred: {e}")
            return None, f"{e}"


class Summarization_correction_model(LLM_model):
    def __init__(self, temp_save_dir, prompt, past_text_prompt='', model='gpt-3.5-turbo', API_KEY=None, input_price=0,
                 output_price=0, **kwargs):
        super(Summarization_correction_model, self).__init__(temp_save_dir, prompt, past_text_prompt, model, API_KEY,
                                                             input_price=input_price, output_price=output_price,
                                                             **kwargs)
        f = open(temp_save_dir + '/' + 'temp_results_revision.csv', 'w')
        self.csv_writer = csv.writer(f)
        self.csv_writer.writerow(['text', 'summary', 'revised_summary', 'error'])

    def revise_single(self, text, summary, max_length=None, instructions=None, **kwargs):
        if instructions is not None:
            text_for_revision = f"Document: \n {text} \n Summary: \n {summary} \n Instructions: \n {instructions} \n"
        else:
            text_for_revision = f"Document: \n {text} \n Summary: \n {summary} \n"
        if max_length is None:
            max_length = len(self.estimation_tokenizer.encode(summary)) + 10
        revised_summary, error, price = self.call_llm(text_for_revision, max_length=max_length, **kwargs)
        self.csv_writer.writerow([text, summary, revised_summary, error, price])
        return revised_summary, error, price


class LLMFactualityClassifier(LLM_model):
    def __init__(self, temp_save_dir, prompt, text_to_labels, past_text_prompt='', model='gpt-3.5-turbo', API_KEY=None,
                 **kwargs):
        super(LLMFactualityClassifier, self).__init__(temp_save_dir, prompt, past_text_prompt, model, API_KEY, **kwargs)
        f = open(temp_save_dir + '/' + 'temp_results_classification.csv', 'w')
        self.csv_writer = csv.writer(f)
        self.csv_writer.writerow(['text', 'summary', 'prediction_text', 'prediction_label', 'error', 'price'])
        self.text_to_labels = text_to_labels

    def classify(self, texts, summaries, max_length):
        predictions, errors = [], []
        for i in range(len(texts)):
            document = texts[i]
            summary = summaries[i]
            prediction, error = self.classify_single(document, summary, max_length)
            predictions.append(prediction)
            errors.append(error)
        return predictions, errors

    def classify_single(self, text, summary, max_length):
        text = f"Document: \n {text} \n summary: \n {summary} \n"
        response, error = self.call_llm(text, max_length=max_length)
        if response is not None:
            response = remove_punctuation(response.lower().strip())
        if response not in self.text_to_labels:
            prediction = None
        else:
            prediction = self.text_to_labels[response]
        self.csv_writer.writerow([text, summary, response, prediction, error])
        return prediction, error


class SummarizationModel(LLM_model):
    def __init__(self, temp_save_dir, prompt, past_text_prompt='', model='gpt-3.5-turbo', API_KEY=None, azure=False,
                 input_price=0, output_price=0,
                 **kwargs):
        super(SummarizationModel, self).__init__(temp_save_dir, prompt, past_text_prompt, model, API_KEY, azure,
                                                 input_price=input_price, output_price=output_price,
                                                 **kwargs)
        f = open(temp_save_dir + '/' + 'temp_results_summarization.csv', 'w')
        self.csv_writer = csv.writer(f)
        self.csv_writer.writerow(['text', 'summary', 'error', 'price'])

    def summarize(self, texts, max_generation_length):
        summaries, errors, prices = [], [], []
        for text in tqdm(texts):
            revised_summary, error, price = self.summarize_single(text, max_generation_length=max_generation_length)
            summaries.append(revised_summary)
            errors.append(error)
            prices.append(price)
            if error is not None:
                print(error)
                time.sleep(3)
        return summaries, errors, prices

    def summarize_single(self, text, max_generation_length):
        text_for_summarization = f"Text: \n {text}"
        summary, error, price = self.call_llm(text_for_summarization, max_length=max_generation_length)
        self.csv_writer.writerow([text, summary, error, price])
        return summary, error, price

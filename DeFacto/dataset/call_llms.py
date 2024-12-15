import google.generativeai as genai
from openai import AzureOpenAI, OpenAI
import openai
import os
import time
import anthropic
from ibm_watsonx_ai.foundation_models import ModelInference
import csv
from transformers import AutoTokenizer, pipeline


def create_backup(temp_save_dir, model):
    temp_save_dir = os.path.join(temp_save_dir, model)
    if not os.path.exists(temp_save_dir):
        os.makedirs(temp_save_dir)
    run_time = time.strftime("%Y%m%d-%H%M%S")
    run_name = 'run_' + run_time
    curr_run_dir = os.path.join(temp_save_dir, run_name)
    os.makedirs(curr_run_dir)
    path_logger = curr_run_dir + '/' + 'logger.txt'
    logger = open(path_logger, 'w')
    temp_save_file = curr_run_dir + '/' + 'temp_save.csv'
    backup = open(temp_save_file, 'w')
    backup = csv.writer(backup)
    return logger, backup


def calc_price(input_tokens, output_tokens, input_price, output_price):
    price = input_tokens * input_price + output_tokens * output_price
    return price / 10 ** 6


class GeminiCaller:
    def __init__(self, model, api_key, temp_save_dir, input_price=0, output_price=0):
        self.model = genai.GenerativeModel(model)
        genai.configure(api_key=api_key)
        self.api_rules_error = 0
        self.other_errors = 0
        self.logger, self.backup = create_backup(temp_save_dir, model)
        self.backup.writerow(["output", "error_type", "error_string", "price"])
        self.input_price = input_price
        self.output_price = output_price

    def call(self, gen_config, input):
        try:
            response = self.model.generate_content(input, generation_config=gen_config)
            if len(response.candidates) == 0:
                self.api_rules_error += 1
                self.logger.write(
                    f"Error occurred, the input was blocked because of {response.prompt_feedback.block_reason.name}")
                self.backup.writerow([None, "Blocked", response.prompt_feedback.block_reason.name, 0])
                return None, response.prompt_feedback.block_reason.name, 0
            price = calc_price(response.usage_metadata['prompt_token_count'],
                               response.usage_metadata['candidates_token_count'],
                               self.input_price, self.output_price)
            if response.candidates[0].finish_reason.name != 'STOP' and \
                    response.candidates[0].finish_reason.name != 'MAX_TOKENS':
                self.api_rules_error += 1
                self.logger.write(
                    f"Error occurred generation was stopped because of {response.candidates[0].finish_reason.name}")
                self.backup.writerow([None, "Stop", response.candidates[0].finish_reason.name, price])
                return None, response.candidates[0].finish_reason.name, price
            self.backup.writerow([response.candidates[0].content, None, None, price])
            return response.text, None, price
        except Exception as e:
            self.other_errors += 1
            self.logger.write(f"Error occurred: {e}")
            return None, f"{e}", 0


class OpenAICaller:
    def __init__(self, model, api_key, temp_save_dir, azure=False,
                 input_price=0, output_price=0):
        self.model = model
        print(f"model: {model}")
        if azure:
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version='2024-02-01',
                azure_endpoint='https://researchopenai2023eastus2.openai.azure.com/',
            )
        else:
            self.client = OpenAI(api_key=api_key)
        self.api_rules_error = 0
        self.other_errors = 0
        if temp_save_dir is not None:
            self.logger, self.backup = create_backup(temp_save_dir, model)
            self.backup.writerow(["output", "error", "price"])
        else:
            self.logger = None
            self.backup = None
        self.input_price = input_price
        self.output_price = output_price

    def call(self, input, timeout=60, max_new_tokens=1000, **kwargs):
        try:
            message = [{
                "role": "user",
                "content": input,
            }]

            response = self.client.chat.completions.create(model=self.model,
                                                           messages=message,
                                                           temperature=0,
                                                           max_tokens=max_new_tokens, timeout=timeout, **kwargs)

            price = calc_price(response.usage.prompt_tokens, response.usage.completion_tokens,
                               self.input_price, self.output_price)
            if self.backup is not None:
                self.backup.writerow([response.choices[0].message.content, None, price])
            return response.choices[0].message.content, None, price
        except openai.OpenAIError as e:
            print(f"Error occurred: {e}")
            if self.logger is not None:
                self.logger.write(f"Error occurred: {e}")
                self.backup.writerow([None, f"{e}", 0])
            self.api_rules_error += 1
            return None, f"{e}", 0
        except Exception as e:
            print(f"Error in output occurred: {e}")
            self.other_errors += 1
            if self.logger is not None:
                self.logger.write(f"Error occurred: {e}")
                self.backup.writerow([None, f"{e}", 0])
            return None, f"{e}", 0


class WatsonCaller:
    def __init__(self, model, iam_token, temp_save_dir, hf_token, project_id="705f3faa-4919-4c89-94e1-0087cf669b5c",
                 input_price=0, output_price=0):
        self.model = ModelInference(
            model_id=model,
            credentials={
                "apikey": iam_token,
                "url": "https://eu-de.ml.cloud.ibm.com"
            },
            project_id=project_id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model, token=hf_token)
        self.api_rules_error = 0
        self.other_errors = 0
        self.input_price = input_price
        self.output_price = output_price
        self.logger, self.backup = create_backup(temp_save_dir, model)
        self.backup.writerow(["output", "error", "price"])

    def call(self, input, max_new_tokens):
        # TODO: complete exceptions handling
        messages = [{"role": "user", "content": input}]
        message = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True, tokenize=False
        )

        output = self.model.generate(message, params={"max_new_tokens": max_new_tokens})
        results = output['results'][0]
        price = calc_price(results['input_token_count'], results['generated_token_count'], self.input_price,
                           self.output_price)
        self.backup.writerow([results['generated_text'], None, price])
        return results['generated_text'], None, price


class AnthropicCaller:
    def __init__(self, model, temp_save_dir, api_key, input_price=0, output_price=0):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.api_rules_error = 0
        self.other_errors = 0
        self.model = model
        self.logger, self.backup = create_backup(temp_save_dir, model)
        self.input_price = input_price
        self.output_price = output_price

    def call(self, input, max_new_tokens):
        # TODO: complete logger and exceptions handling
        message = {"role": "user", "content": [{
            "type": "text",
            "text": input
        }]}
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_new_tokens,
            temperature=0,
            messages=[message]
        )
        return message['content'], None, 0


class ModelCaller:
    def __init__(self, model_id, device_map, hf_token, torch_dtype):
        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map=device_map,
            token=hf_token
        )

    def call(self, input, max_new_tokens):
        messages = [
            {"role": "user", "content": input},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
        )
        output = outputs[0]['generated_text'][-1]['content']
        return output, None, 0

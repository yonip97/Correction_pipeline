import google.generativeai as genai
from google.api_core import retry
from google.generativeai.types import RequestOptions
from google.generativeai.types import GenerationConfig
from openai import AzureOpenAI, OpenAI
import openai
import os
import time
import anthropic
from ibm_watsonx_ai.foundation_models import ModelInference
import csv
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv("/data/home/yehonatan-pe/Correction_pipeline/credentials.env")


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
    def __init__(self, model, temp_save_dir, input_price=0, output_price=0):
        self.model = genai.GenerativeModel(model)
        api_key = os.getenv('GENAI_API_KEY')
        genai.configure(api_key=api_key)
        self.api_rules_error = 0
        self.other_errors = 0
        if temp_save_dir is not None:
            self.logger, self.backup = create_backup(temp_save_dir, model)
            self.backup.writerow(["input", "output", "error", "price"])
        else:
            self.logger = None
            self.backup = None
        self.input_price = input_price
        self.output_price = output_price

    def call(self, input, max_new_tokens,temperature = 0):
        gen_config = GenerationConfig(max_output_tokens=max_new_tokens, temperature=temperature)
        try:
            response = self.model.generate_content(input, generation_config=gen_config,
                                                   request_options=RequestOptions(
                                                       retry=retry.Retry(initial=10, multiplier=2, maximum=60,
                                                                         timeout=300)))
            if len(response.candidates) == 0:
                self.api_rules_error += 1
                if self.logger is not None:
                    self.logger.write(
                        f"Error occurred, the input was blocked because of {response.prompt_feedback.block_reason.name}")
                if self.backup is not None:
                    self.backup.writerow([None, "Blocked", response.prompt_feedback.block_reason.name, 0])
                return None, response.prompt_feedback.block_reason.name, 0
            price = calc_price(response.usage_metadata.prompt_token_count,
                               response.usage_metadata.candidates_token_count,
                               self.input_price, self.output_price)
            if response.candidates[0].finish_reason.name != 'STOP' and \
                    response.candidates[0].finish_reason.name != 'MAX_TOKENS':
                self.api_rules_error += 1
                if self.logger is not None:
                    self.logger.write(
                        f"Error occurred generation was stopped because of {response.candidates[0].finish_reason.name}")
                if self.backup is not None:
                    self.backup.writerow([None, "Stop", response.candidates[0].finish_reason.name, price])
                return None, response.candidates[0].finish_reason.name, price
            if self.backup is not None:
                self.backup.writerow([response.candidates[0].content, None, None, price])
            return response.text, None, price

        except Exception as e:
            self.other_errors += 1
            if self.logger is not None:
                self.logger.write(f"Error occurred: {e}")
            return None, f"{e}", 0


class OpenAICaller:
    def __init__(self, model, temp_save_dir, azure=False, input_price=0, output_price=0):
        self.model = model
        api_key = os.getenv('OPENAI_API_KEY')
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
            self.backup.writerow(["input", "output", "error", "price"])
        else:
            self.logger = None
            self.backup = None
        self.input_price = input_price
        self.output_price = output_price

    def call(self, input, timeout=60, max_new_tokens=2000,temperature = 0, **kwargs):
        max_tries = 8
        wait_time = 10
        for i in range(max_tries):
            try:
                message = [{
                    "role": "user",
                    "content": input,
                }]

                response = self.client.chat.completions.create(model=self.model,
                                                               messages=message,
                                                               temperature=temperature,
                                                               max_tokens=max_new_tokens, timeout=timeout, **kwargs)

                price = calc_price(response.usage.prompt_tokens, response.usage.completion_tokens,
                                   self.input_price, self.output_price)
                if self.backup is not None:
                    self.backup.writerow([input, response.choices[0].message.content, None, price])
                return response.choices[0].message.content, None, price
            except openai.RateLimitError:  # Catch the overuse error (429)
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)  # Wait before retrying
                wait_time *= 2
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

        return None, "max_tries_exceeded", 0


class LlamaApiCaller:
    def __init__(self, model, temp_save_dir, input_price=0.0, output_price=0.0):
        api_key = os.getenv('LLAMA_API_KEY')
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.llama-api.com"
        )
        self.model = model.lower()
        self.api_rules_error = 0
        self.other_errors = 0
        if temp_save_dir is not None:
            self.logger, self.backup = create_backup(temp_save_dir, model)
            self.backup.writerow(["input", "output", "error", "price"])
        else:
            self.logger = None
            self.backup = None
        self.input_price = input_price
        self.output_price = output_price

    def call(self, input, max_new_tokens=2000,temperature = 0):
        message = [{"role": "user", "content": input}]
        max_retries = 8
        wait_time = 5
        for i in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    max_tokens=max_new_tokens,
                    temperature=temperature
                )

                price = calc_price(response.usage.prompt_tokens, response.usage.completion_tokens,
                                   self.input_price, self.output_price)

                output_text = response.choices[0].message.content

                if self.backup is not None:
                    self.backup.writerow([input, output_text, None, price])

                return output_text, None, price

            except Exception as e:
                error_message = str(e)

                if "429" in error_message or "422" in error_message or 500 in error_message:  # Rate limit error
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                    wait_time *= 2
                    time.sleep(wait_time)
                else:
                    self.logger.write(f"API call failed: {error_message}")
                    if self.backup is not None:
                        self.backup.writerow([input, None, error_message, None])
                    return None, error_message, None

        # If max retries exceeded, log error and return failure
        self.logger.write("Max retries exceeded. API call failed.")
        return None, "Max retries exceeded", None



class AnthropicCaller:
    def __init__(self, model, temp_save_dir, input_price=0, output_price=0):
        api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=api_key)
        self.api_rules_error = 0
        self.other_errors = 0
        self.model = model
        if temp_save_dir is not None:
            self.logger, self.backup = create_backup(temp_save_dir, model)
            self.backup.writerow(["input", "output", "error", "price"])
        else:
            self.logger = None
            self.backup = None
        self.input_price = input_price
        self.output_price = output_price

    def call(self, input, max_new_tokens=2000,temperature = 0):
        max_tries = 8
        wait_time = 10
        for i in range(max_tries):
            try:
                message = {"role": "user", "content": input}
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    messages=[message]
                )
                price = calc_price(response.usage.input_tokens, response.usage.output_tokens,
                                   self.input_price, self.output_price)
                if self.backup is not None:
                    self.backup.writerow([input, response.content[0].text, None, price])
                return response.content[0].text, None, price
            except anthropic.RateLimitError as e:
                print(f"Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)  # Wait before retrying
                wait_time *= 2
            except Exception as e:
                print(f"Error in output occurred: {e}")
                self.other_errors += 1
                if self.logger is not None:
                    self.logger.write(f"Error occurred: {e}")
                    self.backup.writerow([None, f"{e}", 0])
                return None, f"{e}", 0


class ModelCallerPipeline:
    def __init__(self, model_id, device_map, torch_dtype):
        hf_token = os.getenv('HF_TOKEN')
        self.pipeline = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch_dtype},
            device_map=device_map,
            token=hf_token
        )

    def call(self, input, max_new_tokens):
        messages = [
            {"role": "user", "content": input[0]},
        ]
        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens)
        output = outputs[0]['generated_text'][-1]['content']
        return output, None, 0


class ModelCallerAutoModel:
    def __init__(self, model_id, device_map, torch_dtype,delimiter = None):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = model
        self.tokenizer = tokenizer
        self.delimiter = delimiter

    def split_on_last_occurrence(self,text: str):
        if not isinstance(text, str):
            return text, None
        idx = text.rfind(self.delimiter)
        if idx == -1:
            return text, ""
        return text[:idx + len(self.delimiter)], text[idx + len(self.delimiter):]

    def call(self, inputs, max_length):
        message_batch = [[{"role": "user", "content": inputs[i]}] for i in range(len(inputs))]
        text_batch = self.tokenizer.apply_chat_template(
            message_batch,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs_batch = self.tokenizer(text_batch, return_tensors="pt", padding=True).to("cuda")
        generated_ids_batch = self.model.generate(
            **model_inputs_batch,
            max_new_tokens=max_length,
        )
        response_batch = self.tokenizer.batch_decode(generated_ids_batch, skip_special_tokens=True)
        if self.delimiter is not None:
            response_batch = [self.split_on_last_occurrence(response)[1] for response in response_batch]
        return response_batch, [None] * len(response_batch), [0] * len(response_batch)

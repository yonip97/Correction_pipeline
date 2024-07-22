from openai import OpenAI, AzureOpenAI
import time
import openai
import tiktoken
import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
import string

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../../')




class ConversationAgentAPI():
    def __init__(self, API_KEY, instructions, revision_prompt, verification_prompt, verification_seperator,
                 verification_desired_outcome, retry_prompt, azure=False,
                 model='gpt-3.5-turbo', sampling_per_second=0.5, **kwargs):
        if azure:
            self.client = AzureOpenAI(
                api_key=API_KEY,
                api_version='2023-09-01-preview',
                azure_endpoint='https://researchopenai2023eastus2.openai.azure.com/')
        else:
            self.client = OpenAI(api_key=API_KEY)
        self.assistant = self.client.beta.assistants.create(
            name="Revision Assistant",
            instructions=instructions,
            model=model
        )
        self.assistant_id = self.assistant.id
        self.time_between_sampling = int(1 / sampling_per_second)
        self.revision_prompt = revision_prompt
        self.verification_prompt = verification_prompt
        self.verification_seperator = verification_seperator
        self.verification_desired_output = verification_desired_outcome
        self.retry_prompt = retry_prompt

    def start_conversation(self):
        assistant_thread = self.client.beta.threads.create()
        self.thread_id = assistant_thread.id

    def end_conversation(self):
        self.client.beta.threads.delete(thread_id=self.thread_id)
        self.thread_id = None

    def send_message(self, content, role="user"):
        message = self.client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role=role,
            content=content
        )
        return message

    def get_reply(self):
        run = self.client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
        )
        while run.status != "completed":
            run = self.client.beta.threads.runs.retrieve(
                thread_id=self.thread_id,
                run_id=run.id
            )
            time.sleep(self.time_between_sampling)
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        latest_message = messages.data[0]
        return latest_message.content[0].text.value

    def create_revision_text(self, text, summary):
        return self.revision_prompt + f"Document: \n {text} \n summary: \n {summary} \n"

    def create_verification_text(self):
        return self.verification_prompt

    def create_retry_text(self):
        return self.retry_prompt

    def revision_successful(self, verification_output):
        if self.verification_seperator is not None:
            verification_output = verification_output.split(self.verification_seperator)[-1].lower().strip()
        return verification_output == self.verification_desired_output

    def extract_conversation(self):
        messages = self.client.beta.threads.messages.list(thread_id=self.thread_id)
        conversation = []
        for message in reversed(messages.data):
            conversation.append(message.content[0].text.value)
        return conversation

    def revision_conversation(self, text, summary):
        self.start_conversation()
        revision_text = self.create_revision_text(text, summary)
        self.send_message(revision_text)
        verification_text = self.create_verification_text()
        self.send_message(verification_text)
        verification_output = self.get_reply()
        if self.revision_successful(verification_output):
            conversation = self.extract_conversation()
            self.end_conversation()
            return conversation
        else:
            retry_text = self.create_retry_text()
            self.send_message(retry_text)
            self.get_reply()
            conversation = self.extract_conversation()
            self.end_conversation()
            return conversation


class ConversationConcatenation():
    def __init__(self, azure, model, API_KEY, revision_prompt, verification_prompt, verification_seperator,
                 verification_desired_output, retry_prompt,
                 revision_generation_length=None, verification_generation_length=None, retry_generation_length=None,
                 user_role="user", system_role="system"):
        self.model = model
        self.azure = azure
        self.API_KEY = API_KEY
        if azure:
            self.client = AzureOpenAI(
                api_key=API_KEY,
                api_version='2023-09-01-preview',
                azure_endpoint='https://researchopenai2023eastus2.openai.azure.com/')
        else:
            self.client = OpenAI(api_key=API_KEY)
        self.revision_prompt = revision_prompt
        if model == 'gpt-3.5-turbo':
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        elif model == 'gpt-4':
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-4')
        elif model == 'gpt-4-turbo':
            self.estimation_tokenizer = tiktoken.encoding_for_model('gpt-4')
        else:
            raise ValueError(f"model {model} not supported")
        self.revision_prompt = revision_prompt
        self.verification_prompt = verification_prompt
        self.verification_seperator = verification_seperator
        self.verification_desired_output = verification_desired_output.lower().strip()
        self.retry_prompt = retry_prompt
        self.revision_generation_length = revision_generation_length
        self.verification_generation_length = verification_generation_length
        self.retry_generation_length = retry_generation_length
        self.usage = {'prompt_tokens': 0, "completion_tokens": 0, "total_tokens": 0}
        self.user_role = user_role
        self.system_role = system_role

    def call_llm(self, message, max_length):
        try:
            # message = [{
            #     "role": role,
            #     "content": content,
            # }]
            if self.model in ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']:
                if self.model == 'gpt-3.5-turbo':
                    model = 'gpt-35-turbo'
                else:
                    model = self.model
                response = self.client.chat.completions.create(model=model,
                                                               messages=message,
                                                               temperature=0,
                                                               max_tokens=max_length, timeout=60)
                self.usage['prompt_tokens'] += response.usage.prompt_tokens
                self.usage['completion_tokens'] += response.usage.completion_tokens
                self.usage['total_tokens'] += response.usage.total_tokens
                return response.choices[0].message.content, None
            else:
                raise ValueError(f"model {self.model} not supported")
        except openai.OpenAIError as e:
            print(f"Error occurred: {e}")
            return None, f"{e}"
        except Exception as e:
            print(f"Error in output occurred: {e}")
            return None, f"{e}"

    # def concat_conversation(self, conversation):
    #     return "\n".join(conversation)
    def create_message(self, conversation):
        messages = []
        for i in range(len(conversation)):
            message = {"content": conversation[i]}
            if i % 2 == 0:
                message["role"] = self.user_role
            else:
                message["role"] = self.system_role
            messages.append(message)
        return messages

    def revision_successful(self, conversation):
        verification_output = conversation[-1]
        if self.verification_seperator is not None:
            verification_output = verification_output.split(self.verification_seperator)[-1].lower().strip()
            verification_output = verification_output.translate(str.maketrans('', '', string.punctuation))
        return verification_output == self.verification_desired_output

    def create_revision_text(self, text, summary):
        return self.revision_prompt + '\n' + f"Document: \n {text} \n summary: \n {summary} \n"

    def create_verification_text(self):
        return self.verification_prompt

    def create_retry_text(self):
        return self.retry_prompt

    def revise(self, text, summary, conversation):
        revision_text = self.create_revision_text(text, summary)
        conversation.append(revision_text)
        messages = self.create_message(conversation)
        revised_summary, error = self.call_llm(message=messages, max_length=self.revision_generation_length)
        conversation.append(revised_summary)
        return revision_text, revised_summary, error, conversation

    def verify(self, conversation):
        verification_text = self.create_verification_text()
        conversation.append(verification_text)
        messages = self.create_message(conversation)
        verification_output, error = self.call_llm(message=messages, max_length=self.verification_generation_length)
        conversation.append(verification_output)
        return verification_output, error, conversation

    def retry(self, conversation):
        retry_text = self.create_retry_text()
        conversation.append(retry_text)
        # conversation_text = self.concat_conversation(conversation)
        messages = self.create_message(conversation)
        retry_output, error = self.call_llm(message=messages, max_length=self.retry_generation_length)
        conversation.append(retry_output)
        return retry_output, error, conversation

    def revision_conversation(self, text, summary):
        conversation = []
        revision_text, revised_summary, error, conversation = self.revise(text, summary, conversation)
        if error is not None:
            return conversation
        verification_output, error, conversation = self.verify(conversation)
        if error is not None or self.revision_successful(conversation):
            return {'revised_summary': revised_summary, 'verification_output': verification_output,
                    'retry_output': None}
        else:
            retry_output, error, conversation = self.retry(conversation)
            return {'revised_summary': revised_summary, 'verification_output': verification_output,
                    'retry_output': retry_output}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision_prompt_path', type=str, default=None)
    parser.add_argument('--verification_prompt_path', type=str, default=None)
    parser.add_argument('--retry_prompt_path', type=str, default=None)
    parser.add_argument('--verification_seperator', type=str, default=None)
    parser.add_argument('--verification_desired_output', type=str, default=None)
    parser.add_argument('--revision_generation_length', type=int, default=1000)
    parser.add_argument('--verification_generation_length', type=int, default=1000)
    parser.add_argument('--retry_generation_length', type=int, default=1000)
    parser.add_argument('--revision_seperator', type=str, default=None)
    parser.add_argument('--retry_seperator', type=str, default=None)
    parser.add_argument('--azure', action='store_true')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--API_key', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    with open(args.revision_prompt_path, 'r') as file:
        text = file.readlines()
        args.revision_prompt = "".join(text).strip()
    with open(args.verification_prompt_path, 'r') as file:
        text = file.readlines()
        args.verification_prompt = "".join(text).strip()
    with open(args.retry_prompt_path, 'r') as file:
        text = file.readlines()
        args.retry_prompt = "".join(text).strip()
    return args


def full_conversations():
    args = parse_args()
    df = pd.read_csv("experiments/revision/data/base_model_50000_documents/base_model_outputs_below_0.5_30.csv",
                     index_col=0)
    df['revised_summary_full_text'] = None
    df['revised_summary'] = None
    df['verification_output_full_text'] = None
    df['verification_output'] = None
    df['retry_output_full_text'] = None
    df['retry_output'] = None
    conversation_model = ConversationConcatenation(azure=args.azure, model=args.model, API_KEY=args.API_key,
                                                   revision_prompt=args.revision_prompt,
                                                   verification_prompt=args.verification_prompt,
                                                   verification_seperator=args.verification_seperator,
                                                   verification_desired_output=args.verification_desired_output,
                                                   retry_prompt=args.retry_prompt,
                                                   revision_generation_length=args.revision_generation_length,
                                                   verification_generation_length=args.verification_generation_length,
                                                   retry_generation_length=args.retry_generation_length)
    for i in tqdm(range(len(df))):
        text = df.iloc[i]['text']
        model_summary = df.iloc[i]['model_summary']
        conversation_outputs = conversation_model.revision_conversation(text, model_summary)
        df.at[i, 'revised_summary_full_text'] = conversation_outputs['revised_summary']
        if args.revision_seperator is not None:
            df.at[i, 'revised_summary'] = conversation_outputs['revised_summary'].split(args.revision_seperator)[
                -1].strip()
        else:
            df.at[i, 'revised_summary'] = conversation_outputs['revised_summary']
        df.at[i, 'verification_output_full_text'] = conversation_outputs['verification_output']
        if args.verification_seperator is not None:
            df.at[i, 'verification_output'] = \
                conversation_outputs['verification_output'].split(args.verification_seperator)[
                    -1].strip().translate(str.maketrans('', '', string.punctuation))
        else:
            df.at[i, 'verification_output'] = conversation_outputs['verification_output'].strip().translate(
                str.maketrans('', '', string.punctuation))
        if conversation_outputs['retry_output'] is not None:
            df.at[i, 'retry_output_full_text'] = conversation_outputs['retry_output']
            if args.retry_seperator is not None:
                df.at[i, 'retry_output'] = conversation_outputs['retry_output'].split(args.retry_seperator)[-1].strip()
            else:
                df.at[i, 'retry_output'] = conversation_outputs['retry_output']
    df.to_csv(args.output_path + '.csv')


def check_classification():
    args = parse_args()
    df = pd.read_csv(
        "experiments/revision/data/base_model_50000_documents/cot_prompts/original/base_model_outputs_below_0.5_30_revised.csv",
        index_col=0)
    df['verification_output_full_text'] = None
    df['verification_output'] = None
    conversation_model = ConversationConcatenation(azure=args.azure, model=args.model, API_KEY=args.API_key,
                                                   revision_prompt=args.revision_prompt,
                                                   verification_prompt=args.verification_prompt,
                                                   verification_seperator=args.verification_seperator,
                                                   verification_desired_output=args.verification_desired_output,
                                                   retry_prompt=args.retry_prompt,
                                                   revision_generation_length=args.revision_generation_length,
                                                   verification_generation_length=args.verification_generation_length,
                                                   retry_generation_length=args.retry_generation_length)
    for i in tqdm(range(len(df))):
        text = df.iloc[i]['text']
        model_summary = df.iloc[i]['model_summary']
        revised_summary = df.iloc[i]['revised_summary_full_text']
        conversation = [conversation_model.create_revision_text(text, model_summary), revised_summary]
        verification_output, error, conversation = conversation_model.verify(conversation)
        df.at[i, 'verification_output_full_text'] = verification_output
        if args.verification_seperator is not None:
            df.at[i, 'verification_output'] = verification_output.split(args.verification_seperator)[
                -1].strip().translate(str.maketrans('', '', string.punctuation))
        else:
            df.at[i, 'verification_output'] = verification_output.strip().translate(
                str.maketrans('', '', string.punctuation))
    df.to_csv(args.output_path + '.csv')


def main():
    check_classification()


if __name__ == "__main__":
    main()
    #
    # assistant = client.beta.assistants.create(
    #     name="My First Assistant",
    #     instructions="You are a helpful assistant",
    #     tools=[{"type": "code_interpreter"}],
    #     model="gpt-3.5-turbo-1106"
    # )
    # #thread = client.beta.threads.create()
    # message = client.beta.threads.messages.create(
    #     thread_id=thread.id,
    #     role="user",
    #     content="Can you give me a 300 words summary about coffee?"
    # )
    # run = client.beta.threads.runs.create(
    #     thread_id=thread.id,
    #     assistant_id=assistant.id,
    # )
    # import time
    #
    # while run.status != "completed":
    #     run = client.beta.threads.runs.retrieve(
    #         thread_id=thread.id,
    #         run_id=run.id
    #     )
    #     print("Run status:", run.status)
    #     time.sleep(3)
    # print(run.usage)
    #
    # messages = client.beta.threads.messages.list(thread_id=thread.id)
    # for message in reversed(messages.data):
    #     print(message.content[0].text.value)
    #
    # message = client.beta.threads.messages.create(
    #     thread_id=thread.id,
    #     role="user",
    #     content="Can you add some additional information about the taste?"
    # )
    # run = client.beta.threads.runs.create(
    #     thread_id=thread.id,
    #     assistant_id=assistant.id,
    # )
    # while run.status != "completed":
    #     run = client.beta.threads.runs.retrieve(
    #         thread_id=thread.id,
    #         run_id=run.id
    #     )
    #     print("Run status:", run.status)
    #     time.sleep(3)
    # print(run.usage)
    # messages = client.beta.threads.messages.list(thread_id=thread.id)
    # for message in reversed(messages.data):
    #     print(message.content[0].text.value)
    #
    # message = client.beta.threads.messages.create(
    #     thread_id=thread.id,
    #     role="user",
    #     content="And how does it compare to tea?"
    # )
    # run = client.beta.threads.runs.create(
    #     thread_id=thread.id,
    #     assistant_id=assistant.id,
    # )
    # while run.status != "completed":
    #     run = client.beta.threads.runs.retrieve(
    #         thread_id=thread.id,
    #         run_id=run.id
    #     )
    #     print("Run status:", run.status)
    #     time.sleep(3)
    # print(run.usage)
    #
    # messages = client.beta.threads.messages.list(thread_id=thread.id)
    # for message in reversed(messages.data):
    #     print(message.content[0].text.value)

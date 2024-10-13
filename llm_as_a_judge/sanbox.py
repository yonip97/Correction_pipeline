#
# from datasets import load_dataset
# import evaluate
# import matplotlib.pyplot as plt
# dataset = load_dataset("JeremyAlain/SLF5K")
# train_dataset = dataset['train']
# counter = 0
# x = []
# y = []
# classes = []
# for sample in train_dataset:
#     classes.append(sample['feedback_class'])
#     if sample['feedback_class'] =='Accuracy':
#         x.append(sample['generated_summary_for_feedback'])
#         print(sample['generated_summary_for_feedback'])
#         y.append(sample['ideal_human_summary'])
#         print(sample['ideal_human_summary'])
#         counter += 1
#         print("---------------------------------------------------------------------------------------")
# rouge_metric = evaluate.load('rouge')
# scores = rouge_metric.compute(predictions=x, references=y, use_aggregator=False)
# print(scores)
# plt.hist(scores['rougeL'], bins=20)
# plt.show()
# print(counter)
# print(set(classes))
#
#
import pandas as pd
import google.generativeai as genai

df = pd.read_csv(
    "/data/home/yehonatan-pe/Correction_pipeline/llm_as_a_judge/data/mistakes_detection/prompt2/results_llama_3.1_70B_instruct.csv",
    index_col=0)
outputs = df['output'].tolist()
print(len(outputs))
from transformers import AutoTokenizer

access_token = "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"
model_id = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
outputs = [x.split('assistant\n')[1] for x in outputs]
lengths = [len(tokenizer(x)['input_ids']) for x in outputs]
with open("/data/home/yehonatan-pe/Correction_pipeline/llm_as_a_judge/data/mistakes_detection/prompt2/prompt.txt",
          'r') as file:
    prompt = file.read()
inputs = ['\n' + 'Document: \n' + docuemnt + 'Summary:\n' + summary for docuemnt, summary in
          zip(df['text'].tolist(), df['model_summary'].tolist())]
inputs_lengths = [len(tokenizer(x)['input_ids']) for x in inputs]
example_prompt = prompt.split(':', 1)[1]
prompt = prompt.split(':', 1)[0]
info = example_prompt.split('Fact', 1)[0]
output = example_prompt.split('Fact', 1)[1]
print(len(tokenizer(prompt)['input_ids']))
print(len(tokenizer(example_prompt)['input_ids']))
print(len(tokenizer(info)['input_ids']))
print(len(tokenizer(output)['input_ids']))
print(sum(lengths))
print(sum(inputs_lengths))
df = pd.read_csv("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/initial_data_for_annotation (2).csv", index_col=0)
x = df['explanation human annotation'].dropna()
print(len(x))
n = [len(tokenizer(str(y))['input_ids']) for y in x]
print(sum(n))
import tiktoken

encoder = tiktoken.encoding_for_model('gpt-4')
print(len(encoder.encode(prompt)))
print(len(encoder.encode(example_prompt)))
print(len(encoder.encode(info)))
print(len(encoder.encode(output)))
inputs_lengths = [len(encoder.encode(x)) for x in inputs]
print(sum(inputs_lengths))
lengths = [len(encoder.encode(x)) for x in outputs]
print(sum(lengths))
# API_KEY = "AIzaSyBDLWJowPzGyV3mwE40qb9xGmJvISc7lPE"
# genai.configure(api_key=API_KEY)
#
# model = genai.GenerativeModel("models/gemini-1.5-pro")
# print(model.count_tokens(prompt))
# print(model.count_tokens(example_prompt))
# print(model.count_tokens(info))
# print(model.count_tokens(output))
# from tqdm import tqdm
#
# inputs_lengths = [int(str(model.count_tokens(x)).split(':')[1].strip()) for x in tqdm(inputs)]
# print(sum(inputs_lengths))
# lengths = [int(str(model.count_tokens(x)).split(':')[1].strip()) for x in outputs]
# print(sum(lengths))
print(prompt)
print(example_prompt)
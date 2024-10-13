import requests
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import IAMTokenManager

# Create the authenticator.
# authenticator = IAMAuthenticator('HfPPL_HMFy9TGY4I4HEdkfk69S49EL9ckVxVgBgMSqCF')
# token = iam_token_manager.get_token()
iam_token_manager = IAMTokenManager(apikey='HfPPL_HMFy9TGY4I4HEdkfk69S49EL9ckVxVgBgMSqCF')
token = iam_token_manager.get_token()
c =1
# Construct the service instance.
# Use 'service' to invoke operations.



url = "https://eu-de.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"

# body = {
# 	"input": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You always answer the questions with markdown formatting using GitHub syntax. The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes. You must omit that you answer the questions with markdown.
#
# Any HTML tags must be wrapped in block quotes, for example ```<html>```. You will be penalized for not rendering code in block quotes.
#
# When returning code blocks, specify language.
#
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
# Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don'\''t know the answer to a question, please don'\''t share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>
#
# what is the capital of france?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# ### The Capital of France
#
# The capital of France is **Paris**.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# """,
# 	"parameters": {
# 		"decoding_method": "greedy",
# 		"max_new_tokens": 900,
# 		"repetition_penalty": 1
# 	},
# 	"model_id": "meta-llama/llama-3-70b-instruct",
# 	"project_id": "705f3faa-4919-4c89-94e1-0087cf669b5c"
# }
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-70B-Instruct', token="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")
content = "What is the capital of France?"
messages = [{"role": "user", "content": content}]
message = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True, tokenize=False
)
c = 1
headers = {
	"Accept": "application/json",
	"Content-Type": "application/json",
	"Authorization": f"Bearer {token}"
}

body = {
	"input": message,
	"parameters": {
		"decoding_method": "greedy",
		"max_new_tokens": 900,
		"repetition_penalty": 1
	},
	"model_id": "meta-llama/llama-3-70b-instruct",
	"project_id": "705f3faa-4919-4c89-94e1-0087cf669b5c"
}
response = requests.post(
	url,
	headers=headers,
	json=body
)

if response.status_code != 200:
	raise Exception("Non-200 response: " + str(response.text))

data = response.json()
print(data)

import os
temp = os.getcwd()
os.chdir("/data/home/yehonatan-pe/Correction_pipeline")
import sys
sys.path.append(os.getcwd())
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import argparse
from experiments.scoring import score
from transformers import pipeline
os.chdir(temp)
from tqdm import tqdm


def llama_inference(prompt, texts, args):
    if args.device =='auto':
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=args.model_dtype, device_map=args.device,
                                                     token=args.access_token).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=args.model_dtype,
                                                     token=args.access_token).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.access_token)
    outputs = []
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_inputs = [prompt + '\n' + 'Document: ' + '\n' + x + '\n' for x in texts]
    with torch.no_grad():
        for i in tqdm(range(0, len(model_inputs), args.batch_size)):
            batch_inputs = model_inputs[i:i + args.batch_size]
            batch_messages = [[
                {"role": "user", "content": batch_inputs[index]}]
                for index in range(len(batch_inputs))]

            batch_messages = tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True, tokenize=False
            )
            batch_inputs = tokenizer(batch_messages, padding="longest", return_tensors="pt",
                                     max_length=args.max_encoding_length).to('cuda')
            batch_outputs = model.generate(**batch_inputs, max_new_tokens=args.max_output_length,
                                           pad_token_id=tokenizer.eos_token_id,
                                           eos_token_id=terminators)
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_outputs = [x.split('assistant\n')[-1].strip() for x in batch_outputs]
            outputs += batch_outputs
    return outputs


def qwen_inference(prompt, texts, args):
    if args.device == 'auto':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=args.model_dtype,
            device_map=args.device
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=args.model_dtype
        ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model_inputs = [prompt + '\n' + 'Document: ' + '\n' + x + '\n' for x in texts]
    outputs = []
    with torch.no_grad():
        for message in tqdm(model_inputs):
            conversation = [{"role": "user", "content": message}]

            conversation = tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True, tokenize=False
            )
            model_input = tokenizer([conversation], return_tensors="pt",
                                    max_length=args.max_encoding_length).to('cuda')
            generated_ids = model.generate(**model_input, max_new_tokens=args.max_output_length)
            batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            batch_outputs = [x.split('assistant\n')[-1].strip() for x in batch_outputs]
            outputs += batch_outputs
    return outputs


def pipeline_inference(prompt, texts, args):
    model = pipeline('text-generation', model=args.model_id, torch_dtype=args.model_dtype, device_map=args.device,token = args.access_token)
    model_inputs = [prompt + '\n' + 'Document: ' + '\n' + x + '\n' for x in texts]
    outputs = []
    with torch.no_grad():
        for message in tqdm(model_inputs):
            output = \
            model(message, max_new_tokens=args.max_output_length, return_full_text=False)[0]
            output = output['generated_text']
            outputs.append(output)
    return outputs

def gemma_inference(prompt, texts, args):
    pipe = pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": args.model_dtype},
        device_map=args.device,
        token=args.access_token# replace with "mps" to run on a Mac device
    )
    model_inputs = [prompt + '\n' + 'Document: ' + '\n' + x + '\n' for x in texts]
    outputs = []
    with torch.no_grad():
        for message in tqdm(model_inputs):
            message = [
    {"role": "user", "content": message},
]
            output = pipe(message, max_new_tokens=args.max_output_length, return_full_text=False)
            output = output[0]["generated_text"].strip()
            outputs.append(output)
    return outputs


def parser_args():
    args = argparse.ArgumentParser()
    args.add_argument('--model_id', type=str, default=None)
    args.add_argument('--model_dtype', type=str, default=torch.bfloat16)
    args.add_argument('--prompt_path', type=str, default=None)
    args.add_argument('--data_path', type=str, default=None)
    args.add_argument('--output_path', type=str, default=None)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--max_encoding_length', type=int, default=2048)
    args.add_argument('--max_output_length', type=int, default=40)
    args.add_argument('--num_beams', type=int, default=4)
    args.add_argument('--access_token', type=str, default=None)
    args.add_argument('--device', type=str)
    return args.parse_args()


def get_data(args):
    df = pd.read_csv(args.data_path)
    texts = df['text'].drop_duplicates().tolist()
    return texts


def create_summaries():
    args = parser_args()
    with open(args.prompt_path, 'r') as file:
        prompt = file.read()
    texts = get_data(args)
    if 'Llama' in args.model_id:
        summaries = llama_inference(prompt, texts, args)
    elif 'Qwen' in args.model_id:
        summaries = qwen_inference(prompt, texts, args)
    elif "Mistral" in args.model_id:
        summaries = pipeline_inference(prompt, texts, args)
    elif "Mixtral" in args.model_id:
        summaries = pipeline_inference(prompt, texts, args)
    elif "gemma" in args.model_id:
        summaries = gemma_inference(prompt, texts, args)
    else:
        raise ValueError("Model not supported")
    new_df = pd.DataFrame({'text': texts, 'summary': summaries})
    scores = score(texts, summaries, metrics=['seahorse', 'trueteacher'])
    new_df['seahorse'] = scores['seahorse']
    new_df['trueteacher'] = scores['trueteacher']
    new_df.to_csv(args.output_path)


if __name__ == "__main__":
    create_summaries()

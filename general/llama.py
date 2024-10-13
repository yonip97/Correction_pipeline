import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLama():
    def __init__(self, model_id: str, device: str, dtype=torch.bfloat16):
        access_token = "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"

        if device == 'auto':
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', token=access_token,
                                                         torch_dtype=dtype).eval()
            self.device = 'cuda'
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token,
                                                         torch_dtype=dtype).to(device).eval()
            self.device = device
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.model = model
        self.tokenizer = tokenizer
        self.terminators = terminators

    def call(self, inputs,tokenizer_kwargs, generation_kwargs,score=False,role='user'):
        try:
            with torch.no_grad():
                messages = [[
                    {"role": role, "content": inputs[index]},
                ] for index in range(len(inputs))]
                messages = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True, tokenize=False
                )
                tokenized_inputs = self.tokenizer(messages,return_tensors="pt",**tokenizer_kwargs).to(
                    self.device)
                if score:
                    outputs = self.model.generate(**tokenized_inputs,
                                                  pad_token_id=self.tokenizer.eos_token_id,
                                                  eos_token_id=self.terminators, **generation_kwargs,return_dict_in_generate=True,output_scores=True,output_logits=True)
                    outputs= outputs.logits[1].detach().cpu().numpy()
                    from scipy.special import softmax
                    outputs = softmax(outputs, axis=1)
                    x = self.tokenizer('Yes')[0]
                    return outputs
                else:
                    outputs = self.model.generate(**tokenized_inputs,
                                                  pad_token_id=self.tokenizer.eos_token_id,
                                                  eos_token_id=self.terminators, **generation_kwargs)
                return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        except Exception as e:
            print(f"Error occurred: {e}")
            return ["Error"] * len(inputs)

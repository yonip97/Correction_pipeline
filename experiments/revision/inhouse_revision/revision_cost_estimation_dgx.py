


from transformers import T5Tokenizer, LlamaTokenizer,AutoTokenizer

def x():
    tokenizer2 = T5Tokenizer.from_pretrained('t5-base')
    tokenizer1 = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-70B',token = "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")
    c = 1
x()
import torch

MODEL_PRICE_MAP = {'gpt-4o': {'input': 5, 'output': 15},
                       'claude-3-5-sonnet-20241022': {'input': 3, 'output': 15},
                       'gemini-1.5-pro': {'input': 1.5, 'output': 5},
                       "llama-3-1-70b-instruct": {'input': 1.8, 'output': 1.8},
                       'llama3.1-405b': {'input': 3.6, 'output': 3.6}}

DTYPE_MAP = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}

import torch
import os

def collate_fn(batch, tokenizer, max_length):
    hypotheses = [f"hypothesis: {sample_dict['hypothesis']}" for sample_dict in batch]
    premises = [f"premise: {sample_dict['premise']}" for sample_dict in batch]
    encoding = tokenizer(text=premises, text_pair=hypotheses, max_length=max_length, truncation='only_first',
                         return_tensors='pt', padding=True)
    labels = [tokenizer(sample_dict['label'])['input_ids'][0] if sample_dict['label'] == '1' else
              tokenizer(sample_dict['label'])['input_ids'][1] for sample_dict in batch]
    return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'],
            'labels': labels}

def find_last_checkpoint(output_dir):
    dirlist = [item for item in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, item))]
    dirlist.remove('runs')
    if len(dirlist) == 0:
        return None
    else:
        last_checkpoint = max([int(item.split('-')[-1]) for item in dirlist])
        last_checkpoint_path = output_dir + '/' + f'checkpoint-{last_checkpoint}'
        return last_checkpoint_path
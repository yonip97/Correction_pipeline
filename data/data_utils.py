# def true_dataset_collate_fn(batch):
#     datasets = [x[0] for x in batch]
#     original_texts = [x[1] for x in batch]
#     generated_text = [x[2] for x in batch]
#     labels = [x[3] for x in batch]
#     return datasets, original_texts, generated_text, labels

def collate_fn(batch):
    datasets = [x['dataset'] for x in batch]
    premises = [sample_dict['premise'] for sample_dict in batch]
    hypotheses = [sample_dict['hypothesis'] for sample_dict in batch]
    labels = [int(sample_dict['label']) for sample_dict in batch]
    return datasets, premises, hypotheses, labels

def tokeinized_collate_fn(batch, tokenizer, max_length):
    hypotheses = [f"hypothesis: {sample_dict['hypothesis']}" for sample_dict in batch]
    premises = [f"premise: {sample_dict['premise']}" for sample_dict in batch]
    encoding = tokenizer(text=premises, text_pair=hypotheses, max_length=max_length, truncation='only_first',
                         return_tensors='pt', padding=True)
    labels = [tokenizer(sample_dict['label'])['input_ids'][0] if sample_dict['label'] == '1' else
              tokenizer(sample_dict['label'])['input_ids'][1] for sample_dict in batch]
    return {'input_ids': encoding['input_ids'], 'attention_mask': encoding['attention_mask'],
            'labels': labels}
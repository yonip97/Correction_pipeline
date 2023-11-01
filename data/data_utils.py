# def true_dataset_collate_fn(batch):
#     datasets = [x[0] for x in batch]
#     original_texts = [x[1] for x in batch]
#     generated_text = [x[2] for x in batch]
#     labels = [x[3] for x in batch]
#     return datasets, original_texts, generated_text, labels

def evaluation_collate_fn(batch):
    datasets = [x['dataset'] for x in batch]
    premises = [sample_dict['premise'] for sample_dict in batch]
    hypotheses = [sample_dict['hypothesis'] for sample_dict in batch]
    labels = [int(sample_dict['label']) for sample_dict in batch]
    return datasets, premises, hypotheses, labels

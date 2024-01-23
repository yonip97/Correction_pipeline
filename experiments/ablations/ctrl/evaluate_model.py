from datasets import load_dataset
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from experiments.ablations.ctrl.utils import SummariesscoredDataset
from torch.utils.data import DataLoader as Dataloader
import pandas as pd
from experiments.scoring import score



def get_test_data():
    xsum_test_set = load_dataset('xsum', split='test')
    texts = [xsum_test_set[i]['document'] for i in range(len(xsum_test_set))]
    summaries = [xsum_test_set[i]['summary'] for i in range(len(xsum_test_set))]
    predictions = [1 for i in range(len(xsum_test_set))]
    test_dataset = SummariesscoredDataset(texts, summaries, predictions)
    return test_dataset


def parserargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', type=str)
    parser.add_argument('--model_checkpoint', type=str)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_length_eval', type=int, default=512)
    parser.add_argument('--generation_max_length', type=int, default=128)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    return args


def collate_fn(batch, tokenizer, max_length):
    texts = [row['text'] for row in batch]
    summaries = [row['summary'] for row in batch]
    predictions = [row['prediction'] for row in batch]
    inputs = tokenizer.encode_plus(
        ["consistent: " if prediction == 1 else "inconsistent: " for prediction in predictions],
        ["summarize: " + text for text in texts], padding=True, truncation='second_only', max_length=max_length,
        return_tensors='pt')
    labels = tokenizer(summaries, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def evaluate():
    args = parserargs()
    metrics = args.metrics.split(',')
    test_dataset = get_test_data()
    device = args.device
    if device == 'auto':
        model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint, device_map='auto')
        device = 'cuda'
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint).to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    dataloader = Dataloader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda x: collate_fn(x, tokenizer, args.max_length_eval), pin_memory=False)
    predictions = []
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_predictions = model.generate(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device),
                                           max_length=args.generation_max_length,
                                           num_beams=args.num_beams, early_stopping=True)
        batch_predictions = tokenizer.batch_decode(batch_predictions, skip_special_tokens=True)
        predictions += batch_predictions
    results = score(metrics=metrics, texts=test_dataset.texts, summaries=predictions)
    results['model_summary'] = predictions
    results['text'] = test_dataset.texts
    results['original_summary'] = test_dataset.summaries

    df = pd.DataFrame.from_records(results)
    if args.save:
        df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    evaluate()

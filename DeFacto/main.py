import os

os.chdir("/data/home/yehonatan-pe/Correction_pipeline")
import sys

sys.path.append(os.getcwd())
from general.t5_trainer import T5_Trainer
import pandas as pd
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainingArguments
from general.utils import find_largest_numbered_dir
from experiments.scoring import score
from general.fragments_metrics import Fragments

def main():
    with open("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/test.jsonl") as f:
        data = [json.loads(line) for line in f]
    for i in range(100):
        print(data[i]['abstract'])
        print(data[i]['candidate'])
        print(data[i]['feedback'].keys())
        print(data[i].keys())
        print("--------------------------------------------------")


class EditDataset(Dataset):
    def __init__(self, texts, summaries, instructions, explanations, revised_summaries):
        self.texts = texts
        self.summaries = summaries
        self.instructions = instructions
        self.explanations = explanations
        self.revised_summaries = revised_summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'summary': self.summaries[idx], 'instruction': self.instructions[idx],
                'explanation': self.explanations[idx], 'revised_summary': self.revised_summaries[idx]}


def collate_fn_revision(batch, tokenizer, max_length):
    texts = [item['text'] for item in batch]
    summaries = [item['summary'] for item in batch]
    instructions = [item['instruction'] for item in batch]
    explanations = [item['explanation'] for item in batch]
    inputs = []
    for i in range(len(batch)):
        input_text = ("Article: " + texts[i], "Candidate: " + summaries[i] + "Instruction: " + instructions[i] +
                      "Explanation: " + explanations[i])
        inputs.append(input_text)
    inputs = tokenizer.batch_encode_plus(inputs, max_length=max_length, padding="longest", truncation="only_first",
                                         return_tensors="pt")
    print(inputs['input_ids'].shape)
    labels = tokenizer([item['revised_summary'] for item in batch], max_length=max_length, padding="longest",
                       truncation=True, return_tensors="pt")
    labels[labels == tokenizer.pad_token_id] = -100
    return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
            'labels': labels['input_ids']}


def score_all_datasets():
    for name in ['train', 'val', 'test']:
        original_summaries = []
        revised_summaries = []
        texts = []
        errors = []
        with open(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}.jsonl") as f:
            data = [json.loads(line) for line in f]
            for x in data:
                errors.append(x['']['has_error'])
                if x['has_error'] == True:
                    texts.append(str(x['article']))
                    revised_summaries.append(str(x['feedback']['summary']))
                    original_summaries.append(str(x['candidate']))
                else:
                    texts.append(str(x['article']))
                    revised_summaries.append(str(x['candidate']))
                    original_summaries.append(str(x['candidate']))
            df = pd.DataFrame(
                {'text': texts, 'model_summary': original_summaries, 'revised_summary': revised_summaries,
                 'error_in_model_summary': errors})
            from general.fragments_metrics import Fragments
            fragments = Fragments()
            original_summaries_metrics = fragments.score(metrics=['density', 'coverage'], texts=texts,
                                                         summaries=original_summaries)
            df['model_summary_density'] = original_summaries_metrics['density']
            df['model_summary_coverage'] = original_summaries_metrics['coverage']
            revised_summaries_metrics = fragments.score(metrics=['density', 'coverage'], texts=texts,
                                                        summaries=revised_summaries)
            df['revised_summary_density'] = revised_summaries_metrics['density']
            df['revised_summary_coverage'] = revised_summaries_metrics['coverage']
            revised_summaries_metrics = score(texts=texts, summaries=revised_summaries,
                                              metrics=['seahorse', 'trueteacher'])
            df['revised_summary_seahorse'] = revised_summaries_metrics['seahorse']
            df['revised_summary_trueteacher'] = revised_summaries_metrics['trueteacher']
            original_summaries_metrics = score(texts=texts, summaries=original_summaries,
                                               metrics=['seahorse', 'trueteacher'])
            df['model_summary_seahorse'] = original_summaries_metrics['seahorse']
            df['model_summary_trueteacher'] = original_summaries_metrics['trueteacher']
            import evaluate
            rouge_metric = evaluate.load('rouge')
            scores = rouge_metric.compute(predictions=revised_summaries, references=original_summaries,
                                          use_aggregator=False)
            df['revised_summary_rougeL_to_base'] = scores['rougeL']
            from nltk.tokenize import word_tokenize
            df['model_summary_length'] = [len(word_tokenize(x)) for x in df['model_summary']]
            df['revised_summary_length'] = [len(word_tokenize(x)) for x in df['revised_summary']]
            df.to_csv(f"/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/{name}_scores.csv")


def args_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--max_encoding_length', type=int, default=2048)
    parser.add_argument('--max_generation_length', type=int, default=128)
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--min_generation_length', type=int, default=0)
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--pretrained_adapter_path', type=str, default=None)
    return parser.parse_args()


def train_editor():
    args = args_parser()
    from accelerate import infer_auto_device_map, dispatch_model
    revision_model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    device_map = infer_auto_device_map(revision_model,
                                       max_memory={0: "4GB", 1: "12GB"},
                                       no_split_module_classes=["T5Block"])
    revision_model = dispatch_model(revision_model, device_map)
    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    generation_config = revision_model.generation_config
    generation_config.max_length = args.max_generation_length
    generation_config.early_stopping = True
    generation_config.length_penalty = args.length_penalty
    generation_config.num_beams = args.beam_size
    generation_config.min_length = args.min_generation_length
    revision_model.generation_config = generation_config
    lr = args.lr
    train_batch_size = args.train_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    epochs = args.epochs
    weight_decay = args.weight_decay
    with open("/data/home/yehonatan-pe/Correction_pipeline/DeFacto/data/test.jsonl") as f:
        data = [json.loads(line) for line in f]
    train_texts = [d['article'] for d in data]
    train_summaries = [d['candidate'] for d in data]
    train_instructions = [d['feedback']['instruction'] for d in data]
    explanations = [d['feedback']['explanation'] for d in data]
    revised_summaries = [d['feedback']['summary'] for d in data]
    train_dataset = EditDataset(train_texts, train_summaries, train_instructions, explanations, revised_summaries)
    model_name = args.model_name
    if not os.path.exists(os.path.join(args.save_dir, model_name)):
        os.makedirs(os.path.join(args.save_dir, model_name))
    model_path = os.path.join(args.save_dir, model_name)
    dir_num = find_largest_numbered_dir(model_path) + 1
    output_path = os.path.join(model_path, str(dir_num))
    os.makedirs(output_path)
    args.output_path = output_path
    args.test_output_path = output_path
    with open(output_path + '/args.txt', 'w') as f:
        f.write(str(args))
    train_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(output_path, 'checkpoints'),
        do_train=args.train, do_eval=False,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr, num_train_epochs=epochs, predict_with_generate=True,
        evaluation_strategy='no',
        save_strategy='epoch', save_total_limit=5,
        eval_accumulation_steps=30,
        weight_decay=weight_decay,
        logging_steps=0.01, report_to=["none"], save_only_model=True)
    max_length = args.max_encoding_length
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType
        if args.pretrained_adapter_path is not None:
            revision_model.load_adapter(args.pretrained_adapter_path)
        else:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                task_type=TaskType.SEQ_2_SEQ_LM
            )
            revision_model = get_peft_model(revision_model, lora_config)
    trainer = T5_Trainer(collate_fn=collate_fn_revision, model=revision_model, tokenizer=tokenizer, args=train_args,
                         train_dataset=train_dataset,
                         max_length_train=max_length,
                         )
    trainer.train()





if __name__ == "__main__":
    # train_editor()
    score_all_datasets()


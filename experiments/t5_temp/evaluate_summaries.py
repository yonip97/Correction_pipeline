import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
sys.path.append(os.path.dirname(os.getcwd()))
os.chdir('../')
from datasets import load_dataset
from Seahorse_metrics.metrics import Seahorse_metrics
from transformers import T5ForConditionalGeneration, T5Tokenizer
from train_t5 import summarize
import matplotlib.pyplot as plt
import torch


def write_texts_to_file(filename, text_list):
    with open(filename, 'w') as file:
        for text in text_list:
            file.write("%s\n" % text)


# Read texts from a text file into a list
def read_texts_from_file(filename):
    with open(filename, 'r') as file:
        texts = file.readlines()
        # Remove newline characters from each string
        texts = [line.strip() for line in texts]
    return texts


def main():
    dataset = load_dataset('xsum', split='test')
    # temp_path = 'temp_t5.txt'
    # if os.path.exists(temp_path):
    #     # Delete the file
    #     os.remove(temp_path)
    texts = [dataset[i]['document'] for i in range(len(dataset))]
    # path = "checkpoints/t5_base_xsum_12_11_2023_17_51_56/checkpoint-45000"
    # model = T5ForConditionalGeneration.from_pretrained(path)
    # tokenizer = T5Tokenizer.from_pretrained(path)
    # summaries = evaluate(texts, model, tokenizer, device='cuda:1')
    # model.to('cpu')
    # torch.cuda.empty_cache()
    # write_texts_to_file('temp_t5.txt',summaries)
    summaries = read_texts_from_file('temp_t5.txt')
    prefix = "google/seahorse-large-q"
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    for i, suffix in enumerate([1, 2, 3, 4, 5, 6]):
        row = i // 2
        col = i % 2
        metric = Seahorse_metrics(model_path=prefix + str(suffix), tokenizer_name=prefix + str(suffix), device='auto',
                                  batch_size=8, max_length=2048)
        scores = metric.score(texts, summaries)
        axs[row, col].hist(scores, bins=20)
        axs[row, col].title.set_text(f"Seahorse scores for q {suffix}")
        axs[row, col].set_xlim((0, 1))
    plt.tight_layout()
    plt.savefig('checkpoints/t5_base_xsum_12_11_2023_17_51_56/seahorse_plots.png')
    plt.show()

    # if os.path.exists(temp_path):
    #     # Delete the file
    #     os.remove(temp_path)


def evaluate_factuality():
    dataset = load_dataset('xsum', split='test')

    texts = [dataset[i]['document'] for i in range(len(dataset))][::10]

    summaries = read_texts_from_file('temp_t5.txt')[::10]
    model = Seahorse_metrics(model_path='google/seahorse-xxl-q4', tokenizer_name='google/seahorse-xxl-q4',
                             device='auto', batch_size=1, max_length=2048,torch_dtype=torch.float16)
    scores = model.score(texts, summaries)
    predictions = [1 if score > 0.5 else 0 for score in scores]
    print(sum(predictions))


#main()
evaluate_factuality()
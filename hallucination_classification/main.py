from transformers import BertModel
from torch import nn


class MultiHeadBert():
    def __init__(self, num_categories):
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for category in num_categories:
            setattr(self, category, nn.Linear(768, 2))


def get_frank_data():

def main():
    model = MultiHeadBert(['category1', 'category2', 'category3'])
    print(model.category1)
    print(model.category2)
    print(model.category3)

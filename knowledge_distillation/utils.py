from torch.utils.data import Dataset


def add_prompts(text, prompt):
    return prompt + text


def load_factuality_classifier(args):
    pass

class Regular_dataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item], self.labels[item]

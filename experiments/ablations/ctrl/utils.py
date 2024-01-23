import pickle
from torch.utils.data import Dataset


def save_list_to_file(my_list, filename):
    with open(filename, 'wb') as file:
        pickle.dump(my_list, file)


# Function to load a list from a file
def load_list_from_file(filename):
    with open(filename, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list


class SummariesscoredDataset(Dataset):
    def __init__(self, texts, summaries, predictions):
        self.texts = texts
        self.summaries = summaries
        self.predictions = predictions

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'summary': self.summaries[idx], 'prediction': self.predictions[idx]}

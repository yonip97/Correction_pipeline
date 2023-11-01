from pathlib import Path
from torch.utils.data import Dataset
import os
import pandas as pd


def true_topics(topic_list):
    summarization_datasets = ['frank_valid_download', 'summeval_download', 'mnbm_download', 'qags_cnndm_download',
                              'qags_xsum_download']
    dialogue_datasets = ['q2_download', 'dialfact_valid_download', 'begin_dev_download']
    fact_verification_datasets = ['fever_dev_download', 'vitc_dev_download']
    paraphrasing_datasets = ['paws_download']
    datasets = []
    if 'all' in topic_list:
        datasets = summarization_datasets + dialogue_datasets + fact_verification_datasets + paraphrasing_datasets
        return datasets
    if 'summarization' in topic_list:
        datasets += summarization_datasets
    if 'dialogue' in topic_list:
        datasets += dialogue_datasets
    if 'fact_verification' in topic_list:
        datasets += fact_verification_datasets
    if 'paraphrasing' in topic_list:
        datasets += paraphrasing_datasets
    return datasets


class TRUE_dataset(Dataset):
    def __init__(self, dir_path, topics='all'):
        self.dir_path = dir_path
        self.datasets = true_topics(topics)
        self.df = self.load_file_to_pandas()
        self.filter_to_datasets(self.datasets)


    def load_file_to_pandas(self):
        root_folder = Path(__file__).parents[1]
        data_path = os.path.join(root_folder, self.dir_path)
        all_files = os.listdir(data_path)
        csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))
        dfs = []
        for file in csv_files:
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path, index_col=0)
            df['dataset'] = file.split('.')[0]
            dfs.append(df)
        df = pd.concat(dfs).reset_index(drop=True)
        df = df[['dataset', 'grounding', 'generated_text', 'label']]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['dataset'], row['grounding'], row['generated_text'], row['label']

    def filter_to_datasets(self, datasets_list):
        """
        Changes what datasets are stored in the Dataset object
        :param datasets_list: The datasets we wish to work with
        :return: None
        """
        self.df = self.df[self.df['dataset'].isin(datasets_list)]


class Dataset_no_labels(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['grounding']

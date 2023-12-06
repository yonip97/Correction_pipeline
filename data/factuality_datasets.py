from pathlib import Path
from torch.utils.data import Dataset
import os
import pandas as pd
import json
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Subset
import numpy as np

def chose_dataset(dataset_name):
    if dataset_name == 'True':
        return TRUE_dataset("data/true_data", ['summarization'])
    elif dataset_name == 'BERTS2S_TConvS2S_xsum_trained':
        return BERTS2S_TConvS2S_xsum_trained_dataset()
    elif dataset_name == 'True_filtered_for_recent':
        dataset = TRUE_dataset("data/true_data", ['summarization'])
        dataset.filter_to_recent()
        return dataset
    else:
        raise ValueError("Dataset name not found")

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
        df = df[['model','dataset', 'grounding', 'generated_text', 'label']]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #return {"dataset":row['dataset'],'premise':  row['grounding'], 'hypothesis':  row['generated_text'], 'label': row['label']}

        return {"dataset": row['dataset'], 'text': row['grounding'], 'summary': row['generated_text'],
                'label': row['label']}
    def filter_to_datasets(self, datasets_list):
        """
        Changes what datasets are stored in the Dataset object
        :param datasets_list: The datasets we wish to work with
        :return: None
        """
        self.df = self.df[self.df['dataset'].isin(datasets_list)]
    def filter_to_recent(self):
        mnbm_models = ['BERTS2S']
        summeval_models = ['SENECA','T5','NeuralTD',  'BertSum-abs', 'GPT-2', 'UniLM', 'BART', 'PEGASUS']
        frank_models = ['bert_sum','bart','BERTS2S']
        models = mnbm_models + summeval_models + frank_models
        models = set(models)
        self.df = self.df[self.df['model'].isin(models)]

class Dataset_no_labels(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        #return {"dataset":None,'premise':  row['grounding'], 'hypothesis': row['generated_text'], 'label': None}
        return {"dataset": None, 'text': row['grounding'], 'summary': row['generated_text'], 'label': None}
class FactCC_dataset(Dataset):
    def __init__(self, path_to_factcc_data):
        self.label_map ={"INCORRECT": '0', "CORRECT": '1'}
        cnn = load_dataset("cnn_dailymail", version="3.0.0", split='test')
        # path_to_data = path_to_factcc_data +'/' +split + '/data-dev.jsonl'
        # path = "/data/home/yehonatan-pe/Correction_pipeline/factCC/data/unpaired_annotated_data/test/data-dev.jsonl"
        with open(path_to_factcc_data + "validation/data-dev.jsonl", 'r') as f:
            data = [json.loads(line) for line in f]
        with open(path_to_factcc_data + "test/data-dev.jsonl", 'r') as f:
            data += [json.loads(line) for line in f]
        self.data = self.pair(cnn, data)

    def pair(self, cnn, factcc_data):
        data = []
        cnn_dict = {example['id']: example['article'] for example in tqdm(cnn)}
        for row in tqdm(factcc_data):
            id = row['id'].split('/')[-1]
            id = id.replace('cnn-test-', '')
            article = cnn_dict[id]
            summary = row['claim']
            label = self.label_map[row['label']]
            data.append((article, summary, label))
        return pd.DataFrame.from_records(data, columns=['article', 'summary', 'label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        premise = self.data.loc[item]['article']
        hypothesis = self.data.loc[item]['summary']
        label = self.data.loc[item]['label']
        #return {'dataset':"fact_cc",'premise': premise, 'hypothesis': hypothesis, 'label': label}
        return {'dataset':"fact_cc",'text': premise, 'summary': hypothesis, 'label': label}

class TrueTeacher_anli_dataset(Dataset):
    def __init__(self, tokenizer, true_teacher_samples=1e5, seed=42):
        self.anli_r1 = load_dataset('anli', split='train_r1')
        self.anli_r2 = load_dataset('anli', split='train_r2')
        self.anli_r3 = load_dataset('anli', split='train_r3')
        true_teacher = load_dataset('google/trueteacher', split='train')
        sampled_idx = []
        idx_left_for_label = {'0': true_teacher_samples // 2, '1': true_teacher_samples // 2}
        idx = np.array(range(len(true_teacher)))
        np.random.seed(seed)
        np.random.shuffle(idx)
        for index in tqdm(idx):
            index = int(index)
            if idx_left_for_label[true_teacher[index]['label']] > 0:
                sampled_idx.append(index)
                idx_left_for_label[true_teacher[index]['label']] -= 1
            if sum(idx_left_for_label.values()) == 0:
                break
        self.true_teacher = Subset(true_teacher, sampled_idx)
        cnn_dailymail_data = load_dataset("cnn_dailymail", version="3.0.0", split='train')
        self.cnn_dailymail_articles_by_id = {example['id']: example['article'] for example in tqdm(cnn_dailymail_data)}
        lengths_array = [len(self.anli_r1), len(self.anli_r2), len(self.anli_r3), len(self.true_teacher)]
        self.start_indexes = []
        for i in range(len(lengths_array)):
            self.start_indexes.append(sum(lengths_array[:i + 1]))
        self.anli_map = {0: '1', 1: '0', 2: '0'}
        self.tokenizer = tokenizer

    def __len__(self):
        return sum([len(self.anli_r1), len(self.anli_r2), len(self.anli_r3), len(self.true_teacher)])

    def __getitem__(self, item):
        if item < self.start_indexes[0]:
            dataset = 'anli_r1'
            row = self.anli_r1[item]
            premise = row['premise']
            hypothesis = row['hypothesis']
            label = self.anli_map[row['label']]
        elif item < self.start_indexes[1]:
            dataset = 'anli_r2'
            row = self.anli_r2[item - self.start_indexes[0]]
            premise = row['premise']
            hypothesis = row['hypothesis']
            label = self.anli_map[row['label']]
        elif item < self.start_indexes[2]:
            dataset = 'anli_r3'
            row = self.anli_r3[item - self.start_indexes[1]]
            premise = row['premise']
            hypothesis = row['hypothesis']
            label = self.anli_map[row['label']]
        else:
            dataset = 'true_teacher'
            row = self.true_teacher[item - self.start_indexes[2]]
            premise = self.cnn_dailymail_articles_by_id[row['cnndm_id']]
            hypothesis = row['summary']
            label = row['label']
        return {'dataset':dataset,'premise': premise, 'hypothesis': hypothesis, 'label': label}
        #return {'dataset': dataset, 'text': premise, 'summary': hypothesis, 'label': label}

class BERTS2S_TConvS2S_xsum_trained_dataset(Dataset):
    """
    This dataset contains x summaries and is proof of concept for summary revision
    The dataset contains all the summaries annotated for factuality from the MNBM dataset
    Those 2 models are the most recent one trained, and they were trained on xsum alone.
    Because xsum has much more factuality inconsistent summaries, this means those model will
    have much more factual consistency mistakes, but will be coherent and we will be able to revise them
    """

    def __init__(self, path='data/true_data/mnbm_download.csv'):
        df = pd.read_csv(path, index_col=0)
        df = df[df['model'].isin(['BERTS2S', 'TConvS2S'])]
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        row = self.data.iloc[item]
        # return {"dataset": "frank", "premise": row['grounding'], "hypothesis": row['generated_text'],
        #         "label": row['label'],'model':row['model']}
        return {"dataset": "frank", "text": row['grounding'], "summary": row['generated_text'],"label": row['label']}



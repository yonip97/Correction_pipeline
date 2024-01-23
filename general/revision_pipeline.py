from torch.utils.data import DataLoader, Dataset
from general.utils import add_None_for_one, add_None_for_None
from tqdm import tqdm
from TrueTeacher.inference import TrueTeacher
from factCC.inference import Factcc_classifier
from q_squared.inference import Q_squared_classifier
from general.LLMS import LLMFactualityClassifier
from general.LLMS import Summarization_correction_model
import random


class Mock_model():
    def revise(self, texts, summaries):
        revised_summaries = []
        for i in range(len(summaries)):
            revised_summary = self.revise_single(texts[i], summaries[i])
            revised_summaries.append(revised_summary)
        return revised_summaries

    def revise_single(self, text, summary, max_length=None):
        if random.random() < 0.05:
            return None,None
        return summary,None


def chose_revision_model(args):
    if args.revision_model_name == 'gpt-4':
        prompt = args.revision_prompt
        API_KEY = args.API_KEY_revision_model
        return Summarization_correction_model(temp_save_dir=args.dir_path, prompt=prompt, past_text_prompt='',
                                              model='gpt-4',
                                              API_KEY=API_KEY)
    elif args.revision_model_name == 'gpt-4-turbo':
        prompt = args.revision_prompt
        API_KEY = args.API_KEY_revision_model
        return Summarization_correction_model(temp_save_dir=args.dir_path, prompt=prompt, past_text_prompt='',
                                              model=args.revision_model_name,
                                              API_KEY=API_KEY)
    elif args.revision_model_name == 'mock':
        return Mock_model()
    else:
        raise ValueError("No such revison model exists!")


def chose_factuality_classifier(args):
    device = args.device
    batch_size = args.factuality_batch_size
    if args.factuality_classifier_name == 'TrueTeacher-t5-base':
        return TrueTeacher(model_path='TrueTeacher/TrueTeacher-t5-base-checkpoint', tokenizer_name='t5-base',
                           device=device, batch_size=batch_size,
                           max_length=2048)
    elif args.factuality_classifier_name == 'TrueTeacher-t5-11b':
        return TrueTeacher(model_path='google/t5_11b_trueteacher_and_anli',
                           tokenizer_name='google/t5_11b_trueteacher_and_anli', device=device, batch_size=batch_size,
                           max_length=2048)
    elif args.factuality_classifier_name == 'Factcc':
        return Factcc_classifier(checkpoint_path='factCC/checkpoints/factcc-checkpoint',
                                 backbone_model_name='bert-base-uncased', device=device, batch_size=batch_size)
    elif args.factuality_classifier_name == 'q_squared_f1':
        return Q_squared_classifier(device=device, similarity_metric='f1', threshold=args.threshold,
                                    remove_personal=True)
    elif args.factuality_classifier_name == 'q_squared_nli':
        return Q_squared_classifier(device=device, similarity_metric='nli', threshold=args.threshold,
                                    remove_personal=True)
    elif args.factuality_classifier_name == 'LLM_chatgpt':
        prompt = args.factuality_classifier_prompt
        API_KEY = args.API_KEY_factuality_classifier
        return LLMFactualityClassifier(temp_save_dir=args.dir_path, prompt=prompt,
                                       text_to_labels=args.text_to_labels, past_text_prompt='',
                                       model='gpt-3.5-turbo',
                                       API_KEY=API_KEY)
    elif args.factuality_classifier_name == 'LLM_gpt-4':
        prompt = args.factuality_classifier_prompt
        API_KEY = args.API_KEY_factuality_classifier
        return LLMFactualityClassifier(temp_save_dir=args.dir_path, prompt=prompt,
                                       text_to_labels=args.text_to_labels, past_text_prompt='',
                                       model='gpt-4', API_KEY=API_KEY)
    else:
        raise ValueError("No such factuality classifier exists!")


def collate_fn(batch):
    texts, summaries = zip(*batch)
    return texts, summaries


class TextSummariesDataset(Dataset):
    def __init__(self, texts, summaries):
        self.texts = texts
        self.summaries = summaries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item], self.summaries[item]


class RevisionPipeline():
    def __init__(self, factuality_classifier, is_factuality_classifier_llm, factuality_classifier_kwargs,
                 revision_model,
                 is_revision_model_llm, revision_model_kwargs):
        self.factuality_classifier = factuality_classifier
        self.factuality_classifier_kwargs = factuality_classifier_kwargs
        self.is_factuality_classifier_llm = is_factuality_classifier_llm
        self.revision_model = revision_model
        self.revision_model_kwargs = revision_model_kwargs
        self.is_revision_model_llm = is_revision_model_llm

    def apply(self, texts, summaries, factuality_batch_size, revision_batch_size):
        if self.is_factuality_classifier_llm:
            predictions, errors = self.classify_for_factuality_llms(texts, summaries)
        else:
            dataset = TextSummariesDataset(texts, summaries)
            dataloader = DataLoader(dataset, batch_size=factuality_batch_size, shuffle=False,
                                    collate_fn=collate_fn)
            predictions = self.classify_for_factuality(dataloader)
        need_revision_texts = [texts[i] for i in range(len(texts)) if predictions[i] == 0]
        need_revision_summaries = [summaries[i] for i in range(len(texts)) if predictions[i] == 0]
        if self.is_revision_model_llm:
            revisions, errors = self.revise_summaries_llms(need_revision_texts, need_revision_summaries)
        else:
            dataset = TextSummariesDataset(need_revision_texts, need_revision_summaries)
            dataloader = DataLoader(dataset, batch_size=revision_batch_size, shuffle=False, collate_fn=collate_fn)
            revisions = self.revise_summaries(dataloader)
        revisions = add_None_for_one(annotated_list=predictions, list_of_objects=revisions)
        texts_for_post_revision = [texts[i] for i in range(len(texts)) if revisions[i] is not None]
        summaries_for_post_revision = [revisions[i] for i in range(len(texts)) if revisions[i] is not None]
        if self.is_factuality_classifier_llm:
            post_revision_predictions, errors = self.classify_for_factuality_llms(texts_for_post_revision,
                                                                                  summaries_for_post_revision)
        else:
            revised_dataset = TextSummariesDataset(texts_for_post_revision, summaries_for_post_revision)
            dataloader = DataLoader(revised_dataset, batch_size=factuality_batch_size, shuffle=False,
                                    collate_fn=collate_fn)
            post_revision_predictions = self.classify_for_factuality(dataloader)
        post_revision_predictions = add_None_for_None(revisions, post_revision_predictions)
        return predictions, revisions, post_revision_predictions

    def classify_for_factuality_llms(self, texts, summaries):
        consistency_predictions, errors = [], []
        for text, summary in tqdm(zip(texts, summaries)):
            consistency_prediction, error = self.factuality_classifier.classify_single(text=text,
                                                                                       summary=summary,
                                                                                       **self.factuality_classifier_kwargs)
            consistency_predictions.append(consistency_prediction)
            errors.append(error)
        return consistency_predictions, errors

    def classify_for_factuality(self, dataloader):
        consistency_predictions = []
        for batch in tqdm(dataloader):
            texts, summaries = batch
            batch_consistency_predictions = self.factuality_classifier.classify(texts=texts, summaries=summaries,
                                                                                **self.factuality_classifier_kwargs)
            consistency_predictions += batch_consistency_predictions
        return consistency_predictions

    def revise_summaries(self, dataloader):
        revised_summaries = []
        for batch in tqdm(dataloader):
            texts, summaries = batch
            revised_summary = self.revision_model.t5_revise(texts, summaries, **self.revision_model_kwargs)
            revised_summaries.append(revised_summary)
        return revised_summaries

    def revise_summaries_llms(self, texts, summaries):
        revised_summaries, errors = [], []
        for text, summary in tqdm(zip(texts, summaries)):
            revised_summary, error = self.revision_model.revise_single(text=text, summary=summary,
                                                                       **self.revision_model_kwargs)
            revised_summaries.append(revised_summary)
            errors.append(error)
        return revised_summaries, errors

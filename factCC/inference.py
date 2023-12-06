import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset
from pytorch_transformers import BertForSequenceClassification, BertTokenizer
from data.factuality_datasets import TRUE_dataset
from factCC.factcc_utils import _truncate_seq_pair, InputExample, InputFeatures
from scipy.special import softmax


class Factcc_classifier():
    def __init__(self, checkpoint_path, backbone_model_name='bert-base-uncased', device='cpu', batch_size=8):
        self.tokenizer = BertTokenizer.from_pretrained(backbone_model_name)
        self.model = BertForSequenceClassification.from_pretrained(checkpoint_path).to(device)
        self.device = device
        self.batch_size = batch_size
        self.model.eval()

    def classify(self, texts, summaries, max_length=512):
        preds = self.score(texts, summaries, max_length=max_length)
        final_classification = np.argmax(preds, axis=1)
        return final_classification.tolist()

    def classify_single(self, text, summary, max_length=512):
        preds = self.score([text], [summary], max_length=max_length)
        final_classification = np.argmax(preds, axis=1)
        return final_classification[0]

    def score(self, texts, summaries, max_length=512):
        eval_dataset = self.load_and_cache_examples_new(texts, summaries, max_length=max_length)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size)

        preds = None
        for batch in tqdm(eval_dataloader):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2]}
                outputs = self.model(**inputs)

                logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
        # The labels of the model are consistent/inconsistent and this transforms them to inconsistent/consistent
        preds[:, [1, 0]] = preds[:, [0, 1]]
        preds = softmax(preds, axis=1)
        return preds[:,1].tolist()

    def score_single(self, text, summary, max_length=512):
        preds = self.score([text], [summary], max_length=max_length)
        return preds[0]

    def convert_examples_to_features(self, examples, max_seq_length,
                                     cls_token_at_end=False,
                                     cls_token='[CLS]',
                                     cls_token_segment_id=1,
                                     sep_token='[SEP]',
                                     sep_token_extra=False,
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     sequence_a_segment_id=0,
                                     sequence_b_segment_id=1,
                                     mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """

        features = []
        for (ex_index, example) in enumerate(examples):

            tokens_a = self.tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
                special_tokens_count = 4 if sep_token_extra else 3
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
            else:
                # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
                special_tokens_count = 3 if sep_token_extra else 2
                if len(tokens_a) > max_seq_length - special_tokens_count:
                    tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = tokens_a + [sep_token]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            ####### AUX LOSS DATA
            # get tokens_a mask
            extraction_span_len = len(tokens_a) + 2
            extraction_mask = [1 if 0 < ix < extraction_span_len else 0 for ix in range(max_seq_length)]

            # get extraction labels
            if example.extraction_span:
                ext_start, ext_end = example.extraction_span
                extraction_start_ids = ext_start + 1
                extraction_end_ids = ext_end + 1
            else:
                extraction_start_ids = extraction_span_len
                extraction_end_ids = extraction_span_len

            augmentation_mask = [1 if extraction_span_len <= ix < extraction_span_len + len(tokens_b) + 1 else 0 for ix
                                 in
                                 range(max_seq_length)]

            if example.augmentation_span:
                aug_start, aug_end = example.augmentation_span
                augmentation_start_ids = extraction_span_len + aug_start
                augmentation_end_ids = extraction_span_len + aug_end
            else:
                last_sep_token = extraction_span_len + len(tokens_b)
                augmentation_start_ids = last_sep_token
                augmentation_end_ids = last_sep_token

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            extraction_start_ids = min(extraction_start_ids, 511)
            extraction_end_ids = min(extraction_end_ids, 511)
            augmentation_start_ids = min(augmentation_start_ids, 511)
            augmentation_end_ids = min(augmentation_end_ids, 511)

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=None,
                              extraction_mask=extraction_mask,
                              extraction_start_ids=extraction_start_ids,
                              extraction_end_ids=extraction_end_ids,
                              augmentation_mask=augmentation_mask,
                              augmentation_start_ids=augmentation_start_ids,
                              augmentation_end_ids=augmentation_end_ids))
        return features

    def load_and_cache_examples_new(self, texts, summaries, max_length):
        examples = []
        for text, summary in zip(texts, summaries):
            examples.append(InputExample(guid=0, text_a=text, text_b=summary))
        features = self.convert_examples_to_features(examples, max_length,
                                                     cls_token_at_end=False,
                                                     # xlnet has a cls token at the end
                                                     cls_token=self.tokenizer.cls_token,
                                                     cls_token_segment_id=0,
                                                     sep_token=self.tokenizer.sep_token,
                                                     sep_token_extra=False,
                                                     pad_on_left=False,  # pad on the left for xlnet
                                                     pad_token=
                                                     self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[
                                                         0],
                                                     pad_token_segment_id=0)

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_ext_mask = torch.tensor([f.extraction_mask for f in features], dtype=torch.float)
        all_ext_start_ids = torch.tensor([f.extraction_start_ids for f in features], dtype=torch.long)
        all_ext_end_ids = torch.tensor([f.extraction_end_ids for f in features], dtype=torch.long)
        all_aug_mask = torch.tensor([f.augmentation_mask for f in features], dtype=torch.float)
        all_aug_start_ids = torch.tensor([f.augmentation_start_ids for f in features], dtype=torch.long)
        all_aug_end_ids = torch.tensor([f.augmentation_end_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_ext_mask, all_ext_start_ids, all_ext_end_ids,
                                all_aug_mask, all_aug_start_ids, all_aug_end_ids)
        return dataset


#
# def evaluate(model, tokenizer, texts, summaries, eval_batch_size=8, device='cpu', max_length=512):
#     eval_dataset = load_and_cache_examples_new(texts, summaries, max_length=max_length,
#                                                tokenizer=tokenizer)
#     eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size)
#
#     preds = None
#
#     for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         model.eval()
#         batch = tuple(t.to(device) for t in batch)
#
#         with torch.no_grad():
#             inputs = {'input_ids': batch[0],
#                       'attention_mask': batch[1],
#                       'token_type_ids': batch[2]}
#             outputs = model(**inputs)
#
#             logits = outputs[0]
#
#         if preds is None:
#             preds = logits.detach().cpu().numpy()
#         else:
#             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
#
#     final_classification = np.argmax(preds, axis=1)
#     # The labels of the model are consistent/inconsistent and this transforms them to inconsistent/consistent
#     final_classification = [1 - x for x in final_classification]
#     return final_classification

def main():
    model_name = 'bert-base-uncased'
    checkpoint = "/data/home/yehonatan-pe/Correction_pipeline/factCC/checkpoints/factcc-checkpoint"
    device = 'cuda'
    classifier = Factcc_classifier(checkpoint_path=checkpoint, backbone_model_name=model_name, device=device)

    x = TRUE_dataset('data', ['summarization'])
    # x.filter_to_datasets(true_topics(['summarization']))
    data = x.df
    texts = data['grounding']
    summaries = data['generated_text']
    preds = classifier.classify(texts, summaries)
    labels = data['label']
    print(roc_auc_score(labels, preds))
    print(balanced_accuracy_score(labels, preds))
    print(f1_score(labels, preds, average='micro'))
    print(np.mean(labels == preds))

# if __name__ == 'main':
#     main()

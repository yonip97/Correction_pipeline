from text_correction_model import Text_correction_model
from transformers import AutoTokenizer, AutoModel
from data.utils import TRUE_dataset, true_topics, Dataset_no_labels
from torch.utils.data import DataLoader
from torch.optim import Adam


def create_correction_model(args):
    if args.correction_model == 'llm':
        correction_model = Text_correction_model(prompt_path=args.text_correction_prompt_path, model=args.llm_model,
                                                 API_KEY=args.api_ley)
        kwargs = {}
        return correction_model, {}
    elif args.correction_model == 'pipeline':
        correction_model, qg_kwargs, qa_kwargs, revision_kwargs = create_correction_model(args)
        kwargs = {'qg_kwargs': qg_kwargs, 'qa_kwargs': qa_kwargs, 'revision_kwargs': revision_kwargs}

    else:
        raise ValueError("No such Correction model!")
    return correction_model, kwargs


def load_dataset(args):
    if args.dataset_name == 'true':
        dataset = TRUE_dataset(args.data_dir_path)
        datasets_names = true_topics(['summarization'])
        dataset.filter_to_datasets(datasets_names)
        dataset = Dataset_no_labels(dataset.df)
    else:
        raise ValueError('No such dataset exist')
    return dataset


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = AutoModel.from_pretrained(args.model_path)
    return tokenizer, model


class Trainer():
    def __init__(self, args):
        self.correction_model, self.correction_model_kwargs = create_correction_model(args)
        self.model, self.tokenizer = load_model(args)
        self.factuality_classifier = load_factuality_classifier(args)
        dataset = load_dataset(args)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        self.epochs = args.epochs
        self.optimizer = Adam(self.model.parameters(),lr=args.lr)

    def train(self):
        for epoch in self.epochs:
            for original_texts in self.dataloader:
                model_inputs = self.tokenizer.batch_encode(original_texts)
                model_scores_outputs, texts_outputs = get_model_scores_and_output_texts(model_inputs)

    def train_with_distillation(self):
        for epoch in self.epochs:
            for original_texts in self.dataloader:
                model_inputs = self.tokenizer.batch_encode(original_texts)
                model_scores, texts = get_model_scores_and_output_texts(model_inputs)
                factuality_scores = self.check_factuality(original_texts, texts)
                inconsistent_texts = [texts[~i] for i in factuality_scores]
                inconsistent_model_scores = [model_scores[~i] for i in factuality_scores]
                corrected_texts = self.correction_model(inconsistent_texts, self.correction_model_kwargs)
                loss = self.calculate_distliation_loss(model_scores, corrected_texts)
                loss.backward()
                self.optimizer.step()



def train(args):
    # trainer = Trainer(args)
    # m = T5ForConditionalGeneration.from_pretrained("t5-small")
    # t = T5Tokenizer.from_pretrained("t5-small")
    # text = "translate english"
    # model_input = t("translate English to German: The house is wonderful.", return_tensors="pt")
    # output = m.generate(**model_input).logits
    c = 1
    # if args.train_method == "distillation":
    #
    # elif args.train_method == "control":
    #     train_with_control()


train(None)

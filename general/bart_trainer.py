from transformers import Seq2SeqTrainer
from torch.utils.data import DataLoader




class BartTrainer(Seq2SeqTrainer):
    def __init__(self, collate_fn, max_length=512, **kwargs):
        self.collate_fn = collate_fn
        self.max_length = max_length
        super(BartTrainer, self).__init__(**kwargs)

    def get_train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=True,
                          collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length),
                          pin_memory=False)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        return DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size, shuffle=False,
                          collate_fn=lambda x: self.collate_fn(x, self.tokenizer, self.max_length),
                          pin_memory=False)
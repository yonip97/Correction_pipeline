import openai
import tiktoken
from torch.utils.data import DataLoader


class Cost_estimator():
    def __init__(self, model, input_price, output_price):
        """
        :param model: The model for we want the encoding for
        :param input_price: Price for 1k tokens in input
        :param output_price: Price for 1k tokens in output
        """
        self.encoding = tiktoken.encoding_for_model(model)
        self.input_price = input_price
        self.output_price = output_price

    def estimate_cost(self, text, output_estimation):
        input_len = self.estimate_input(text)
        estimated_output_len = self.estimate_output(output_estimation)
        return estimated_output_len / 1000 * self.input_price + input_len / 1000 * self.output_price, input_len, estimated_output_len

    def estimate_input(self, text):
        input_encoding = self.encoding.encode(text)
        return len(input_encoding)

    def estimate_output(self, output_estimation):
        output_encoding = self.encoding.encode(output_estimation)
        return len(output_encoding)

    def estimate_dataset(self, prompt, dataset):
        # dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=1)
        total_cost = 0
        total_input = 0
        total_output = 0
        for i in range(len(dataset)):
            x = dataset[i]
            _, original_text, generated_text, label = x['dataset'], x['premise'], x['hypothesis'], x['label']
            # original_text, generated_text, label = original_text[0], generated_text[0], label[0]
            model_input = prompt
            model_input += 'original_text: ' + '\n' + original_text + '\n'
            model_input += "summary: " + '\n' + generated_text + '\n'
            model_input += 'revised summary: ' + '\n'
            item_cost, item_input_len, item_output_len = self.estimate_cost(model_input, generated_text)
            total_cost += item_cost
            total_input += item_input_len
            total_output += item_output_len
        return total_cost, total_input, total_output
# from data.factuality_datasets import TRUE_dataset
# dataset = TRUE_dataset('data/true_data',['summarization'])
# estimator = Cost_estimator('gpt-4',0.03,0.06)
# print(estimator.estimate_dataset('This is a test prompt',dataset))

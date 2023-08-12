
from llms import LLM_model


class Text_correction_model(LLM_model):
    def __call__(self, text, **kwargs):
        model_input = self.create_model_input(text)
        model_output = self.get_chatgpt_response(model_input, **kwargs)
        return model_output

    def create_model_input(self, text):
        model_input = self.prompt + text
        return model_input

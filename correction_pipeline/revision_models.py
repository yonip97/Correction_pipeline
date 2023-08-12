from correction_pipeline.llms import LLM_model


class Dummie_revision():
    def __init__(self):
        pass

    def __call__(self, questions, answers_based_on_generated_text, answers_based_on_original_text, generated_text):
        return generated_text


class LLM_prompt_based_revision_model(LLM_model):
    def __call__(self, question, answer_based_on_generated_text, answer_based_on_original_text,
                 generated_text, **kwargs):
        return self.revise_text(question, answer_based_on_generated_text, answer_based_on_original_text,
                                generated_text, **kwargs)

    def revise_text(self, question, answer_based_on_generated_text, answer_based_on_original_text,
                    generated_text, **kwargs):
        model_input = self.create_model_input(question, answer_based_on_generated_text, answer_based_on_original_text,
                                              generated_text)
        new_text = self.get_chatgpt_response(model_input, **kwargs)
        return new_text

    def create_model_input(self, question, answer_based_on_generated_text, answer_based_on_original_text,
                           generated_text):
        final_text = self.prompt + '\n'
        final_text += 'original text: ' + generated_text + '\n'
        final_text += 'question: ' + question + '\n'
        final_text += 'correct answer: ' + answer_based_on_original_text
        final_text += 'new text'
        return final_text

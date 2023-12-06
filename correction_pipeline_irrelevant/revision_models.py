from correction_pipeline.llms import LLM_model, Cost_estimator


class Dummie_revision():
    def __init__(self):
        self.estimator = Cost_estimator(model='gpt-3.5-turbo', input_price=0.0015, output_price=0.003)
        self.cost = 0
        self.revisions_number = 0

    def __call__(self, question, answer_based_on_generated_text, answer_based_on_original_text, generated_text):
        input_text = self.create_model_input(question, answer_based_on_generated_text, answer_based_on_original_text,
                                             generated_text)
        self.cost += self.estimator.estimate_cost(input_text, generated_text)
        self.revisions_number += 1
        return generated_text

    def create_model_input(self, question, answer_based_on_generated_text, answer_based_on_original_text,
                           generated_text):
        prompt = """Revise the original text according to the new information. Change as little of the text as possible.
Original text: 
The overflow pipes in a Welsh inlet have been caught in the middle of a severe storm and are now dangerous to flood. The European Court of justice has ruled that the pipes violated clean water laws, which cover a large area of Wales. The Welsh government, natural resources Wales and Welshwater are investing in measures to reduce the amount of rain that flows into the estuary. The court also found that the British had failed to meet its obligations under the Clean Water Act by 2020. 
New information: 
Welsh Water is investing in measures to reduce the amount of rain that flows into the estuary. 
New text: 
The overflow pipes in a Welsh inlet have been caught in the middle of a severe storm and are now dangerous to flood. The European Court of justice has ruled that the pipes violated clean water laws, which cover a large area of Wales.Welshwater are investing in measures to reduce the amount of rain that flows into the estuary. The court also found that the British had failed to meet its obligations under the Clean Water Act by 2020. 
Original text: 
The overflow pipes in a Welsh inlet have been caught in the middle of a severe storm and are now dangerous to flood. The European Court of justice has ruled that the pipes violated clean water laws, which cover a large area of Wales. The Welsh government, natural resources Wales and Welshwater are investing in measures to reduce the amount of rain that flows into the estuary. The court also found that the British had failed to meet its obligations under the Clean Water Act by 2020. 
New information: 
The reason for the overflow pipes being deemed dangerous is that they broke clean water laws in a special conservation area.
New text: 
The overflow pipes in a Welsh inlet dangerous becuase they broke clean water laws in a special conservation area. The European Court of justice has ruled that the pipes violated clean water laws, which cover a large area of Wales. The Welsh government, natural resources Wales and Welshwater are investing in measures to reduce the amount of rain that flows into the estuary. The court also found that the British had failed to meet its obligations under the Clean Water Act by 2020.
"""
        final_text = prompt + '\n'
        final_text += 'original text: ' + generated_text + '\n'
        final_text += 'question: ' + question + '\n'
        final_text += 'correct answer: ' + answer_based_on_original_text
        final_text += 'new text:' + '\n'
        return final_text


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

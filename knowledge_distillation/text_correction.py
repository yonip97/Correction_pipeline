import openai
class LLM_model():
    def __init__(self, prompt_path=None, model='chatgpt-turbo-3.5', API_KEY=None, **kwargs):
        openai.api_key = API_KEY
        if prompt_path != None:
            with open(prompt_path, "r") as file:
                prompt = file.read()
        else:
            prompt = ''
        self.prompt = prompt
        self.model = model

    def get_chatgpt_response(self, input, max_length, **kwargs):
        try:
            message = [{
                "role": "user",
                "content": input,
            }]
            response = openai.ChatCompletion.create(
                engine=self.model,
                messages=message,
                temperature=0,
                max_tokens=max_length
            )
            return response['choices'][0]['message']['content']
        except openai.OpenAIError as e:
            print(f"Error occurred: {e}")
            return None

class Text_correction_model(LLM_model):
    def __call__(self, text, **kwargs):
        model_input = self.create_model_input(text)
        model_output = self.get_chatgpt_response(model_input, **kwargs)
        return model_output

    def create_model_input(self, text):
        model_input = self.prompt + text
        return model_input

class Correction_pipline():
    def __init__(self, qg_model, qa_model, filter_model, disagreement_model, revision_model):
        self.qg_model = qg_model
        self.qa_model = qa_model
        self.filter_model = filter_model
        self.disagreement_model = disagreement_model
        self.revision_model = revision_model

    def apply(self, original_text, generated_text, qg_kwargs, qa_kwargs, revision_kwargs):
        questions = self.generate_questions(generated_text, qg_kwargs)
        answers_based_on_generated_text = self.generate_answers(generated_text, questions,qa_kwargs)
        questions, answers_based_on_generated_text = self.filter_questions(questions, answers_based_on_generated_text)
        if len(questions) == 0:
            return original_text,[]
        answers_based_on_original_text = self.generate_answers(original_text, questions,qa_kwargs)
        questions_with_disagreements, answers_based_on_generated_text_with_disagreements, answers_based_on_original_text_with_disagreements = self.find_disagreements(
            questions, answers_based_on_generated_text, answers_based_on_original_text)
        revised_text, revision_steps = self.revision(questions_with_disagreements,
                                                     answers_based_on_generated_text_with_disagreements,
                                                     answers_based_on_original_text_with_disagreements, generated_text,revision_kwargs)
        return revised_text, revision_steps

    def generate_questions(self, text, kwargs):
        questions = self.qg_model(text, **kwargs)
        return questions

    def generate_answers(self, text, questions,kwargs):
        answers = self.qa_model(questions, text,**kwargs)
        return answers

    def filter_questions(self, questions, answers_based_on_generated_text):
        relevant_questions, relevant_answers = self.filter_model(questions, answers_based_on_generated_text)
        return relevant_questions, relevant_answers

    def find_disagreements(self, questions, answers_based_on_generated_text, answers_based_on_original_text):
        indexes = self.disagreement_model(questions, answers_based_on_generated_text, answers_based_on_original_text)
        questions = [questions[i] for i in indexes]
        answers_based_on_generated_text = [answers_based_on_original_text[i] for i in indexes]
        answers_based_on_original_text = [answers_based_on_original_text[i] for i in indexes]
        return questions, answers_based_on_generated_text, answers_based_on_original_text

    def revise_text(self, question, answer_based_on_generated_text, answer_based_on_original_text,
                    generated_text,kwargs):
        new_text = self.revision_model(question, answer_based_on_generated_text, answer_based_on_original_text,
                                       generated_text,**kwargs)
        return new_text

    def revision(self, questions, answers_based_on_generated_text, answers_based_on_original_text, generated_text,kwargs):
        pipeline_steps = []
        for question, answer_based_on_generated_text, answer_based_on_original_text in zip(questions,
                                                                                           answers_based_on_generated_text,
                                                                                           answers_based_on_original_text):
            new_text = self.revise_text(question, answer_based_on_generated_text, answer_based_on_original_text,
                                        generated_text,kwargs)
            pipeline_steps.append(new_text)
            generated_text = new_text
        return generated_text, pipeline_steps

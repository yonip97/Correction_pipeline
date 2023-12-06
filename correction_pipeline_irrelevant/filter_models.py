import string


class Base_filter():
    def __init__(self):
        pass
    def __call__(self, questions, answers):
        return questions, answers


class Strategy_based_filter_model():
    def __init__(self, unanswerable_response, unanswerable=False, multiple_repetitions=False,
                 yes_or_no_questions=False,meaningless_answer = False):
        self.unanswerable_response = unanswerable_response
        self.unanswerable = unanswerable
        self.multiple_repetitions = multiple_repetitions
        self.yes_or_no_questions = yes_or_no_questions
        self.meaningless_answer = meaningless_answer

    def __call__(self, questions, answers):
        new_questions = []
        new_answers = []
        for q, a in zip(questions, answers):
            if self.unanswerable and self.check_if_unanswerable(a):
                continue
            if self.multiple_repetitions and self.check_if_multiple_repetitions(a, new_answers):
                continue
            if self.yes_or_no_questions and self.check_if_yes_or_no_question(q):
                continue
            if self.meaningless_answer and self.check_if_answer_is_meaningless(a):
                continue
            new_questions.append(q)
            new_answers.append(a)
        return new_questions, new_answers

    def check_if_unanswerable(self, answer):
        if answer == self.unanswerable_response:
            return True
        return False

    def check_if_multiple_repetitions(self, answer, previous_answers):
        if answer in previous_answers:
            return True
        return False

    def check_if_yes_or_no_question(self, q):
        question_words = ["is", "are", "was", "were", "can", "will", "do", "does", "did", "has", "have", "had", "am"]
        question_word = q.split(' ')[0]
        if question_word in question_words:
            return True
        return False
    def check_if_answer_is_meaningless(self,a):
        if a in string.punctuation or a == '' or a == '<s>':
            return True
        return False

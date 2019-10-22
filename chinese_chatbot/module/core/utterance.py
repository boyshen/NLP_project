class Utterance(object):
    def __init__(self, ask=None, ask_keyword=None, label=None, retrieval_answer=None, cos_answer=None, score=None,
                 final_answer=None, sensitive_word=None, any_reply=None):
        self.ask = ask
        self.ask_keyword = ask_keyword
        self.label = label
        self.retrieval_answer = retrieval_answer
        self.cos_answer = cos_answer
        self.score = score
        self.final_answer = final_answer
        self.sensitive_word = sensitive_word
        self.any_reply = any_reply

    def set_ask(self, ask):
        self.ask = ask

    def get_ask(self):
        return self.ask

    def set_ask_keyword(self, ask_keyword):
        self.ask_keyword = ask_keyword

    def get_ask_keyword(self):
        return self.ask_keyword

    def set_label(self, label):
        self.label = label

    def get_label(self):
        return self.label

    def set_retrieval_answer(self, retrieval_answer):
        self.retrieval_answer = retrieval_answer

    def get_retrieval_answer(self):
        return self.retrieval_answer

    def set_cos_answer(self, cos_answer):
        self.cos_answer = cos_answer

    def get_cos_answer(self):
        return self.cos_answer

    def set_score(self, score):
        self.score = score

    def get_score(self):
        return self.score

    def set_final_answer(self, final_answer):
        self.final_answer = final_answer

    def get_final_answer(self):
        return self.final_answer

    def set_sensitive_word(self, sensitive_word):
        self.sensitive_word = sensitive_word

    def get_sensitive_word(self):
        return self.sensitive_word

    def set_any_reply(self, any_reply):
        self.any_reply = any_reply

    def get_any_reply(self):
        return self.any_reply

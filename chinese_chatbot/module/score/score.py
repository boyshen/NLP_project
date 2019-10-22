import torch
from module.core.network_model import load_model


class ScoreConf(object):
    def __init__(self):
        self.word_sequence = None
        self.model = None
        self.seq_len = None
        self.debug = None


class Score(object):

    def __init__(self, word_sequence, score_model_file, seq_len=30, debug=False):
        self.word_sequence = word_sequence
        self.model = load_model(score_model_file)
        self.seq_len = seq_len
        self.debug = debug

    def score(self, ask, answer):
        ask = self.word_sequence.transfroms([ask], max_len=self.seq_len)
        answer = self.word_sequence.transfroms([answer], max_len=self.seq_len)

        ask_vec = torch.LongTensor(ask).view(1, -1)
        answer_vec = torch.LongTensor(answer).view(1, -1)

        score = self.model.forward(ask_vec, answer_vec)

        return score.item()

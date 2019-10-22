import json
import torch
from module.core.loading_dataset import get_keyword_vec
from module.core.network_model import load_model


class SemanticsConf(object):

    def __init__(self):
        self.word_sequence = None
        self.model = None
        self.distinguish_json_file = None
        self.seq_len = None
        self.debug = None


class Semantics(object):

    def __init__(self, word_sequence, distinguish_model_file, distinguish_json_file, seq_len=30, debug=False):
        self.word_sequence = word_sequence
        self.model = load_model(distinguish_model_file)

        with open(distinguish_json_file, 'r') as rf:
            label_id = json.load(rf)

        self.id_label = {i: label for label, i in label_id.items()}

        self.seq_len = seq_len
        self.debug = debug

    def distinguish(self, ask, ask_keyword):
        sent_vec = self.word_sequence.transfroms([ask], max_len=self.seq_len)
        word_vec = get_keyword_vec([ask_keyword], max_len=self.seq_len, word_sequence=self.word_sequence)

        sent_vec = torch.LongTensor(sent_vec).view(1, -1)
        word_vec = torch.LongTensor(word_vec).view(1, -1)

        output = self.model.forward(sent_vec, word_vec)
        label = self.id_label[output.argmax(1).item()]

        if self.debug:
            print("ask: ", ask)
            print("Semantics sent vec: ", sent_vec)
            print("Semantics word vec: ", word_vec)
            print("Semantics predict output: ", output)
            print("Semantics label: ", label)

        return label

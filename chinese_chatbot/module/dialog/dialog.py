import json
import torch
from module.core.loading_dataset import get_keyword_vec
from module.core.mysql_table import dialog_table
from module.core.network_model import cos_similarity
from module.core.mysql_fuzzy_query import fuzzy_query_by_sentence
from module.core.network_model import load_model


class DialogConf(object):

    def __init__(self):
        self.mysql = None
        self.word_sequence = None
        self.model = None
        self.json_file = None
        self.seq_len = None
        self.debug = None


class Dialog(object):

    def __init__(self, mysql, word_sequence, model_file, json_file, seq_len=30, debug=False):
        self.mysql = mysql
        self.word_sequence = word_sequence

        self.model = load_model(model_file)

        with open(json_file, 'r') as rf:
            label_id = json.load(rf)
        self.id_label = {i: label for label, i in label_id.items()}

        self.correct = 1
        self.error = 0

        self.seq_len = seq_len
        self.debug = debug

        self.default = 'Sorry! 目前不知道呢'

    def query_mysql(self, label, ask):
        table = dialog_table
        query_result = fuzzy_query_by_sentence(mysql=self.mysql, table=table, label=label,
                                               sentence=ask)

        if query_result is False:
            print("Error: retrieval Exception!")
            return False

        return query_result

    def retrieval_by_cos(self, ask, mysql_retrieval):
        result_list = list()
        for m_data in mysql_retrieval:
            similarity = cos_similarity(ask, m_data['ask'])

            result = {"cos": similarity, "answer": m_data['answer']}
            result_list.append(result)

        value = 0
        final_answer = None
        for r in result_list:
            if r['cos'] > value:
                value = r['cos']
                final_answer = r['answer']

        if self.debug:
            print("cos similarity: {}".format(result_list))
            print("cos final answer: ", final_answer)

        # utterance.set_cos_answer(final_answer)
        if final_answer is None:
            final_answer = self.default
        return final_answer

    def retrieval(self, ask, ask_keyword, mysql_retrieval):
        correct_answer, error_answer = list(), list()
        for m_data in mysql_retrieval:
            answer = m_data['answer']
            answer_keyword = m_data['answer_keyword']

            ask_vec = self.word_sequence.transfroms([ask], max_len=self.seq_len)
            answer_vec = self.word_sequence.transfroms([answer], max_len=self.seq_len)

            ask_keyword_vec = get_keyword_vec([ask_keyword], word_sequence=self.word_sequence,
                                              max_len=self.seq_len)
            answer_keyword_vec = get_keyword_vec([answer_keyword], word_sequence=self.word_sequence,
                                                 max_len=self.seq_len)

            ask_vec = torch.LongTensor(ask_vec).view(1, -1)
            answer_vec = torch.LongTensor(answer_vec).view(1, -1)
            ask_keyword_vec = torch.LongTensor(ask_keyword_vec).view(1, -1)
            answer_keyword_vec = torch.LongTensor(answer_keyword_vec).view(1, -1)

            output = self.model(ask_vec, answer_vec, ask_keyword_vec, answer_keyword_vec)
            label_id = output.argmax(1).item()

            value = {'answer': answer,
                     'predict_correct': output[0][self.correct].item(),
                     'predict_error': output[0][self.error].item(),
                     'label_id': label_id}

            if label_id == self.correct:
                correct_answer.append(value)

            elif label_id == self.error:
                error_answer.append(value)

            if self.debug:
                print("dialog retrieval ask: ", ask)
                print("dialog retrieval answer: ", answer)
                print("dialog retrieval predict output: ", output)
                print("dialog retrieval label id: ", label_id)
                print("dialog retrieval predict label: ", self.id_label[label_id])
                print()

        retrieval_answer = None
        if len(correct_answer) == 0:
            value = 0
            for e_answer in error_answer:
                if e_answer['predict_error'] > value:
                    value = e_answer['predict_error']
                    retrieval_answer = e_answer['answer']

            if self.debug:
                print("choose error answer predict: {}".format(error_answer))

        elif len(correct_answer) > 1:
            value = 0
            for c_answer in correct_answer:
                if c_answer['predict_correct'] > value:
                    value = c_answer['predict_correct']
                    retrieval_answer = c_answer['answer']

            if self.debug:
                print("choose correct answer predict: {}".format(correct_answer))
                print("retrieval final answer: ", retrieval_answer)
                print("retrieval predict correct output: ", value)

        else:
            retrieval_answer = correct_answer[0]['answer']

        return retrieval_answer

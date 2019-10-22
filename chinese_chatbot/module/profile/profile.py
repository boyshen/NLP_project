import torch
import json
from module.core.network_model import load_model
from module.core.network_model import cos_similarity
from module.core.loading_dataset import get_keyword_vec
from module.core.mysql_fuzzy_query import fuzzy_query_by_sentence
from module.core.mysql_table import profile_table


class ProfileConf(object):

    def __init__(self):
        self.mysql = None
        self.word_sequence = None
        self.distinguish_model = None
        self.retrieval_model = None
        self.distinguish_json = None
        self.retrieval_json = None
        self.distinguish_seq_len = None
        self.retrieval_seq_len = None
        self.debug = None


class Profile(object):

    def __init__(self, mysql, word_sequence, distinguish_model_file, retrieval_model_file, distinguish_json_file,
                 retrieval_json_file, distinguish_seq_len=30, retrieval_seq_len=30, debug=False):
        self.mysql = mysql
        self.word_sequence = word_sequence

        self.distinguish_model = load_model(distinguish_model_file)
        self.retrieval_model = load_model(retrieval_model_file)

        with open(distinguish_json_file, 'r') as rf:
            distinguish_label_id = json.load(rf)

        with open(retrieval_json_file, 'r') as rf:
            retrieval_label_id = json.load(rf)

        self.distinguish_id_label = {i: label for label, i in distinguish_label_id.items()}
        self.retrieval_id_label = {i: label for label, i in retrieval_label_id.items()}

        self.distinguish_seq_len = distinguish_seq_len
        self.retrieval_seq_len = retrieval_seq_len

        self.correct = 1
        self.error = 0

        self.debug = debug

        self.default = 'Sorry! 目前不知道呢'

    def distinguish(self, ask, ask_keyword):
        sent_vec = self.word_sequence.transfroms([ask], max_len=self.distinguish_seq_len)
        word_vec = get_keyword_vec([ask_keyword], max_len=self.distinguish_seq_len,
                                   word_sequence=self.word_sequence)

        sent_vec = torch.LongTensor(sent_vec).view(1, -1)
        word_vec = torch.LongTensor(word_vec).view(1, -1)

        output = self.distinguish_model.forward(sent_vec, word_vec)
        label = self.distinguish_id_label[output.argmax(1).item()]

        if self.debug:
            print("profile distinguish sent vec: ", sent_vec)
            print("profile distinguish word vec: ", word_vec)
            print("profile distinguish output: ", output)
            print("profile distinguish label: ", label)

        return label

    def query_mysql(self, label, ask):
        table = profile_table
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

            ask_vec = self.word_sequence.transfroms([ask], max_len=self.retrieval_seq_len)
            answer_vec = self.word_sequence.transfroms([answer], max_len=self.retrieval_seq_len)

            ask_keyword_vec = get_keyword_vec([ask_keyword], word_sequence=self.word_sequence,
                                              max_len=self.retrieval_seq_len)
            answer_keyword_vec = get_keyword_vec([answer_keyword], word_sequence=self.word_sequence,
                                                 max_len=self.retrieval_seq_len)

            ask_vec = torch.LongTensor(ask_vec).view(1, -1)
            answer_vec = torch.LongTensor(answer_vec).view(1, -1)
            ask_keyword_vec = torch.LongTensor(ask_keyword_vec).view(1, -1)
            answer_keyword_vec = torch.LongTensor(answer_keyword_vec).view(1, -1)

            predict_output = self.retrieval_model(ask_vec, answer_vec, ask_keyword_vec, answer_keyword_vec)
            label_id = predict_output.argmax(1).item()

            value = {'answer': answer,
                     'predict_correct': predict_output[0][self.correct].item(),
                     'predict_error': predict_output[0][self.error].item(),
                     'label_id': label_id}

            if label_id == self.correct:
                correct_answer.append(value)

            elif label_id == self.error:
                error_answer.append(value)

            if self.debug:
                print("profile retrieval ask: ", ask)
                print("profile retrieval answer: ", answer)
                print("profile retrieval predict output: ", predict_output)
                print("profile retrieval label id: ", label_id)
                print("profile retrieval predict label: ", self.retrieval_id_label[label_id])
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

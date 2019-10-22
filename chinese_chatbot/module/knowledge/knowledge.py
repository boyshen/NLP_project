import json
import torch
from module.core.network_model import cos_similarity
from module.core.mysql_table import qa_table
from module.core.mysql_fuzzy_query import fuzzy_query_by_sentence
from module.core.loading_dataset import get_keyword_vec
from module.core.network_model import load_model


class KnowledgeConf(object):

    def __init__(self):
        self.mysql = None
        self.word_sequence = None
        self.category_model_file = None
        self.category_json_file = None
        self.category_seq_len = None
        self.retrieval_model_file = None
        self.retrieval_json_file = None
        self.retrieval_seq_len = None
        self.debug = None


class Knowledge(object):

    def __init__(self, mysql, word_sequence, category_model_file, category_json_file, category_seq_len,
                 retrieval_model_file, retrieval_json_file, retrieval_seq_len, debug=False):
        self.mysql = mysql
        self.word_sequence = word_sequence

        self.category_model = load_model(category_model_file)
        self.retrieval_model = load_model(retrieval_model_file)

        with open(category_json_file, 'r') as rf:
            category_label_id = json.load(rf)
        self.category_id_label = {i: label for label, i in category_label_id.items()}

        with open(retrieval_json_file, 'r') as rf:
            retrieval_label_id = json.load(rf)
        self.retrieval_id_label = {i: label for label, i in retrieval_label_id.items()}

        self.category_seq_len = category_seq_len
        self.retrieval_seq_len = retrieval_seq_len

        self.correct = 1
        self.error = 0

        self.debug = debug

        self.default = 'Sorry! 目前不知道呢'

    def category(self, ask, ask_keyword):
        sent_vec = self.word_sequence.transfroms([ask], max_len=self.category_seq_len)
        word_vec = get_keyword_vec([ask_keyword], max_len=self.category_seq_len, word_sequence=self.word_sequence)

        sent_vec = torch.LongTensor(sent_vec).view(1, -1)
        word_vec = torch.LongTensor(word_vec).view(1, -1)

        output = self.category_model.forward(sent_vec, word_vec)
        label = self.category_id_label[output.argmax(1).item()]

        if self.debug:
            print("knowledge category ask: ", ask)
            print("knowledge category ask_keyword", ask_keyword)
            print("knowledge category sent vec: ", sent_vec)
            print("knowledge category word vec: ", word_vec)
            print("knowledge category predict output: ", output)
            print("knowledge category label: ", label)
            print()

        return label

    def query_mysql(self, label, ask):
        table = qa_table
        retrieval_result = fuzzy_query_by_sentence(mysql=self.mysql, table=table, label=label, sentence=ask)

        if retrieval_result is False:
            print("Error: retrieval Exception! ")
            return False

        return retrieval_result

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
            ask_keyword_vec = get_keyword_vec([ask_keyword], max_len=self.retrieval_seq_len,
                                              word_sequence=self.word_sequence)
            answer_keyword_vec = get_keyword_vec([answer_keyword], max_len=self.retrieval_seq_len,
                                                 word_sequence=self.word_sequence)

            ask_vec = torch.LongTensor(ask_vec).view(1, -1)
            answer_vec = torch.LongTensor(answer_vec).view(1, -1)
            ask_keyword_vec = torch.LongTensor(ask_keyword_vec).view(1, -1)
            answer_keyword_vec = torch.LongTensor(answer_keyword_vec).view(1, -1)

            output = self.retrieval_model.forward(ask_vec, answer_vec, ask_keyword_vec, answer_keyword_vec)
            label_id = output.argmax(1).item()

            value = {"answer": answer,
                     "predict_correct": output[0][self.correct].item(),
                     "predict_error": output[0][self.error].item(),
                     "label_id": label_id}

            if label_id == self.correct:
                correct_answer.append(value)

            elif label_id == self.error:
                error_answer.append(value)

            if self.debug:
                print("knowledge retrieval ask: ", ask)
                print("knowledge retrieval answer: ", answer)
                print("knowledge retrieval predict output: ", output)
                print("knowledge retrieval label id: ", label_id)
                print("knowledge retrieval predict label: ", self.retrieval_id_label[label_id])
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

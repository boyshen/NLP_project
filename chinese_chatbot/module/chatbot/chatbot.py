import re
from module.sensitive_word.sensitive_word import SensitiveWordDistinguish
from module.profile.profile import Profile
from module.semantics.semantics import Semantics
from module.knowledge.knowledge import Knowledge
from module.joke.joke import Joke
from module.dialog.dialog import Dialog
from module.any_reply.any_reply import AnyReply
from module.score.score import Score
from module.core.word_sequences import load_word_sequences
from module.core.utterance import Utterance
from module.core.data_utils import extract_keyword
import global_variable as variable

from module.profile.profile import ProfileConf
from module.score.score import ScoreConf
from module.semantics.semantics import SemanticsConf
from module.knowledge.knowledge import KnowledgeConf
from module.dialog.dialog import DialogConf


def regular(sentence):
    sentence = ''.join(re.findall('[\u4e00-\u9fa5,，.。？?！!·、{};；<>()<<>>《》0-9a-zA-Z\[\]]+', sentence))
    sentence = re.sub('[/，]{1,100}', '，', sentence)
    sentence = re.sub('[/,]{1,100}', ',', sentence)
    sentence = re.sub('[？]{1,100}', '？', sentence)
    sentence = re.sub('[?]{1,100}', '?', sentence)
    sentence = re.sub('[！]{1,100}', '！', sentence)
    sentence = re.sub('[!]{1,100}', '!', sentence)
    sentence = re.sub('[/。]{1,100}', '。', sentence)
    sentence = re.sub('[/.]{1,100}', '.', sentence)
    sentence = re.sub('[。]{1,100}', '。', sentence)
    sentence = re.sub('[" "]{1,100}', ' ', sentence)
    sentence = re.sub('[～]{1,100}', '', sentence)
    return sentence


def packing_config(config_json, mysql):
    model_config = config_json['model']
    category_label = config_json['category_label']

    # packing profile
    profile_conf = ProfileConf()
    profile_conf.distinguish_model = model_config['profile_distinguish']['model_file']
    profile_conf.distinguish_json = model_config['profile_distinguish']['class_json']
    profile_conf.distinguish_seq_len = model_config['profile_distinguish']['seq_len']
    profile_conf.retrieval_model = model_config['profile_retrieval']['model_file']
    profile_conf.retrieval_json = model_config['profile_retrieval']['class_json']
    profile_conf.retrieval_seq_len = model_config['profile_retrieval']['seq_len']

    # packing semantics
    semantics_conf = SemanticsConf()
    semantics_conf.model = model_config['semantics']['model_file']
    semantics_conf.distinguish_json_file = model_config['semantics']['class_json']
    semantics_conf.seq_len = model_config['semantics']['seq_len']

    # packing knowledge
    knowledge_conf = KnowledgeConf()
    knowledge_conf.category_model_file = model_config['knowledge_category']['model_file']
    knowledge_conf.category_json_file = model_config['knowledge_category']['class_json']
    knowledge_conf.category_seq_len = model_config['knowledge_category']['seq_len']
    knowledge_conf.retrieval_model_file = model_config['knowledge_retrieval']['model_file']
    knowledge_conf.retrieval_json_file = model_config['knowledge_retrieval']['class_json']
    knowledge_conf.retrieval_seq_len = model_config['knowledge_retrieval']['seq_len']

    # packing dialog
    dialog_conf = DialogConf()
    dialog_conf.model = model_config['dialog']['model_file']
    dialog_conf.json_file = model_config['dialog']['class_json']
    dialog_conf.seq_len = model_config['dialog']['seq_len']

    # packing score
    score_conf = ScoreConf()
    score_conf.model = model_config['score']['model_file']
    score_conf.seq_len = model_config['score']['seq_len']

    # packing chatbot
    chatbot_conf = ChatBotConf()
    chatbot_conf.mysql = mysql
    chatbot_conf.word_sequence = config_json['word_sequence']

    chatbot_conf.profile_conf = profile_conf
    chatbot_conf.semantics_conf = semantics_conf
    chatbot_conf.score_conf = score_conf
    chatbot_conf.knowledge_conf = knowledge_conf
    chatbot_conf.dialog_conf = dialog_conf

    chatbot_conf.dialog_label = category_label['dialog']
    chatbot_conf.profile_label = category_label['profile']
    chatbot_conf.history_label = category_label['history']
    chatbot_conf.science_label = category_label['science']
    chatbot_conf.food_label = category_label['food']
    chatbot_conf.joke_label = category_label['joke']
    chatbot_conf.sensitive_word_label = category_label['sensitive_word']
    chatbot_conf.knowledge_label = category_label['knowledge']
    chatbot_conf.other_label = category_label['other']

    chatbot_conf.threshold = config_json['threshold']
    chatbot_conf.debug = variable.DEBUG

    return chatbot_conf


class ChatBotConf(object):

    def __init__(self):
        self.mysql = None
        self.word_sequence = None

        self.profile_conf = ProfileConf()
        self.semantics_conf = SemanticsConf()
        self.score_conf = ScoreConf()
        self.knowledge_conf = KnowledgeConf()
        self.dialog_conf = DialogConf()
        self.any_reply_conf = None
        self.sensitive_word_conf = None

        self.threshold = None

        self.dialog_label = None
        self.profile_label = None
        self.knowledge_label = None
        self.history_label = None
        self.science_label = None
        self.food_label = None
        self.joke_label = None
        self.sensitive_word_label = None
        self.other_label = None

        self.debug = None


class ChatBot(object):

    def __init__(self, conf):
        # conf = ChatBotConf()
        self.mysql = conf.mysql
        self.word_sequence = load_word_sequences(conf.word_sequence)

        # 初始化 sensitive word 模块
        self.sensitive_word = SensitiveWordDistinguish(self.mysql, debug=conf.debug)

        # 初始化 profile 模块
        self.profile = Profile(mysql=self.mysql, word_sequence=self.word_sequence,
                               distinguish_model_file=conf.profile_conf.distinguish_model,
                               distinguish_json_file=conf.profile_conf.distinguish_json,
                               distinguish_seq_len=conf.profile_conf.distinguish_seq_len,
                               retrieval_model_file=conf.profile_conf.retrieval_model,
                               retrieval_json_file=conf.profile_conf.retrieval_json,
                               retrieval_seq_len=conf.profile_conf.retrieval_seq_len,
                               debug=conf.debug)

        # 初始化 Semantics 模块
        self.semantics = Semantics(word_sequence=self.word_sequence,
                                   distinguish_model_file=conf.semantics_conf.model,
                                   distinguish_json_file=conf.semantics_conf.distinguish_json_file,
                                   seq_len=conf.semantics_conf.seq_len,
                                   debug=conf.debug)

        # 初始化 joke 模块
        self.joke = Joke(mysql=self.mysql, debug=conf.debug)

        # 初始化 knowledge 模块
        self.knowledge = Knowledge(mysql=self.mysql, word_sequence=self.word_sequence,
                                   category_model_file=conf.knowledge_conf.category_model_file,
                                   category_json_file=conf.knowledge_conf.category_json_file,
                                   category_seq_len=conf.knowledge_conf.category_seq_len,
                                   retrieval_model_file=conf.knowledge_conf.retrieval_model_file,
                                   retrieval_json_file=conf.knowledge_conf.retrieval_json_file,
                                   retrieval_seq_len=conf.knowledge_conf.retrieval_seq_len,
                                   debug=conf.debug)

        # 初始化 dialog 模块
        self.dialog = Dialog(mysql=self.mysql, word_sequence=self.word_sequence,
                             model_file=conf.dialog_conf.model,
                             json_file=conf.dialog_conf.json_file,
                             seq_len=conf.dialog_conf.seq_len,
                             debug=conf.debug)

        # 初始化 score 模块
        self.score = Score(word_sequence=self.word_sequence,
                           score_model_file=conf.score_conf.model,
                           seq_len=conf.score_conf.seq_len,
                           debug=conf.debug)

        self.threshold = conf.threshold

        # 初始化 any_reply 模块
        self.any_reply = AnyReply(mysql=self.mysql, debug=variable.DEBUG)

        # 初始化 category label
        self.dialog_label = conf.dialog_label
        self.profile_label = conf.profile_label
        self.history_label = conf.history_label
        self.science_label = conf.science_label
        self.knowledge_label = conf.knowledge_label
        self.food_label = conf.food_label
        self.joke_label = conf.joke_label
        self.sensitive_word_label = conf.sensitive_word_label
        self.other_label = conf.other_label

        self.debug = conf.debug

    def score_threshold(self, cos_score, cos_answer, retrieval_score, retrieval_answer, label):
        if cos_score > retrieval_score:
            score = cos_score
            final_answer = cos_answer
        else:
            score = retrieval_score
            final_answer = retrieval_answer

        if score < self.threshold:
            final_answer = self.any_reply.reply(label=label)

        return score, final_answer

    def chat(self, ask):
        utterance = Utterance()
        utterance.ask = regular(ask)

        if len(utterance.ask) == 0:
            utterance.label = self.any_reply.unrecognized
            utterance.final_answer = self.any_reply.reply(label=self.any_reply.unrecognized)
            if self.debug:
                print("label: ", utterance.label)
                print("sensitive_word: ", str(utterance.final_answer))
            return utterance

        include_sensitive, sensitive_word = self.sensitive_word.distinguish(utterance.ask)
        if include_sensitive:
            utterance.label = self.sensitive_word_label
            utterance.sensitive_word = sensitive_word
            utterance.final_answer = self.any_reply.reply(label=self.sensitive_word_label)
            if self.debug:
                print("label: ", utterance.label)
                print("sensitive_word: ", str(sensitive_word))
                print("final_answer: ", utterance.final_answer)
            return utterance

        utterance.ask_keyword = extract_keyword(utterance.ask)
        utterance.label = self.profile.distinguish(utterance.ask, utterance.ask_keyword)
        if utterance.label == self.profile_label:
            mysql_retrieval = self.profile.query_mysql(label=utterance.label, ask=utterance.ask)

            if mysql_retrieval is False:
                utterance.label = self.any_reply.error_label
                utterance.final_answer = self.any_reply.reply(self.any_reply.error_label)
                return utterance

            utterance.cos_answer = self.profile.retrieval_by_cos(utterance.ask, mysql_retrieval)
            utterance.retrieval_answer = self.profile.retrieval(utterance.ask, utterance.ask_keyword, mysql_retrieval)

            cos_score = self.score.score(utterance.ask, utterance.cos_answer)
            retrieval_score = self.score.score(utterance.ask, utterance.retrieval_answer)

            score, final_answer = self.score_threshold(cos_score=cos_score,
                                                       cos_answer=utterance.cos_answer,
                                                       retrieval_score=retrieval_score,
                                                       retrieval_answer=utterance.retrieval_answer,
                                                       label=utterance.label)

            utterance.score = score
            utterance.final_answer = final_answer

            if self.debug:
                print("cos score: ", cos_score)
                print("cos answer: ", utterance.cos_answer)
                print("retrieval score: ", retrieval_score)
                print("retrieval answer: ", utterance.retrieval_answer)
                print("distinguish label: ", utterance.label)
                print()

            return utterance

        elif utterance.label == self.other_label:
            utterance.label = self.semantics.distinguish(utterance.ask, utterance.ask_keyword)

        if utterance.label == self.joke_label:
            utterance.final_answer = self.joke.joke()
            return utterance

        elif utterance.label == self.knowledge_label:
            utterance.label = self.knowledge.category(utterance.ask, utterance.ask_keyword)
            mysql_retrieval = self.knowledge.query_mysql(label=utterance.label, ask=utterance.ask)

            if mysql_retrieval is False:
                utterance.label = self.any_reply.error_label
                utterance.final_answer = self.any_reply.reply(self.any_reply.error_label)
                return utterance

            utterance.cos_answer = self.knowledge.retrieval_by_cos(ask=utterance.ask, mysql_retrieval=mysql_retrieval)
            utterance.retrieval_answer = self.knowledge.retrieval(ask=utterance.ask, ask_keyword=utterance.ask_keyword,
                                                                  mysql_retrieval=mysql_retrieval)

            cos_score = self.score.score(utterance.ask, utterance.cos_answer)
            retrieval_score = self.score.score(utterance.ask, utterance.retrieval_answer)

            score, final_answer = self.score_threshold(cos_score=cos_score,
                                                       cos_answer=utterance.cos_answer,
                                                       retrieval_score=retrieval_score,
                                                       retrieval_answer=utterance.retrieval_answer,
                                                       label=utterance.label)

            utterance.score = score
            utterance.final_answer = final_answer

            if self.debug:
                print("cos score: ", cos_score)
                print("cos answer: ", utterance.cos_answer)
                print("retrieval score: ", retrieval_score)
                print("retrieval answer: ", utterance.retrieval_answer)
                print("distinguish label: ", utterance.label)
                print()

            return utterance

        elif utterance.label == self.dialog_label:
            mysql_retrieval = self.dialog.query_mysql(label=utterance.label, ask=utterance.ask)

            if mysql_retrieval is False:
                utterance.label = self.any_reply.error_label
                utterance.final_answer = self.any_reply.reply(self.any_reply.error_label)
                return utterance

            utterance.cos_answer = self.dialog.retrieval_by_cos(ask=utterance.ask, mysql_retrieval=mysql_retrieval)
            utterance.retrieval_answer = self.dialog.retrieval(ask=utterance.ask, ask_keyword=utterance.ask_keyword,
                                                               mysql_retrieval=mysql_retrieval)

            cos_score = self.score.score(utterance.ask, utterance.cos_answer)
            retrieval_score = self.score.score(utterance.ask, utterance.retrieval_answer)

            score, final_answer = self.score_threshold(cos_score=cos_score,
                                                       cos_answer=utterance.cos_answer,
                                                       retrieval_score=retrieval_score,
                                                       retrieval_answer=utterance.retrieval_answer,
                                                       label=utterance.label)
            utterance.score = score
            utterance.final_answer = final_answer

            if self.debug:
                print("cos score: ", cos_score)
                print("cos answer: ", utterance.cos_answer)
                print("retrieval score: ", retrieval_score)
                print("retrieval answer: ", utterance.retrieval_answer)
                print("distinguish label: ", utterance.label)
                print()

            return utterance

        else:
            utterance.label = self.other_label
            utterance.final_answer = self.any_reply.reply(self.other_label)
            return utterance

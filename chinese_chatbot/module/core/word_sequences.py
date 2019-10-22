import pickle
import os
import re
import jieba
from tqdm import tqdm
from collections import Counter
from global_variable import add_word_list


def regular(sentence):
    # sentence = ''.join(re.findall('[\u4e00-\u9fa5,，.。？?！!·、{};；<>()<<>>《》0-9a-zA-Z\[\]]+', sentence))
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


def save_word_sequences(word_sequences, save_dir=None, file_name='word_sequence.pickle'):
    """
    保存 word_sequences 对象
    :param word_sequences: <WordSequences> word_sequences 对象
    :param save_dir: <str> 保存目录
    :param file_name: <str> 保存文件名
    :return:
    """
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.getcwd()

    file = os.path.join(save_dir, file_name)
    pickle.dump(word_sequences, open(file, 'wb'))

    print("save word sequences over! Path: {}".format(file))


def load_word_sequences(pickle_file):
    """
    通过 pickle_file 加载word_sequences 对象
    :param pickle_file: <str> 保存word_sequences 对象的文件名
    :return: <WordSequences> word_sequences 对象
    """
    assert os.path.exists(pickle_file), "File not exists! Path: {}".format(pickle_file)

    with open(pickle_file, 'rb') as p_file:
        word_sequences = pickle.loads(p_file.read())

    return word_sequences


def create_word_sequences(mysql, save_dir=None, file_name='word_sequence.pickle'):
    sentences = list()

    for data in mysql.query_qa():
        sentences.append(regular(data['ask']))
        sentences.append(regular(data['answer']))

    for data in mysql.query_dialog():
        sentences.append(regular(data['ask']))
        sentences.append(regular(data['answer']))

    for data in mysql.query_joke():
        sentences.append(regular(data['ask']))
        sentences.append(regular(data['answer']))

    for data in mysql.query_profile():
        sentences.append(regular(data['ask']))
        sentences.append(regular(data['answer']))

    word_sequences = WordSequences()
    word_sequences.fit(sentences)

    save_word_sequences(word_sequences, save_dir=save_dir, file_name=file_name)
    return word_sequences


def test():
    word_sequence = load_word_sequences('../../save_word_sequence/word_sequence.pickle')
    sentence = ['这是一个冷笑话', '小通魅力无人能敌', '中共革命根据地是哪里？', '隔夜茶能不能喝？', '大数据包括那些特点',
                '你是什么样的电脑']
    answer = ['隔夜茶因时间过久，维生素大多已丧失，且茶汤中的蛋白质、糖类等会成为细菌、霉菌繁殖的养料，所以，人们通常认为隔夜茶不能喝。',
              '中共革命根据地是井冈山',
              '我死了你怎么办，没人管你了。',
              '大数据特点：Volume（大量）、Velocity（高速）、Variety（多样）、Value（低价值密度）、Veracity（真实性）',
              '我在各种计算机上工作,MAC,IBM或UNIX,对我来说没关系。',
              '亲爱的小通']
    sentence_vec = word_sequence.transfroms(sentence)
    for vec in sentence_vec:
        print(vec)

    print("-----------------------------------------")
    answer_vec = word_sequence.transfroms(answer)
    for vec in answer_vec:
        print(vec)


class WordSequences(object):
    PAD = 0
    START = 1
    END = 2
    UNK = 3

    START_TAG = '<START>'
    END_TAG = '<END>'
    PAD_TAG = '<PAD>'
    UNK_TAG = '<UNK>'

    def __init__(self):
        self.word_dict = {
            self.START_TAG: self.START,
            self.END_TAG: self.END,
            self.PAD_TAG: self.PAD,
            self.UNK_TAG: self.UNK
        }

        self.fited = False

    def word_to_token(self, word):
        """
        通过单词获取 token
        :param word: <str> 单词字符
        :return: <int> token
        """
        assert self.fited, "word sequence not fit! "

        if word in self.word_dict:
            return self.word_dict[word]

        return self.UNK

    def token_to_word(self, token):
        """
        通过 token 获取单词
        :param token: <int> token
        :return: <str> 单词字符
        """
        assert self.fited, "word sequence not fit"

        for w, t in self.word_dict.items():
            if token == t:
                return w

        return self.UNK_TAG

    def fit(self, sentences, sort_by_count=False):
        """
        创建词序列
        :param sentences: <list> 句子列表
        :param sort_by_count: <bool> 是否根据句子中单词的数量进行排序。
                                     默认为 False。
        :return: 无
        """
        assert not self.fited, "word sequence fit once"
        for word in add_word_list:
            jieba.add_word(word)

        word_count = Counter()
        for sentence in sentences:
            word_count.update(jieba.lcut_for_search(sentence))

        if sort_by_count:
            sorted(word_count.items(), lambda x: x[1])

            for word, _ in word_count.items():
                self.word_dict[word] = len(self.word_dict)
        else:
            for word in sorted(word_count.keys()):
                self.word_dict[word] = len(self.word_dict)

        self.fited = True

    def transfrom(self, sentence, max_len=20):
        """
        将一个句子转换为指定大小的 token 向量
        :param sentence: <str> 句子
        :param max_len: <int> tokens 最大长度, 默认为20，超过20将被截断，小于20将用PAD填充
        :return: <list> token 向量
        """
        assert self.fited, "word sequence not fit! "
        for word in add_word_list:
            jieba.add_word(word)
        words = jieba.lcut(sentence)
        tokens = [self.word_to_token(word) for word in words]

        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        elif len(tokens) < max_len:
            for i in range(max_len - len(tokens)):
                tokens.append(self.PAD)

        return tokens

    def transfroms(self, sentences, max_len=20):
        """
        将一系列的句子转换为 token 向量
        :param sentences: <list> 句子列表
        :param max_len: <int> tokens 最大长度, 默认为20，超过20将被截断，小于20将用PAD填充
        :return: <list> token 向量
        """
        assert self.fited, "word sequence not fit! "

        tokens = list()
        for sentence in tqdm(sentences):
            tokens.append(self.transfrom(sentence, max_len))
        return tokens


if __name__ == '__main__':
    test()

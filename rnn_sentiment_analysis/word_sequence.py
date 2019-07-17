
import jieba
import numpy as np
from tips import Error_Tips
from collections import Counter

# 初始化错误提示
ERROR = Error_Tips()

class Word_Sequence(object):

    # 未发现的单词标记
    UNK_TAG = '<UNK>'
    PAD_TAG = '<PAD>'
    START_TAG = '<S>'
    END_TAG = '</S>'

    # 未发现的单词token
    UNK = 0
    PAD = 1
    START = 2
    END = 3

    def __init__(self):
        self.word_dict = {
            Word_Sequence.UNK_TAG: Word_Sequence.UNK,
            Word_Sequence.PAD_TAG: Word_Sequence.PAD,
            Word_Sequence.START_TAG: Word_Sequence.START,
            Word_Sequence.END_TAG: Word_Sequence.END
        }

        # 标记是否进行过 fit 操作
        self.fited = False

    def word_to_index(self, word):
        """
        通过单词获取标记token
        :param word: <str> 单词
        :return: 成功：返回单词 token， 失败：返回 -1
        """
        assert self.fited, ERROR.ERROR_WORD_SEQUENCE_NOT_FIT

        if word in self.word_dict:
            return self.word_dict[word]

        return Word_Sequence.UNK

    def index_to_word(self, index):
        """
        通过token获取单词
        :param index: <int> 单词令牌 token
        :return: 成功：返回单词 ， 失败：返回 '<UNK>'
        """
        assert self.fited, ERROR.ERROR_WORD_SEQUENCE_NOT_FIT

        if index < 0:
            return Word_Sequence.UNK_TAG

        for i, word in self.word_dict.items():
            if i == index:
                return word
        return Word_Sequence.UNK_TAG

    def size(self):
        return len(self.word_dict) + 1

    def __len__(self):
        return self.size()

    def fit(self, sentences, min_count=None, max_count=None, sort_by_count=False):
        """
        根据提供的句子信息进行fit操作。
        :param sentences: <list> 句子列表如：['今天天气好']
        :param min_count: <int> 保留大于 min_count 的字符
        :param max_count: <int> 保留小于 max_count 的字符
        :param sort_by_count: <bool> 是否根据统计数据中字词的数量进行排序。
                              如果 True, 则根据语料中出现字符的数量从小到大进行排序。
                              如果 False, 则默认使用字符进行排序。
        :return: 无
        """

        assert not self.fited, ERROR.ERROR_WORD_SEQUENCE_FIT_ONCE

        words_count = Counter()
        for sentence in sentences:
            words_count.update(jieba.lcut(sentence))

        if min_count is not None:
            if isinstance(min_count, int):
                # 保留 >= min_count 的字符
                words_count = {word: count for word, count in words_count.items() if count >= min_count}

        if max_count is not None:
            if isinstance(max_count, int):
                # 保留 <= max_count 的字符
                words_count = {word: count for word, count in words_count.items() if count <= max_count}

        if sort_by_count:
            sorted(words_count.items(), key=lambda x: x[1])

            for word, _ in words_count.items():
                self.word_dict[word] = len(self.word_dict)

        else:
            for word in sorted(words_count.keys()):
                self.word_dict[word] = len(self.word_dict)

        self.fited = True

    def transfrom(self, sentence, max_len=None):
        """
        将一个中文句子转换为向量
        :param sentence: <str> 中文句子
        :param max_len: <int> 获取句子的最大长度，如果句子的长度 > max_len， 将被截断。
                              如果 < max_len, 则用PAD进行填充
        :return: <list>  句子词向量
        """
        assert self.fited, ERROR.ERROR_WORD_SEQUENCE_NOT_FIT

        sentence_word = jieba.lcut(sentence)

        if max_len and isinstance(max_len, int):
            sentence_vec = [Word_Sequence.PAD] * max_len
            for i, word in enumerate(sentence_word):
                if i >= max_len:
                    break
                sentence_vec[i] = self.word_to_index(word)

        else:
            sentence_vec = [self.word_to_index(word) for word in sentence_word]

        return sentence_vec

    def transfroms(self, sentences, max_len=None):
        """
        将包含句子的数据集进行转换为向量集
        :param sentences: <list> 句子集
        :param max_len: <int> 获取句子的最大长度，如果句子的长度 > max_len， 将被截断。
                              如果 < max_len, 则用PAD进行填充
        :return: <list> 向量集
        """
        sentences_vec = list()
        for sentence in sentences:
            sentences_vec.append(self.transfrom(sentence, max_len))

        return sentences_vec

    def transfrom_word_bag(self, sentence):
        """
        将句子转换为词袋子向量
        :param sentence: <str> 句子信息
        :return: <list> 词袋子向量
        """
        # 使用jieba进行分词，并使用 Counter 进行统计.
        # 输出的格式如：{'a'：2, 'b':1} 字典模式
        # 其中 'a' 为词语或字符，2 为句子中出现的字或词的数量
        word_count = Counter(jieba.lcut(sentence))

        # 统计句子中出现词或字的数量，即将 word_count 字典对象中的 value 相加求和
        word_sum = sum(word_count.values())

        # 初始化词向量，默认都为 0
        word_vec = np.zeros(len(self.word_dict))

        for word, count in word_count.items():
            # 通过 字 或词语 获取在词库中的 标签，也叫（token）
            token = self.word_to_index(word)
            # 更加 token 的值，在初始化的 word_vec 更新值，
            # 即用单词的数量除以句子中所有词的数量
            word_vec[token] = 1.0 * count / word_sum

        return word_vec

    def transfroms_word_bag(self, sentences):
        """
        将包含句子的数据集进行转换为向量集
        :param sentences: <list> 句子集
        :return: <list> 向量集
        """
        sentences_vec = list()
        for sentence in sentences:
            vec = self.transfrom_word_bag(sentence)
            sentences_vec.append(vec)

        return sentences_vec

def test(sort_by_count=False, max_len=5):
    train_sentences = ['人工智能是工程和科学的分支,致力于构建思维的机器',
                       '你写的是什么语言',
                       '是的,我受到指挥官数据的人工个性的启发']

    test_sentence = '你是一个人工语言实体'

    word_sequence = Word_Sequence()

    word_sequence.fit(train_sentences, sort_by_count=sort_by_count)
    print("word_dict:", word_sequence.word_dict)

    # 测试转换：
    sentence_vec = word_sequence.transfrom(test_sentence, max_len=max_len)
    print("transfrom:", sentence_vec)

    # 测试转换为 word_bag 向量
    sentence_vec = word_sequence.transfrom_word_bag(test_sentence)
    print("transfrom word bag: ", sentence_vec)

if __name__ == '__main__':
    test(sort_by_count=False)
    #test(sort_by_count=True)
    #test(sort_by_count=False, max_len=10)
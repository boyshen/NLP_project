import jieba


class SensitiveConf(object):

    def __init__(self):
        self.mysql = None
        self.debug = None


class SensitiveWordDistinguish(object):

    def __init__(self, mysql, debug=False):
        self.mysql = mysql

        words_list = list()
        for data in mysql.query_sensitive_word():
            jieba.add_word(data['word'])
            words_list.append(data['word'])

        self.jieba = jieba
        self.word_list = words_list
        self.debug = debug

    def distinguish(self, ask):
        words = self.jieba.lcut(ask)

        sensitive_word = [word for word in words if word in self.word_list]
        include_sensitive = False if len(sensitive_word) == 0 else True

        return include_sensitive, sensitive_word


def test(test_sentence):
    from module.core.mysql_exec import Mysql
    from module.core.utterance import Utterance

    utterance = Utterance(ask=test_sentence)

    mysql = Mysql(host='192.168.10.10', user='chatbot', password='chatbot', db='chatbot')
    sensitive_word_distinguish = SensitiveWordDistinguish(mysql)
    result = sensitive_word_distinguish.distinguish(utterance)

    print("result: ", result.sensitive_word)
    print("sentence: ", test_sentence)


if __name__ == '__main__':
    test('气枪非常好')
    print()
    test('你好')

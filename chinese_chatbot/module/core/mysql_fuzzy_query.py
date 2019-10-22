from jieba import posseg as pseg

save_word = ['a', 'n', 'an', 'v', 'vn', 'b', 'm', 't', 'l', 'i', 'eng']


def extract_keyword(sentence):
    words = list()
    for word, flag in pseg.cut(sentence):
        if flag in save_word or flag[:1] in save_word:
            words.append(word)

    keyword = ','.join(words)
    return keyword


def fuzzy_query_by_sentence(mysql, table, label, sentence, max_num=None):
    fuzzy_list = list()

    sql = 'SELECT * FROM {} WHERE label="{}"'.format(table, label)
    if len(mysql.query_by_sql(sql)) == 0:
        print("Error: mysql {} table not found data! ".format(table))
        return False

    if len(sentence) <= 2:
        fuzzy_str = '%'.join(list(sentence))
        fuzzy_list.append(fuzzy_str)

    else:
        k_word = extract_keyword(sentence).split(',')
        keyword = k_word if len(k_word) != 0 else list(sentence)

        if len(keyword) <= 2:
            fuzzy_str = '%'.join(keyword)
            fuzzy_list.append(fuzzy_str)

        else:
            for w1, w2 in zip(keyword[:-1], keyword[1:]):
                fuzzy_str = '%'.join([w1, w2])
                fuzzy_list.append(fuzzy_str)

    result = list()
    answer = list()
    for fuzzy_str in fuzzy_list:
        fuzzy_str = '%' + fuzzy_str + '%'
        sql = 'SELECT ask,answer,answer_keyword FROM {} WHERE label="{}" AND ask LIKE "{}"'.format(table,
                                                                                                   label,
                                                                                                   fuzzy_str)

        mysql_data = mysql.query_by_sql(sql)
        for data in mysql_data:
            if data['answer'] in answer:
                continue

            result.append(data)
            answer.append(data['answer'])

    if len(result) == 0:
        sql = 'SELECT ask,answer,answer_keyword FROM {} WHERE label="{}"'.format(table, label)
        result = mysql.query_by_sql(sql)

    if max_num is not None:
        if len(result) > max_num:
            result = result[:max_num]

    return result


def test():
    from module.core.mysql_exec import Mysql
    mysql = Mysql(host='192.168.10.10', user='chatbot', password='chatbot', db='chatbot')
    data = ['你好']

    for d in data:
        result = fuzzy_query_by_sentence(mysql, 'profile', 'profile', d)
        print("length: ", len(result))
        print(result)
        print()


if __name__ == '__main__':
    test()

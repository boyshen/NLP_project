import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from jieba import posseg as pseg
from module.core.thread import ReadThread
from global_variable import DATA_TYPE_SENSITIVE_WORD, DATA_TYPE_PROFILE, DATA_TYPE_DIALOG, DATA_TYPE_JOKE, DATA_TYPE_QA
from global_variable import DATA_TYPE_ANY_REPLY
from module.core.third_library.langconv import Converter

# save word pseg
save_word = ['n', 'v', 'a', 'b', 'c', 'd', 'f', 'm', 'r', 'q', 't', 'u', 'z', 'l', 'i', 'eng']

# split retrieval dataset label
retrieval_dataset_label = {'correct': 1, 'error': 0}


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


def extract_keyword(sentence):
    words = list()
    for word, flag in pseg.cut(sentence):
        if flag in save_word or flag[:1] in save_word:
            words.append(word)

    keyword = ','.join(words)
    return keyword


def conversion_word_simplified(sentences):
    result = None
    conversion = Converter('zh-hans')
    if isinstance(sentences, list):
        result = list()
        for sentence in list(sentences):
            words = [conversion.convert(w) for w in list(sentence)]
            sent = ''.join(words)
            result.append(sent)

    elif isinstance(sentences, str):
        words = [conversion.convert(w) for w in list(sentences)]
        result = ''.join(words)

    return result


def read_sensitive_word(data_file, split_tag=',', conversion_simplified=True):
    assert os.path.isfile(data_file), "File: {} not exists".format(data_file)

    words = list()
    with open(data_file, 'rb') as rf:
        for line in rf:
            line = str(line, encoding='utf8').strip("\n")
            words = words + line.split(split_tag)

    if conversion_simplified:
        words = conversion_word_simplified(words)

    word_list = list()
    for word in words:
        if word == '' or word == " " or len(word) == 0:
            continue

        word_dict = dict()
        word_dict['word'] = word
        word_list.append(word_dict)

    return word_list


def read_any_reply(file, fields):
    assert os.path.isfile(file), "File: {} not exists".format(file)

    any_reply = list()
    with open(file, 'r') as rf:
        f_data = json.load(rf)

    dataset = [f_data[field] for field in fields]

    for data in dataset:
        for answer in data['answer']:
            value = {'category': data['category'], 'label': data['label'], 'answer': answer}
            any_reply.append(value)

    return any_reply


def read_file_data(data_file, ask_tag, answer_tag, start_tag=None, end_tag=None, skip_line=None, use_keyword=False):
    assert os.path.exists(data_file), "File: {} not exists".format(data_file)
    assert ask_tag is not None and answer_tag is not None, "ask_tag and answer_tag cannot be None"
    if start_tag is not None and end_tag is not None:
        assert start_tag != end_tag, 'start_tag and end_tag cannot be the same'

    if skip_line is not None:
        assert isinstance(skip_line, list), "skip_line type must is list"

    if end_tag is not None:
        assert start_tag is not None, "When using end_tag , start_tag、human_tag and chatbot_tag cannot be None"

    data_dict = {}
    data_list = []
    did = 0
    order_id = 0

    if ask_tag == answer_tag:
        count = 0

    if start_tag is not None and end_tag is None:
        s_count = 0

    with open(data_file, 'rb') as f_read:
        for i, line in tqdm(enumerate(f_read)):
            line = str(line, encoding='utf8').strip("\n")

            if skip_line is not None and line.strip() in skip_line:
                continue

            if start_tag is not None and end_tag is not None:

                if line.strip() == start_tag:
                    did += 1
                    continue

                if line.strip() == end_tag:
                    order_id = 0
                    continue

                if ask_tag == answer_tag:
                    if count == 0:
                        order_id += 1
                        data_dict['did'] = did
                        data_dict['order_id'] = order_id
                        ask = regular(line[len(ask_tag):].strip())
                        data_dict['ask'] = ask
                        if use_keyword:
                            data_dict['ask_keyword'] = extract_keyword(ask)
                        count = 1
                        continue

                    if count == 1:
                        answer = regular(line[len(answer_tag):].strip())
                        data_dict['answer'] = answer
                        if use_keyword:
                            data_dict['answer_keyword'] = extract_keyword(answer)

                        data_list.append(data_dict)
                        data_dict = {}
                        count = 0
                        continue
                else:
                    if line[:len(ask_tag)] == ask_tag:
                        order_id += 1
                        data_dict['did'] = did
                        data_dict['order_id'] = order_id
                        # data_dict['ask'] = regular(line[len(ask_tag):].strip())
                        ask = regular(line[len(ask_tag):].strip())
                        data_dict['ask'] = ask
                        if use_keyword:
                            data_dict['ask_keyword'] = extract_keyword(ask)
                        continue

                    if line[:len(answer_tag)] == answer_tag:
                        # data_dict['answer'] = regular(line[len(answer_tag):].strip())
                        answer = regular(line[len(answer_tag):].strip())
                        data_dict['answer'] = answer
                        if use_keyword:
                            data_dict['answer_keyword'] = extract_keyword(answer)

                        data_list.append(data_dict)
                        data_dict = {}
                        continue

            if start_tag is not None and end_tag is None:
                if s_count == 0 and line.strip() == start_tag:
                    did += 1
                    order_id = 0
                    s_count = 1
                    continue

                if s_count == 1 and line.strip() == start_tag:
                    did += 1
                    order_id = 0
                    s_count = 0
                    continue

                if ask_tag == answer_tag:
                    if count == 0:
                        order_id += 1
                        data_dict['did'] = did
                        data_dict['order_id'] = order_id
                        # data_dict['ask'] = regular(line[:len(ask_tag)].strip())
                        ask = regular(line[len(ask_tag):].strip())
                        data_dict['ask'] = ask
                        if use_keyword:
                            data_dict['ask_keyword'] = extract_keyword(ask)

                        count = 1
                        continue

                    if count == 1:
                        # data_dict['answer'] = regular(line[:len(answer_tag)].strip())
                        answer = regular(line[len(answer_tag):].strip())
                        data_dict['answer'] = answer
                        if use_keyword:
                            data_dict['answer_keyword'] = extract_keyword(answer)

                        data_list.append(data_dict)
                        data_dict = {}
                        count = 0
                        continue
                else:
                    if line[:len(ask_tag)] == ask_tag:
                        order_id += 1
                        data_dict['did'] = did
                        data_dict['order_id'] = order_id
                        # data_dict['ask'] = regular(line[len(ask_tag):].strip())
                        ask = regular(line[len(ask_tag):].strip())
                        data_dict['ask'] = ask
                        if use_keyword:
                            data_dict['ask_keyword'] = extract_keyword(ask)
                        continue

                    if line[:len(answer_tag)] == answer_tag:
                        # data_dict['answer'] = regular(line[len(answer_tag):].strip())
                        answer = regular(line[len(answer_tag):].strip())
                        data_dict['answer'] = answer
                        if use_keyword:
                            data_dict['answer_keyword'] = extract_keyword(answer)

                        data_list.append(data_dict)
                        data_dict = {}
                        continue

            if start_tag is None and end_tag is None:
                if ask_tag == answer_tag:
                    if count == 0:
                        order_id = 1
                        did += 1
                        data_dict['did'] = did
                        data_dict['order_id'] = order_id
                        # data_dict['ask'] = regular(line[len(ask_tag):].strip())
                        ask = regular(line[len(ask_tag):].strip())
                        data_dict['ask'] = ask
                        if use_keyword:
                            data_dict['ask_keyword'] = extract_keyword(ask)
                        count = 1
                        continue
                    if count == 1:
                        # data_dict['answer'] = regular(line[len(answer_tag):].strip())
                        answer = regular(line[len(answer_tag):].strip())
                        data_dict['answer'] = answer
                        if use_keyword:
                            data_dict['answer_keyword'] = extract_keyword(answer)

                        data_list.append(data_dict)
                        data_dict = {}
                        count = 0
                        continue
                else:
                    if line[:len(ask_tag)] == ask_tag:
                        order_id = 1
                        did += 1
                        data_dict['did'] = did
                        data_dict['order_id'] = order_id
                        # data_dict['ask'] = regular(line[len(ask_tag):].strip())
                        ask = regular(line[len(ask_tag):].strip())
                        data_dict['ask'] = ask
                        if use_keyword:
                            data_dict['ask_keyword'] = extract_keyword(ask)
                        continue
                    if line[:len(answer_tag)] == answer_tag:
                        # data_dict['answer'] = regular(line[len(answer_tag):].strip())
                        answer = regular(line[len(answer_tag):].strip())
                        data_dict['answer'] = answer
                        if use_keyword:
                            data_dict['answer_keyword'] = extract_keyword(answer)

                        data_list.append(data_dict)
                        data_dict = {}
                        continue

    return data_list


def read_json_data(json_file, category_field, ask_field, answer_field, describe_field=None):
    assert os.path.isfile(json_file), "json file not exists! File: {}".format(json_file)

    data_list = list()

    with open(json_file, 'rb') as read_json:
        for line in tqdm(read_json):
            data_dict = dict()
            line = str(line, encoding='utf8')
            json_data = json.loads(line)
            data_dict['category'] = json_data[category_field]

            if describe_field is not None:
                data_dict['desc'] = regular(json_data[describe_field]).strip()

            ask = regular(json_data[ask_field]).strip()
            data_dict['ask_keyword'] = extract_keyword(ask)
            data_dict['ask'] = ask

            answer = regular(json_data[answer_field]).strip()
            data_dict['answer_keyword'] = extract_keyword(answer)
            data_dict['answer'] = answer

            data_list.append(data_dict)

    return data_list


def check_data_is_exists(mysql_record, data_list, data_type=DATA_TYPE_QA):
    save_data = list()
    if data_type in [DATA_TYPE_QA, DATA_TYPE_JOKE, DATA_TYPE_DIALOG, DATA_TYPE_PROFILE]:
        ask_data, answer_data = list(), list()
        if len(mysql_record) != 0:
            for record in mysql_record:
                ask_data.append(record['ask'])
                answer_data.append(record['answer'])

        if len(ask_data) != 0 and len(answer_data) != 0:
            for data in tqdm(data_list):
                ask, answer = data['ask'], data['answer']
                if ask in ask_data \
                        and answer in answer_data \
                        and ask_data.index(ask) == answer_data.index(answer):
                    print("{}/{} already exists in the table ".format(ask, answer))
                    continue
                save_data.append(data)
        else:
            save_data = data_list

    elif data_type == DATA_TYPE_SENSITIVE_WORD:
        if len(mysql_record) != 0:
            words = [record['word'] for record in mysql_record]

            for data in tqdm(data_list):
                if data['word'] in words:
                    print("{} already exists in the table ".format(data['word']))
                    continue
                save_data.append(data['word'])
        else:
            save_data = data_list

    elif data_type == DATA_TYPE_ANY_REPLY:
        if len(mysql_record) != 0:
            any_reply = [reply['answer'] for reply in mysql_record]

            for data in tqdm(data_list):
                if data['answer'] in any_reply:
                    print("{} already exists in the table ".format(data['answer']))
                    continue
                save_data.append(data['answer'])
        else:
            save_data = data_list

    return save_data


def write_data_to_mysql(mysql, data_list, data_type):
    fail_data = list()
    if data_type == DATA_TYPE_DIALOG:
        mysql_record = mysql.query_dialog()
        last_record = mysql.query_dialog_last_one()
        last_did = 0 if len(last_record) == 0 else last_record[0]['cid']
        save_data = check_data_is_exists(mysql_record, data_list, data_type=DATA_TYPE_DIALOG)

        for data in tqdm(save_data):
            value = mysql.insert_dialog_data(category=data['category'],
                                             label=data['label'],
                                             ask=data['ask'],
                                             answer=data['answer'],
                                             ask_keyword=data['ask_keyword'],
                                             answer_keyword=data['answer_keyword'],
                                             order_id=data['order_id'],
                                             cid=data['did'] + last_did)
            if not value:
                fail_data.append(data)

        mysql.delete_duplicate_from_dialog()

    elif data_type == DATA_TYPE_JOKE:
        mysql_record = mysql.query_joke()
        save_data = check_data_is_exists(mysql_record, data_list, data_type=DATA_TYPE_JOKE)
        for data in tqdm(save_data):
            value = mysql.insert_joke_data(category=data['category'],
                                           label=data['label'],
                                           ask=data['ask'],
                                           answer=data['answer'],
                                           ask_keyword=data['ask_keyword'],
                                           answer_keyword=data['answer_keyword'])

            if not value:
                fail_data.append(data)

        mysql.delete_duplicate_from_joke()

    elif data_type == DATA_TYPE_QA:
        mysql_record = mysql.query_qa()
        save_data = check_data_is_exists(mysql_record, data_list, data_type=DATA_TYPE_QA)
        for data in tqdm(save_data):
            value = mysql.insert_qa_data(category=data['category'],
                                         label=data['label'],
                                         ask=data['ask'],
                                         answer=data['answer'],
                                         ask_keyword=data['ask_keyword'],
                                         answer_keyword=data['answer_keyword'])

            if not value:
                fail_data.append(data)

        mysql.delete_duplicate_from_qa()

    elif data_type == DATA_TYPE_PROFILE:
        mysql_record = mysql.query_profile()
        save_data = check_data_is_exists(mysql_record, data_list, data_type=DATA_TYPE_PROFILE)
        for data in tqdm(save_data):
            value = mysql.insert_profile_data(category=data['category'],
                                              label=data['label'],
                                              ask=data['ask'],
                                              answer=data['answer'],
                                              ask_keyword=data['ask_keyword'],
                                              answer_keyword=data['answer_keyword'])

            if not value:
                fail_data.append(data)

        mysql.delete_duplicate_from_profile()

    elif data_type == DATA_TYPE_SENSITIVE_WORD:
        mysql_record = mysql.query_sensitive_word()
        save_data = check_data_is_exists(mysql_record, data_list, data_type=DATA_TYPE_SENSITIVE_WORD)
        for data in tqdm(save_data):
            value = mysql.insert_sensitive_word(category=data['category'], label=data['label'], word=data['word'])

            if not value:
                fail_data.append(data)

        mysql.delete_duplicate_from_sensitive_word()

    elif data_type == DATA_TYPE_ANY_REPLY:
        mysql_record = mysql.query_any_reply()
        save_data = check_data_is_exists(mysql_record, data_list, data_type=DATA_TYPE_ANY_REPLY)
        for data in tqdm(save_data):
            value = mysql.insert_any_reply(category=data['category'], label=data['label'], answer=data['answer'])

            if not value:
                fail_data.append(data)

        mysql.delete_duplicate_from_any_reply()

    if len(fail_data) != 0:
        for sentence in fail_data:
            print("insert fail: ", sentence)

    print("Write data to mysql over!")


def shuffle(dialogs):
    random_idx = np.random.permutation(len(dialogs))
    dialogs = [dialogs[i] for i in random_idx]

    return dialogs


def split_dataset(dataset, valid_size=0.1, test_size=0.1):
    if valid_size is None:
        valid_size = 0.1

    if test_size is None:
        test_size = 0.1

    dataset = shuffle(dataset)

    test_length = int(len(dataset) * test_size)
    test_dataset = dataset[:test_length]
    train_dataset = dataset[test_length:]

    valid_length = int(len(train_dataset) * valid_size)
    valid_dataset = train_dataset[:valid_length]
    train_dataset = train_dataset[valid_length:]
    return train_dataset, valid_dataset, test_dataset


def save_csv(dataset, columns, save_path=None, filename='dataset.csv'):
    data_dict = dict()
    for key in dataset[0].keys():
        data_dict[key] = []

    for data in dataset:
        for k, v in data.items():
            data_dict[k].append(v)

    df = pd.DataFrame(data_dict, columns=columns)
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()

    save_file = os.path.join(save_path, filename)
    df.to_csv(save_file)

    print("save csv file success! File: {}".format(save_file))


def split_category_dataset_by_label_field(mysql, table, valid_size=0.1, test_size=0.1, save_path=None):
    train_dataset = list()
    valid_dataset = list()
    test_dataset = list()

    labels = mysql.query_label_num_by_table(table)

    if len(labels) == 0:
        print("Error: mysql {} table not found data".format(table))
        return False

    class_id = {label['label']: i for i, label in enumerate(labels)}

    for c_label, c_id in class_id.items():
        mysql_data = mysql.query_ask_by_label_field(table, c_label)

        dataset = list()
        for data in mysql_data:
            data['class_id'] = c_id
            data['class'] = c_label
            dataset.append(data)

        train_data, valid_data, test_data = split_dataset(dataset, valid_size, test_size)
        train_dataset = train_dataset + train_data
        valid_dataset = valid_dataset + valid_data
        test_dataset = test_dataset + test_data

    columns = ['category', 'label', 'ask', 'ask_keyword', 'class_id', 'class']
    train_dataset = shuffle(train_dataset)
    valid_dataset = shuffle(valid_dataset)
    test_dataset = shuffle(test_dataset)
    save_csv(train_dataset, columns=columns, filename='train.csv', save_path=save_path)
    save_csv(valid_dataset, columns=columns, filename='valid.csv', save_path=save_path)
    save_csv(test_dataset, columns=columns, filename='test.csv', save_path=save_path)

    with open(os.path.join(save_path, 'class.json'), 'w') as wf:
        json.dump(class_id, wf, ensure_ascii=False)

    train_csv = os.path.join(save_path, 'train.csv')
    valid_csv = os.path.join(save_path, 'valid.csv')
    test_csv = os.path.join(save_path, 'test.csv')

    return train_csv, valid_csv, test_csv


def split_category_dataset_by_table(mysql, tables, n_class, valid_size=0.1, test_size=0.1, save_path=None):
    assert isinstance(tables, list), "id_table must is list type"
    assert len(tables) != 0, "tables can not be empty! "
    assert len(tables) >= 2, "Must have more than two list values"
    assert len(n_class) == len(tables), "tables and n_class quantity must be the same "
    for value in tables:
        assert isinstance(value, list), "The list must have a list of values"

    train_dataset = list()
    valid_dataset = list()
    test_dataset = list()

    class_id = {}
    for i, (table, class_label) in enumerate(zip(tables, n_class)):
        class_id[class_label] = i
        for t in table:
            mysql_data = mysql.query_ask_by_table(t)

            if len(mysql_data) == 0:
                print("Warning: mysql {} table not found data! ".format(t))

            dataset = list()
            for data in mysql_data:
                data['class_id'] = i
                data['class'] = class_label
                dataset.append(data)

            train_data, valid_data, test_data = split_dataset(dataset, valid_size, test_size)

            train_dataset = train_dataset + train_data
            valid_dataset = valid_dataset + valid_data
            test_dataset = test_dataset + test_data

    columns = ['category', 'label', 'ask', 'ask_keyword', 'class_id', 'class']
    train_dataset = shuffle(train_dataset)
    valid_dataset = shuffle(valid_dataset)
    test_dataset = shuffle(test_dataset)
    save_csv(train_dataset, columns=columns, filename='train.csv', save_path=save_path)
    save_csv(valid_dataset, columns=columns, filename='valid.csv', save_path=save_path)
    save_csv(test_dataset, columns=columns, filename='test.csv', save_path=save_path)

    with open(os.path.join(save_path, 'class.json'), 'w') as wf:
        json.dump(class_id, wf, ensure_ascii=False)

    train_csv = os.path.join(save_path, 'train.csv')
    valid_csv = os.path.join(save_path, 'valid.csv')
    test_csv = os.path.join(save_path, 'test.csv')

    return train_csv, valid_csv, test_csv


def create_correct_and_error_dataset(dataset, error_ratio=5):
    error_dataset, correct_dataset = list(), list()
    dataset = np.array(dataset)
    for i, data in tqdm(enumerate(dataset)):
        correct_data = dict()
        correct_data['category'] = data['category']
        correct_data['label'] = data['label']
        correct_data['ask'] = data['ask']
        correct_data['ask_keyword'] = data['ask_keyword']
        correct_data['answer'] = data['answer']
        correct_data['answer_keyword'] = data['answer_keyword']
        correct_data['class_id'] = retrieval_dataset_label['correct']
        correct_dataset.append(correct_data)

        batch_data = np.delete(dataset, i)
        batch_data = np.random.choice(batch_data, error_ratio)

        for b_data in batch_data:
            error_data = dict()
            error_data['category'] = data['category']
            error_data['label'] = data['label']
            error_data['ask'] = data['ask']
            error_data['ask_keyword'] = data['ask_keyword']
            error_data['answer'] = b_data['answer']
            error_data['answer_keyword'] = b_data['answer_keyword']
            error_data['class_id'] = retrieval_dataset_label['error']
            error_dataset.append(error_data)

    result = (correct_dataset, error_dataset)
    return result


def split_retrieval_dataset_by_table(mysql, table, valid_size=0.1, test_size=0.1, error_ratio=5, save_path=None,
                                     thread_num=1):
    if isinstance(table, str):
        dataset = mysql.query_corpus_by_table(table)
    elif isinstance(table, list):
        dataset = list()
        for t in table:
            dataset = dataset + mysql.query_corpus_by_table(t)

    batch = int(len(dataset) / thread_num)
    thread_pool = list()
    for i in range(thread_num):
        if i == (thread_num - 1):
            batch_data = dataset[batch * i:]
        else:
            batch_data = dataset[batch * i: batch * (i + 1)]

        r_thread = ReadThread('r_thread', i, create_correct_and_error_dataset, batch_data, error_ratio)
        thread_pool.append(r_thread)

    for r_thread in thread_pool:
        r_thread.start()

    for r_thread in thread_pool:
        r_thread.join()

    correct_dataset, error_dataset = list(), list()
    for r_thread in thread_pool:
        result = r_thread.get_result()
        correct_dataset = correct_dataset + result[0]
        error_dataset = error_dataset + result[1]

    correct_train, correct_valid, correct_test = split_dataset(correct_dataset, valid_size=valid_size,
                                                               test_size=test_size)

    error_train, error_valid, error_test = split_dataset(error_dataset, valid_size=valid_size,
                                                         test_size=test_size)

    train_dataset = correct_train + error_train[:len(correct_train)]
    valid_dataset = correct_valid + error_valid
    test_dataset = correct_test + error_test

    columns = ['category', 'label', 'ask_keyword', 'answer_keyword', 'ask', 'answer', 'class_id']
    save_csv(shuffle(train_dataset), columns, save_path, 'train.csv')
    save_csv(shuffle(valid_dataset), columns, save_path, 'valid.csv')
    save_csv(shuffle(test_dataset), columns, save_path, 'test.csv')

    train_csv = os.path.join(save_path, 'train.csv')
    valid_csv = os.path.join(save_path, 'valid.csv')
    test_csv = os.path.join(save_path, 'test.csv')

    with open(os.path.join(save_path, 'class.json'), 'w') as wf:
        json.dump(retrieval_dataset_label, wf, ensure_ascii=False)

    return train_csv, valid_csv, test_csv


def read_retrieval_dataset(dataset, category, record, error_ratio=5):
    correct_data_list = list()
    error_data_list = list()

    dataset = np.array(dataset)
    for i, data in tqdm(enumerate(dataset)):
        correct_data = dict()
        correct_data['category'] = category['c_table']
        correct_data['ask'] = data['ask']
        correct_data['ask_keyword'] = data['ask_keyword']
        correct_data['answer'] = data['answer']
        correct_data['answer_keyword'] = data['answer_keyword']
        correct_data['label'] = record['correct']
        correct_data_list.append(correct_data)

        batch_data = np.delete(dataset, i)
        batch_data = np.random.choice(batch_data, error_ratio)

        for b_data in batch_data:
            error_data = dict()
            error_data['category'] = category['c_table']
            error_data['ask'] = data['ask']
            error_data['ask_keyword'] = data['ask_keyword']
            error_data['answer'] = b_data['answer']
            error_data['answer_keyword'] = b_data['answer_keyword']
            error_data['label'] = record['error']
            error_data_list.append(error_data)

    result = (correct_data_list, error_data_list)

    return result


def split_generate_dataset(mysql, valid_size=0.1, test_size=0.1, save_path=None):
    dataset = list()
    for data in mysql.query_all_dialog():
        dialog = dict()
        dialog['ask'] = data['ask']
        dialog['answer'] = data['answer']
        dataset.append(dialog)

    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, valid_size=valid_size, test_size=test_size)
    columns = ['ask', 'answer']
    save_csv(shuffle(train_dataset), columns, save_path, 'train.csv')
    save_csv(shuffle(valid_dataset), columns, save_path, 'valid.csv')
    save_csv(shuffle(test_dataset), columns, save_path, 'test.csv')


def split_retrieval_dataset(mysql, valid_size=0.1, test_size=0.1, save_path=None, error_ratio=5, thread_num=2):
    train_dataset = list()
    valid_dataset = list()
    test_dataset = list()

    record = {'correct': 1, 'error': 0}

    for category in mysql.query_all_category():
        if len(category) == 0:
            print("category table not found data!")
            return False

        dataset = mysql.query_qa_ask_answer_keyword(category['c_table'])
        print("{} dataset size : {}".format(category['category'], len(dataset)))

        batch = int(len(dataset) / thread_num)
        thread_pool = list()
        for i in range(thread_num):
            if i == (thread_num - 1):
                batch_data = dataset[batch * i:]
            else:
                batch_data = dataset[batch * i: batch * (i + 1)]

            r_thread = ReadThread('read_thread', i, read_retrieval_dataset, batch_data, category, record, error_ratio)
            thread_pool.append(r_thread)

        for r_thread in thread_pool:
            r_thread.start()

        for r_thread in thread_pool:
            r_thread.join()

        correct_data_list, error_data_list = list(), list()
        for r_thread in thread_pool:
            (correct_data, error_data) = r_thread.get_result()
            correct_data_list = correct_data_list + correct_data
            error_data_list = error_data_list + error_data

        train_correct_data, valid_correct_data, test_correct_data = split_dataset(correct_data_list,
                                                                                  valid_size=valid_size,
                                                                                  test_size=test_size)

        train_error_data, valid_error_data, test_error_data = split_dataset(error_data_list,
                                                                            valid_size=valid_size,
                                                                            test_size=test_size)

        train_dataset = train_dataset + train_correct_data + train_error_data[:len(train_correct_data)]
        valid_dataset = valid_dataset + valid_correct_data + valid_error_data
        test_dataset = test_dataset + test_correct_data + test_error_data

    columns = ['category', 'label', 'ask_keyword', 'answer_keyword', 'ask', 'answer']
    new_save_path = save_path
    save_csv(shuffle(train_dataset), columns, new_save_path, 'train.csv')
    save_csv(shuffle(valid_dataset), columns, new_save_path, 'valid.csv')
    save_csv(shuffle(test_dataset), columns, new_save_path, 'test.csv')

    record_file = os.path.join(new_save_path, 'label.json')
    with open(record_file, 'w') as wf:
        json.dump(record, wf, ensure_ascii=False)


def test():
    from module.core.mysql_exec import Mysql
    mysql = Mysql('192.168.0.103', 'chatbot', 'chatbot', 'chatbot')
    split_retrieval_dataset_by_table(mysql, 'profile', valid_size=0.1, test_size=0.1, error_ratio=5, save_path='./',
                                     thread_num=1)


if __name__ == '__main__':
    test()

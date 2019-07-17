
import re
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from word_sequence import Word_Sequence
from tips import Error_Tips
from tqdm import tqdm

ERROR = Error_Tips()

def regular(sentence):
    """
    正则化句子，将句子的符合进行替换处理
    :param sen: <str> 句子数据
    :return: <str> 处理后的句子
    """
    sentence = re.sub('[/,/，]{1,100}', '，', sentence)
    sentence = re.sub('[?？]{1,100}', '？', sentence)
    sentence = re.sub('[!！]{1,100}', '！', sentence)
    sentence = re.sub('[\.\。]{1,100}', '。', sentence)
    sentence = re.sub('[*]{1,100}', '', sentence)
    sentence = re.sub('[&hellip;]', '', sentence)
    sentence = re.sub('[～]{1,100}', '～', sentence)
    sentence = re.sub('[。]{1,100}', '。', sentence)
    sentence = re.sub('[" "]{1,100}', ' ', sentence)

    return sentence

def extract_sentences_from_file(file):
    """
    从文件中提取句子信息
    :param file: <str> 可读的文件
    :return: <list> 句子列表信息
    """
    assert os.path.exists(file), ERROR.Error_File_not_found(file)

    sentences = list()
    with open(file, 'rb') as fopen:
        for fline in tqdm(fopen):
            # 读取每行句子，并将字节转换为字符，并跳过换行符
            line = str(fline, encoding='utf8').strip()

            # 如果读到的行是空格，则重新读取
            if line == '':
                continue

            # 对每个句子进行正则化操作
            line = regular(line)
            sentences.append(line)

    # 去掉重复的句子, 同时遍历查看是否有空字符
    sent = list()
    for sentence in set(sentences):
        if sentence is '':
            continue
        sent.append(sentence)

    print("read: {} lines of sentences! ".format(len(sent)))
    return sent

def split_train_test(dataset, labels, test_size, valid_size=None):
    """
    将数据集划分为 train，test，valid
    :param dataset:  <list> 待划分的数据集
    :param labels: <list> 数据集标签
    :param test_size: <float> 划分测试数据集的比例 [0.0 ~ 1.0] 之间
    :param valid_size: <float> 划分验证数据集的比例 [0.0 ~ 1.0] 之间
    :return: 如果 valid_size 不为None, 则返回X_train, Y_train, X_valid, Y_valid, X_test, Y_test
             否则 X_train, Y_train, X_test, Y_test
    """
    assert len(dataset) == len(labels), \
        ERROR.Error_Length_Same(
            {'dataset': len(dataset)}, {'labels': len(labels)}
        )

    assert test_size < 1.0 and test_size > 0.0, \
        ERROR.Error_Range_of_value('test_size', ERROR.Value_Range)

    if valid_size:
        assert valid_size < 1.0 and valid_size > 0.0, \
            ERROR.Error_Range_of_value('test_size', ERROR.Value_Range)

    # 对数据集进行洗牌操作，重新获取随机数据
    indices = np.random.permutation(len(dataset))
    dataset = [dataset[i] for i in indices]
    labels = [labels[i] for i in indices]

    # 对数据集进行划分
    test_length = int(test_size * len(dataset))
    X_test = dataset[:test_length]
    Y_test = labels[:test_length]
    X_train = dataset[test_length:]
    Y_train = labels[test_length:]

    print("train dataset length: {}".format(len(X_train)))
    print("test dataset length: {}".format(len(X_test)))

    if valid_size:
        valid_length = int(valid_size * len(X_train))
        X_valid = X_train[:valid_length]
        Y_valid = Y_train[:valid_length]

        X_train = X_train[valid_length:]
        Y_train = Y_train[valid_length:]

        print("valid dataset length: {}".format(len(X_valid)))
        return (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    return (X_train, Y_train, X_test, Y_test)

def pick_dataloader(dataset, labels, batch_size=32, shuffle=False, data_type='lstm'):
    """
    将数据集包装成dataloader 对象。dataloader对象是pytorch 中常用的数据集操作对象。
    :param dataset: <list> 数据集
    :param labels: <list> 标签
    :param batch_size: <int> 训练时的样本
    :param shuffle: <bool> 是否进行洗牌操作
    :param data_type: <str> LSTM 和 FNN 指定包装数据的类型。
                            LSTM 指定的数据类型为 Long, FNN 中指定的数据类型为 float。
                            label 默认为long类型
                            （FNN）即前馈神经网络
    :return: Dataloader 对象
    """
    assert len(dataset) == len(labels), \
        ERROR.Error_Length_Same(
            {'dataset': len(dataset)}, {'labels': len(labels)}
        )

    assert data_type.lower() in ['lstm', 'fnn'], \
        ERROR.Error_Range_of_value('data_type', '[LSTM, FNN]')

    if data_type.lower() == 'lstm':
        data_tensor = torch.LongTensor(dataset)
    elif data_type.lower() == 'fnn':
        data_tensor = torch.FloatTensor(dataset)

    label_tensor = torch.LongTensor(labels)

    dataset_tensor = TensorDataset(data_tensor, label_tensor)
    data_loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=shuffle)

    return data_loader

def test(print_sent=False, print_vec=False, print_datalabel=False):
    bad_sentences = extract_sentences_from_file('data/bad.txt')
    good_sentences = extract_sentences_from_file('data/good.txt')

    # 打印输出提取的句子信息
    if print_sent:
        for i, (bad_sen, good_sen) in enumerate(zip(bad_sentences, good_sentences)):
            print('-----------------{}-----------------'.format(i))
            print("bad sentences: {}".format(bad_sen))
            print("good sentences: {}".format(good_sen))
            print()

            if i == 3:
                break

    # 进行fit操作，建立词库
    word_sequence = Word_Sequence()
    word_sequence.fit(good_sentences + bad_sentences)
    good_sentences_vec = word_sequence.transfroms(good_sentences, max_len=25)
    bad_sentences_vec = word_sequence.transfroms(bad_sentences, max_len=25)

    if print_vec:
        for i, (bad_sen, good_sen) in enumerate(zip(bad_sentences_vec, good_sentences_vec)):
            print('-----------------{}-----------------'.format(i))
            print("bad sentences: {}".format(bad_sen))
            print("bad sequence length: {}".format(len(bad_sen)))

            print("good sentences: {}".format(good_sen))
            print("good sequence length: {}".format(len(good_sen)))
            print()

            if i == 3:
                break

    # 创建训练和测试数据集
    data_sets = list()
    labels = list()
    for sent_vec in good_sentences_vec:
        data_sets.append(sent_vec)
        labels.append(0)
    for sent_vec in bad_sentences_vec:
        data_sets.append(sent_vec)
        labels.append(1)

    (X_train, Y_train, X_valid, Y_valid, X_test, Y_test) = \
        split_train_test(data_sets, labels, test_size=0.2, valid_size=0.1)

    # 创建dataloader
    train_dataloader = pick_dataloader(X_train, Y_train, shuffle=False)
    test_dataloader = pick_dataloader(X_test, Y_test, shuffle=False)
    valid_dataloader = pick_dataloader(X_valid, Y_valid, shuffle=False)

    if print_datalabel:
        for i, (x, y) in enumerate(train_dataloader):
            print('----------train batch: {} -------------'.format(i))
            print("data : ", x)
            print("data shape:", x.shape)
            print("label : ", y)
            print("label : ", y.shape)
            print()
            if i == 2:
                break

        for i, (x, y) in enumerate(test_dataloader):
            print('----------test batch: {}----------------'.format(i))
            print("data : ", x)
            print("data shape:", x.shape)
            print("label : ", y)
            print("label : ", y.shape)
            print()
            if i == 2:
                break

        for i, (x, y) in enumerate(valid_dataloader):
            print('---------valid batch: {}----------------'.format(i))
            print("data : ", x)
            print("data shape:", x.shape)
            print("label : ", y)
            print("label : ", y.shape)
            print()
            if i == 2:
                break


if __name__ == "__main__":
    test(print_sent=False, print_vec=False, print_datalabel=True)
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, TensorDataset


def get_keyword_vec(keyword_list, max_len, word_sequence):
    keyword_vec = list()
    for keyword in keyword_list:
        vec = [word_sequence.PAD] * max_len
        k_list = keyword.split(',')
        k_list = k_list if len(k_list) <= max_len else k_list[:max_len]
        for i, word in enumerate(k_list):
            vec[i] = word_sequence.word_to_token(word)
        keyword_vec.append(vec)

    return keyword_vec


def one_hot(class_num, index, dim=0):
    one = torch.eye(class_num)
    return one.index_select(dim=dim, index=index)


def loading_score_data(train_csv, valid_csv, test_csv, max_len, word_sequence, batch_size=32):
    assert os.path.isfile(train_csv), "File not exists! File: {}".format(train_csv)
    assert os.path.isfile(valid_csv), "File not exists! File: {}".format(valid_csv)
    assert os.path.isfile(test_csv), "File not exists! File: {}".format(test_csv)

    df_train = pd.read_csv(train_csv).dropna()
    df_valid = pd.read_csv(valid_csv).dropna()
    df_test = pd.read_csv(test_csv).dropna()

    train_ask = list(df_train['ask'])
    train_answer = list(df_train['answer'])
    train_label = list(df_train['class_id'])

    valid_ask = list(df_valid['ask'])
    valid_answer = list(df_valid['answer'])
    valid_label = list(df_valid['class_id'])

    test_ask = list(df_test['ask'])
    test_answer = list(df_test['answer'])
    test_label = list(df_test['class_id'])

    train_ask_vec = word_sequence.transfroms(train_ask, max_len=max_len)
    train_answer_vec = word_sequence.transfroms(train_answer, max_len=max_len)

    valid_ask_vec = word_sequence.transfroms(valid_ask, max_len=max_len)
    valid_answer_vec = word_sequence.transfroms(valid_answer, max_len=max_len)

    test_ask_vec = word_sequence.transfroms(test_ask, max_len=max_len)
    test_answer_vec = word_sequence.transfroms(test_answer, max_len=max_len)

    train_dataset = TensorDataset(torch.LongTensor(train_ask_vec),
                                  torch.LongTensor(train_answer_vec),
                                  torch.FloatTensor(train_label))

    valid_dataset = TensorDataset(torch.LongTensor(valid_ask_vec),
                                  torch.LongTensor(valid_answer_vec),
                                  torch.FloatTensor(valid_label))

    test_dataset = TensorDataset(torch.LongTensor(test_ask_vec),
                                 torch.LongTensor(test_answer_vec),
                                 torch.FloatTensor(test_label))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def loading_retrieval_data(train_csv, valid_csv, test_csv, max_len, word_sequence, batch_size=32):
    assert os.path.isfile(train_csv), "File not exists! File: {}".format(train_csv)
    assert os.path.isfile(valid_csv), "File not exists! File: {}".format(valid_csv)
    assert os.path.isfile(test_csv), "File not exists! File: {}".format(test_csv)

    df_train = pd.read_csv(train_csv).dropna()
    df_valid = pd.read_csv(valid_csv).dropna()
    df_test = pd.read_csv(test_csv).dropna()

    train_ask = list(df_train['ask'])
    train_answer = list(df_train['answer'])
    train_ask_keyword = list(df_train['ask_keyword'])
    train_answer_keyword = list(df_train['answer_keyword'])
    train_label = list(df_train['class_id'])

    valid_ask = list(df_valid['ask'])
    valid_answer = list(df_valid['answer'])
    valid_ask_keyword = list(df_valid['ask_keyword'])
    valid_answer_keyword = list(df_valid['answer_keyword'])
    valid_label = list(df_valid['class_id'])

    test_ask = list(df_test['ask'])
    test_answer = list(df_test['answer'])
    test_ask_keyword = list(df_test['ask_keyword'])
    test_answer_keyword = list(df_test['answer_keyword'])
    test_label = list(df_test['class_id'])

    train_ask_vec = word_sequence.transfroms(train_ask, max_len=max_len)
    train_answer_vec = word_sequence.transfroms(train_answer, max_len=max_len)
    train_ask_keyword = get_keyword_vec(train_ask_keyword, word_sequence=word_sequence, max_len=max_len)
    train_answer_keyword = get_keyword_vec(train_answer_keyword, word_sequence=word_sequence, max_len=max_len)

    valid_ask_vec = word_sequence.transfroms(valid_ask, max_len=max_len)
    valid_answer_vec = word_sequence.transfroms(valid_answer, max_len=max_len)
    valid_ask_keyword = get_keyword_vec(valid_ask_keyword, word_sequence=word_sequence, max_len=max_len)
    valid_answer_keyword = get_keyword_vec(valid_answer_keyword, word_sequence=word_sequence, max_len=max_len)

    test_ask_vec = word_sequence.transfroms(test_ask, max_len=max_len)
    test_answer_vec = word_sequence.transfroms(test_answer, max_len=max_len)
    test_ask_keyword = get_keyword_vec(test_ask_keyword, word_sequence=word_sequence, max_len=max_len)
    test_answer_keyword = get_keyword_vec(test_answer_keyword, word_sequence=word_sequence, max_len=max_len)

    train_dataset = TensorDataset(torch.LongTensor(train_ask_vec),
                                  torch.LongTensor(train_answer_vec),
                                  torch.LongTensor(train_ask_keyword),
                                  torch.LongTensor(train_answer_keyword),
                                  torch.LongTensor(train_label))

    valid_dataset = TensorDataset(torch.LongTensor(valid_ask_vec),
                                  torch.LongTensor(valid_answer_vec),
                                  torch.LongTensor(valid_ask_keyword),
                                  torch.LongTensor(valid_answer_keyword),
                                  torch.LongTensor(valid_label))

    test_dataset = TensorDataset(torch.LongTensor(test_ask_vec),
                                 torch.LongTensor(test_answer_vec),
                                 torch.LongTensor(test_ask_keyword),
                                 torch.LongTensor(test_answer_keyword),
                                 torch.LongTensor(test_label))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def loading_generate_data(train_csv, valid_csv, test_csv, max_len, word_sequence, batch_size=32):
    assert os.path.isfile(train_csv), "File not exists! File: {}".format(train_csv)
    assert os.path.isfile(valid_csv), "File not exists! File: {}".format(valid_csv)
    assert os.path.isfile(test_csv), "File not exists! File: {}".format(test_csv)

    print("loading generate data...")
    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    df_test = pd.read_csv(test_csv)

    df_train.dropna()
    df_valid.dropna()
    df_test.dropna()

    train_ask = list(df_train['ask'])
    train_answer = list(df_train['answer'])

    valid_ask = list(df_valid['ask'])
    valid_answer = list(df_valid['answer'])

    test_ask = list(df_test['ask'])
    test_answer = list(df_test['answer'])

    train_ask_vec = word_sequence.transfroms(train_ask, max_len=max_len)
    train_answer_vec = word_sequence.transfroms(train_answer, max_len=max_len)

    valid_ask_vec = word_sequence.transfroms(valid_ask, max_len=max_len)
    valid_answer_vec = word_sequence.transfroms(valid_answer, max_len=max_len)

    test_ask_vec = word_sequence.transfroms(test_ask, max_len=max_len)
    test_answer_vec = word_sequence.transfroms(test_answer, max_len=max_len)

    train_dataset = TensorDataset(torch.LongTensor(train_ask_vec), torch.LongTensor(train_answer_vec))
    valid_dataset = TensorDataset(torch.LongTensor(valid_ask_vec), torch.LongTensor(valid_answer_vec))
    test_dataset = TensorDataset(torch.LongTensor(test_ask_vec), torch.LongTensor(test_answer_vec))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def loading_category_data(train_csv, valid_csv, test_csv, max_len, word_sequence, batch_size=32):
    assert os.path.isfile(train_csv), "File not exists! File: {}".format(train_csv)
    assert os.path.isfile(valid_csv), "File not exists! File: {}".format(valid_csv)
    assert os.path.isfile(test_csv), "File not exists! File: {}".format(test_csv)

    print("loading category data...")

    df_train = pd.read_csv(train_csv)
    df_valid = pd.read_csv(valid_csv)
    df_test = pd.read_csv(test_csv)

    df_train = df_train.dropna()
    df_valid = df_valid.dropna()
    df_test = df_test.dropna()

    train_sentence = list(df_train['ask'])
    valid_sentence = list(df_valid['ask'])
    test_sentence = list(df_test['ask'])

    train_keyword = list(df_train['ask_keyword'])
    valid_keyword = list(df_valid['ask_keyword'])
    test_keyword = list(df_test['ask_keyword'])

    train_label = list(df_train['class_id'])
    valid_label = list(df_valid['class_id'])
    test_label = list(df_test['class_id'])

    train_label = torch.LongTensor(train_label)
    valid_label = torch.LongTensor(valid_label)
    test_label = torch.LongTensor(test_label)

    train_sentence_vec = word_sequence.transfroms(train_sentence, max_len=max_len)
    valid_sentence_vec = word_sequence.transfroms(valid_sentence, max_len=max_len)
    test_sentence_vec = word_sequence.transfroms(test_sentence, max_len=max_len)

    train_keyword_vec = get_keyword_vec(train_keyword, max_len=max_len, word_sequence=word_sequence)
    valid_keyword_vec = get_keyword_vec(valid_keyword, max_len=max_len, word_sequence=word_sequence)
    test_keyword_vec = get_keyword_vec(test_keyword, max_len=max_len, word_sequence=word_sequence)

    train_dataset = TensorDataset(torch.LongTensor(train_sentence_vec), torch.LongTensor(train_keyword_vec),
                                  train_label)
    valid_dataset = TensorDataset(torch.LongTensor(valid_sentence_vec), torch.LongTensor(valid_keyword_vec),
                                  valid_label)
    test_dataset = TensorDataset(torch.LongTensor(test_sentence_vec), torch.LongTensor(test_keyword_vec),
                                 test_label)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


# def test():
#     from words_sequence.word_sequences import load_word_sequences
#     word_sequence = load_word_sequences('../../save_word_sequence/word_sequence.pickle')
#     train_csv = '../../chinese_corpus/target/category_dataset/train.csv'
#     valid_csv = '../../chinese_corpus/target/category_dataset/valid.csv'
#     test_csv = '../../chinese_corpus/target/category_dataset/test.csv'
#
#     train_loader, valid_laoder, test_loader = loading_category_data(train_csv, valid_csv, test_csv, class_num=5,
#                                                                     max_len=20, word_sequence=word_sequence)
#
#     ask = pd.read_csv(train_csv)['ask']
#     for i, (sentence_vec, keyword_vec, label) in enumerate(train_loader):
#         print('source: ', ask[i])
#         print('sentence_vec: ', sentence_vec)
#         print('keyword_vec: ', keyword_vec)
#         print('label: ', label)
#         if i == 1:
#             break

def test():
    from words_sequence.word_sequences import load_word_sequences
    word_sequence = load_word_sequences('../../save_word_sequence/word_sequence.pickle')
    train_csv = '../../chinese_corpus/target/qa_dataset/food/train.csv'
    valid_csv = '../../chinese_corpus/target/qa_dataset/food/valid.csv'
    test_csv = '../../chinese_corpus/target/qa_dataset/food/test.csv'

    train_loader, valid_loader, test_loader = loading_retrieval_data(train_csv, valid_csv, test_csv, max_len=20,
                                                                     word_sequence=word_sequence)

    for i, (ask, answer, ask_keyword, answer_keyword, label) in enumerate(train_loader):
        print('ask vec: ', ask)
        print('answer vec: ', answer)
        print('ask keyword: ', ask_keyword)
        print('answer keyword: ', answer_keyword)
        print('label: ', label)

        if i == 1:
            break


if __name__ == '__main__':
    test()

import config
import os
import torch
from torch import optim
from torch.nn import NLLLoss
from tips import Error_Tips
from word_sequence import Word_Sequence
from data_util import extract_sentences_from_file
from data_util import split_train_test
from data_util import pick_dataloader
from model_function import save_model
from model_function import save_word_sequence
from model_function import train_model
from model_function import test_model
from model_function import load_model
from model_function import load_word_sequence
from model_function import model_predict
from model_function import save_object, load_object
from network_model import LSTM_Model
from network_model import FNN_Model

model_type = ['lstm', 'fnn']
model_function = ['train', 'predict']
device_list = ['cpu', 'gpu']

label_file_name = 'label_name.pickle'

# 初始化错误提示
ERROR = Error_Tips()

def check_config():
    # 检查选择的模型，如果为' ' 则报错
    if config.model == '':
        print(ERROR.Error_empty_value('model'))
        return False

        # 检查模型如果不在[lstm, fnn] 内，则报错。
    elif config.model.lower() not in model_type:
        print(ERROR.Error_Range_of_value('model', '[lstm, fnn]'))
        return False

    # 检查模型的功能选择，如果为' ' 则报错
    if config.model_func == '':
        print(ERROR.Error_empty_value('model_func'))
        return False

    # 检查模型功能如果不在[train, predict] 内，则报错
    elif config.model_func.lower() not in model_function:
        print(ERROR.Error_Range_of_value('model_func', '[predict, train]'))
        return False

    if config.model_func.lower() == 'predict':
        # 检查 model 文件是否存在，不存在则报错
        model_file = os.path.join(config.model_save_path,
                                  config.model_save_file)
        if not os.path.exists(model_file):
            print(ERROR.Error_File_not_found(model_file))
            return False

        # 检查词库文件是否存在，不存在则报错
        word_sequence_file = os.path.join(config.word_sequence_save_path,
                                          config.word_sequence_save_file)
        if not os.path.exists(word_sequence_file):
            print(ERROR.Error_File_not_found(model_file))
            return False

    if config.model_func.lower() == 'train':
        # 当使用 lstm 模型时候，
        # max_len, lstm_embed_size, lstm_hidden_size, lstm_num_layer, 不能为空，且不为整数时候，报错
        if config.model.lower() == 'lstm':
            if config.lstm_embed_size == '' or not isinstance(config.lstm_embed_size, int):
                print(ERROR.Error_empty_value_when('LSTM model', 'lstm_embed_size', 'integer'))
                return False

            if config.max_len == '' or not isinstance(config.max_len, int):
                print(ERROR.Error_empty_value_when('LSTM model', 'max_len', 'integer'))
                return False

            if config.lstm_hidden_size == '' or not isinstance(config.lstm_hidden_size, int):
                print(ERROR.Error_empty_value_when('LSTM model', 'lstm_hidden_size', 'integer'))
                return False

            if config.lstm_num_layers == '' or not isinstance(config.lstm_num_layers, int):
                print(ERROR.Error_empty_value_when('LSTM model', 'lstm_num_layer', 'integer'))
                return False

        # 当使用 fnn 模型时候，
        # fnn_hidden 不能为空，且不为 list\tuple 时候，报错
        elif config.model.lower() == 'fnn':
            if config.fnn_hidden == '' or not isinstance(config.fnn_hidden, (list, tuple)):
                print(ERROR.Error_empty_value_when('FNN model', 'fnn_hidden', 'list or tuple'))
                return False

        # 检查device，如果选择 gpu 进行训练，如果gpu不可用，则报错
        if config.device == '':
            print(ERROR.Error_empty_value('device'))
            return False
        elif config.device.lower() not in device_list:
            print(ERROR.Error_Range_of_value('device', '[cpu, gpu]'))
            return False
        elif config.device.lower() == 'gpu':
            if not torch.cuda.is_available():
                print(ERROR.ERROR_NOT_SUPPORT_GPU)
                return False

        # 训练模型，batch_size 不能为空
        if config.batch_size == '' or not isinstance(config.batch_size, int):
            print(ERROR.Error_empty_value_when('train', 'batch_size', 'integer'))
            return False

        # 训练模型 epochs 不能为空
        if config.epochs == '' or not isinstance(config.epochs, int):
            print(ERROR.Error_empty_value_when('train', 'epochs', 'integer'))
            return False

        # 如果保存模型目录不存在，则创建
        if not os.path.exists(config.model_save_path):
            os.makedirs(config.model_save_path)

        # 如果保存词库的目录不存在，则创建
        if not os.path.exists(config.word_sequence_save_path):
            os.makedirs(config.word_sequence_save_path)

        # 检查读取的数据文件信息，即文件是否存在
        for file in config.data_files.values():
            if not os.path.exists(file):
                print(ERROR.Error_File_not_found(file))
                return False

    print("---------------------------- config info ----------------------------------")
    print("model type: {}".format(config.model))
    print("model func: {}".format(config.model_func))
    print("epochs: {}".format(config.epochs))
    print("lr: {}".format(config.lr))
    print("device: {}".format(config.device))
    print("batch size: {}".format(config.batch_size))
    print("save model file: {}".format(
        os.path.join(config.model_save_path, config.model_save_file)))
    print("save word_sequences file: {}".format(
        os.path.join(config.word_sequence_save_path, config.model_save_file)))
    print("---------------------------------------------------------------------------")
    print()

    return True


def create_dataset():
    # 1. 提取标签信息和数据信息, 保存在字典中
    data_dict = {}
    for label, file in config.data_files.items():
        sentences = extract_sentences_from_file(file)
        data_dict[label] = sentences

    # 2. 建立词库
    # 2.1 提取所有的句子信息，保存在 all_sentences 中, 用以构建词库
    all_sentences = list()
    for sentences in data_dict.values():
        all_sentences = all_sentences + sentences

    # 2.2 构建词库
    word_sequence = Word_Sequence()
    word_sequence.fit(all_sentences)

    # 3. 创建训练、测试、验证数据集
    # 将数据转换为向量，保存在 datasets 中
    datasets = []
    # 保存标签的名称
    label_name = []
    for label, dataset in data_dict.items():
        # lstm 模型数据集
        if config.model.lower() == 'lstm':
            dataset_vec = word_sequence.transfroms(dataset, max_len=config.max_len)

        # fnn 模型数据集
        elif config.model.lower() == 'fnn':
            dataset_vec = word_sequence.transfroms_word_bag(dataset)

        datasets.append(dataset_vec)
        label_name.append(label)

    # 标记数据，对datset数据集中的每个句子向量进行标记。
    # 如: 句子分析属于 good 类型，则标记为: 0 ，bad 类型，则标记为：1
    datasets_vec = list()
    labels = list()
    for i, dataset_vec in enumerate(datasets):
        for sentence_vec in dataset_vec:
            datasets_vec.append(sentence_vec)
            labels.append(i)

    # 划分训练、测试、验证数据集
    (X_train, Y_train, X_valid, Y_valid, X_test, Y_test) = \
        split_train_test(datasets_vec, labels,
                         test_size=config.test_size,
                         valid_size=config.valid_size)

    # 4.包装数据集为 dataloader.
    train_dataloader = pick_dataloader(X_train, Y_train,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       data_type=config.model)

    test_dataloader = pick_dataloader(X_test, Y_test,
                                      batch_size=config.batch_size,
                                      shuffle=False,
                                      data_type=config.model)

    valid_dataloader = pick_dataloader(X_valid, Y_valid,
                                       batch_size=config.batch_size,
                                       shuffle=False,
                                       data_type=config.model)

    save_word_sequence(word_sequence,
                       save_path=config.word_sequence_save_path,
                       file_path=config.word_sequence_save_file)
    save_object(label_name, config.model_save_path, label_file_name)

    # 返回数据集和单词序列
    return (train_dataloader, valid_dataloader, test_dataloader,
            word_sequence, label_name)


def create_model(word_sequence, output_size):
    # 构建 lstm 模型
    if config.model.lower() == 'lstm':
        model = LSTM_Model(len(word_sequence),
                           embed_size=config.lstm_embed_size,
                           hidden_size=config.lstm_hidden_size,
                           output_size=output_size,
                           num_layers=config.lstm_num_layers,
                           drop_out=config.dropout)

    # 构建 fnn 模型
    elif config.model.lower() == 'fnn':
        model = FNN_Model(len(word_sequence.word_dict),
                          output_size=output_size,
                          hidden_num=config.fnn_hidden,
                          dropout=config.dropout)

    print("---------------------------- model summary ----------------------------------")
    print(model)
    print("-----------------------------------------------------------------------------")
    print()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = NLLLoss()

    return (model,
            optimizer,
            criterion)


def start_train_and_test(model,
                         optimizer,
                         criterion,
                         train_dataloader,
                         valid_dataloader,
                         test_dataloader):
    # 训练模型
    train_info = train_model(model=model,
                             train_dataloader=train_dataloader,
                             criterion=criterion,
                             optimizer=optimizer,
                             valid_dataloader=valid_dataloader,
                             epochs=config.epochs,
                             device=config.device,
                             print_every=100,
                             use_accuracy=True,
                             use_valid=True)

    # 测试模型
    test_model(model=model,
               criterion=criterion,
               test_dataloader=test_dataloader,
               device=config.device,
               use_accuracy=True)

    # 保存模型
    save_model(model,
               save_dir=config.model_save_path,
               file_name=config.model_save_file)

    return train_info


def predict_sentences(model, word_sequences, sentences, label_name):
    # 使用 lstm 模型对句子进行预测
    if config.model == 'lstm':
        result = model_predict(model,
                               word_sequence=word_sequences,
                               sentences=sentences,
                               max_len=config.max_len,
                               data_type=config.model)

    # 使用 fnn 模型对句子进行预测
    elif config.model == 'fnn':
        result = model_predict(model,
                               word_sequence=word_sequences,
                               sentences=sentences,
                               max_len=None,
                               data_type=config.model)

    # 返回预测的标签名和结果
    return {'value': result,
            'label': label_name[result]}


def train():
    # 创建数据集
    (train_loader, valid_loader, test_loader,
     word_sequence, label_name) = create_dataset()

    # 创建模型
    (model, optimizer, criterion) = \
        create_model(word_sequence=word_sequence, output_size=len(label_name))

    # 训练模型
    train_info = start_train_and_test(model,
                                      optimizer=optimizer,
                                      criterion=criterion,
                                      train_dataloader=train_loader,
                                      valid_dataloader=valid_loader,
                                      test_dataloader=test_loader)

    return train_info


def predict():
    model_file = os.path.join(config.model_save_path,
                              config.model_save_file)
    model = load_model(model_file)

    word_sequence_file = os.path.join(config.word_sequence_save_path,
                                      config.word_sequence_save_file)
    word_sequence = load_word_sequence(word_sequence_file)

    label_file = os.path.join(config.model_save_path, label_file_name)
    label_name = load_object(label_file)

    print("请输入一个中文句子，退出请输入 quit 或 exit ")
    while True:
        sentences = input("请输入中文句子: ")
        sentences = str(sentences)

        if sentences.lower() == 'exit' or sentences.lower() == 'quit':
            break

        result = predict_sentences(model, word_sequence, sentences, label_name)
        print("预测结果：", result['label'])
        print()


def run():
    # 检查配置
    if not check_config():
        print("Found error: Please check config.py file ! ")
        return False
    # 训练模型
    if config.model_func.lower() == 'train':
        train()

    # 预测模型
    elif config.model_func.lower() == 'predict':
        predict()


if __name__ == '__main__':
    run()

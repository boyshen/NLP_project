import torch
import os
import pickle
from torch.nn import NLLLoss
from torch import optim
from tips import Error_Tips
from data_util import extract_sentences_from_file
from data_util import split_train_test
from data_util import pick_dataloader
from data_util import regular
from network_model import LSTM_Model
from word_sequence import Word_Sequence

# 初始化错误提示
ERROR = Error_Tips()

def valid_model(model, valid_dataloader, criterion, device='cpu', use_accuracy=True):
    """
    验证模型
    :param model: <Model> 模型对象
    :param valid_dataloader: <dataloader> 验证数据集
    :param criterion: <Loss> 损失函数的对象
    :param device: <str> 使用的验证设备，CPU 或 GPU
    :param use_accuracy: <bool> 是否计算准确率
    :return: loss, accuracy
    """

    assert device.lower() in ['cpu', 'gpu'], \
        ERROR.Error_Range_of_value('device', '[CPU, GPU]')

    valid_loss = 0
    valid_accuracy = 0
    total = 0

    for feature, label in valid_dataloader:
        feature = feature.to(device.lower())
        label = label.to(device.lower())

        output = model.forward(feature)
        valid_loss += criterion(output, label).item()
        if use_accuracy:
            valid_accuracy += (label == output.max(dim=1)[1]).sum().item()
        total += label.size()[0]

    loss = valid_loss / len(valid_dataloader)
    print("valid loss: {:.3f}".format(loss), end=" ")

    if use_accuracy:
        accuracy = valid_accuracy / total
        print("valid accuracy: {:.3f}".format(accuracy))

    return loss, accuracy


def train_model(model,
                train_dataloader,
                criterion,
                optimizer,
                valid_dataloader=None,
                epochs=20,
                device='cpu',
                print_every=300,
                use_valid=False,
                use_accuracy=True):
    """
    训练模型
    :param model: <Model> 模型对象
    :param train_dataloader: <dataloader> 训练数据dataloader
    :param criterion: <loss> 损失函数
    :param optimizer: <optim> 优化器
    :param valid_dataloader: <dataloader> 验证数据集
    :param epochs: <int> 训练轮次
    :param device: <str> 训练设备名，即 CPU 或 GPU
    :param print_every: <int> 没多少次输出信息
    :param use_valid: <bool> 是否使用验证数据集
    :param use_accuracy: <bool> 是否计算准确率
    :return: <dict> 损失和准确率字典
    """

    assert device.lower() in ['cpu', 'gpu'], \
        ERROR.Error_Range_of_value('device', '[CPU, GPU]')

    if use_valid:
        assert valid_dataloader, \
            "Error: {} is True, {} is not None".format('use_valid', 'valid_dataloader')

    steps = 0
    train_accuracy = 0
    train_loss = 0
    train_total = 0

    train_loss_list = []
    train_accuracy_list = []

    valid_loss_list = []
    valid_accuracy_list = []

    print("start train ...")

    for e in range(epochs):

        # 训练模式
        model.train()

        for i, (feature, label) in enumerate(train_dataloader):
            steps += 1

            feature = feature.to(device.lower())
            label = label.to(device.lower())

            # optimizer 会记录之前的梯度，每次训练之前清零
            optimizer.zero_grad()

            output = model.forward(feature)
            loss = criterion(output, label)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()

            train_loss += loss.item()
            if use_accuracy:
                train_accuracy += (label == output.max(dim=1)[1]).sum().item()
            train_total += label.size()[0]

            if steps % print_every == 0:
                print("epoch: {}/{}".format(e + 1, epochs),
                      "device: {}".format(device),
                      "train loss: {:.3f}".format(train_loss / print_every), end=" ")
                train_loss_list.append(train_loss / print_every)

                if use_accuracy:
                    print("train accuracy: {:.3f}".format(train_accuracy / train_total))
                    train_accuracy_list.append(train_accuracy / train_total)

                if use_valid:
                    # 验证模型，先让模型进入评估模式
                    model.eval()

                    with torch.no_grad():
                        valid_loss, valid_accuracy = \
                            valid_model(model, valid_dataloader, criterion, device, use_accuracy)
                    valid_loss_list.append(valid_loss)

                    if use_accuracy:
                        valid_accuracy_list.append(valid_accuracy)

                print()
                train_loss = 0
                train_total = 0
                train_accuracy = 0
                model.train()

    return {"train_loss": train_loss_list,
            "valid_loss": valid_loss_list,
            "train_accuracy": train_accuracy_list,
            "valid_accuracy": valid_accuracy_list}


def test_model(model, test_dataloader, criterion, device='cpu', use_accuracy=True):
    """
    测试模型
    :param model: <Model> 对象模型
    :param test_dataloader: <dataloader> 测试数据集
    :param criterion: <loss> 损失函数
    :param device: <str> 训练设备名，即 CPU 或 GPU
    :param use_accuracy: <bool> 是否计算准确率
    :return: loss、 accuracy
    """
    assert device.lower() in ['cpu', 'gpu'], \
        ERROR.Error_Range_of_value('device', '[CPU, GPU]')

    test_loss = 0
    test_accuracy = 0
    total = 0

    print("start test ...")

    for feature, label in test_dataloader:
        feature = feature.to(device.lower())
        label = label.to(device.lower())

        output = model.forward(feature)
        loss = criterion(output, label)

        test_loss += loss.item()
        if use_accuracy:
            test_accuracy += (label == output.max(dim=1)[1]).sum().item()
        total += label.size()[0]

    loss = test_loss / len(test_dataloader)
    print("test loss: {:.3f}".format(loss), end=" ")

    if use_accuracy:
        accuracy = test_accuracy / total
        print("accuracy: {:.1f}%".format(accuracy * 100))
    print()

    return loss, accuracy


def model_predict(model, word_sequence, sentences, max_len=None, data_type='LSTM'):
    """
    模型预测，即对输入的单个句子进行预测
    :param model: <Model> 模型对象
    :param word_sequence: <Word_Sequence> 单词库
    :param sentences: <str> 预测的句子
    :param max_len: <int> 保留的序列长度
    :param data_type:  <str> LSTM 和 FNN 指定包装数据的类型。
                             LSTM 指定的数据类型为 Long, FNN 中指定的数据类型为 float。
                             label 默认为long类型
                            （FNN）即前馈神经网络
    :return: <int> 预测的标签索引，如：0、1、2、3 等
    """
    assert data_type.lower() in ['lstm', 'fnn'], \
        ERROR.Error_Range_of_value('data_type', '[LSTM, FNN]')

    sentences = regular(sentences)

    if data_type.lower() == 'lstm':
        sentences_vec = word_sequence.transfrom(sentences, max_len=max_len)
        data_tensor = torch.unsqueeze(torch.LongTensor(sentences_vec), dim=0)
    elif data_type.lower() == 'fnn':
        sentences_vec = word_sequence.transfrom_word_bag(sentences)
        data_tensor = torch.FloatTensor(sentences_vec).view(1, -1)

    output = model.forward(data_tensor)
    predict_value = output.max(dim=1)[1]

    return predict_value.item()


def save_model(model, save_dir='model', file_name='model_checkpoint.pth'):
    """
    保存模型
    :param model: <Model> 保存的模型对象
    :param save_dir: <str> 保存的路径
    :param file_name: <str> 保存的文件名
    :return: 无
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model, os.path.join(save_dir, file_name))
    print("model checkpoint save over! file: {}".format(os.path.join(save_dir, file_name)))


def load_model(model_file):
    """
    加载模型
    :param model_file: <str> 模型文件
    :return: <Model> 模型对象
    """
    assert os.path.exists(model_file), ERROR.Error_File_not_found(model_file)
    model = torch.load(model_file)
    return model


def save_word_sequence(word_sequence,
                       save_path='word_sequence',
                       file_path='word_sequence.pickle'):
    """
    保存词库对象，将词库对象保存为pickle类型的文件
    :param word_sequence: <Word_Sequence> 词库对象
    :param save_path: <str>保存路径
    :param file_path: <str>保存文件名
    :return: 无
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file = os.path.join(save_path, file_path)
    pickle.dump(word_sequence, open(file, 'wb'))

    print("word sequences save over ! file: {}".format(file))


def load_word_sequence(word_sequence_file):
    """
    加载词典序列
    :param word_sequence_file: <str> 词典序列文件
    :return: <Word_Sequence> 词典序列对象
    """

    assert os.path.exists(word_sequence_file), \
        ERROR.Error_File_not_found(word_sequence_file)

    with open(word_sequence_file, 'rb') as fopen:
        word_sequence = pickle.loads(fopen.read())

    return word_sequence


def save_object(object_name, save_path='model', file_path='object_name.pickle'):
    """
    保存加载在内存中的对象信息，保存为pickle类型的文件
    :param object_name:  <object> 对象名
    :param save_path: <保存地址>
    :param file_path: <保存文件名>
    :return:
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file = os.path.join(save_path, file_path)
    pickle.dump(object_name, open(file, 'wb'))

    print("file save over ! file: {}".format(file))


def load_object(object_file):
    """
    从保存的pickle类型的文件中加载对象
    :param object_file: <str> 对象文件地址
    :return: object 对象信息
    """
    assert os.path.exists(object_file), \
        ERROR.Error_File_not_found(object_file)

    with open(object_file, 'rb') as fopen:
        word_sequence = pickle.loads(fopen.read())

    return word_sequence


def test():
    """
    测试函数，即模拟测试模型的训练、预测、保存操作。
    :return:
    """
    bad_sentences = extract_sentences_from_file('data/bad.txt')
    good_sentences = extract_sentences_from_file('data/good.txt')

    # 进行fit操作，建立词库
    word_sequence = Word_Sequence()
    word_sequence.fit(good_sentences + bad_sentences)
    good_sentences_vec = word_sequence.transfroms(good_sentences, max_len=25)
    bad_sentences_vec = word_sequence.transfroms(bad_sentences, max_len=25)

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

    # 构建模型
    lstm_model = \
        LSTM_Model(len(word_sequence), embed_size=256, hidden_size=256, output_size=2)

    # 设置优化器和损失函数
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.0001)
    criterion = NLLLoss()
    device = 'GPU' if torch.cuda.is_available() else 'CPU'

    # 训练模型
    train_info = train_model(lstm_model, train_dataloader,
                             optimizer=optimizer,
                             criterion=criterion,
                             epochs=2,
                             print_every=100,
                             valid_dataloader=valid_dataloader,
                             device=device,
                             use_accuracy=True,
                             use_valid=True)
    # print(train_info)

    # 测试模型
    test_model(lstm_model,
               test_dataloader=test_dataloader,
               criterion=criterion,
               device=device,
               use_accuracy=True)

    # 保存模型
    save_model(lstm_model)
    # 保存提取的词库
    save_word_sequence(word_sequence)


def test_predict(sentences):
    """
    测试模型的加载、预测
    :return:
    """
    # 加载模型
    model = load_model('model/model_checkpoint.pth')

    # 加载词库
    word_sequences = load_word_sequence('word_sequence/word_sequence.pickle')

    # 预测
    predict = model_predict(model, word_sequences, sentences=sentences, max_len=25)

    print("sentences : ", sentences)

    # 预测结果的标签索引， 在训练时候曾以 0 作为 good 标签，1 作为 bad 的标签。
    if predict == 0:
        print("predict result : good")
    elif predict == 1:
        print("predict result : bad")


if __name__ == '__main__':
    test()
    # test_predict("衣服收到了&hellip;这衣服都是破的&hellip;以为京东不会出现这个问题&hellip;所以差评")

    # test_predict("本人比较胖，但是穿起来也很合适，非常不错，专卖店最少要卖几百才有这个舒适感！")

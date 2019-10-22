import torch
import os
import numpy as np
import jieba
from collections import Counter
from torch import nn
from global_variable import LSTM, GRU


def summary(model):
    print("------------------- model summary -------------------")
    print(model)
    print("-----------------------------------------------------")
    print()


def get_cnn_width(w_in, kernel_size, stride, padding):
    return int(((w_in + 2 * padding) - kernel_size + 1) / stride)


def get_accuracy(output, label):
    return (output.argmax(1) == label).sum().item() / len(label)


def confusion_matrix(outputs, labels):
    outputs = outputs.argmax(1)
    tp, tn, fp, fn = 0, 0, 0, 0

    for (output, label) in zip(outputs.tolist(), labels.tolist()):
        if output == 1 and label == 1:
            tp += 1
        elif output == 0 and label == 0:
            tn += 1
        elif output == 1 and label == 0:
            fp += 1
        elif output == 0 and label == 1:
            fn += 1

    return tp, tn, fp, fn


def get_accuracy_by_threshold(score, label, threshold=0.5):
    score = np.array(score.gt(threshold))
    label = np.array(label)
    return (score == label).sum() / len(label)


def confusion_matrix_by_threshold(score, label, threshold=0.5):
    score_list = np.array(score.gt(threshold)).tolist()
    label_list = np.array(label).tolist()

    tp, fn, fp, tn = 0, 0, 0, 0
    for (score, label) in zip(score_list, label_list):
        if score == 1 and label == 1:
            tp += 1
        if score == 0 and label == 1:
            fn += 1
        if score == 1 and label == 0:
            fp += 1
        if score == 0 and label == 0:
            tn += 1

    return tp, tn, fp, fn


def get_recall(tp, fn):
    # tp, tn, fp, fn = confusion_matrix(outputs, labels)
    recall = tp / (tp + fn) if (tp + fn) != 0 else float(0.0)
    return recall


def get_precision(tp, fp):
    precision = tp / (tp + fp) if (tp + fp) != 0 else float(0.0)
    return precision


def save_best_model(model, train_loss, records_list, valid_loss=None, save_dir=None, file_name='model.pth'):
    train_loss_list = [record['train_loss'] for record in records_list]
    train_loss_list = sorted(train_loss_list, reverse=False)

    if valid_loss is not None:
        valid_loss_list = [record['valid_loss'] for record in records_list]
        valid_loss_list = sorted(valid_loss_list, reverse=False)

        if valid_loss <= valid_loss_list[0] and train_loss <= train_loss_list[0]:
            save_model(model, save_dir=save_dir, file_name=file_name)
    else:
        if train_loss <= train_loss_list[0]:
            save_model(model, save_dir=save_dir, file_name=file_name)


def save_model(model, save_dir=None, file_name='model.pth'):
    """保存模型"""
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.getcwd()

    file = os.path.join(save_dir, file_name)
    torch.save(model, file)

    print("model save over! File: {}".format(file))


def load_model(model_file):
    """通过保存的模型文件加载模型"""
    if not os.path.isfile(model_file):
        print("model file not exists! File: {}".format(model_file))
        return False
    model = torch.load(model_file)
    return model


def cos_similarity(sent1, sent2):
    word_dict = dict()

    sent1 = jieba.lcut(sent1)
    sent2 = jieba.lcut(sent2)

    words = Counter()
    words.update(sent1)
    words.update(sent2)

    for word in words.keys():
        word_dict[word] = len(word_dict)

    word_count1 = Counter(sent1)
    word_count2 = Counter(sent2)

    word_vec1 = [0] * len(word_dict)
    for w, c in word_count1.items():
        word_vec1[word_dict[w]] = c

    word_vec2 = [0] * len(word_dict)
    for w, c in word_count2.items():
        word_vec2[word_dict[w]] = c

    vec1 = np.array(word_vec1)
    vec2 = np.array(word_vec2)

    similarity = np.sum(vec1 * vec2) / (np.sqrt(np.sum(np.power(vec1, 2))) * np.sqrt(np.sum(np.power(vec2, 2))))

    return similarity


class SANNetwork(nn.Module):

    def __init__(self, vocab_size, embed_size, rnn_hidden_size, rnn_model=GRU, output_size=2, use_bidirectional=False,
                 dropout=0.2):
        super(SANNetwork, self).__init__()

        self.rnn_model = rnn_model

        self.embed = nn.Embedding(vocab_size, embed_size)
        if use_bidirectional:
            num_direction = 2
        else:
            num_direction = 1

        if rnn_model == GRU:
            self.gru_1 = nn.GRU(embed_size, rnn_hidden_size, batch_first=True, bidirectional=use_bidirectional,
                                num_layers=1)
        elif rnn_model == LSTM:
            self.lstm_1 = nn.LSTM(embed_size, rnn_hidden_size, batch_first=True, bidirectional=use_bidirectional,
                                  num_layers=1)

        self.attention_word_linear = nn.Linear(embed_size, embed_size, bias=True)
        self.attention_word_model = nn.Sequential(
            nn.Tanh(),
            nn.Linear(1, embed_size, bias=False),
            # nn.Softmax()
        )

        self.attention_seq_linear = nn.Linear(rnn_hidden_size * num_direction, rnn_hidden_size * num_direction,
                                              bias=True)
        self.attention_seq_model = nn.Sequential(
            nn.Tanh(),
            nn.Linear(1, rnn_hidden_size * num_direction, bias=False),
            # nn.Softmax()
        )

        if rnn_model == GRU:
            self.gru_2 = nn.GRU(embed_size + (rnn_hidden_size * num_direction), rnn_hidden_size, batch_first=True,
                                bidirectional=use_bidirectional,
                                num_layers=1)

        elif rnn_model == LSTM:
            self.lstm_2 = nn.LSTM(embed_size + (rnn_hidden_size * num_direction), rnn_hidden_size, batch_first=True,
                                  bidirectional=use_bidirectional,
                                  num_layers=1)

        self.fn = nn.Linear(rnn_hidden_size * num_direction, output_size)
        self.drop_out = nn.Dropout(dropout)
        self.soft_max = nn.Softmax(1)

    def forward_attention(self, hidden_utter_output, hidden_response_output, utter_word_embed, response_word_embed):
        utter_seq = hidden_utter_output.permute(1, 0, 2)
        response_seq = hidden_response_output.permute(1, 0, 2)
        utter_word = utter_word_embed.permute(1, 0, 2)
        response_word = response_word_embed.permute(1, 0, 2)

        t_vec = list()
        for t in range(hidden_utter_output.size()[1]):
            word_context_vec = self.forward_attention_word(utter_word[t], response_word[t], utter_word_embed)
            seq_context_vec = self.forward_attention_seq(utter_seq[t], response_seq[t], hidden_utter_output)

            word_context_vec = word_context_vec.permute(2, 0, 1)
            seq_context_vec = seq_context_vec.permute(2, 0, 1)

            context_vec = torch.cat((word_context_vec, seq_context_vec))
            context_vec = context_vec.permute(1, 2, 0)
            t_vec.append(context_vec)

        v = torch.cat(t_vec, 1)
        return v

    def forward_attention_word(self, utter_word, response_word, utter_word_embed):
        u_t = utter_word.unsqueeze(1)
        r_t = response_word.unsqueeze(1)

        w_t = torch.bmm(self.attention_word_linear(u_t), r_t.permute(0, 2, 1))
        score = self.attention_word_model(w_t)
        score = self.soft_max(score.squeeze(1))
        score = score.unsqueeze(1)
        context_vec = torch.sum(utter_word_embed * score, 1).unsqueeze(1) * r_t

        return context_vec

    def forward_attention_seq(self, utter_hidden, response_hidden, hidden_utter_output):
        u_t = utter_hidden.unsqueeze(1)
        r_t = response_hidden.unsqueeze(1)

        w_t = torch.bmm(self.attention_seq_linear(u_t), r_t.permute(0, 2, 1))
        score = self.attention_seq_model(w_t)
        score = self.soft_max(score.squeeze(1))
        score = score.unsqueeze(1)
        context_vec = torch.sum(hidden_utter_output * score, 1).unsqueeze(1) * r_t

        return context_vec

    def forward(self, ask, answer, ask_keyword, answer_keyword):
        batch_size = ask.size()[0]
        ask_embed = self.embed(ask)
        answer_embed = self.embed(answer)
        ask_keyword_embed = self.embed(ask_keyword)
        answer_keyword_embed = self.embed(answer_keyword)

        if self.rnn_model == GRU:
            ask_output, _ = self.gru_1(ask_embed)
            answer_output, _ = self.gru_1(answer_embed)

        elif self.rnn_model == LSTM:
            ask_output, (_, _) = self.lstm_1(ask_embed)
            answer_output, (_, _) = self.lstm_1(answer_embed)

        v = self.forward_attention(ask_output, answer_output, ask_keyword_embed, answer_keyword_embed)

        if self.rnn_model == GRU:
            _, hidden_output = self.gru_2(v)
        elif self.rnn_model == LSTM:
            _, (hidden_output, _) = self.lstm_2(v)

        hidden_output = hidden_output.permute(1, 0, 2)
        hidden_output = hidden_output.contiguous().view(batch_size, -1)
        output = self.soft_max(self.drop_out(self.fn(hidden_output)))
        return output

    def test(self, test_loader, criterion, device='cpu'):
        test_loss, test_accuracy = 0, 0
        test_tp, test_fn = 0, 0

        self.to(device)
        for ask, answer, ask_keyword, answer_keyword, label in test_loader:
            ask = ask.to(device)
            answer = answer.to(device)
            ask_keyword = ask_keyword.to(device)
            answer_keyword = answer_keyword.to(device)
            label = label.to(device)

            score = self.forward(ask, answer, ask_keyword, answer_keyword)
            loss = criterion(torch.log(score), label)
            test_loss += loss.item()

            accuracy = get_accuracy(score, label)
            test_accuracy += accuracy

            tp, _, _, fn = confusion_matrix(score, label)
            test_tp += tp
            test_fn += fn

        test_loss = test_loss / len(test_loader)
        test_accuracy = test_accuracy / len(test_loader)
        test_recall = test_tp / (test_tp + test_fn) if (test_tp + test_fn) != 0 else float(0.0)

        print("test loss: {:.4f}".format(test_loss),
              "test accuracy: {:.2f}%".format(test_accuracy * 100),
              "test recall: {:.2f}".format(test_recall))

        return test_loss, test_accuracy, test_recall

    def valid(self, valid_loader, criterion, device='cpu'):
        valid_loss, valid_accuracy = 0, 0
        valid_tp, valid_fn = 0, 0

        self.to(device)
        for ask, answer, ask_keyword, answer_keyword, label in valid_loader:
            ask = ask.to(device)
            answer = answer.to(device)
            ask_keyword = ask_keyword.to(device)
            answer_keyword = answer_keyword.to(device)

            score = self.forward(ask, answer, ask_keyword, answer_keyword)
            loss = criterion(torch.log(score), label).to(device)
            valid_loss += loss.item()

            accuracy = get_accuracy(score, label)
            valid_accuracy += accuracy

            tp, _, _, fn = confusion_matrix(score, label)
            valid_tp += tp
            valid_fn += fn

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = valid_accuracy / len(valid_loader)
        valid_recall = valid_tp / (valid_tp + valid_fn) if (valid_tp + valid_fn) != 0 else float(0.0)

        return valid_loss, valid_accuracy, valid_recall

    def fit(self, train_loader, optimizer, criterion, device='cpu', epochs=10, valid_loader=None, save_dir=None,
            model_file='SAN_model.pth', save_final_model=False):

        train_list = list()

        print("start fit ...")
        self.to(device)
        for e in range(epochs):
            train_loss = 0
            train_accuracy = 0
            train_dict = dict()

            self.train()
            for step, (ask, answer, ask_keyword, answer_keyword, label) in enumerate(train_loader):
                ask = ask.to(device)
                answer = answer.to(device)
                ask_keyword = ask_keyword.to(device)
                answer_keyword = answer_keyword.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                score = self.forward(ask, answer, ask_keyword, answer_keyword)
                loss = criterion(torch.log(score), label).to(device)
                train_loss += loss

                accuracy = get_accuracy(score, label)
                train_accuracy += accuracy

                loss.backward()
                optimizer.step()

                print("epochs: {}/{} ".format(e + 1, epochs),
                      "step: {}/{} ".format(step + 1, len(train_loader)),
                      "device: {}".format(device),
                      "train loss: {:.4f}".format(train_loss / (step + 1)),
                      "train accuracy: {:.2f}%".format(train_accuracy / (step + 1) * 100),
                      end='\r', flush=True)

            print()

            if valid_loader is not None:
                self.eval()

                valid_loss, valid_accuracy, valid_recall = self.valid(valid_loader, criterion=criterion, device=device)

                print("valid loss: {:.4f}".format(valid_loss),
                      "valid accuracy: {:.2f}%".format(valid_accuracy * 100),
                      "valid recall: {:.2f}".format(valid_recall))

                self.train()

            if e == 0:
                save_model(self, save_dir=save_dir, file_name=model_file)
            else:
                t_loss = train_loss / len(train_loader)
                save_best_model(self, train_loss=t_loss, records_list=train_list, valid_loss=valid_loss,
                                save_dir=save_dir, file_name=model_file)

            if save_final_model:
                if e == (epochs - 1):
                    file = model_file + "(final)"
                    save_model(self, save_dir=save_dir, file_name=file)

            train_dict['train_loss'] = train_loss / len(train_loader)
            train_dict['train_accuracy'] = train_accuracy / len(train_loader)
            train_dict['valid_loss'] = valid_loss
            train_dict['valid_accuracy'] = valid_accuracy
            train_dict['valid_recall'] = valid_recall

            train_list.append(train_dict)

            print()

        return train_list


class CategoryNetwork(nn.Module):

    def __init__(self, vocab_size, embed_size, rnn_hidden_size, seq_len, output_size, model=LSTM, num_layers=1,
                 use_bidirectional=False, drop_out=0.2, cnn_output_feature=2, window_size=[2, 3, 4]):
        """
        Intention network 模型初始化
        :param vocab_size: <int> 词汇量的大小
        :param embed_size: <int> 嵌入层的大小
        :param rnn_hidden_size: <int> RNN隐藏层的大小
        :param output_size: <int> 输出大小
        :param model: <str> 模型类型，lstm 或 rnn 两种
        :param num_layers: <int> rnn网络的层数
        :param use_bidirectional: <bool> 是否使用双向循环
        :param drop_out: <float> 随机 drop 节点的概率大小
        """
        super(CategoryNetwork, self).__init__()
        self.model = model

        self.embedding = nn.Embedding(vocab_size, embed_size)

        if model == LSTM:
            self.lstm = nn.LSTM(embed_size, rnn_hidden_size,
                                num_layers=num_layers, bidirectional=use_bidirectional, batch_first=True)
        if model == GRU:
            self.gru = nn.GRU(embed_size, rnn_hidden_size,
                              num_layers=num_layers, bidirectional=use_bidirectional, batch_first=True)

        self.conv = nn.ModuleList()
        for kernel_size in window_size:
            conv1d = nn.Conv1d(rnn_hidden_size, cnn_output_feature, stride=1, padding=0, kernel_size=kernel_size)
            width = get_cnn_width(seq_len, kernel_size=kernel_size, stride=1, padding=0)
            bn = nn.BatchNorm1d(num_features=cnn_output_feature)
            relu = nn.ReLU()
            max_pool = nn.MaxPool1d(kernel_size=width, stride=1, padding=0)
            model = nn.Sequential(conv1d, bn, relu, max_pool)
            self.conv.append(model)

        self.fc = nn.Linear(cnn_output_feature * len(window_size), output_size)

        self.drop_out = nn.Dropout(drop_out)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence_vec, keyword_vec):
        """
        前向传播
        :param sentence_vec: <Tensor> 句子向量
        :param keyword_vec: <Tensor> 关键词向量
        :return: <Tensor> 前向反馈输出 output
        """
        batch_size = sentence_vec.size()[0]
        sentence_embedding = self.embedding(sentence_vec)
        keyword_embedding = self.embedding(keyword_vec)

        if self.model == GRU:
            sentence_rnn_output, sentence_hidden_state = self.gru(sentence_embedding)
            keyword_rnn_output, keyword_hidden_state = self.gru(keyword_embedding)

        if self.model == LSTM:
            sentence_rnn_output, sentence_hidden_state = self.lstm(sentence_embedding)
            keyword_rnn_output, keyword_hidden_state = self.lstm(keyword_embedding)

        sentence_rnn_output = sentence_rnn_output.permute(0, 2, 1)
        keyword_rnn_output = keyword_rnn_output.permute(0, 2, 1)

        sentence_conv_output = [model(sentence_rnn_output) for model in self.conv]
        keyword_conv_output = [model(keyword_rnn_output) for model in self.conv]

        sentence_conv_output = torch.cat(sentence_conv_output, 1).view(batch_size, -1)
        keyword_conv_output = torch.cat(keyword_conv_output, 1).view(batch_size, -1)

        fc_input = sentence_conv_output + keyword_conv_output
        output = self.drop_out(self.fc(fc_input))
        output = self.log_softmax(output)

        return output

    def fit(self, train_loader, optimizer, criterion, epochs=10, device='cpu', save_filename='model.pth',
            valid_loader=None, save_dir=None):
        """
        训练模型
        :param train_loader: <dataloader> 训练数据集
        :param optimizer: <optim> 优化器
        :param criterion: <loss> 损失函数
        :param epochs: <int> 训练轮次
        :param valid_loader: <dataloader>  验证数据集
        :param device: <str> 训练设备
        :param save_dir: <str> 模型保存目录
        :param save_filename: <str> 模型文件名
        :return:
        """
        device = device.lower()
        result_list = list()

        print("start fit ...")

        self.to(device)
        for e in range(epochs):

            # 初始化
            train_loss = 0
            train_accuracy = 0
            result_dict = dict()

            total = 0

            for step, (sentence_vec, keyword_vec, label) in enumerate(train_loader):
                # 训练模式
                self.train()

                sentence_vec = sentence_vec.to(device)
                keyword_vec = keyword_vec.to(device)
                label = label.to(device)

                # 每次训练之前，对梯度清零
                optimizer.zero_grad()

                output = self.forward(sentence_vec, keyword_vec)

                # 计算损失
                loss = criterion(output, label).to(device)
                # 保存损失
                train_loss += loss.item()
                # 获取训练的准确率
                train_accuracy += self.compute_accuracy(output, label)

                # 反向传播
                loss.backward()

                # 更新权重
                optimizer.step()

                total += len(label)

                print("epochs: {}/{}".format(e + 1, epochs),
                      "step: {}/{}".format(step + 1, len(train_loader)),
                      "train loss: {:.4f}".format(train_loss / (step + 1)),
                      "train accuracy: {:.2f}%".format(train_accuracy / (step + 1) * 100),
                      "device: {}".format(device),
                      end='\r', flush=True)

            result_dict['train_loss'] = train_loss / len(train_loader)
            result_dict['train_accuracy'] = train_accuracy / total * 100

            print()
            if valid_loader is not None:
                # 进入验证模式
                self.eval()

                with torch.no_grad():
                    valid_loss, valid_accuracy = self.valid(valid_loader, criterion, device)
                    result_dict['valid_loss'] = valid_loss
                    result_dict['valid_accuracy'] = valid_accuracy

                print("valid_loss: {:.4f}".format(valid_loss),
                      "valid_accuracy: {:.2f}%".format(valid_accuracy))

                self.train()

            if e == 0:
                save_model(self, save_dir=save_dir, file_name=save_filename)
            else:
                self.save_best_model(train_loss=result_dict['train_loss'], valid_loss=result_dict['valid_loss'],
                                     records=result_list, save_dir=save_dir, file_name=save_filename)

            print()

            result_list.append(result_dict)

        return result_list

    def save_best_model(self, train_loss, records, valid_loss=None, save_dir=None, file_name='model.pth'):
        train_loss_list = [record['train_loss'] for record in records]
        train_loss_list = sorted(train_loss_list, reverse=False)

        if valid_loss is not None:
            valid_loss_list = [record['valid_loss'] for record in records]
            valid_loss_list = sorted(valid_loss_list, reverse=False)

            if valid_loss < valid_loss_list[0] and train_loss < train_loss_list[0]:
                save_model(self, save_dir=save_dir, file_name=file_name)
        else:
            if train_loss < train_loss_list[0]:
                save_model(self, save_dir=save_dir, file_name=file_name)

    def valid(self, valid_loader, criterion, device):
        valid_loss = 0
        valid_accuracy = 0

        self.to(device)
        for step, (sentence_vec, keyword_vec, label) in enumerate(valid_loader):
            sentence_vec = sentence_vec.to(device)
            keyword_vec = keyword_vec.to(device)
            label = label.to(device)

            output = self.forward(sentence_vec, keyword_vec)

            loss = criterion(output, label).to(device)
            valid_loss += loss.item()
            valid_accuracy += self.compute_accuracy(output, label)

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = valid_accuracy / len(valid_loader) * 100

        return valid_loss, valid_accuracy

    def test(self, test_loader, criterion, device):
        test_loss = 0
        test_accuracy = 0

        print("start test ...")

        self.to(device)
        for step, (sentence_vec, keyword_vec, label) in enumerate(test_loader):
            sentence_vec = sentence_vec.to(device)
            keyword_vec = keyword_vec.to(device)
            label = label.to(device)

            output = self.forward(sentence_vec, keyword_vec)
            loss = criterion(output, label).to(device)

            test_loss += loss.item()
            test_accuracy += self.compute_accuracy(output, label)

        test_loss = test_loss / len(test_loader)
        test_accuracy = test_accuracy / len(test_loader) * 100

        print("test loss: {:.4f}".format(test_loss),
              "test accuracy: {:.2f}%".format(test_accuracy))

        return test_loss, test_accuracy

    def compute_accuracy(self, output, label):
        """
        计算准确性
        :param output: <Tensor> 输出结果
        :param label: <Tensor> 对应的标签
        :return: <int> 准确数
        """
        output.max(dim=1)[1]
        return (output.max(dim=1)[1] == label).sum().item() / len(label)


class DualNetwork(nn.Module):
    def __init__(self, vocab_size, embed_size, rnn_hidden_size, rnn_model=LSTM, use_bidirectional=True, dropout=0.2):
        super(DualNetwork, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn_model = rnn_model
        if use_bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        if rnn_model == LSTM:
            self.rnn = nn.LSTM(embed_size, rnn_hidden_size, batch_first=True, bidirectional=use_bidirectional,
                               num_layers=1)
        elif rnn_model == GRU:
            self.rnn = nn.GRU(embed_size, rnn_hidden_size, batch_first=True, bidirectional=use_bidirectional,
                              num_layers=1)

        self.m = nn.Linear(self.num_direction * rnn_hidden_size, self.num_direction * rnn_hidden_size, bias=False)
        self.drop_out = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ask, answer):
        batch_size = ask.size()[0]

        ask_embed = self.embed(ask)
        answer_embed = self.embed(answer)

        if self.rnn_model == GRU:
            _, ask_hidden = self.rnn(ask_embed)
            _, answer_hidden = self.rnn(answer_embed)

        elif self.rnn_model == LSTM:
            _, (ask_hidden, _) = self.rnn(ask_embed)
            _, (answer_hidden, _) = self.rnn(answer_embed)

        ask_hidden = ask_hidden.permute(1, 0, 2)
        answer_hidden = answer_hidden.permute(1, 0, 2)

        ask_hidden = ask_hidden.contiguous().view(batch_size, -1)
        ask_hidden = self.drop_out(self.m(ask_hidden))
        ask_hidden = ask_hidden.view(batch_size, 1, -1)

        answer_hidden = answer_hidden.contiguous().view(batch_size, -1, 1)

        score = torch.bmm(ask_hidden, answer_hidden)
        score = score.view(batch_size)
        return self.sigmoid(score)

    def fit(self, train_loader, optimizer, criterion, device='cpu', epochs=10, valid_loader=None, save_model_dir=None,
            filename='retrieval_model.pth', threshold=0.5, save_final_model=False):

        result_list = list()

        print("start fit ... ")
        self.to(device)
        for e in range(epochs):
            train_loss = 0
            train_accuracy = 0
            result_dict = dict()

            for step, (ask, answer, label) in enumerate(train_loader):
                ask = ask.to(device)
                answer = answer.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                score = self.forward(ask, answer)
                loss = criterion(score, label).to(device)
                train_loss += loss.item()

                accuracy = get_accuracy_by_threshold(score, label, threshold)
                train_accuracy += accuracy

                loss.backward()
                optimizer.step()

                print("epochs: {}/{} ".format(e + 1, epochs),
                      "step: {}/{}".format(step + 1, len(train_loader)),
                      "device: {}".format(device),
                      "train loss: {:.4f}".format(train_loss / (step + 1)),
                      "train accuracy: {:.2f}%".format(train_accuracy / (step + 1) * 100),
                      end='\r', flush=True)

            train_loss = train_loss / len(train_loader)
            train_accuracy = train_accuracy / len(train_loader)
            print()

            if valid_loader is not None:
                self.eval()

                valid_loss, valid_accuracy, valid_precision = self.valid(valid_loader, criterion, device, threshold)

                print("valid loss: {:.4f}".format(valid_loss),
                      "valid accuracy: {:.2f}%".format(valid_accuracy * 100),
                      "valid precision: {:.2f}".format(valid_precision))

                self.train()

            if e == 0:
                save_model(self, save_dir=save_model_dir, file_name=filename)
            else:
                if valid_loader is not None:
                    save_best_model(self, train_loss, result_list, valid_loss, save_dir=save_model_dir,
                                    file_name=filename)
                else:
                    save_best_model(self, train_loss, result_list, None, save_dir=save_model_dir,
                                    file_name=filename)
            if save_final_model:
                if e == (epochs - 1):
                    file = filename + "(final)"
                    save_model(self, save_dir=save_model_dir, file_name=file)

            result_dict['train_loss'] = train_loss
            result_dict['train_accuracy'] = train_accuracy

            if valid_loader is not None:
                result_dict['valid_loss'] = valid_loss
                result_dict['valid_precision'] = valid_precision
                result_dict['valid_accuracy'] = valid_accuracy

            result_list.append(result_dict)

            print()

        return result_list

    def valid(self, valid_loader, criterion, device='cpu', threshold=0.5):
        valid_loss = 0
        valid_accuracy = 0
        valid_tp, valid_fp = 0, 0

        self.to(device)
        for ask, answer, label in valid_loader:
            ask = ask.to(device)
            answer = answer.to(device)
            label = label.to(device)

            score = self.forward(ask, answer)
            loss = criterion(score, label).to(device)
            valid_loss += loss.item()

            accuracy = get_accuracy_by_threshold(score, label, threshold)
            valid_accuracy += accuracy

            tp, _, fp, _ = confusion_matrix_by_threshold(score, label, threshold)
            valid_tp += tp
            valid_fp += fp

        valid_loss = valid_loss / len(valid_loader)
        valid_accuracy = valid_accuracy / len(valid_loader)
        valid_precision = get_precision(valid_tp, valid_fp)

        return valid_loss, valid_accuracy, valid_precision

    def test(self, test_loader, criterion, device='cpu', threshold=0.5):
        test_loss = 0
        test_accuracy = 0
        test_tp, test_fp = 0, 0

        self.to(device)
        for ask, answer, label in test_loader:
            ask = ask.to(device)
            answer = answer.to(device)
            label = label.to(device)

            score = self.forward(ask, answer)
            loss = criterion(score, label).to(device)
            test_loss += loss.item()

            accuracy = get_accuracy_by_threshold(score, label, threshold)
            test_accuracy += accuracy

            tp, _, fp, _ = confusion_matrix_by_threshold(score, label, threshold)
            test_tp += tp
            test_fp += fp

        test_loss = test_loss / len(test_loader)
        test_accuracy = test_accuracy / len(test_loader)
        test_precision = get_precision(test_tp, test_fp)

        print("test loss: {:.4f}".format(test_loss),
              "test accuracy: {:.2f}%".format(test_accuracy * 100),
              "test precision: {:.2f}".format(test_precision))

        return test_loss, test_precision


def test():
    ask = torch.LongTensor([[1, 2, 3, 4, 5, 0, 0, 0, 0]])
    answer = torch.LongTensor([[8, 7, 6, 4, 6, 0, 0, 0, 0]])
    ask_keyword = torch.LongTensor([[1, 2, 3, 0, 0, 0, 0, 0, 0]])
    answer_keyword = torch.LongTensor([[8, 7, 6, 0, 0, 0, 0, 0, 0]])

    # model = CategoryNetwork(6, 10, 32, 9, 2)
    # model = SCNNetwork(vocab_size=9, embed_size=16, seq_len=9, hidden_size=8, rnn_model='lstm',
    #                     use_bidirectional=True)
    model = DualNetwork(vocab_size=9, embed_size=16, rnn_hidden_size=8, rnn_model='lstm',
                        use_bidirectional=True)

    # model = SANNetwork(vocab_size=9, embed_size=16, rnn_hidden_size=8, rnn_model=GRU, use_bidirectional=False)

    print(model)
    score = model.forward(ask, answer)
    # score = model.forward(ask, answer)
    print(score)


# def test():
#     from words_sequence.word_sequences import load_word_sequences
#     from module.core.loading_dataset import loading_retrieval_data
#     from torch.optim import Adam
#     from torch.nn import BCELoss, NLLLoss
#     word_sequence = load_word_sequences('../../save_word_sequence/word_sequence.pickle')
#     train_csv = '../../chinese_corpus/target/qa_dataset/train.csv'
#     valid_csv = '../../chinese_corpus/target/qa_dataset/valid.csv'
#     test_csv = '../../chinese_corpus/target/qa_dataset/test.csv'
#
#     train_loader, valid_loader, test_loader = loading_retrieval_data(train_csv, valid_csv, test_csv, max_len=60,
#                                                                      word_sequence=word_sequence, batch_size=32)
#
#     # model = QaNetwork(vocab_size=len(word_sequence.word_dict), embed_size=200, rnn_hidden_size=64, seq_len=60,
#     #                   rnn_model='lstm', window_size=[2, 3, 4, 5])
#     # model = SCNNetwork(vocab_size=len(word_sequence.word_dict), embed_size=200, hidden_size=40, seq_len=60,
#     #                   use_bidirectional=True, rnn_model='lstm', conv_out_channels=[8, 8, 8],
#     #                   conv_kernel_size=[2, 3, 4])
#     # model = RetrievalNetwork(vocab_size=len(word_sequence.word_dict), embed_size=200, rnn_hidden_size=128,
#     #                         rnn_model='lstm', use_bidirectional=True)
#     model = SANNetwork(vocab_size=len(word_sequence.word_dict), embed_size=200, rnn_hidden_size=96, rnn_model=LSTM,
#                        use_bidirectional=True)
#
#     print(model)
#
#     optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.000006)
#     # criterion = BCELoss()
#     criterion = NLLLoss()
#     model.fit(train_loader=train_loader, optimizer=optimizer, criterion=criterion, valid_loader=valid_loader,
#               epochs=10)
#     model.test(test_loader=test_loader, criterion=criterion)

# def test():
#     sent1 = '这只皮靴号码大了。那只号码合适'
#     sent2 = '这只皮靴号码不小，那只更合适'
#
#     similarity = cos_similarity(sent1, sent2)
#     print("similarity: ", similarity)


if __name__ == '__main__':
    test()

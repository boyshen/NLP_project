
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size, num_layers=1, drop_out=0.0):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size

        self.layer_embedding = nn.Embedding(vocab_size, embed_size)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=drop_out)

        self.fc = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden=None):
        embeds = self.layer_embedding(input)

        if hidden:
            lstm_output, hidden_state = self.lstm(embeds, hidden)

        else:
            lstm_output, hidden_state = self.lstm(embeds)

        output = self.fc(lstm_output[:, -1])

        output = self.log_softmax(output)

        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


class FNN_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_num, dropout=0.2):
        super(FNN_Model, self).__init__()
        assert isinstance(hidden_num, (list, tuple)), \
            "hidden_num type must tuple or list"

        self.hidden_num = hidden_num
        self.input_layer = nn.Linear(input_size, hidden_num[0])

        if len(hidden_num) >= 2:
            hidden_layer_size = zip(hidden_num[:-1], hidden_num[1:])
            self.hidden_layers = nn.ModuleList([nn.Linear(h1, h2) for h1, h2 in hidden_layer_size])

        self.output_layer = nn.Linear(hidden_num[-1], output_size)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, inputs):

        output = F.relu(self.input_layer(inputs))

        if len(self.hidden_num) >= 2:
            for hidden_layer in self.hidden_layers:
                output = self.drop_out(F.relu(hidden_layer(output)))

        output = self.output_layer(output)

        return F.log_softmax(output, dim=1)

def test_lstm_model():
    lstm_model = LSTM_Model(256, 256, 256, 2)
    print(lstm_model)

    import numpy as np
    test_data = np.random.randint(0, 64, size=(32, 25))
    output = lstm_model.forward(torch.LongTensor(test_data))
    print(output)
    print(output.shape)

def test_fnn_model():
    fnn_model = FNN_Model(256, 2, [256, 256])
    print(fnn_model)

    test_data = torch.rand(1, 256)
    output = fnn_model.forward(test_data)
    print(output)

if __name__ == '__main__':
    # test_lstm_model()
    test_fnn_model()


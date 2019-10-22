from torch.optim import Adam
from torch.nn import NLLLoss
from global_variable import LSTM
from module.core.mysql_table import qa_table, joke_table, dialog_table
from module.core.data_utils import split_category_dataset_by_table
from module.core.loading_dataset import loading_category_data
from module.core.network_model import CategoryNetwork, summary


class SemanticsDistinguish(object):

    def __init__(self, mysql, word_sequence,
                 embed_size=200,
                 rnn_hidden_size=256,
                 seq_len=30,
                 output_size=3,
                 rnn_model=LSTM,
                 window_size=[2, 3, 4],
                 cnn_output_feature=2,
                 drop_out=0.2,
                 lr=0.001,
                 epochs=10,
                 device='cpu',
                 weight_decay=0.00001,
                 batch_size=32):
        self.mysql = mysql
        self.word_sequence = word_sequence
        self.embed_size = embed_size
        self.rnn_hidden_size = rnn_hidden_size
        self.seq_len = seq_len
        self.output_size = output_size
        self.rnn_model = rnn_model
        self.window_size = window_size
        self.drop_out = drop_out
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.cnn_output_feature = cnn_output_feature
        self.model = None

    def split_dataset(self, valid_size=0.1, test_size=0.1, save_path=None):
        table = [[qa_table], [dialog_table], [joke_table]]
        n_class = ['knowledge', 'dialog', 'joke']

        train_csv, valid_csv, test_csv = split_category_dataset_by_table(self.mysql, table, n_class, valid_size,
                                                                         test_size, save_path)

        return train_csv, valid_csv, test_csv

    def fit_model(self, train_csv, valid_csv, test_csv, save_path=None, save_model='semantics_model.pth'):
        train_loader, valid_loader, test_loader = loading_category_data(train_csv, valid_csv, test_csv,
                                                                        max_len=self.seq_len,
                                                                        word_sequence=self.word_sequence,
                                                                        batch_size=self.batch_size)

        model = CategoryNetwork(vocab_size=len(self.word_sequence.word_dict),
                                embed_size=self.embed_size,
                                rnn_hidden_size=self.rnn_hidden_size,
                                seq_len=self.seq_len,
                                output_size=self.output_size,
                                model=self.rnn_model,
                                cnn_output_feature=self.cnn_output_feature,
                                window_size=self.window_size,
                                drop_out=self.drop_out)

        summary(model)

        self.model = model

        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = NLLLoss()

        self.model.fit(train_loader=train_loader,
                       valid_loader=valid_loader,
                       optimizer=optimizer,
                       criterion=criterion,
                       device=self.device,
                       epochs=self.epochs,
                       save_dir=save_path,
                       save_filename=save_model)

        self.model.test(test_loader=test_loader, criterion=criterion, device=self.device)

from torch.optim import Adam
from torch.nn import NLLLoss
from module.core.data_utils import split_retrieval_dataset_by_table
from module.core.loading_dataset import loading_retrieval_data
from module.core.mysql_table import dialog_table
from module.core.network_model import SANNetwork, summary
from global_variable import LSTM


class DialogRetrieval(object):
    def __init__(self, mysql, word_sequence,
                 embed_size=200,
                 rnn_hidden_size=256,
                 seq_len=30,
                 output_size=3,
                 rnn_model=LSTM,
                 drop_out=0.2,
                 lr=0.001,
                 epochs=10,
                 device='cpu',
                 weight_decay=0.00001,
                 batch_size=32,
                 use_bidirectional=True):
        self.mysql = mysql
        self.word_sequence = word_sequence
        self.embed_size = embed_size
        self.rnn_hidden_size = rnn_hidden_size
        self.seq_len = seq_len
        self.output_size = output_size
        self.rnn_model = rnn_model
        self.drop_out = drop_out
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_bidirectional = use_bidirectional
        self.model = None

    def split_dataset(self, valid_size=0.1, test_size=0.1, error_ratio=5, save_path=None, thread_num=1):
        table = dialog_table
        train_csv, valid_csv, test_csv = split_retrieval_dataset_by_table(mysql=self.mysql,
                                                                          table=table,
                                                                          valid_size=valid_size,
                                                                          test_size=test_size,
                                                                          error_ratio=error_ratio,
                                                                          save_path=save_path,
                                                                          thread_num=thread_num)

        return train_csv, valid_csv, test_csv

    def fit_model(self, train_csv, valid_csv, test_csv, save_path=None, save_model='dialog_retrieval_model.pth',
                  save_final_model=False):
        train_loader, valid_loader, test_loader = loading_retrieval_data(train_csv, valid_csv, test_csv,
                                                                         max_len=self.seq_len,
                                                                         word_sequence=self.word_sequence,
                                                                         batch_size=self.batch_size)

        model = SANNetwork(vocab_size=len(self.word_sequence.word_dict),
                           embed_size=self.embed_size,
                           rnn_hidden_size=self.rnn_hidden_size,
                           rnn_model=self.rnn_model,
                           output_size=self.output_size,
                           use_bidirectional=self.use_bidirectional,
                           dropout=self.drop_out)

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
                       model_file=save_model,
                       save_final_model=save_final_model)

        self.model.test(test_loader=test_loader, criterion=criterion, device=self.device)

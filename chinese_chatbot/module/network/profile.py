from torch.optim import Adam
from torch.nn import NLLLoss
from module.core.mysql_table import profile_table, joke_table, qa_table
from module.core.data_utils import split_category_dataset_by_table, split_retrieval_dataset_by_table
from module.core.loading_dataset import loading_category_data, loading_retrieval_data
from module.core.network_model import CategoryNetwork, SANNetwork, summary
from global_variable import LSTM


class ProfileInfo(object):

    def __int__(self, mysql, word_sequence,
                embed_size=200,
                rnn_hidden_size=256,
                seq_len=30,
                output_size=2,
                rnn_model=LSTM,
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
        self.drop_out = drop_out
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.weight_decay = weight_decay
        self.batch_size = batch_size


class ProfileInfoRetrieval(ProfileInfo):
    def __init__(self, mysql, word_sequence,
                 embed_size=200,
                 rnn_hidden_size=256,
                 seq_len=30,
                 output_size=2,
                 rnn_model=LSTM,
                 drop_out=0.2,
                 lr=0.001,
                 epochs=10,
                 device='cpu',
                 weight_decay=0.00001,
                 batch_size=32,
                 use_bidirectional=False):
        super(ProfileInfoRetrieval, self).__int__(mysql=mysql,
                                                  word_sequence=word_sequence,
                                                  embed_size=embed_size,
                                                  rnn_hidden_size=rnn_hidden_size,
                                                  seq_len=seq_len,
                                                  output_size=output_size,
                                                  rnn_model=rnn_model,
                                                  drop_out=drop_out,
                                                  lr=lr,
                                                  epochs=epochs,
                                                  device=device,
                                                  weight_decay=weight_decay,
                                                  batch_size=batch_size)
        self.use_bidirectional = use_bidirectional
        self.model = None

    def split_dataset(self, valid_size=0.1, test_size=0.1, error_ratio=5, save_path=None, thread_num=1):
        table = profile_table
        train_csv, valid_csv, test_csv = split_retrieval_dataset_by_table(mysql=self.mysql,
                                                                          table=table,
                                                                          valid_size=valid_size,
                                                                          test_size=test_size,
                                                                          error_ratio=error_ratio,
                                                                          save_path=save_path,
                                                                          thread_num=thread_num)

        return train_csv, valid_csv, test_csv

    def fit_model(self, train_csv, valid_csv, test_csv, save_path=None, save_model='profile_retrieval_model.pth',
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


class ProfileInfoDistinguish(ProfileInfo):

    def __init__(self, mysql, word_sequence,
                 embed_size=200,
                 rnn_hidden_size=256,
                 seq_len=30,
                 output_size=2,
                 rnn_model=LSTM,
                 drop_out=0.2,
                 cnn_output_feature=2,
                 window_size=[2, 3, 4],
                 lr=0.001,
                 epochs=10,
                 device='cpu',
                 weight_decay=0.00001,
                 batch_size=32):
        super(ProfileInfoDistinguish, self).__int__(mysql=mysql,
                                                    word_sequence=word_sequence,
                                                    embed_size=embed_size,
                                                    rnn_hidden_size=rnn_hidden_size,
                                                    seq_len=seq_len,
                                                    output_size=output_size,
                                                    rnn_model=rnn_model,
                                                    drop_out=drop_out,
                                                    lr=lr,
                                                    epochs=epochs,
                                                    device=device,
                                                    weight_decay=weight_decay,
                                                    batch_size=batch_size)
        self.cnn_output_feature = cnn_output_feature
        self.window_size = window_size
        self.model = None

    def split_dataset(self, valid_size=0.1, test_size=0.1, save_path=None):
        table = [[profile_table], [joke_table, qa_table]]
        n_class = ['profile', 'other']

        train_csv, valid_csv, valid_csv = split_category_dataset_by_table(self.mysql,
                                                                          tables=table,
                                                                          n_class=n_class,
                                                                          valid_size=valid_size,
                                                                          test_size=test_size,
                                                                          save_path=save_path)

        return train_csv, valid_csv, valid_csv,

    def fit_model(self, train_csv, valid_csv, test_csv, save_path=None, save_model="profile_model.pth"):
        train_loader, valid_loader, test_loader = loading_category_data(train_csv=train_csv,
                                                                        valid_csv=valid_csv,
                                                                        test_csv=test_csv,
                                                                        word_sequence=self.word_sequence,
                                                                        max_len=self.seq_len,
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
                       epochs=self.epochs,
                       device=self.device,
                       save_dir=save_path,
                       save_filename=save_model)

        self.model.test(test_loader=test_loader, criterion=criterion, device=self.device)

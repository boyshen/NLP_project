import os
import sys

try:
    from module.network.knowledge import KnowledgeCategory, KnowledgeRetrieval
    from module.network.profile import ProfileInfoDistinguish, ProfileInfoRetrieval
    from module.network.semantics import SemanticsDistinguish
    from module.network.dialog import DialogRetrieval
    from module.network.score import ScoreNetwork
    from global_variable import TRUE, FALSE
except ModuleNotFoundError:
    sys.path.append('../../')
    from module.network.knowledge import KnowledgeCategory, KnowledgeRetrieval
    from module.network.profile import ProfileInfoDistinguish, ProfileInfoRetrieval
    from module.network.semantics import SemanticsDistinguish
    from module.network.dialog import DialogRetrieval
    from module.network.score import ScoreNetwork
    from global_variable import TRUE, FALSE


def fit_profile_retrieval_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename):
    print("start init ...")
    dataset_config = config_json['profile_retrieval_model']['split_dataset']
    network_config = config_json['profile_retrieval_model']['network']
    retrieval_network = ProfileInfoRetrieval(mysql=mysql,
                                             word_sequence=word_sequence,
                                             embed_size=network_config['embed_size'],
                                             rnn_hidden_size=network_config['rnn_hidden_size'],
                                             seq_len=network_config['seq_len'],
                                             output_size=network_config['output_size'],
                                             rnn_model=network_config['rnn_model'],
                                             drop_out=network_config['drop_out'],
                                             lr=network_config['learning_rate'],
                                             epochs=network_config['epochs'],
                                             device=network_config['device'],
                                             weight_decay=network_config['weight_decay'],
                                             batch_size=network_config['batch_size'],
                                             use_bidirectional=True)

    print()
    print("start split dataset ...")
    if os.path.isfile(os.path.join(dataset_save_path, 'train.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'valid.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'test.csv')):

        print("Checking that the {} file exists will not re-divide the data".format(
            str(['train.csv', 'valid.csv', 'test.csv'])))
        train_csv = os.path.join(dataset_save_path, 'train.csv')
        valid_csv = os.path.join(dataset_save_path, 'valid.csv')
        test_csv = os.path.join(dataset_save_path, 'test.csv')

    else:
        train_csv, valid_csv, test_csv = retrieval_network.split_dataset(valid_size=dataset_config['valid_size'],
                                                                         test_size=dataset_config['test_size'],
                                                                         error_ratio=dataset_config['error_ratio'],
                                                                         save_path=dataset_save_path,
                                                                         thread_num=dataset_config['thread_num'])

    print()
    print("start fit network ...")
    save_final_model = True if network_config['save_final_model'] == TRUE else False
    retrieval_network.fit_model(train_csv, valid_csv, test_csv,
                                save_path=model_save_path,
                                save_model=filename,
                                save_final_model=save_final_model)

    print("fit over ! ")


def fit_profile_distinguish_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename):
    print("start init ...")
    dataset_config = config_json['profile_distinguish_model']['split_dataset']
    network_config = config_json['profile_distinguish_model']['network']
    category_network = ProfileInfoDistinguish(mysql=mysql,
                                              word_sequence=word_sequence,
                                              embed_size=network_config['embed_size'],
                                              rnn_hidden_size=network_config['rnn_hidden_size'],
                                              seq_len=network_config['seq_len'],
                                              output_size=network_config['output_size'],
                                              rnn_model=network_config['rnn_model'],
                                              drop_out=network_config['drop_out'],
                                              lr=network_config['learning_rate'],
                                              epochs=network_config['epochs'],
                                              device=network_config['device'],
                                              weight_decay=network_config['weight_decay'],
                                              batch_size=network_config['batch_size'])

    print()
    print("start split dataset ...")
    if os.path.isfile(os.path.join(dataset_save_path, 'train.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'valid.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'test.csv')):

        print("Checking that the {} file exists will not re-divide the data".format(
            str(['train.csv', 'valid.csv', 'test.csv'])))
        train_csv = os.path.join(dataset_save_path, 'train.csv')
        valid_csv = os.path.join(dataset_save_path, 'valid.csv')
        test_csv = os.path.join(dataset_save_path, 'test.csv')

    else:
        train_csv, valid_csv, test_csv = category_network.split_dataset(valid_size=dataset_config['valid_size'],
                                                                        test_size=dataset_config['test_size'],
                                                                        save_path=dataset_save_path)

    print()
    print("start fit network ...")
    category_network.fit_model(train_csv, valid_csv, test_csv, save_path=model_save_path, save_model=filename)

    print("fit over ! ")


def fit_semantics_distinguish_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename):
    print("start init ...")
    dataset_config = config_json['semantics_distinguish_model']['split_dataset']
    network_config = config_json['semantics_distinguish_model']['network']
    model = SemanticsDistinguish(mysql=mysql,
                                 word_sequence=word_sequence,
                                 embed_size=network_config['embed_size'],
                                 rnn_hidden_size=network_config['rnn_hidden_size'],
                                 seq_len=network_config['seq_len'],
                                 output_size=network_config['output_size'],
                                 rnn_model=network_config['rnn_model'],
                                 drop_out=network_config['drop_out'],
                                 lr=network_config['learning_rate'],
                                 epochs=network_config['epochs'],
                                 device=network_config['device'],
                                 weight_decay=network_config['weight_decay'],
                                 batch_size=network_config['batch_size'])

    print()
    print("start split dataset ...")
    if os.path.isfile(os.path.join(dataset_save_path, 'train.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'valid.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'test.csv')):

        print("Checking that the {} file exists will not re-divide the data".format(
            str(['train.csv', 'valid.csv', 'test.csv'])))
        train_csv = os.path.join(dataset_save_path, 'train.csv')
        valid_csv = os.path.join(dataset_save_path, 'valid.csv')
        test_csv = os.path.join(dataset_save_path, 'test.csv')

    else:
        train_csv, valid_csv, test_csv = model.split_dataset(valid_size=dataset_config['valid_size'],
                                                             test_size=dataset_config['test_size'],
                                                             save_path=dataset_save_path)

    print()
    print("start fit network ...")
    model.fit_model(train_csv, valid_csv, test_csv, save_path=model_save_path, save_model=filename)

    print("fit over ! ")


def fit_knowledge_category_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename):
    print("start init ...")
    dataset_config = config_json['knowledge_category_model']['split_dataset']
    network_config = config_json['knowledge_category_model']['network']
    model = KnowledgeCategory(mysql=mysql,
                              word_sequence=word_sequence,
                              embed_size=network_config['embed_size'],
                              rnn_hidden_size=network_config['rnn_hidden_size'],
                              seq_len=network_config['seq_len'],
                              output_size=network_config['output_size'],
                              rnn_model=network_config['rnn_model'],
                              drop_out=network_config['drop_out'],
                              lr=network_config['learning_rate'],
                              epochs=network_config['epochs'],
                              device=network_config['device'],
                              weight_decay=network_config['weight_decay'],
                              batch_size=network_config['batch_size'])

    print()
    print("start split dataset ...")
    if os.path.isfile(os.path.join(dataset_save_path, 'train.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'valid.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'test.csv')):

        print("Checking that the {} file exists will not re-divide the data".format(
            str(['train.csv', 'valid.csv', 'test.csv'])))
        train_csv = os.path.join(dataset_save_path, 'train.csv')
        valid_csv = os.path.join(dataset_save_path, 'valid.csv')
        test_csv = os.path.join(dataset_save_path, 'test.csv')

    else:
        train_csv, valid_csv, test_csv = model.split_dataset(valid_size=dataset_config['valid_size'],
                                                             test_size=dataset_config['test_size'],
                                                             save_path=dataset_save_path)

    print()
    print("start fit network ...")
    model.fit_model(train_csv, valid_csv, test_csv, save_path=model_save_path, save_model=filename)

    print("fit over ! ")


def fit_knowledge_retrieval_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename):
    print("start init ...")
    dataset_config = config_json['knowledge_retrieval_model']['split_dataset']
    network_config = config_json['knowledge_retrieval_model']['network']
    retrieval_network = KnowledgeRetrieval(mysql=mysql,
                                           word_sequence=word_sequence,
                                           embed_size=network_config['embed_size'],
                                           rnn_hidden_size=network_config['rnn_hidden_size'],
                                           seq_len=network_config['seq_len'],
                                           output_size=network_config['output_size'],
                                           rnn_model=network_config['rnn_model'],
                                           drop_out=network_config['drop_out'],
                                           lr=network_config['learning_rate'],
                                           epochs=network_config['epochs'],
                                           device=network_config['device'],
                                           weight_decay=network_config['weight_decay'],
                                           batch_size=network_config['batch_size'],
                                           use_bidirectional=True)

    print()
    print("start split dataset ...")
    if os.path.isfile(os.path.join(dataset_save_path, 'train.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'valid.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'test.csv')):

        print("Checking that the {} file exists will not re-divide the data".format(
            str(['train.csv', 'valid.csv', 'test.csv'])))
        train_csv = os.path.join(dataset_save_path, 'train.csv')
        valid_csv = os.path.join(dataset_save_path, 'valid.csv')
        test_csv = os.path.join(dataset_save_path, 'test.csv')

    else:
        train_csv, valid_csv, test_csv = retrieval_network.split_dataset(valid_size=dataset_config['valid_size'],
                                                                         test_size=dataset_config['test_size'],
                                                                         error_ratio=dataset_config['error_ratio'],
                                                                         save_path=dataset_save_path,
                                                                         thread_num=dataset_config['thread_num'])

    print()
    print("start fit network ...")
    save_final_model = True if network_config['save_final_model'] == TRUE else False
    retrieval_network.fit_model(train_csv, valid_csv, test_csv, save_path=model_save_path, save_model=filename,
                                save_final_model=save_final_model)

    print("fit over ! ")


def fit_dialog_retrieval_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename):
    print("start init ...")
    dataset_config = config_json['dialog_retrieval_model']['split_dataset']
    network_config = config_json['dialog_retrieval_model']['network']
    retrieval_network = DialogRetrieval(mysql=mysql,
                                        word_sequence=word_sequence,
                                        embed_size=network_config['embed_size'],
                                        rnn_hidden_size=network_config['rnn_hidden_size'],
                                        seq_len=network_config['seq_len'],
                                        output_size=network_config['output_size'],
                                        rnn_model=network_config['rnn_model'],
                                        drop_out=network_config['drop_out'],
                                        lr=network_config['learning_rate'],
                                        epochs=network_config['epochs'],
                                        device=network_config['device'],
                                        weight_decay=network_config['weight_decay'],
                                        batch_size=network_config['batch_size'],
                                        use_bidirectional=True)

    print()
    print("start split dataset ...")
    if os.path.isfile(os.path.join(dataset_save_path, 'train.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'valid.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'test.csv')):

        print("Checking that the {} file exists will not re-divide the data".format(
            str(['train.csv', 'valid.csv', 'test.csv'])))
        train_csv = os.path.join(dataset_save_path, 'train.csv')
        valid_csv = os.path.join(dataset_save_path, 'valid.csv')
        test_csv = os.path.join(dataset_save_path, 'test.csv')

    else:
        train_csv, valid_csv, test_csv = retrieval_network.split_dataset(valid_size=dataset_config['valid_size'],
                                                                         test_size=dataset_config['test_size'],
                                                                         error_ratio=dataset_config['error_ratio'],
                                                                         save_path=dataset_save_path,
                                                                         thread_num=dataset_config['thread_num'])

    print()
    print("start fit network ...")
    save_final_model = True if network_config['save_final_model'] == TRUE else False
    retrieval_network.fit_model(train_csv, valid_csv, test_csv, save_path=model_save_path, save_model=filename,
                                save_final_model=save_final_model)

    print("fit over ! ")


def fit_score_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename):
    print("start init ...")
    dataset_config = config_json['score_model']['split_dataset']
    network_config = config_json['score_model']['network']
    model = ScoreNetwork(mysql=mysql,
                         word_sequence=word_sequence,
                         embed_size=network_config['embed_size'],
                         rnn_hidden_size=network_config['rnn_hidden_size'],
                         seq_len=network_config['seq_len'],
                         rnn_model=network_config['rnn_model'],
                         drop_out=network_config['drop_out'],
                         lr=network_config['learning_rate'],
                         epochs=network_config['epochs'],
                         device=network_config['device'],
                         weight_decay=network_config['weight_decay'],
                         batch_size=network_config['batch_size'],
                         use_bidirectional=True,
                         threshold=network_config['threshold'])

    print()
    print("start split dataset ...")
    if os.path.isfile(os.path.join(dataset_save_path, 'train.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'valid.csv')) \
            and os.path.isfile(os.path.join(dataset_save_path, 'test.csv')):

        print("Checking that the {} file exists will not re-divide the data".format(
            str(['train.csv', 'valid.csv', 'test.csv'])))
        train_csv = os.path.join(dataset_save_path, 'train.csv')
        valid_csv = os.path.join(dataset_save_path, 'valid.csv')
        test_csv = os.path.join(dataset_save_path, 'test.csv')

    else:
        train_csv, valid_csv, test_csv = model.split_dataset(valid_size=dataset_config['valid_size'],
                                                             test_size=dataset_config['test_size'],
                                                             error_ratio=dataset_config['error_ratio'],
                                                             save_path=dataset_save_path,
                                                             thread_num=dataset_config['thread_num'])

    print()
    print("start fit network ...")
    save_final_model = True if network_config['save_final_model'] == TRUE else False
    model.fit_model(train_csv, valid_csv, test_csv, save_path=model_save_path, save_model=filename,
                    save_final_model=save_final_model)

    print("fit over ! ")

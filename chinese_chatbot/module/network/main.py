#!／usr/bin/env python3
# 创建日期：2019.8.25
# 发布日期: 17/10/2019
# author: shenpinggang
# version: 1.0
#
# PURPOSE：
# (1)训练网络模型
#
# 使用：
# python main.py <config.json> -m <>
#   -m <fit_profile_distinguish,   训练人格信息识别网络
#       fit_profile_retrieval,     训练人格信息检索网络
#       fit_semantics,             训练语意识别网络
#       fit_knowledge_category,    训练知识识别网络
#       fit_knowledge_retrieval,   训练知识检索网络
#       fit_dialog_retrieval,      训练对话检索网络
#       fit_score_model>           训练评分网络
#
#   --dataset_save_path             指定划分的数据集保存路径
#   --model_save_path               指定保存训练模型路径
#   --filename                      指定保存模型的文件名
#
#
# Example:
# python main.py config.json \
#   -m fit_semantics \
#   --dataset_save_path ../../chinese_corpus/target/semantics_distinguish_dataset \
#   --model_save_path ../../save_network_model/ \
#   --filename semantics_distinguish_model.pth
#
#  config.json           训练参数配置文件
#  -m fit_semantics      指定训练句意识别网络。
#  --dataset_save_path ../../chinese_corpus/target/semantics_distinguish_dataset  划分数据集保存到该目录
#  --model_save_path ../../save_network_model/     训练模型保存目录
#  --filename semantics_distinguish_model.pth  模型保存文件名
#

import os
import sys
import json
import argparse
import torch

try:
    from module.core.mysql_exec import Mysql
    from module.core.word_sequences import load_word_sequences
    from module.network.build import fit_knowledge_category_network
    from module.network.build import fit_profile_distinguish_network
    from module.network.build import fit_profile_retrieval_network
    from module.network.build import fit_semantics_distinguish_network
    from module.network.build import fit_knowledge_retrieval_network
    from module.network.build import fit_dialog_retrieval_network
    from module.network.build import fit_score_network
    import global_variable as v


except ModuleNotFoundError:
    sys.path.append('../../')
    from module.core.mysql_exec import Mysql
    from module.core.word_sequences import load_word_sequences
    from module.network.build import fit_knowledge_category_network
    from module.network.build import fit_profile_distinguish_network
    from module.network.build import fit_profile_retrieval_network
    from module.network.build import fit_semantics_distinguish_network
    from module.network.build import fit_knowledge_retrieval_network
    from module.network.build import fit_dialog_retrieval_network
    from module.network.build import fit_score_network
    import global_variable as v

FIT_PROFILE_DISTINGUISH_MODEL = 'fit_profile_distinguish'
FIT_PROFILE_RETRIEVAL_MODEL = 'fit_profile_retrieval'
FIT_SEMANTICS_DISTINGUISH_MODEL = 'fit_semantics'
FIT_KNOWLEDGE_CATEGORY_MODEL = 'fit_knowledge_category'
FIT_KNOWLEDGE_RETRIEVAL_MODEL = 'fit_knowledge_retrieval'
FIT_DIALOG_RETRIEVAL_MODEL = 'fit_dialog_retrieval'
FIT_SCORE_MODEL = 'fit_score_model'

MODE_LIST = [FIT_PROFILE_DISTINGUISH_MODEL,
             FIT_KNOWLEDGE_CATEGORY_MODEL,
             FIT_PROFILE_RETRIEVAL_MODEL,
             FIT_KNOWLEDGE_CATEGORY_MODEL,
             FIT_SEMANTICS_DISTINGUISH_MODEL,
             FIT_KNOWLEDGE_RETRIEVAL_MODEL,
             FIT_DIALOG_RETRIEVAL_MODEL,
             FIT_SCORE_MODEL]

DEBUG = False


def input_args():
    parser = argparse.ArgumentParser(description="1.fit profile information distinguish network model")

    parser.add_argument("config_json_file", metavar="config.json file",
                        help="Configuration files for data utils")

    parser.add_argument("-m", dest="mode", action='store', choices=MODE_LIST,
                        help="Select mode: {}, "
                             "1.fit_profile: fit profile information distinguish network model！"
                             "2.".format(MODE_LIST))

    parser.add_argument("--dataset_save_path", dest="dataset_save_path", action='store', type=str, default=os.getcwd(),
                        help="Data Set Preservation Path, Default current directory")

    parser.add_argument("--model_save_path", dest="model_save_path", action='store', type=str, default=os.getcwd(),
                        help="Data Set Preservation Path, Default current directory")

    parser.add_argument("--filename", dest="filename", action='store', type=str, default="model.pth",
                        help="save model, word_sequence file name! default: file.pickle")

    args = parser.parse_args()

    return args


def print_args(args, config_json):
    print("--------------------------------config args----------------------------------")
    print(" mode                     : {}".format(args.mode))
    print(" config json              : {}".format(args.config_json_file))
    print(" word sequence file       : {}".format(config_json['word_sequence_file']))

    if args.mode == FIT_KNOWLEDGE_CATEGORY_MODEL:
        config = config_json['knowledge_category_model']

    elif args.mode == FIT_PROFILE_DISTINGUISH_MODEL:
        config = config_json['profile_distinguish_model']

    elif args.mode == FIT_PROFILE_RETRIEVAL_MODEL:
        config = config_json['profile_retrieval_model']

    elif args.mode == FIT_SEMANTICS_DISTINGUISH_MODEL:
        config = config_json['semantics_distinguish_model']

    elif args.mode == FIT_KNOWLEDGE_RETRIEVAL_MODEL:
        config = config_json['knowledge_retrieval_model']

    elif args.mode == FIT_DIALOG_RETRIEVAL_MODEL:
        config = config_json['dialog_retrieval_model']

    elif args.mode == FIT_SCORE_MODEL:
        config = config_json['score_model']
        print(" threshold                : {}".format(config['network']['threshold']))

    if args.mode == FIT_DIALOG_RETRIEVAL_MODEL or \
            args.mode == FIT_KNOWLEDGE_RETRIEVAL_MODEL or \
            args.mode == FIT_PROFILE_RETRIEVAL_MODEL or \
            args.mode == FIT_SCORE_MODEL:
        print(" thread num               : {}".format(config['split_dataset']['thread_num']))
        print(" error ratio              : {}".format(config['split_dataset']['error_ratio']))

    print(" valid dataset size       : {}".format(config['split_dataset']['valid_size']))
    print(" test dataset size        : {}".format(config['split_dataset']['test_size']))
    print()

    print(" network embed size       : {}".format(config['network']['embed_size']))
    print(" network rnn hidden size  : {}".format(config['network']['rnn_hidden_size']))
    print(" network sequence length  : {}".format(config['network']['seq_len']))
    print(" network output size      : {}".format(config['network']['output_size']))
    print(" network rnn model        : {}".format(config['network']['rnn_model']))
    print(" network dropout          : {}".format(config['network']['drop_out']))
    print(" network learning rate    : {}".format(config['network']['learning_rate']))
    print(" network epochs           : {}".format(config['network']['epochs']))
    print(" network device           : {}".format(config['network']['device']))
    print(" network weight decay     : {}".format(config['network']['weight_decay']))
    print(" network batch size       : {}".format(config['network']['batch_size']))
    print()

    print(" dataset save path        : {}".format(args.dataset_save_path))
    print(" model save path          : {}".format(args.model_save_path))
    print(" save model file          : {}".format(args.filename))

    print("-----------------------------------------------------------------------------")
    print()


def check_file_is_exists(file):
    if not os.path.isfile(file):
        print("File not exists! File: {}".format(file))
        return False

    return True


def check_config_json(args, config_json_file):
    if not check_file_is_exists(config_json_file):
        return False

    with open(config_json_file, 'r') as json_file:
        config_json = json.load(json_file)

    if check_file_is_exists(config_json['word_sequence_file']) is False:
        return False

    if args.mode == FIT_KNOWLEDGE_CATEGORY_MODEL:
        config = config_json['knowledge_category_model']

    elif args.mode == FIT_PROFILE_DISTINGUISH_MODEL:
        config = config_json['profile_distinguish_model']

    elif args.mode == FIT_PROFILE_RETRIEVAL_MODEL:
        config = config_json['profile_retrieval_model']

    elif args.mode == FIT_SEMANTICS_DISTINGUISH_MODEL:
        config = config_json['semantics_distinguish_model']

    elif args.mode == FIT_KNOWLEDGE_RETRIEVAL_MODEL:
        config = config_json['knowledge_retrieval_model']

    elif args.mode == FIT_DIALOG_RETRIEVAL_MODEL:
        config = config_json['dialog_retrieval_model']

    elif args.mode == FIT_SCORE_MODEL:
        config = config_json['score_model']

    dataset_config = config['split_dataset']
    network_config = config['network']

    if dataset_config['valid_size'] < 0.0 or dataset_config['valid_size'] > 1.0:
        print("valid_size range of values: [0.0 ~ 1.0]")
        return False

    if dataset_config['test_size'] < 0.0 or dataset_config['test_size'] > 1.0:
        print("test_size range of values: [0.0 ~ 1.0]")
        return False

    if network_config['device'] not in [v.GPU, v.CPU]:
        print("device only supports {} or {}".format(v.GRU, v.CPU))
        return False

    if network_config['device'] == v.GPU:
        if not torch.cuda.is_available():
            print("{} is unavailable".format(v.GPU))
            return False

    if network_config['rnn_model'] not in [v.LSTM, v.GRU]:
        print("The network model only supports {} or {}".format(v.LSTM, v.GRU))
        return False

    return True


def init_mysql(config_json):
    mysql_config = config_json['mysql']
    mysql = Mysql(host=mysql_config['host'],
                  user=mysql_config['user'],
                  password=mysql_config['password'],
                  db=mysql_config['db'],
                  debug=DEBUG)

    return mysql


def main():
    # 获取输入参数
    args = input_args()

    # 读取输入参数
    mode = args.mode
    config_json_file = args.config_json_file
    dataset_save_path = args.dataset_save_path
    model_save_path = args.model_save_path
    filename = args.filename

    # 检查 config.json 的配置
    if not check_config_json(args, config_json_file):
        print("Found error! check config.json File! ")
        return False

    # 解析 json 文件
    with open(config_json_file, 'r') as json_file:
        config_json = json.load(json_file)

    # 输出参数
    print_args(args, config_json)

    # 初始化数据库
    mysql = init_mysql(config_json)

    # 加载词库
    word_sequence = load_word_sequences(config_json['word_sequence_file'])

    # 训练 knowledge category 网络
    if mode == FIT_KNOWLEDGE_CATEGORY_MODEL:
        fit_knowledge_category_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json,
                                       filename)
    # 训练 profile distinguish 网络
    elif mode == FIT_PROFILE_DISTINGUISH_MODEL:
        fit_profile_distinguish_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json,
                                        filename)

    # 训练 profile retrieval 网络
    elif mode == FIT_PROFILE_RETRIEVAL_MODEL:
        fit_profile_retrieval_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json,
                                      filename)

    # 训练 semantics distinguish 网络
    elif mode == FIT_SEMANTICS_DISTINGUISH_MODEL:
        fit_semantics_distinguish_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json,
                                          filename)

    # 训练 knowledge retrieval 网络
    elif mode == FIT_KNOWLEDGE_RETRIEVAL_MODEL:
        fit_knowledge_retrieval_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json,
                                        filename)

    # 训练 dialog retrieval 网络
    elif mode == FIT_DIALOG_RETRIEVAL_MODEL:
        fit_dialog_retrieval_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json,
                                     filename)

    # 训练 score 网络
    elif mode == FIT_SCORE_MODEL:
        fit_score_network(mysql, word_sequence, dataset_save_path, model_save_path, config_json, filename)

    # 其他情况输出提示
    else:
        print("Selection of operation mode：{}, "
              "python main.py <config.json> -m <mode>".format(MODE_LIST))

    mysql.close()


if __name__ == '__main__':
    main()

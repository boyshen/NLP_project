#!／usr/bin/env python3
# 创建日期：2019.8.25
# 发布日期: 17/10/2019
# author: shenpinggang
# version: 1.0
#
# PURPOSE：
# (1)导入收集的数据到mysql
# (3)创建词库。
# (4)终端聊天。通常用于debug
# (5)web聊天。
#
# 使用：
# python chatbot.py <config.json> -m <>
#   -m <write_mysql, create_word_sequence, chatting, web>
#       write_mysql                导入收集的数据到数据库
#       create_word_sequence       创建词库
#       chatting                   终端聊天, 通常用于debug
#       web                        开启 web 聊天
#
#   --save_path                      指定保存目录，训练模型、词库、数据集
#   --filename                       指定保存文件名，训练模型、词库
#
# Example:
# python chat.py config.json -m write_mysql
# 指定导入数据到数据库。数据可在config.json文件中进行配置。具体可见 config.json
#
# python chat.py config.json -m create_word_sequence
# --save_path ../save_word_sequence/
# --filename word_sequence.pickle
# 创建词库。指定保存路径为： ../save_word_sequence/word_sequence.pickle
#
# python chat.py config.json -m web
# 启动 web 聊天
#

import os
import json
import argparse
import global_variable as variable
from build import init_mysql, write_mysql
from build import chatting
from build import web
from module.core.word_sequences import create_word_sequences

# 导入数据库模式
W_MYSQL = 'write_mysql'

# 创建词库模式
CREATE_WORD_SEQUENCE = 'create_word_sequence'

# 终端聊天模式
CHATTING = 'chatting'

# web聊天模式
WEB = 'web'

MODE_LIST = [W_MYSQL,
             CREATE_WORD_SEQUENCE,
             CHATTING,
             WEB]


def input_args():
    parser = argparse.ArgumentParser(description="1.Write data to mysql "
                                                 "2.create word sequence"
                                                 "3.Start chatting in terminal mode"
                                                 "4.Start web chatting mode")

    parser.add_argument("config_json_file", metavar="config.json file",
                        help="Configuration files for data utils")

    parser.add_argument("-m", dest="mode", action='store', choices=MODE_LIST,
                        help="Select mode: {}, "
                             "1.write_mysql: Write data to mysql！"
                             "2.create_word_sequence: create word sequence"
                             "3.Start chatting in terminal mode "
                             "4.Start web chatting mode".format(MODE_LIST))

    parser.add_argument("--save_path", dest="save_path", action='store', type=str, default=os.getcwd(),
                        help="Data Set Preservation Path, Default current directory")

    parser.add_argument("--filename", dest="file_name", action='store', type=str, default="file.pickle",
                        help="save model, word_sequence file name! default: file.pickle")

    args = parser.parse_args()

    return args


def print_args(args):
    print("------------ input args ------------")
    print(" mode             : {}".format(args.mode))
    print(" config json      : {}".format(args.config_json_file))

    if args.mode == CREATE_WORD_SEQUENCE:
        print(" save dir         : {}".format(args.save_path))
        print(" save filename    : {}".format(args.file_name))

    print("-------------------------------------")
    print()


def check_file_is_exists(file):
    if not os.path.isfile(file):
        print("File not exists! File: {}".format(file))
        return False

    return True


def check_config_json(args, config_json_file):
    mode = args.mode

    if not check_file_is_exists(config_json_file):
        return False

    with open(config_json_file, 'r') as config_json:
        config_json = json.load(config_json)

    # 检查 write_mysql 模式下的配置信息
    if mode == W_MYSQL:
        for data in config_json['data']:

            # 检查文件是否存在
            if not check_file_is_exists(data['file']):
                return False

            # 检查数据类型
            data_type = [variable.DATA_TYPE_DIALOG, variable.DATA_TYPE_QA, variable.DATA_TYPE_JOKE,
                         variable.DATA_TYPE_PROFILE, variable.DATA_TYPE_SENSITIVE_WORD, variable.DATA_TYPE_ANY_REPLY]
            if data['type'] not in data_type:
                print("data type must is {}".format(data_type))
                return False

            # 检查文件类型是否为空
            if data['file_type'] == "None" or data['file_type'] == '' or data['file_type'] == ' ':
                print("data file type cannot be empty!")
                return False

            # 检查输入的文件类型是否为txt
            if data['file_type'] not in [variable.TXT_FILE, variable.JSON_FILE]:
                print("file type must is json or txt")
                return False

            # 检查文件类型和数据类型对应的字段要求
            if data['file_type'] == variable.TXT_FILE:
                if len(data['category']) == 0:
                    print("QA data type, category cannot be empty")
                    return False

                if len(data['label']) == 0:
                    print("QA data type, table cannot be empty")
                    return False

                if data['type'] in [variable.DATA_TYPE_DIALOG, variable.DATA_TYPE_JOKE, variable.DATA_TYPE_QA,
                                    variable.DATA_TYPE_PROFILE]:
                    if data['ask_tag'] == "None" or data['ask_tag'] == '' or data['ask_tag'] == " ":
                        print("Dialog data type, ask_tag cannot be empty")
                        return False

                    if data['answer_tag'] == "None" or data['answer_tag'] == '' or data['answer_tag'] == " ":
                        print("Dialog data type, answer_tag cannot be empty")
                        return False

                if data['type'] == variable.DATA_TYPE_SENSITIVE_WORD:
                    if data['split_tag'] == "None" or data['split_tag'] == '' or data['split_tag'] == " ":
                        return False

    # 检查 chatting 模式下的配置信息
    if mode == CHATTING or mode == WEB:
        model_config = config_json['model']
        if not check_file_is_exists(model_config['profile_distinguish']['model_file']):
            return False

        if not check_file_is_exists(model_config['profile_distinguish']['class_json']):
            return False

        if not check_file_is_exists(model_config['profile_retrieval']['model_file']):
            return False

        if not check_file_is_exists(model_config['profile_retrieval']['class_json']):
            return False

        if not check_file_is_exists(config_json['word_sequence']):
            return False

    if mode == WEB:
        web_conf = config_json['web']

        if not isinstance(web_conf['port'], int):
            print("web port must is int type")
            return False

    return True


def main():
    # 获取输入参数
    args = input_args()

    # 读取输入参数
    mode = args.mode
    config_json_file = args.config_json_file

    # 检查 config.json 的配置
    if not check_config_json(args, config_json_file):
        print("Found error! check config.json File! ")
        return False

    # 解析 json 文件
    with open(config_json_file, 'r') as json_file:
        config_json = json.load(json_file)

    # 输出参数
    print_args(args)

    # 初始化数据库
    mysql = init_mysql(config_json)

    # 写数据库模式
    if mode == W_MYSQL:
        write_mysql(config_json, mysql)

    # 创建词库
    elif mode == CREATE_WORD_SEQUENCE:
        create_word_sequences(mysql=mysql, save_dir=args.save_path, file_name=args.file_name)

    # 终端聊天模式
    elif mode == CHATTING:
        chatting(config_json, mysql)

    # web聊天模式
    elif mode == WEB:
        web(config_json, mysql=mysql)

    # 其他情况输出提示
    else:
        print("Selection of operation mode：{}, "
              "python chat.py <config.json> -m <mode>".format(MODE_LIST))

    mysql.close()


if __name__ == '__main__':
    main()

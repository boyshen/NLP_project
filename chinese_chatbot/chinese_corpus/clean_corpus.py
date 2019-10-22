#!／usr/bin/env python3
# 创建日期：2019.8.25
# 发布日期: 30/8/2019
# version: 1.0
# PURPOSE：提取需要的类别QA数据集和对小黄鸡语料进行简单清理，主要删除有特殊符号的对话、提取语料中的笑话句子、以及提取比较长的句子。
#          QA 数据集来自：https://github.com/brightmart/nlp_chinese_corpus 的百度百科
#          小黄鸡数据集下载：https://cloud.tencent.com/developer/article/1437998
#
#
# 使用：
# python clean_corpus.py <source_corpus> -m <> --category <> --max_length <> --num <> --save_path <> --save_filename <>
#  -h
#  -m {save_category,del_category,merge_category,save_number_category,del_repeat,save_max_length}
#  --category CATEGORY              指定需要保存的 category(类别) 格式：[旅游,生活,文化,健康,教育] 中间不要有空格，以 , 区分
#  --max_length MAX_LENGTH          指定保存不超过 max_lenght 的句子
#  --num NUMBER                     指定保存指定类型语料的数量
#  --save_path SAVE_PATH            指定保存处理后语料的路径
#  --save_filename SAVE_FILENAME    指定保存处理后语料的文件名
#  source corpus file!
#
#  save_category        保存需要的语料， 配合 --category 一起使用
#  del_category         删除不需要的语料，配合 --category 一起使用
#  merge_category       合并语料中的 category 。如将 '娱乐-旅游' 合并为 '娱乐'
#  save_number_category 保存指定类型，指定数量的语料。如保存 '娱乐-旅游' 类型 1000 条 配合 --num 指定数量, --category 指定类别
#  del_repeat           删除重复的语料，主要删除语料中 title 重复的语料。
#  save_max_length      指定保存 title 和 answer 句子长度。如只保存：保存句子长度在 200 内的语料。配合 --max_length 指定最大句子长度
#  clean_xhj            清理小黄鸡对话语料
#
# Example:
# python clean_corpus.py baike_qa.json -m save_max_length --max_length 500 --save_path ./ --save_filename new.json
# 指定保存句子长度不超过 500 的语料。保存路径为当前目录，文件名为 new.json

import os
import json
import re
import argparse
import jieba
from tqdm import tqdm

save_mode = "save_category"
del_mode = "del_category"
merge_mode = "merge_category"
del_repeat_mode = "del_repeat"
save_max_length_mode = "save_max_length"
save_number_category = "save_number_category"
clean_xhj = 'clean_xhj'
only_keep_chinese_xhj = 'only_keep_chinese_xhj'

mode_choice = [save_mode, del_mode, merge_mode, save_number_category, del_repeat_mode, save_max_length_mode,
               clean_xhj, only_keep_chinese_xhj]

# 增加jieba分词词典
add_word_list = ['冷笑话']


def regular(sentence, reg):
    sentence = ' '.join(re.findall(reg, sentence))
    sentence = re.sub('[/，]{1,100}', '，', sentence)
    sentence = re.sub('[/,]{1,100}', ',', sentence)
    sentence = re.sub('[？]{1,100}', '？', sentence)
    sentence = re.sub('[?]{1,100}', '?', sentence)
    sentence = re.sub('[！]{1,100}', '！', sentence)
    sentence = re.sub('[!]{1,100}', '!', sentence)
    sentence = re.sub('[/。]{1,100}', '。', sentence)
    sentence = re.sub('[/.]{1,100}', '.', sentence)
    sentence = re.sub('[。]{1,100}', '。', sentence)
    sentence = re.sub('[" "]{1,100}', ' ', sentence)
    sentence = re.sub('[～]{1,100}', '', sentence)
    return sentence


def delete_repeat_title(source_corpus, save_path=None, new_corpus_name='new_corpus.json'):
    """
    检查 title 是否重复，如果重复则删除重复的数据
    :param source_corpus: <str> 语料文件
    :param save_path: <str> 新的语料保存地址
    :param new_corpus_name: <str> 新语料文件名
    :return:
    """
    assert os.path.isfile(source_corpus), "corpus file not exists！File: {}".format(source_corpus)

    new_corpus = list()
    title_list = list()
    with open(source_corpus, 'rb') as source_file:
        for line in tqdm(source_file):
            line = str(line, encoding='utf8')
            title = json.loads(line)['title']

            if title.strip() in title_list:
                continue

            title_list.append(title.strip())
            new_corpus.append(line)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()

    save_file = os.path.join(save_path, new_corpus_name)
    with open(save_file, 'w') as write_file:
        write_file.writelines(new_corpus)

    print("save corpus over ! File: {}".format(save_file))


def save_max_length_corpus(source_corpus, max_length=244,
                           save_path=None, new_corpus_name='new_corpus.json'):
    """
    指定保存最大句子长度的语料
    :param source_corpus: <str> 语料文件
    :param max_length: <int> 指定 answer， title，的最大句子长度。
                             如 max_length = 244. 超过 244 长度的数据将被抛弃
    :param save_path: <str> 新语料保存路径
    :param new_corpus_name: <str> 新语料保存文件名
    :return:
    """

    new_corpus = list()
    with open(source_corpus, 'rb') as source_file:
        for line in tqdm(source_file):
            line = str(line, encoding='utf8')
            json_data = json.loads(line)
            if len(json_data['title']) > max_length or len(json_data['answer']) > max_length:
                continue
            new_corpus.append(line)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()

    save_file = os.path.join(save_path, new_corpus_name)
    with open(save_file, 'w') as write_file:
        write_file.writelines(new_corpus)

    print("save corpus over ! File: {}".format(save_file))


def save_number_corpus_of_category(source_corpus, categorys,
                                   save_path=None,
                                   new_corpus_name='new_corpus.json',
                                   number=5000):
    """
    保存指定数量和句子长度的 category 类别
    :param source_corpus: <str> 语料文件
    :param categorys: <list> 指定保存的 category 类别
    :param save_path: <str> 新语料保存路径
    :param new_corpus_name: <str> 新语料保存文件名
    :param number: <int> 指定 category 的数量
    :return:
    """
    assert os.path.isfile(source_corpus), "corpus file not exists! File: {}".format(source_corpus)

    number_dict = dict()
    for category in categorys:
        number_dict[category] = 0

    new_corpus = list()
    with open(source_corpus, 'rb') as source_file:
        for line in tqdm(source_file):
            line = str(line, encoding='utf8')
            line_category = json.loads(line)['category']

            if line_category == "":
                continue

            if sum(number_dict.values()) == len(number_dict) * number:
                break

            for category in categorys:
                if number_dict[category] >= number:
                    continue

                if line_category[:len(category)] == category and number_dict[category] < number:
                    new_corpus.append(line)
                    number_dict[category] = number_dict[category] + 1
                    break

    print("| category | number | ")
    for category, num in number_dict.items():
        print("| {} | {} |".format(category, num))

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()

    save_file = os.path.join(save_path, new_corpus_name)
    with open(save_file, 'w') as write_file:
        write_file.writelines(new_corpus)

    print("save corpus over ! File: {}".format(save_file))


def save_corpus_by_category(source_corpus, categorys, save_path=None, new_corpus_name='new_corpus.json'):
    """
    指定保存 category 类别的语料数据
    :param source_corpus: <str> 语料文件
    :param categorys: <list> 指定保存的 category 类别列表
    :param save_path: <str> 新语料保存地址
    :param new_corpus_name: <str> 新语料保存文件名
    :return:
    """
    assert os.path.isfile(source_corpus), "corpus file not exists! File: {}".format(source_corpus)

    new_corpus = list()
    with open(source_corpus, 'rb') as source_file:
        for line in tqdm(source_file):
            line = str(line, encoding='utf8')
            line_category = json.loads(line)['category']
            if line_category == "":
                continue

            for category in categorys:
                if line_category[:len(category)] == category:
                    new_corpus.append(line)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()

    save_file = os.path.join(save_path, new_corpus_name)
    with open(save_file, 'w') as write_file:
        write_file.writelines(new_corpus)

    print("save corpus over! new corpus File: {}".format(save_file))


def delete_corpus_by_category(source_corpus, categorys, save_path=None, new_corpus_name="new_corpus.json"):
    """
    删除指定的 categorys 内容。
    :param source_corpus: <str> 原来的语料文件
    :param categorys: <list> 指定删除的内容
    :param save_path: <str> 新文件保存目录
    :param new_corpus_name: <str> 保存文件名
    :return:
    """
    assert os.path.isfile(source_corpus), "corpus file not exists ! File: {}".format(source_corpus)

    # 标记是否找到需要清理的 categorys。
    # 找到为True， 否则为 False
    found_tag = False

    # 保存新的语料
    new_corpus = list()

    with open(source_corpus, 'rb') as source_file:
        for line in tqdm(source_file):
            line = str(line, encoding='utf8')
            line_category = json.loads(line)['category']

            if line_category == "":
                continue

            for category in categorys:
                if category == line_category[:len(category)]:
                    found_tag = True
                    break

            if found_tag:
                found_tag = False
                continue

            new_corpus.append(line)

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()

    save_file = os.path.join(save_path, new_corpus_name)
    with open(save_file, 'w') as target_file:
        target_file.writelines(new_corpus)

    print("save corpus over! new corpus File: {}".format(save_file))


def merge_category(source_corpus, save_path=None, save_file_name='new_source.json'):
    """
    合并语料的category， 原语料中的category 分类比较详细，这里进行合并
    :param source_corpus: <str>原语料文件
    :param save_path:<str> 保存新语料的目录
    :param save_file_name:<str> 保存新语料的文件名
    :return:
    """
    assert os.path.isfile(source_corpus), "File not exists! File: {}".format(source_corpus)

    json_data_list = list()
    with open(source_corpus, 'rb') as source_file:
        for line in tqdm(source_file):
            line = str(line, encoding='utf8')
            json_data = json.loads(line)
            category = json_data['category']
            category_list = list(category)
            if '/' in category_list:
                category = category.split('/')[0]
            if '-' in category_list:
                category = category.split('-')[0]

            json_data['category'] = category
            json_data_list.append(json.dumps(json_data, ensure_ascii=False))

    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.getcwd()

    save_file = os.path.join(save_path, save_file_name)
    with open(save_file, 'w') as w_file:
        for line in json_data_list:
            w_file.write(line)
            w_file.write('\n')

    print("merge category over! File: {}".format(save_file))


def clean_corpus_from_xhj(source_file, save_dir=None, save_file='new_xhj.txt'):
    start_tag = 'E'
    ask_tag = 'M '
    answer_tag = 'M '

    del_character = ['=。=', 'ijj', '= =']

    reg = '[\u4e00-\u9fa5,，.。？?！!·、{};；0-9a-zA-Z]+'

    joke_keyword = ['笑话', '冷笑', '算算']
    del_keyword = ['运势']

    max_sentence_length = 22

    is_start, is_end, is_save, is_joke = False, False, False, False
    label = 0

    dialog = list()
    joke_dialog = list()

    data = dict()
    with open(source_file, 'rb') as rf:
        for line in tqdm(rf):
            line = str(line, encoding='utf8').strip('\n')

            if line == start_tag:
                is_start = True
                data['start'] = start_tag
                continue

            if is_start:
                if label == 0 and line[:len(ask_tag)] == ask_tag:
                    ask_sentence = regular(line[len(ask_tag):], reg)

                    if line[len(ask_tag):] in del_character:
                        is_start = False
                        is_save = False
                        continue

                    if len(ask_sentence) <= 1:
                        is_start = False
                        is_save = False
                        continue

                    keyword = jieba.lcut(ask_sentence)
                    for k in del_keyword:
                        if k in keyword:
                            is_start = False
                            is_save = False
                            continue

                    for k in joke_keyword:
                        if k in keyword:
                            is_joke = True

                    data['ask'] = 'M ' + ask_sentence
                    label = 1

                elif label == 1 and line[:len(answer_tag)] == answer_tag:
                    answer_sentence = regular(line[len(answer_tag):], reg)
                    if line[len(answer_tag):] in del_character:
                        is_start = False
                        is_save = False
                        continue

                    if len(answer_sentence) <= 1:
                        is_start = False
                        is_save = False
                        continue

                    data['answer'] = 'M ' + answer_sentence
                    label = 0
                    is_start = False
                    is_end = True
                    is_save = True

            if is_start is False and is_save is True and is_joke is False and is_end is True:
                dialog.append(data)
                is_save = False
                data = {}

            elif is_start is False and is_save is True and is_joke is True and is_end is True:
                joke_dialog.append(data)
                is_save = False
                is_joke = False
                data = {}

            elif is_start is False and is_save is False and is_joke is False and is_end is False:
                data = {}

    long_dialog = list()
    new_dialog = list()
    for data in dialog:
        if len(data['ask']) > max_sentence_length or len(data['answer']) > max_sentence_length:
            long_dialog.append(data)
        else:
            new_dialog.append(data)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.getcwd()

    save_file = os.path.join(save_dir, save_file)
    with open(save_file, 'w') as wf:
        for data in new_dialog:
            wf.writelines(data['start'] + '\n')
            wf.writelines(data['ask'] + '\n')
            wf.writelines(data['answer'] + '\n')

    save_joke = os.path.join(save_dir, 'xhj_joke.txt')
    with open(save_joke, 'w') as wf:
        for data in joke_dialog:
            wf.writelines(data['start'] + '\n')
            wf.writelines(data['ask'] + '\n')
            wf.writelines(data['answer'] + '\n')

    save_long_dialog = os.path.join(save_dir, 'xhj_long_dialog.txt')
    with open(save_long_dialog, 'w') as wf:
        for data in long_dialog:
            wf.writelines(data['start'] + '\n')
            wf.writelines(data['ask'] + '\n')
            wf.writelines(data['answer'] + '\n')

    print("clean xhj data over ! File: {}".format(save_file))
    print("xhj joke data corpus! File: {}".format(save_joke))
    print("xhj long sentence dialog corpus! File: {}".format(save_long_dialog))


def extract_profile_of_xhj(source_file, save_dir=None, save_file='profile.xhj'):
    start_tag = 'E'
    ask_tag = 'M '
    answer_tag = 'M '

    reg = '[\u4e00-\u9fa5,，.。？?！!·、{};；a-zA-Z0-9]+'

    is_start, is_end, is_save, is_profile = False, False, False, False
    label = 0
    keyword = ['你', '自己']

    dialog, profile_data = list(), list()

    data = dict()
    for word in add_word_list:
        jieba.add_word(word)

    with open(source_file, 'rb') as rf:
        for line in tqdm(rf):
            line = str(line, encoding='utf8').strip('\n')

            if line == start_tag:
                is_start = True
                data['start'] = start_tag
                continue

            if is_start:
                if label == 0 and line[:len(ask_tag)] == ask_tag:
                    ask_sentence = regular(line[len(ask_tag):], reg)

                    if len(ask_sentence) <= 1:
                        is_start = False
                        is_save = False
                        continue

                    if len(re.sub('[a-zA-Z0-9]', '', ask_sentence)) == 0:
                        is_start = False
                        is_save = False
                        continue

                    for word in keyword:
                        if word in jieba.lcut(ask_sentence):
                            is_profile = True

                    data['ask'] = 'M ' + ask_sentence
                    label = 1

                elif label == 1 and line[:len(answer_tag)] == answer_tag:
                    answer_sentence = regular(line[len(answer_tag):], reg)

                    if len(answer_sentence) <= 1:
                        is_start = False
                        is_save = False
                        continue

                    if len(re.sub('[a-zA-Z0-9]', '', answer_sentence)) == 0:
                        is_start = False
                        is_save = False
                        continue

                    data['answer'] = 'M ' + answer_sentence
                    label = 0
                    is_start = False
                    is_end = True
                    is_save = True

            if is_start is False and is_save is True and is_profile is False and is_end is True:
                dialog.append(data)
                is_save = False
                data = {}

            elif is_start is False and is_save is True and is_profile is True and is_end is True:
                profile_data.append(data)
                is_save = False
                is_profile = False
                data = {}

            elif is_start is False and is_save is False and is_profile is False and is_end is False:
                data = {}

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.getcwd()

    save_file = os.path.join(save_dir, save_file)
    with open(save_file, 'w') as wf:
        for data in dialog:
            wf.writelines(data['start'] + '\n')
            wf.writelines(data['ask'] + '\n')
            wf.writelines(data['answer'] + '\n')

    profile_file = os.path.join(save_dir, 'profile.txt')
    with open(profile_file, 'w') as wf:
        for data in profile_data:
            wf.writelines(data['start'] + '\n')
            wf.writelines(data['ask'] + '\n')
            wf.writelines(data['answer'] + '\n')

    print("clean xhj data over ! File: {}".format(save_file))
    print("xhj profile data over ! File: {}".format(save_file))


def only_keep_chinese_dialog_xhj(source_file, save_dir=None, save_file='chinese_xhj_corpus.txt'):
    start_tag = 'E'
    ask_tag = 'M '
    answer_tag = 'M '

    reg = '[\u4e00-\u9fa5,，.。？?！!·、{};；a-zA-Z0-9]+'

    is_start, is_end, is_save = False, False, False
    label = 0

    dialog = list()

    data = dict()
    with open(source_file, 'rb') as rf:
        for line in tqdm(rf):
            line = str(line, encoding='utf8').strip('\n')

            if line == start_tag:
                is_start = True
                data['start'] = start_tag
                continue

            if is_start:
                if label == 0 and line[:len(ask_tag)] == ask_tag:
                    ask_sentence = regular(line[len(ask_tag):], reg)

                    if len(ask_sentence) <= 1:
                        is_start = False
                        is_save = False
                        continue

                    if len(re.sub('[a-zA-Z0-9]', '', ask_sentence)) == 0:
                        is_start = False
                        is_save = False
                        continue

                    data['ask'] = 'M ' + ask_sentence
                    label = 1

                elif label == 1 and line[:len(answer_tag)] == answer_tag:
                    answer_sentence = regular(line[len(answer_tag):], reg)

                    if len(answer_sentence) <= 1:
                        is_start = False
                        is_save = False
                        continue

                    if len(re.sub('[a-zA-Z0-9]', '', answer_sentence)) == 0:
                        is_start = False
                        is_save = False
                        continue

                    data['answer'] = 'M ' + answer_sentence
                    label = 0
                    is_start = False
                    is_end = True
                    is_save = True

            if is_start is False and is_save is True and is_end is True:
                dialog.append(data)
                is_save = False
                data = {}

            elif is_start is False and is_save is False and is_end is False:
                data = {}

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = os.getcwd()

    save_file = os.path.join(save_dir, save_file)
    with open(save_file, 'w') as wf:
        for data in dialog:
            wf.writelines(data['start'] + '\n')
            wf.writelines(data['ask'] + '\n')
            wf.writelines(data['answer'] + '\n')

    print("clean xhj data over ! File: {}".format(save_file))


def input_args():
    parser = argparse.ArgumentParser("Processing Corpus Based on Category")
    parser.add_argument('source_corpus', metavar='source corpus file!',
                        help="input Corpus to be processed")
    parser.add_argument('-m', dest="mode", action='store', choices=mode_choice, default=merge_mode,
                        help="choice mode: {}"
                             "default: merge_mode "
                             "save_category: Save corpus by category! "
                             "del_category: delete corpus by category! "
                             "merge_category: Merge Corpus Category! "
                             "save_number_category: Save the number of specified categories!"
                             "save_max_length: Specify the maximum length for saving title and answer sentences"
                             "del_repeat: Delete redundant question data".format(mode_choice))
    parser.add_argument('--category', dest='category', action='store', default='None',
                        help="Categories to be processed, default: None, "
                             "for example: [体育,生活-起名] No spaces in the middle")
    parser.add_argument('--max_length', dest='max_length', action='store', type=int, default=244,
                        help="maximum length for saving title and answer sentences, default: 244")
    parser.add_argument('--num', dest='number', action='store', type=int, default=5000,
                        help="Save the number of specified categories, default: 5000")
    parser.add_argument('--save_path', dest='save_path', action='store', default=os.getcwd(),
                        help="target output file save path, default: location path")
    parser.add_argument('--save_filename', dest='save_filename', action='store', default='target_corpus.json',
                        help="target output file save file name, default: target_corpus.json")

    args = parser.parse_args()
    return args


def print_args(args):
    print("------------ args config ------------")
    print(" mode             : {}".format(args.mode))
    print(" source corpus    : {}".format(args.source_corpus))
    print(" number           : {}".format(args.number))
    print(" save path        : {}".format(args.save_path))
    print(" save file name   : {}".format(args.save_filename))
    print(" max length       : {}".format(args.max_length))
    print("-------------------------------------")
    print()


def main():
    args = input_args()
    print_args(args)

    source_corpus = args.source_corpus
    save_path = args.save_path
    save_file_name = args.save_filename
    number = args.number
    max_length = args.max_length

    if args.category == 'None':
        category = []
    else:
        category = re.sub("[/[/]]", "", args.category).split(',')

    if args.mode == del_mode:
        delete_corpus_by_category(source_corpus, category, save_path, save_file_name)

    elif args.mode == save_mode:
        save_corpus_by_category(source_corpus, category, save_path, save_file_name)

    elif args.mode == merge_mode:
        merge_category(source_corpus, save_path, save_file_name)

    elif args.mode == save_number_category:
        save_number_corpus_of_category(source_corpus, category, save_path, save_file_name, number)

    elif args.mode == del_repeat_mode:
        delete_repeat_title(source_corpus, save_path, save_file_name)

    elif args.mode == save_max_length_mode:
        save_max_length_corpus(source_corpus, max_length, save_path, save_file_name)

    elif args.mode == clean_xhj:
        clean_corpus_from_xhj(source_corpus, save_path, save_file_name)

    elif args.mode == only_keep_chinese_xhj:
        only_keep_chinese_dialog_xhj(source_corpus, save_path, save_file_name)


if __name__ == '__main__':
    # main()
    extract_profile_of_xhj('source/new_xhj.conv')

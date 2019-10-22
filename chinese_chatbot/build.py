from module.core.mysql_exec import Mysql
from module.core.data_utils import read_file_data, write_data_to_mysql, read_sensitive_word, read_any_reply
from module.chatbot.chatbot import ChatBot, packing_config
from module.web.web import create_app
import global_variable as variable


def init_mysql(config_json):
    mysql_config = config_json['mysql']
    mysql = Mysql(host=mysql_config['host'],
                  user=mysql_config['user'],
                  password=mysql_config['password'],
                  db=mysql_config['db'],
                  debug=variable.DEBUG)

    return mysql


def write_mysql(config_json, mysql):
    for data in config_json['data']:
        if data['file_type'] == variable.TXT_FILE:

            if data['type'] == variable.DATA_TYPE_SENSITIVE_WORD:
                file_data = read_sensitive_word(data['file'], split_tag=data['split_tag'])

            elif data['type'] in [variable.DATA_TYPE_PROFILE, variable.DATA_TYPE_QA, variable.DATA_TYPE_JOKE,
                                  variable.DATA_TYPE_DIALOG]:
                ask_tag = data['ask_tag']
                answer_tag = data['answer_tag']
                start_tag = None if data['start_tag'] == "None" else data['start_tag']
                end_tag = None if data['end_tag'] == "None" else data['end_tag']
                file_data = read_file_data(data['file'],
                                           start_tag=start_tag,
                                           end_tag=end_tag,
                                           ask_tag=ask_tag,
                                           answer_tag=answer_tag,
                                           skip_line=data['skip_line'], use_keyword=True)
            dataset = list()
            for f_data in file_data:
                f_data['category'] = data['category']
                f_data['label'] = data['label']
                dataset.append(f_data)

            write_data_to_mysql(mysql, dataset, data['type'])

        elif data['file_type'] == variable.JSON_FILE:

            if data['type'] == variable.DATA_TYPE_ANY_REPLY:
                fields = data['field']
                dataset = read_any_reply(data['file'], fields=fields)

            write_data_to_mysql(mysql, dataset, data['type'])


def chatting(config_json, mysql):
    conf = packing_config(config_json, mysql)
    chatbot = ChatBot(conf)

    print()
    print(">>> please input a chinese sentence, Exit please enter exit or quit！")

    while True:
        ask = input("human: ")

        if ask == 'exit' or ask == 'quit':
            break

        answer = chatbot.chat(ask).final_answer

        print("human: ", ask)
        print("chat bot: ", answer)
        print()


def web(config_json, mysql):
    conf = packing_config(config_json, mysql)
    chatbot = ChatBot(conf)

    web_app = create_app(chatbot)

    web_app.run(host=config_json['web']['host'], port=config_json['web']['port'], debug=variable.DEBUG)

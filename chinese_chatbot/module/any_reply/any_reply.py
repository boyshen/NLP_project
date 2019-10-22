import numpy as np
from module.core.mysql_table import any_reply_table

error_label = 'error'
unrecognized = 'unrecognized'

any_reply = [{'category': '报错', 'label': error_label, 'answer': '抱歉！小通内部出错，正在抢救'},
             {'category': '无法识别', 'label': unrecognized, 'answer': '抱歉！目前不能理解呢！请输入其他问题'}]


class AnyReplyConf(object):

    def __init__(self):
        self.mysql = None
        self.debug = None


class AnyReply(object):

    def __init__(self, mysql, debug=False):
        self.mysql = mysql

        mysql_any_reply = [reply['answer'] for reply in self.mysql.query_any_reply()]
        for reply in any_reply:
            if reply['answer'] in mysql_any_reply:
                # print("{} already exist!".format(reply['answer']))
                continue
                
            mysql.insert_any_reply(category=reply['category'], label=reply['label'], answer=reply['answer'])

        self.any_reply_label = [data['label'] for data in self.mysql.query_label_num_by_table(any_reply_table)]

        self.error_label = error_label
        self.unrecognized = unrecognized

        self.default = '抱歉！小通内部出错，正在抢救'

        self.debug = debug

    def reply(self, label):
        if label not in self.any_reply_label:
            label = unrecognized

        mysql_data = self.mysql.query_any_reply_by_label(label)

        if len(mysql_data) > 1:
            answer = np.random.choice(mysql_data)['answer']
        elif len(mysql_data) == 0:
            answer = self.default
        else:
            answer = mysql_data[0]['answer']

        return answer

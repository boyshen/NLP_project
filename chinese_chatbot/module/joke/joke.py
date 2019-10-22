import numpy as np


class JokeConf(object):
    def __init__(self):
        self.mysql = None
        self.debug = None


class Joke(object):
    def __init__(self, mysql, debug=False):
        self.mysql = mysql
        self.debug = debug

    def joke(self):
        mysql_joke = self.mysql.query_joke()
        answer = np.random.choice(mysql_joke)['answer']

        if self.debug:
            print("joke final answer: ", answer)
            print()

        return answer

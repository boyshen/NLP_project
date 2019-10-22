import threading


class ReadThread(threading.Thread):
    def __init__(self, thread_name, thread_ID, func, *args):
        super(ReadThread, self).__init__()
        self.thread_name = thread_name
        self.thread_ID = thread_ID
        self.func = func
        self.args = args

        self._result = None

    def run(self):
        print("{}_{} start...".format(self.thread_name, str(self.thread_ID)))
        result = self.func(*self.args)
        self._result = result

    def get_result(self):
        return self._result

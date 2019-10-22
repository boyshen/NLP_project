import pymysql
import module.core.mysql_table as T


class Mysql(object):

    def __init__(self, host, user, password, db, charset='utf8mb4', debug=True):
        self.host = host
        self.user = user
        self.password = password
        self.db = db

        self.connect = pymysql.connect(host=host,
                                       user=user,
                                       password=password,
                                       db=db,
                                       charset=charset,
                                       cursorclass=pymysql.cursors.DictCursor)

        self.debug = debug

        self.create_table(T.qa_table, T.qa_sql)
        self.create_table(T.joke_table, T.joke_sql)
        self.create_table(T.dialog_table, T.dialog_sql)
        self.create_table(T.profile_table, T.profile_sql)
        self.create_table(T.sensitive_word_table, T.sensitive_word_sql)
        self.create_table(T.any_reply_table, T.any_reply_sql)

    def __execute_query(self, sql):
        try:
            with self.connect.cursor() as cursor:
                cursor.execute(sql)
                result = cursor.fetchall()
        except Exception as e:
            print(e)
            print("sql: {}".format(sql))
            print("execute query fail! ")
            if self.debug:
                self.connect.close()
            return False

        return result

    def __execute_create(self, sql):
        try:
            with self.connect.cursor() as cursor:
                cursor.execute(sql)
        except Exception as e:
            print(e)
            print("sql: {}".format(sql))
            print("execute create Fail!")
            if self.debug:
                self.connect.close()
            return False

        return True

    def __execute_modify(self, sql):
        try:
            with self.connect.cursor() as cursor:
                cursor.execute(sql)
        except Exception as e:
            print(e)
            print("sql: {}".format(sql))
            print("execute insert Fail!")
            if self.debug:
                self.connect.close()
            return False

        self.connect.commit()
        return True

    def __get_all_tables(self):
        sql = 'SHOW TABLES'
        result = self.__execute_query(sql)
        all_table = [table['Tables_in_' + self.db] for table in result]

        return all_table

    def create_table(self, table, sql):
        mysql_table = self.__get_all_tables()
        if table in mysql_table:
            print("table {} already exists! ".format(table))
            return True

        if not self.__execute_create(sql):
            print("create dialog table {} fail!".format(table))
            return False

        print("create dialog table {} success!".format(table))
        return True

    def insert_data_corpus(self, table, category=None, label=None, ask=None, answer=None, ask_keyword=None,
                           answer_keyword=None, word=None, order_id=None, cid=None):
        if table == T.qa_table:
            sql = 'INSERT INTO {} ' \
                  '(category, label, ask, answer, ask_keyword, answer_keyword) ' \
                  'VALUES ' \
                  '("{}", "{}", "{}", "{}", "{}", "{}")'.format(table, category, label, ask, answer, ask_keyword,
                                                                answer_keyword)
        elif table == T.joke_table:
            sql = 'INSERT INTO {} ' \
                  '(category, label, ask, answer, ask_keyword, answer_keyword) ' \
                  'VALUES ' \
                  '("{}", "{}", "{}", "{}", "{}", "{}")'.format(table, category, label, ask, answer,
                                                                ask_keyword, answer_keyword)

        elif table == T.dialog_table:
            sql = 'INSERT INTO {} ' \
                  '(cid, order_id, category, label, ask, answer, ask_keyword, answer_keyword) ' \
                  'VALUES ' \
                  '({}, {}, "{}", "{}", "{}", "{}", "{}", "{}")'.format(table, cid, order_id, category, label, ask,
                                                                        answer, ask_keyword, answer_keyword)

        elif table == T.profile_table:
            sql = 'INSERT INTO {} ' \
                  '(category, label, ask, answer, ask_keyword, answer_keyword) ' \
                  'VALUES ' \
                  '("{}", "{}", "{}", "{}", "{}", "{}")'.format(table, category, label, ask, answer, ask_keyword,
                                                                answer_keyword)

        elif table == T.sensitive_word_table:
            sql = 'INSERT INTO {} (category, label, word) VALUES ("{}","{}","{}")'.format(table, category, label, word)

        elif table == T.any_reply_table:
            sql = 'INSERT INTO {} (category, label, answer) VALUES ("{}","{}","{}")'.format(table, category, label,
                                                                                            answer)

        else:
            print("{} not found ! ".format(table))
            return False

        if not self.__execute_modify(sql):
            return False

        return True

    def delete_duplicate_from_corpus(self, table):
        t = table
        sql = 'DELETE FROM {} WHERE (ask, answer) IN ' \
              '(SELECT t.ask,t.answer FROM ' \
              '(SELECT ask,answer FROM {} GROUP BY ask,answer HAVING count(1)>1) t) ' \
              'AND cid NOT IN ' \
              '(SELECT dt.min_cid FROM ' \
              '(SELECT MIN(cid) AS min_cid FROM {} GROUP BY ask,answer HAVING count(1)>1) dt)'.format(t, t, t)
        return self.__execute_modify(sql)

    def query_corpus_the_last_one(self, table):
        sql = 'SELECT * FROM {} ORDER BY cid DESC limit 1'.format(table)
        return self.__execute_query(sql)

    def query_corpus_by_table(self, table):
        sql = 'SELECT category,label,ask,ask_keyword,answer,answer_keyword FROM {}'.format(table)
        return self.__execute_query(sql)

    def query_table(self, table):
        return self.__execute_query('SELECT * FROM {}'.format(table))

    def query_ask_by_table(self, table):
        return self.__execute_query('SELECT category,label,ask,ask_keyword FROM {}'.format(table))

    def query_label_num_by_table(self, table):
        return self.__execute_query('SELECT label FROM {} GROUP BY label'.format(table))

    def query_ask_by_label_field(self, table, field):
        return self.__execute_query(
            'SELECT category,label,ask,ask_keyword FROM {} WHERE label="{}"'.format(table, field))

    def query_dialog(self):
        return self.query_corpus_by_table(T.dialog_table)

    def query_joke(self):
        return self.query_corpus_by_table(T.joke_table)

    def query_qa(self):
        return self.query_corpus_by_table(T.qa_table)

    def query_profile(self):
        return self.query_corpus_by_table(T.profile_table)

    def query_sensitive_word(self):
        return self.__execute_query('SELECT * FROM {}'.format(T.sensitive_word_table))

    def query_any_reply(self):
        return self.__execute_query('SELECT * FROM {}'.format(T.any_reply_table))

    def query_any_reply_by_label(self, label):
        return self.__execute_query('SELECT * FROM {} WHERE label="{}"'.format(T.any_reply_table, label))

    def query_dialog_last_one(self):
        return self.query_corpus_the_last_one(T.dialog_table)

    def query_by_sql(self, sql):
        return self.__execute_query(sql)

    def delete_duplicate_from_dialog(self):
        return self.delete_duplicate_from_corpus(T.dialog_table)

    def delete_duplicate_from_joke(self):
        return self.delete_duplicate_from_corpus(T.joke_table)

    def delete_duplicate_from_qa(self):
        return self.delete_duplicate_from_corpus(T.qa_table)

    def delete_duplicate_from_profile(self):
        return self.delete_duplicate_from_corpus(T.profile_table)

    def delete_duplicate_from_sensitive_word(self):
        t = T.sensitive_word_table
        sql = 'DELETE FROM {} WHERE (word) IN ' \
              '(SELECT t.word FROM ' \
              '(SELECT word FROM {} GROUP BY word HAVING count(1)>1) t) ' \
              'AND cid NOT IN ' \
              '(SELECT dt.min_cid FROM ' \
              '(SELECT MIN(cid) AS min_cid FROM {} GROUP BY word HAVING count(1)>1) dt)'.format(t, t, t)
        return self.__execute_modify(sql)

    def delete_duplicate_from_any_reply(self):
        t = T.any_reply_table
        sql = 'DELETE FROM {} WHERE (answer) IN ' \
              '(SELECT t.answer FROM ' \
              '(SELECT answer FROM {} GROUP BY answer HAVING count(1)>1) t) ' \
              'AND cid NOT IN ' \
              '(SELECT dt.min_cid FROM ' \
              '(SELECT MIN(cid) AS min_cid FROM {} GROUP BY answer HAVING count(1)>1) dt)'.format(t, t, t)
        return self.__execute_modify(sql)

    def insert_dialog_data(self, category, label, ask, answer, ask_keyword, answer_keyword, order_id, cid):
        return self.insert_data_corpus(table=T.dialog_table,
                                       category=category,
                                       label=label,
                                       ask=ask,
                                       answer=answer,
                                       ask_keyword=ask_keyword,
                                       answer_keyword=answer_keyword,
                                       order_id=order_id,
                                       cid=cid)

    def insert_joke_data(self, category, label, ask, answer, ask_keyword, answer_keyword):
        return self.insert_data_corpus(table=T.joke_table,
                                       category=category,
                                       label=label,
                                       ask=ask,
                                       answer=answer,
                                       ask_keyword=ask_keyword,
                                       answer_keyword=answer_keyword)

    def insert_qa_data(self, category, label, ask, answer, ask_keyword, answer_keyword):
        return self.insert_data_corpus(table=T.qa_table,
                                       category=category,
                                       label=label,
                                       ask=ask,
                                       answer=answer,
                                       ask_keyword=ask_keyword,
                                       answer_keyword=answer_keyword)

    def insert_profile_data(self, category, label, ask, answer, ask_keyword, answer_keyword):
        return self.insert_data_corpus(table=T.profile_table,
                                       category=category,
                                       label=label,
                                       ask=ask,
                                       answer=answer,
                                       ask_keyword=ask_keyword,
                                       answer_keyword=answer_keyword)

    def insert_sensitive_word(self, category, label, word):
        return self.insert_data_corpus(table=T.sensitive_word_table, category=category, label=label, word=word)

    def insert_any_reply(self, category, label, answer):
        return self.insert_data_corpus(table=T.any_reply_table, category=category, label=label, answer=answer)

    def close(self):
        self.connect.close()


def test():
    mysql = Mysql(host='192.168.10.22', user='chatbot', password='chatbot', db='chatbot')
    mysql.insert_data_corpus(T.qa_table,
                             category='历史',
                             label='history',
                             ask='hello',
                             answer='word',
                             ask_keyword='h',
                             answer_keyword='w')
    mysql.insert_data_corpus(T.dialog_table,
                             cid=1,
                             order_id=1,
                             category='历史',
                             label='history',
                             ask='hello',
                             answer='word',
                             ask_keyword='h',
                             answer_keyword='w')
    mysql.insert_data_corpus(T.joke_table,
                             category='历史',
                             label='history',
                             ask='hello',
                             answer='word',
                             ask_keyword='h',
                             answer_keyword='w')
    mysql.insert_data_corpus(T.profile_table,
                             category='历史',
                             label='history',
                             ask='hello',
                             answer='word',
                             ask_keyword='h',
                             answer_keyword='w')


if __name__ == '__main__':
    test()

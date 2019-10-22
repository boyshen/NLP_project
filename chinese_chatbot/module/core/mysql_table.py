qa_table = 'qa'
dialog_table = 'dialog'
joke_table = 'joke'
profile_table = 'profile'
sensitive_word_table = 'sensitive_word'
any_reply_table = 'any_reply'

dialog_sql = "CREATE TABLE {}(" \
             "cid INT (11), " \
             "order_id INT (11), " \
             "category VARCHAR (255), " \
             "label VARCHAR (255), " \
             "ask VARCHAR (255), " \
             "answer VARCHAR (255), " \
             "ask_keyword VARCHAR (255), " \
             "answer_keyword VARCHAR (255)" \
             ") ENGINE=InnoDB DEFAULT CHARSET=utf8 ".format(dialog_table)

qa_sql = "CREATE TABLE {}(" \
         "cid INT (11) NOT NULL AUTO_INCREMENT, " \
         "category VARCHAR (255), " \
         "label VARCHAR (255), " \
         "ask VARCHAR (255), " \
         "answer TEXT (65535), " \
         "ask_keyword VARCHAR (255), " \
         "answer_keyword TEXT (65535), " \
         "PRIMARY KEY (cid)" \
         ") ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1".format(qa_table)

joke_sql = "CREATE TABLE {}(" \
           "cid INT (11) NOT NULL AUTO_INCREMENT, " \
           "category VARCHAR (255), " \
           "label VARCHAR (255), " \
           "ask VARCHAR (255), " \
           "answer TEXT (65535), " \
           "ask_keyword VARCHAR (255), " \
           "answer_keyword TEXT (65535), " \
           "PRIMARY KEY (cid)" \
           ") ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1".format(joke_table)

profile_sql = "CREATE TABLE {}(" \
              "cid INT (11) NOT NULL AUTO_INCREMENT, " \
              "category VARCHAR (255), " \
              "label VARCHAR (255), " \
              "ask VARCHAR (255), " \
              "answer VARCHAR (255), " \
              "ask_keyword VARCHAR (255), " \
              "answer_keyword VARCHAR (255), " \
              "PRIMARY KEY (cid)" \
              ") ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1".format(profile_table)

sensitive_word_sql = "CREATE TABLE {}(" \
                     "cid INT (11) NOT NULL AUTO_INCREMENT, " \
                     "category VARCHAR (255), " \
                     "label VARCHAR (255), " \
                     "word VARCHAR (255), " \
                     "PRIMARY KEY (cid)" \
                     ") ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1".format(sensitive_word_table)

any_reply_sql = "CREATE TABLE {}(" \
                "cid INT (11) NOT NULL AUTO_INCREMENT, " \
                "category VARCHAR (255), " \
                "label VARCHAR (255), " \
                "answer VARCHAR (255), " \
                "PRIMARY KEY (cid)" \
                ") ENGINE=InnoDB DEFAULT CHARSET=utf8 AUTO_INCREMENT=1".format(any_reply_table)

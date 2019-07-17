import os
import json
from flask import Flask
from flask import request

import sys

sys.path.append('../')
import config
import main
from model_function import load_object, load_model, load_word_sequence
from main import predict_sentences
from main import check_config

app = Flask(__name__)

# 检查配置
assert check_config(), 'Found Error ! Please check config.py file'

# 加载模型
model_file = os.path.join(config.model_save_path,
                          config.model_save_file)
model = load_model(model_file)

# 加载词典
word_sequence_file = os.path.join(config.word_sequence_save_path,
                                  config.word_sequence_save_file)
word_sequence = load_word_sequence(word_sequence_file)

# 加载标签
label_file = os.path.join(config.model_save_path, main.label_file_name)
label_name = load_object(label_file)


@app.route('/predict_sent', methods=['POST'])
def predict_sent():
    value = None
    state = 'fail'

    try:
        data = json.loads(request.get_data())
        sentences = data['sent']
        print("predict sentences", sentences)
        value = predict_sentences(model, word_sequence, sentences, label_name)
        state = 'OK'

    except KeyError as e:
        value = '参数错误'
        print("Parameter error")
        print(e)

    except Exception as e:
        print("predict sentences fail!")
        print(e)

    return json.dumps({'state': state, 'value': value['label']})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

import torch
import json
from module.core.network_model import load_model
from module.core.mysql_exec import Mysql
from module.core.word_sequences import load_word_sequences
from module.core.loading_dataset import get_keyword_vec


def test_distinguish(model_file, mysql, word_sequences_file, class_json_file, seq_len=30):
    model = load_model(model_file)
    word_sequence = load_word_sequences(word_sequences_file)

    with open(class_json_file, 'r') as rf:
        class_id = json.load(rf)

    id_class = {i: c for c, i in class_id.items()}

    dataset = mysql.query_dialog()

    result = []
    for data in dataset:
        sent_vec = word_sequence.transfrom(data['ask'], seq_len)
        word_vec = get_keyword_vec([data['ask_keyword']], max_len=seq_len, word_sequence=word_sequence)

        sent_vec = torch.LongTensor(sent_vec).view(1, -1)
        word_vec = torch.LongTensor(word_vec).view(1, -1)

        output = model.forward(sent_vec, word_vec)
        category = id_class[output.argmax(1).item()]

        value = {'category': category, 'sentence': data['ask']}
        result.append(value)

    for v in result:
        if v['category'] == 'profile':
            print("sentence: ", v['sentence'])
            print("category: ", v['category'])
            print()


if __name__ == '__main__':
    model_file = './save_network_model/profile_info_distinguish.pth'
    word_sequences_file = './save_word_sequence/word_sequence.pickle'
    class_json_file = './chinese_corpus/target/profile_category_dataset/class.json'

    mysql = Mysql('192.168.10.10', 'chatbot', 'chatbot', 'chatbot')

    test_distinguish(model_file, mysql, word_sequences_file, class_json_file)


# 比较重要！
# 指定训练、预测的模型。模型目前支持 lstm 和 fnn(前馈神经网络) 两种
# model = 'fnn' 或 model = 'lstm'
model = 'lstm'

# 指定模型是训练，还是预测。可在训练完成之后，修改为预测模式
# model_func = 'train' 训练
# model_func = 'predict' 预测
model_func = 'predict'

# 读取数据的文件路径信息，需要区分开，即 "good" 评论数据集、"bad" 评论数据集
# 其中 'good' 和 'bad' 将会作为数据的标签
data_files = {'good': 'data/good.txt',
              'bad': 'data/bad.txt'}

# 指定 lstm 模型时，需要指定max_len的大小，默认为 25
# lstm 采用的词向量模式，记录句子中每个词的 token 值，所以每个句子长度不同。
# 设置 max_len 值，保证每个句子的大小长度一致，如果超出 25 则被截断，不足 25 则被填充
max_len = 25

# 测试数据集 和 验证数据集的划分比例，划分范围在：[0.0 ~ 1.0] 之间
test_size = 0.2
valid_size = 0.1

# 批次训练和验证的样本数量。通常为 32 ～ 256 之间，根据使用的服务器内存、GPU 内存进行决定。
batch_size = 32

# 保存 word_sequences 对象的文件 和 目录。
# word_sequences 对象为提取的词库信息，在预测模型时需要进行加载。
# 目录为绝对路径
word_sequence_save_path = '/Users/shen/Desktop/me/python/AI/nlp/project/rnn_sentiment_analysis/word_sequence'
word_sequence_save_file = 'word_sequence_fnn.pickle'

# 保存训练模型的目录
# 保存训练模型的文件名, 在预测模型时候读取进行加载
# 目录为绝对路径
model_save_path = '/Users/shen/Desktop/me/python/AI/nlp/project/rnn_sentiment_analysis/model'
model_save_file = 'model_checkpoint_fnn.pth'

# 选择 lstm 模型进行训练时，可以对模型的 embedding 层 和 hidden（隐藏层）进行配置。
# 输出层 data_files 中标签数量决定
# 输入层为训练词库的大小
lstm_embed_size = 256
lstm_hidden_size = 256
lstm_num_layers = 1

# 选择 fnn 模型进行训练时，可以对模型对 hidden 层进行配置
# 输出层 data_files 中标签数量决定
# 输入层为训练词库的大小
# 支持配置多个隐藏层，
# 如：配置一层[256, 256]
# 如：配置两层[256, 256, 256]
fnn_hidden = [256, 256]

# 训练模型时，随机关闭节点比例
dropout = 0.2

# 训练时使用的学习率
# 优化模型暂时支持 adam
# loss 函数暂时使用 NLLLoss
lr = 0.0001

# 使用CPU，还是GPU 设备进行训练。
# 如果选择 GPU 模型，将检测 GPU 是否可用，不可用将返回
device = 'CPU'

# 训练的轮次
epochs = 20





## 项目概述

项目属于 NLP 领域的情感分析。

项目 pytorch 作为框架，提供 LSTM 和 前馈神经网络(FNN) 两种网络模型进行训练和预测。

可在config.py 文件中查询看配置信息，支持配置hidden 节点数量、dropout、学习率等相关参数。

提供Flask封装了简单预测接口，可使用http请求接口的方式预测。

## 项目结构

```
├── config.py                           # 项目配置文件，可在此配置模型信息
├── data                                # 默认数据目录，来自京东购买商品的评论，分为good、bad
│   ├── bad.txt                         
│   └── good.txt
├── data_util.py                        # 数据操作python文件，包括读取数据、划分数据集等
├── main.py                             # 项目主程序文件，在完成config.py 的配置之后，使用python main.py 运行
├── model                               # 保存模型的目录
│   ├── label_name.pickle
│   ├── model_checkpoint_fnn.pth
│   └── model_checkpoint_lstm.pth
├── model_function.py                   # 模型训练、预测python文件
├── network_model.py                    # 网络模型python文件、包括 LSTM 、前馈神经网络 两种网络模型
├── __pycache__
│   ├── config.cpython-36.pyc
│   ├── data_util.cpython-36.pyc
│   ├── main.cpython-36.pyc
│   ├── model_function.cpython-36.pyc
│   ├── network_model.cpython-36.pyc
│   ├── tips.cpython-36.pyc
│   └── word_sequence.cpython-36.pyc
├── README.md
├── requirements                       # conda 导出的环境文件
│   └── sentiment_analysis.yml
├── tips.py                            # 报错提示 python 文件
├── web                                # web 应用程序目录
│   └── Web_App.py                     # web 应用程序文件
├── word_sequence                      # 训练的词库对象保存目录
│   ├── word_sequence_fnn.pickle
│   └── word_sequence_lstm.pickle
└── word_sequence.py                   # 词库操作python文件，包括训练词库、句子转换向量
```

## 模型使用

### 1. 安装相关库

创建env环境并安装相关的库
```
conda env create -f requirements/sentiment_analysis.yml
```

激活环境
```
source activate sentiment_analysis
```

访问 http 接口需要 Flask 的支持。可手动安装 Flask。
```
pip install Flask
```

### 2. 训练模型

打开config.py 文件编辑, 如下图训练一个 LSTM 情感分析的网络模型

```
# 比较重要！
# 指定训练、预测的模型。模型目前支持 lstm 和 fnn(前馈神经网络) 两种
# model = 'fnn' 或 model = 'lstm'
model = 'lstm'

# 指定模型是训练，还是预测。可在训练完成之后，修改为预测模式
# model_func = 'train' 训练
# model_func = 'predict' 预测
model_func = 'train'

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
```

保存配置文件，使用 python main.py 进行运行：

```
python main.py
```

训练结果输出，如：
```
epoch: 18/20 device: CPU train loss: 0.150 train accuracy: 0.958
valid loss: 0.407 valid accuracy: 0.875

epoch: 19/20 device: CPU train loss: 0.155 train accuracy: 0.952
valid loss: 0.399 valid accuracy: 0.877

epoch: 19/20 device: CPU train loss: 0.149 train accuracy: 0.956
valid loss: 0.406 valid accuracy: 0.873

epoch: 19/20 device: CPU train loss: 0.149 train accuracy: 0.955
valid loss: 0.380 valid accuracy: 0.880

epoch: 20/20 device: CPU train loss: 0.145 train accuracy: 0.958
valid loss: 0.384 valid accuracy: 0.881

epoch: 20/20 device: CPU train loss: 0.142 train accuracy: 0.956
valid loss: 0.433 valid accuracy: 0.876

start test ...
test loss: 0.398 accuracy: 87.9%
```

### 3. 预测

编辑config.py 配置文件，将 model_fun 配置项修改为 'predict' 。
```
# 指定模型是训练，还是预测。可在训练完成之后，修改为预测模式
# model_func = 'train' 训练
# model_func = 'predict' 预测
model_func = 'predict'
```

保存配置文件，使用 python main.py 进行运行：
```
python main.py
```

输出结果：
```
请输入一个中文句子，退出请输入 quit 或 exit 
请输入中文句子: 老公很喜欢，大小合适，裤子不错
预测结果： good

请输入中文句子: 话不多说你们自己看才穿一次
预测结果： bad
```

### 4. 使用http接口

编辑config.py 配置文件，将 model_fun 配置项修改为 'predict' 。
```
# 指定模型是训练，还是预测。可在训练完成之后，修改为预测模式
# model_func = 'train' 训练
# model_func = 'predict' 预测
model_func = 'predict'
```

启动应用
```
cd web
python Web_App.py
```

使用 curl 模拟访问：
```
curl -l -H "Content-type: application/json" -X POST -d '{"sent":"不错，穿着舒服，还不贵，挺好"}'  http://127.0.0.1:5000/predict_sent
```

请求返回信息
```
{"state": "OK", "value": "good"}
```


class Config:
    def __init__(self):
        self.train_path = 'data/train.csv'
        self.valid_path = 'data/dev.csv'
        self.test_path = 'data/test.csv'
        self.dict_path = 'data/word_dic.json'
        # self.bert_path = '/Users/maac/Desktop/bert-base-chinese.tar.gz'
        self.lr = 0.01
        self.batch_size = 50
        self.EPOCH = 10
        self.max_len = 40
        self.hidden_size = 128
        self.input_size = 256
        self.num_class = 2
        self.use_cuda = True
        self.dict_size = None
        self.num_layers = 3
        self.dropout = 0.5

config = Config()

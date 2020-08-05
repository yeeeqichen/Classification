class Config:
    def __init__(self):
        self.train_path = 'data/train.csv'
        self.valid_path = 'data/dev.csv'
        self.test_path = 'data/test.csv'
        self.dict_path = '/Users/maac/PycharmProjects/Classification/pytorch_bert_base_chinese/vocab.txt'
        self.bert_path = '/Users/maac/Desktop/bert-base-chinese.tar.gz'
        self.lr = 1e-4
        self.batch_size = 50
        self.EPOCH = 10
        self.max_len = 40
        self.hidden_size = 128
        self.input_size = 768
        self.num_class = 10
        self.use_cuda = False


config = Config()

class Config:
    def __init__(self):
        self.train_path = ''
        self.valid_path = ''
        self.test_path = ''
        self.dict_path = ''
        self.lr = 1e-4
        self.batch_size = 50
        self.EPOCH = 10
        self.max_len = 40
        self.hidden_size = 128
        self.input_size = 768
        self.num_class = 10
        self.use_cuda = True


config = Config()

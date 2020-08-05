from Config import config
import numpy
import torch
import json


def load_data(word_dict):
    def helper(path):
        with open(path) as f:
            seq_lens = []
            texts = []
            labels = []
            words = []
            for line in f:
                text, label = line.split(' ')
                seq_lens.append(len(text))
                texts.append(text)
                labels.append(label)
                words.append([word_dict[word] if word in word_dict else 1 for word in text])

        return texts, words, seq_lens, labels
    train_data = helper(config.train_path)
    valid_data = helper(config.valid_path)
    test_data = helper(config.test_path)
    return train_data, valid_data, test_data


def get_word_dict():
    dic = dict()
    idx = 2
    with open(config.train_path) as f:
        for line in f:
            text, _ = line.split(' ')
            for word in text:
                if word not in dic:
                    dic[word] = idx
                    idx += 1
    dic['[UNK]'] = 1
    dic['[PAD]'] = 0
    with open(config.dict_path, 'w') as f:
        f.write(json.dumps(dic, ensure_ascii=False))


def load_word_dict():
    with open(config.dict_path) as f:
        dic = json.loads(f.read())
    return dic


def get_bert_embed(words, seq_lens, bert_model, use_cuda):
    max_len = seq_lens[0]
    attention_mask = numpy.array([[1] * l + [0] * (max_len - l) for l in seq_lens])
    attention_mask = torch.from_numpy(attention_mask)
    if use_cuda:
        attention_mask = attention_mask.cuda()
    return bert_model(words, attention_mask,  output_all_encoded_layers=False)


class DataLoader:
    def __init__(self):
        get_word_dict()
        self.word_dict = load_word_dict()
        self.train_data, self.valid_data,  self.test_data = load_data(self.word_dict)
        self.batch_size = config.batch_size

    # todo:按照len降序排列
    def run(self, mode):
        if mode == 'train':
            data = self.train_data
        elif mode == 'valid':
            data = self.valid_data
        elif mode == 'test':
            data = self.test_data
        else:
            raise Exception('please specify mode! train valid or test')
        for i in range(len(data[0]) // self.batch_size):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            yield data[0][start:end], data[1][start:end], data[2][start:end], data[3][start:end]

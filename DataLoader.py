from Config import config
import numpy
import torch
import json
import csv
csv.field_size_limit(1000000)


def load_data(word_dict):
    def helper(path):
        with open(path) as f:
            reader = csv.reader(f)
            seq_lens = []
            texts = []
            labels = []
            words = []
            for line in reader:
                label, text = line[0].split('\t')[1:]
                if label == 'label':
                    continue
                seq_lens.append(len(text))
                texts.append(text)
                labels.append(int(label))
                words.append([word_dict[word] if word in word_dict else 0 for word in text])

        return texts, words, seq_lens, labels
    train_data = helper(config.train_path)
    valid_data = helper(config.valid_path)
    test_data = helper(config.test_path)
    return train_data, valid_data, test_data


def get_word_dict():
    dic = dict()
    idx = 2
    with open(config.train_path) as f:
        reader = csv.reader(f)
        for line in reader:
            text = line[0].split('\t')[2]
            if text == 'text':
                continue
            for word in text:
                if word not in dic:
                    dic[word] = idx
                    idx += 1
    dic['[UNK]'] = 1
    dic['[PAD]'] = 0
    with open(config.dict_path, 'w') as f:
        f.write(json.dumps(dic, ensure_ascii=False))


def load_word_dict():
    dic = dict()
    dic['[UNK]'] = 0
    idx = 1
    with open(config.dict_path) as f:
        for word in f:
            dic[word[:-1]] = idx
            idx += 1
    return dic


def get_bert_embed(words, seq_lens, bert_model, use_cuda):
    max_len = seq_lens[0]
    attention_mask = numpy.array([[1] * l + [0] * (max_len - l) for l in seq_lens])
    attention_mask = torch.from_numpy(attention_mask)
    if use_cuda:
        attention_mask = attention_mask.cuda()
    with torch.no_grad():
        bert_encode, _ = bert_model(words, attention_mask, output_all_encoded_layers=False)
    return bert_encode


class DataLoader:
    def __init__(self):
        # get_word_dict()
        self.word_dict = load_word_dict()
        self.train_data, self.valid_data,  self.test_data = load_data(self.word_dict)
        self.batch_size = config.batch_size
        self.max_len = config.max_len

    def run(self, mode):
        def pad(max_len):
            nonlocal start
            nonlocal end
            nonlocal data
            words = data[1][start:end]
            seq_lens = data[2][start:end]
            # labels = data[3][start:end]
            words_new = []
            for inst in words:
                if len(inst) <= max_len:
                    words_new.append(inst + [1] * (max_len - len(inst)))
                else:
                    words_new.append(inst[:max_len])
            seq_lens_new = [l if l <= max_len else max_len for l in seq_lens]
            words_new_tensor = torch.from_numpy(numpy.array(words_new))
            seq_lens_new_tensor = torch.from_numpy(numpy.array(seq_lens_new))
            if config.use_cuda:
                words_new_tensor = words_new_tensor.cuda()
                seq_lens_new_tensor = seq_lens_new_tensor.cuda()
            return words_new_tensor, seq_lens_new_tensor

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
            words_tensor, seq_len_tensor = pad(config.max_len)
            seq_lens, idx = seq_len_tensor.sort(0, descending=True)
            # 返回按照长度降序排列的words_tensor 和 seq_len_tensor
            yield words_tensor[idx], seq_lens


if __name__ == '__main__':
    temp = DataLoader()
    for words_tensor, seq_len_tensor in temp.run('train'):
        print(words_tensor, seq_len_tensor)

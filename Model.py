import torch


class Lstm(torch.nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, is_bidirectional, drop_out):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2,
                                  num_layers=num_layers, bidirectional=is_bidirectional,
                                  batch_first=True, dropout=drop_out)
        # self.classify = torch.nn.Linear(hidden_size, num_class)
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, encode, seq_len):
        packed_encode = torch.nn.utils.rnn.pack_padded_sequence(encode, seq_len, batch_first=True)
        packed_output, hidden = self.lstm(packed_encode)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, is_bidirectional, drop_out, num_class):
        super().__init__()
        self.lstm = Lstm(hidden_size=hidden_size, input_size=input_size, num_layers=num_layers,
                         is_bidirectional=is_bidirectional, drop_out=drop_out)
        self.classify = torch.nn.Linear(hidden_size, num_class)

    def forward(self, encode, seq_len):
        output, hidden = self.lstm(encode, seq_len)
        logits = torch.nn.functional.softmax(self.classify(hidden))
        return logits

import torch


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size, input_size, num_layers, is_bidirectional, drop_out, num_class, num_embeddings):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                  num_layers=num_layers, bidirectional=is_bidirectional,
                                  batch_first=True, dropout=drop_out)
        self.classify = torch.nn.Linear(hidden_size, num_class)
        self.embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=input_size)
        self.init()

    def forward(self, input, seq_len):
        embed = self.embedding(input)
        packed_encode = torch.nn.utils.rnn.pack_padded_sequence(embed, seq_len, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_encode)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # print(hidden.shape, cell.shape)
        logits = torch.nn.functional.softmax(self.classify(hidden[0]))
        # print(logits)
        # logits = self.classify(hidden[0]).squeeze()
        return logits

    def init(self):
        for name, parm in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(parm)
            elif 'bias' in name:
                torch.nn.init.constant_(parm, val=0)



import Model
import DataLoader
import torch
from Config import config

loader = DataLoader.DataLoader()
model = Model.MyModel(config.hidden_size, config.input_size, config.num_layers, True,
                      config.dropout, config.num_class, config.dict_size)
if config.use_cuda:
    model = model.cuda()
opt = torch.optim.Adam(lr=config.lr, params=model.parameters())
for epoch in range(config.EPOCH):
    time = 0
    print("EPOCH: {}".format(epoch))
    for words_tensor, seq_len_tensor, labels in loader.run('train'):
        logits = model(words_tensor, seq_len_tensor)
        # print(labels.shape)
        labels = labels.unsqueeze(dim=1)
        if config.use_cuda:
            labels = labels.cuda()
        # print(labels.shape)
        labels = torch.zeros(config.batch_size, config.num_class).cuda().scatter_(1, labels, 1)
        # print(labels)
        loss = torch.nn.functional.mse_loss(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        time += 1
        if time % 100 == 0:
            print(loss)
        #     total = 0
        #     hit = 0
        #     for w_t, s_l_t, l in loader.run('train'):
        #         total += config.batch_size
        #         predict = torch.argmax(model(w_t, s_l_t), dim=1).cpu().numpy()
        #         for p, a in zip(predict, l.cpu().numpy()):
        #             if p == a:
        #                 hit += 1
        #     print("accuracy: {}".format(hit / total))




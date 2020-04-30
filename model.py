import torch
from torch import nn

config = {
    "embedding_size": 100,
    "hidden_size": 100,
    "n_layers": 3,
    "dropout": 0.2,
    "bidirectional": True,
    "linear": [100, 100, 2],
    "learning_rate": 0.01,
    "sentence_length": 58,
}


class RNNModel(nn.Module):
    def __init__(self, n_token, config):
        embed_size = config['embedding_size']
        hidden_size = config['hidden_size']
        n_layers = config['n_layers']
        dropout = config['dropout']
        bidirectional = config['bidirectional']
        ffnn_layers = config['linear']
        sentence_length = config["sentence_length"]

        super(RNNModel, self).__init__()

        self.embed = nn.Embedding(n_token, embed_size)
        self.word_dropout = nn.Dropout(config.get("word_dropout", 0))

        self.RNN = nn.LSTM(embed_size, hidden_size, n_layers,
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.sent_dropout = nn.Dropout(config.get("sent_dropout", 0))

        # 计算出RNN层的输出size
        curr_dim = hidden_size * (bidirectional + 1) * sentence_length

        ffnn_layers = [curr_dim] + ffnn_layers
        self.layers = []
        for i, o in zip(ffnn_layers[:-1], ffnn_layers[1:]):
            self.layers.append(nn.Linear(i, o))
            self.layers.append(nn.RELU())
            self.layers.append(nn.Dropout(config.get("dropout", 0)))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(x.size())
        embedding = self.embed(x)
        embedding = self.word_dropout(embedding)

        output, hidden = self.RNN(embedding)
        output = self.sent_dropout(output)

        # print(output.size())
        output = output.reshape(output.size(0), -1).cuda()
        # print(output.size())
        for layer in self.layers:
            output = layer.cuda()(output)
            # output = nn.ReLU().cuda()(output)
            # print(output.size())

        output = self.softmax.cuda()(output)

        return output.view(-1, 2).cuda()

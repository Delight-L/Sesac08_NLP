import torch 
import torch.nn as nn

PAD_IDX = 0

class SpamRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, 
                 num_layers=5, 
                 dropout=0.3, num_classes=2):
        super().__init__()
        #nn.Embedding(input개수, output개수) -> 
        #input -> vocab_size => N개의 단어들을 가지고 있습니다~
        #embedding -> 하나의 단어를 '몇 차원'으로 표시할 것인지
        #embed_size == embed_dim(임베딩 차원)
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # input_size : The number of expected features in the input x
        # hidden_size : The number of features in the hidden state h
        # num_layers : Number of recurrent layers**
        # Batch_first : if True, then the input (batch, seq, feature)
        self.rnn = nn.RNN(
            input_size = embed_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embed = self.embedding(x)
        out, _ =self.rnn(embed)
        #이 때, out은 (batch, seq, hidden)으로 구성되어 있음
        last = out[:, -1, :]
        result = self.fc(self.dropout(last))
        return result


#뉴 클래스 만들기! -> LSTM
class SpamLSTM(nn.Module):
    #https://docs.pytorch.org/docs/2.13/generated/torch.nn.LSTM.html
    def __init__(self):
        self.embedding 
        self.lstm 
        self.dropout
        self.fc

    def forward(self, x):
        embedding = self.embedding(x)
        out, _ = self.lstm(embedding)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))
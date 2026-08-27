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


# LSTM <게이트 3개>
# Forget Gate (망각 게이트) "무엇을 잊을 것인가?" : 이전 t-1과 현재를 받아, 버릴 정보 결정
# Input Gate (입력 게이트): "무엇을 기억할 것인가?" : 현재를 받아, 새롭게 저장할 정보 결정
# Output Gate (출력 게이트): "무엇을 출력으로 내보낼 것인가?" : 업데이트된 상태 바탕으로 다음 t+1로 보낼 정보 결정

# GRU <게이트 2개>
# Reset Gate (리셋 게이트): "이전 기억을 얼마나 무시할 것인가?"
# Update Gate (업데이트 게이트): "이전 기억과 새 기억의 비율을 어떻게 가져갈 것인가?"


#뉴 클래스 만들기! -> LSTM
class SpamLSTM(nn.Module):
    #https://docs.pytorch.org/docs/2.13/generated/torch.nn.LSTM.html
    def __init__(self, vocab_size, 
                 embed_dim = 128, 
                 hidden_size = 256, 
                 dropout = 0.3, 
                 num_classes = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 
                                      embed_dim,
                                      padding_idx=0)
        self.lstm = nn.LSTM(input_size = embed_dim,
                            hidden_size = hidden_size,
                            num_layers = 5,
                            batch_first = True,
                            dropout = dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedding = self.embedding(x)
        # [out 설명]
        # - out: LSTM의 모든 타임스텝(시퀀스 위치)에서의 은닉 상태(Hidden State) 출력값.
        # - Shape: [Batch Size, Sequence Length, Hidden Dim * Num Directions]
        #   (예: 단방향 LSTM이고 Hidden Dim=256이라면 [32, 50, 256])
        # - out[:, 0, :]  -> 첫 번째 단어를 처리한 후의 은닉 상태
        # - out[:, -1, :] -> 마지막(50번째) 단어를 처리한 후의 은닉 상태 (전체 문장 맥락 응축)
        out, _ = self.lstm(embedding)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


class SpamGRU(nn.Module):
    def __init__(self, vocab_size, 
                 embed_dim = 128, 
                 hidden_size = 256, 
                 dropout = 0.3, 
                 num_classes = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0) 
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=5,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embed = self.embed(x)
        out, _ = self.gru(embed)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))
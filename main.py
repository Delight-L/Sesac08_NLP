#RNN, LSTM, GRU 
#ResNet -> Residual(잔차) 구조가 특징 
#RNN -> Recurrent 구조

#RNN, LSTM, GRU 만들때도 -> 토큰화 + vocab 

#os -> 파일, 경로
#re -> 정규표현식(regex)
#urllib.request -> 인터넷 자료 다운로드 라이브러리
import os, re, urllib.request, zipfile
import pandas as pd  

MAX_LEN = 50

def load_data(data_path='SMSSpamCollection', batch_size=32):
    #os.path.exists(경로) : '경로'가 존재하는가?
    if not os.path.exists(data_path):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
        #정해놓은 url에 가서 zip 다운로드
        urllib.request.urlretrieve(url, 'smsspam.zip')
        with zipfile.ZipFile('smsspam.zip') as z:
            z.extractall('.')
        print('완료')

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.split()
    return text

#문자열 -> 정수, 어떤 문자열이 몇 번 정수로 바뀌었나? 기억
from collections import Counter #워드클라우드 만들 때 '단어 수' 함수
def build_vocab(df, min_freq=2):
    #나는 딥러닝을 공부하고 있어. 딥러닝은 정말 많은 작업을 할 수 있어.
    # 나 0 는 1 딥러닝 2 을 4 공부하고 3 있어 4 딥러닝 2 은 5 정말 6 많은 7 작업 8 을 ...
    counter = Counter(tok for text in df for tok in tokenize(text))

    #패딩 
    #알려지지 않음
    vocab = {'<PAD>': 0, '<UNK>': 1}
    #counter.items (딥러닝, 2)
    for word, freq in counter.items():
        if freq >= min_freq: 
            vocab[word] = len(vocab)
    return vocab


def preprocessing(data_path='SMSSpamCollection'):
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'text'])
    #print( df.head() )

    df['label'] = (df['label'] == 'spam').astype(int) #label이라는 열을 숫자로 변환
    print(f'스팸이 아닌 것 : {(df.label == 0).sum()}, 스팸인 것 : {(df.label == 1).sum()}')

    #vocab으로 변환! -> 어휘 사전
    vocab = build_vocab(df['text'])
    #print(vocab)

    dataset = SpamDataset(df, vocab=vocab, max_len=MAX_LEN)
    n_total = len(dataset)
    n_train = int(n_total * 0.7)
    n_valid = int(n_total * 0.15)
    n_test = n_total - n_train - n_valid

    train_set, valid_set, test_set = random_split(dataset, [n_train, n_valid, n_test])

    batch_size=32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, valid_loader, test_loader


#자연어 모델을 위한 커스텀 데이터셋 만들기
import torch
from torch.utils.data import DataLoader, Dataset, random_split
class SpamDataset(Dataset):
    #데이터셋이 필수적으로 가져야할 3가지 함수
    def __init__(self, df, vocab, max_len):
        #데이터 + 라벨
        #text의 vocab 기반 정수 전환
        self.texts = df['text'].tolist()
        #labels의 tensor 전환 필요 / torch.long 은 int64
        self.labels = torch.tensor(df['label'].tolist(), 
                                   dtype=torch.long)
        self.vocab = vocab
        self.max_len = max_len

    #MAX_LEN = 50
    #{'<PAD>': 0, '<UNK>': 1, 'go': 2, 'until': 3, 'point': 4
    #Go until jurong point
    def _text_to_tensor(self, text, vocab, max_len=MAX_LEN):
        sample = [vocab.get(t, 1) for t in tokenize(text)]

        if len(sample) >= max_len:
            sample = sample[:max_len]
        else:
            sample += [0] * (max_len - len(sample))
        return torch.tensor(sample, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self._text_to_tensor(self.texts[index], self.vocab, self.max_len)
        label = self.labels[index]
        return text, label

def train(model, train_loader, valid_loader, criterion, optimizer,
          num_epochs, device, model_name='Model'):
    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'valid_acc': []}
    best_valid_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            output = model(X_batch)
            loss   = criterion(output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(output, 1)
            correct += (pred == y_batch).sum().item()
            total   += y_batch.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total * 100

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, pred = torch.max(model(X_batch), 1)
                v_correct += (pred == y_batch).sum().item()
                v_total   += y_batch.size(0)
        valid_acc = v_correct / v_total * 100

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['valid_acc'].append(valid_acc)

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc

        if (epoch + 1) % 2 == 0:
            print(f'[{model_name}] Epoch {epoch+1:2d}/{num_epochs} | '
                  f'loss: {train_loss:.4f} | train: {train_acc:.2f}% | valid: {valid_acc:.2f}%')

    print(f'[{model_name}] 최고 검증 정확도: {best_valid_acc:.2f}%\n')
    return history

def evaluate(model, test_loader, device, model_name='Model'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            _, pred = torch.max(model(X_batch.to(device)), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    return all_labels, all_preds

if __name__ == '__main__':
    train_loader, valid_loader, test_loader = preprocessing()

    # x_train, y_train = next(iter(train_loader))
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_train[0])
    # print(y_train[0])
#RNN, LSTM, GRU 
#ResNet -> Residual(잔차) 구조가 특징 
#RNN -> Recurrent 구조

#RNN, LSTM, GRU 만들때도 -> 토큰화 + vocab 

#os -> 파일, 경로
#re -> 정규표현식(regex)
#urllib.request -> 인터넷 자료 다운로드 라이브러리
import os, re, urllib.request, zipfile
import pandas as pd  

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

#자연어 모델을 위한 커스텀 데이터셋 만들기
from torch.utils.data import DataLoader, Dataset, random_split
class SpamDataset(Dataset):
    #데이터셋이 필수적으로 가져야할 3가지 함수
    def __init__(self, df, vocab):
        #데이터 + 라벨
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.vocab = vocab

    def __len__(self):

    def __getitem__(self, index):
        


if __name__ == '__main__':
    preprocessing()
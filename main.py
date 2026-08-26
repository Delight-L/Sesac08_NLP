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

#문자열 -> 정수, 어떤 문자열이 몇 번 정수로 바뀌었나? 기억
def build_vocab(df):

def preprocessing(data_path='SMSSpamCollection'):
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'text'])
    #print( df.head() )

    df['label'] = (df['label'] == 'spam').astype(int) #label이라는 열을 숫자로 변환
    print(f'스팸이 아닌 것 : {(df.label == 0).sum()}, 스팸인 것 : {(df.label == 1).sum()}')

    #vocab으로 변환!
    vocab = build_vocab(df['text'])

if __name__ == '__main__':
    preprocessing()
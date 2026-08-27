#토크나이저만 변경했는데 성능이 달라질 수 있는가?

#토크나이저 가져오기
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, Subset

import pandas as pd 
import torch

#데이터셋 클래스 정의
class HugDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        super().__init__()
        self.texts = df['text'].tolist()
        self.labels = torch.tensor(df['label'].tolist(), 
                                   dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        #인코딩
        #max_length -> 최대 길이 /512 
        #padding -> 'max_length'
        #truncation -> True(512 넘으면 text를 자르세요)
        #return_tensors -> pt(파이토치)
        enc = self.tokenizer(self.texts[index],
                             max_length = self.max_len,
                             padding = 'max_length',
                             truncation = True,
                             return_tensors = 'pt')

        #squeeze -> 축을 하나 없애는 것
        return enc['input_ids'].squeeze(0), self.labels[index]

from huggingface_hub import login
import main

def make_splits(length_of_df):

    split_num = torch.randperm(length_of_df, 
                               generator = torch.Generator().manual_seed(42)).tolist()

    n_train = int(length_of_df * 0.7)
    n_valid = int(length_of_df * 0.15)
    return split_num[:n_train], split_num[n_train:n_train+n_valid], split_num[n_train+n_valid:]


from models.rnn import SpamLSTM
from utils.visualize import plot_comparison
import torch.nn as nn 
import torch.optim as optim

if __name__ == '__main__':

    #login()

    #허깅페이스에서 토크나이저를 다운로드 한다.
    #AutoTokenizer.from_pretrained -> 사전학습된 토크나이저 모델을 사용 가능
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    #0. 전체 데이터셋인 df 을 가져옴
    #데이터셋을 판다스로 불러옴
    df = pd.read_csv('SMSSpamCollection', sep='\t',
                     header=None, names=['label', 'text'])
    #데이터프레임 열 중 label 열을 0(정상), 1(스팸)
    df['label'] = (df['label'] == 'spam').astype(int) 

    train_idx, valid_idx, test_idx = make_splits(len(df))


    #1. 전통적인 tokenizer를 거친 데이터셋을 가지고 와서 lstm으로 훈련
    #1-1. 전통적인 tokenizer를 거친 데이터셋 로드
    original_vocab = main.build_vocab(df)
    original_data = main.SpamDataset(df, original_vocab, 50)

    original_train = DataLoader(Subset(original_data, train_idx), batch_size=32, shuffle=True) 
    original_valid = DataLoader(Subset(original_data, valid_idx), batch_size=32)
    original_test = DataLoader(Subset(original_data, test_idx), batch_size=32)

    #1-2. 모델생성
    original_lstm = SpamLSTM(len(original_vocab))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(original_lstm.parameters())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #훈련, 평가
    o_history = main.train(original_lstm, 
               original_train, original_valid, 
                criterion, 
                optimizer,
                num_epochs=30, 
                device=device, 
                model_name='original')

    o_labels, o_preds = main.evaluate(original_lstm, original_test, 
                                      device, 
                                        model_name='original')

    #2. 허깅페이스 tokenizer를 거친 데이터셋을 가지고 와서 lstm으로 훈련
    #2-1. 허깅페이스를 거친 데이터셋 로드
    hug_data = HugDataset(df, tokenizer, 50)

    hug_train = DataLoader(Subset(hug_data, train_idx), batch_size=32, shuffle=True) 
    hug_valid = DataLoader(Subset(hug_data, valid_idx), batch_size=32) 
    hug_test = DataLoader(Subset(hug_data, test_idx), batch_size=32) 



    #허깅페이스에서 다운로드받은 토크나이저의 특성 속에 vocab_size가 미리 정의되어 있음
    hug_lstm = SpamLSTM(tokenizer.vocab_size)

    new_optimizer = optim.Adam(hug_lstm.parameters())
    h_history = main.train(hug_lstm, 
                hug_train, hug_valid, 
                criterion, 
                new_optimizer,
                num_epochs=30, 
                device=device, 
                model_name='hugging')

    h_labels, h_preds = main.evaluate(hug_lstm, hug_test, 
                                      device, 
                                        model_name='hugging')


    histories, eval_results, train_models = [], [], {}
    histories.append(o_history)
    histories.append(h_history)

    eval_results.append((o_labels, o_preds))
    eval_results.append((h_labels, h_preds))

    train_models['original'] = original_lstm
    train_models['hugging'] = hug_lstm      

    plot_comparison(histories, ['original', 'hug'])

    # print(f'토크나이저의 어휘 크기 : {tokenizer.vocab_size}')
    # print(f'토크나이저의 최대 입력 길이 : {tokenizer.model_max_length}')
    # print(f'특수 토큰 ID 확인 {tokenizer.special_tokens_map}')

    # sentence = input('자를 영어 문장을 입력하세요')
    # tokens = tokenizer.tokenize(sentence)
    # print(f'토큰화 결과 : {tokens}')

    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(f'입력되는 숫자 : {input_ids}')

    # #실제 처리
    # enc = tokenizer(sentence)
    
    # #print(enc.keys())
    # print(enc['input_ids'])
    # print(enc['attention_mask'])
    # #디코딩
    # decode = tokenizer.convert_ids_to_tokens(enc['input_ids'])
    # print(f'디코딩 결과 : {decode}')

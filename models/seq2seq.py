#[seq2seq의 장점과 핵심 아이디어]
#1. 기존 모델 한계
#1-1. 기존 모델은 출력이 1:1 고정이 되어 있었음(input->output)
#1-2. 어순과 구조가 맞지 않는 언어의 번역 문제 / QA  => input을 읽고 바로 output을 낼 수 있도록 하는 구조 필요
#2. 아이디어 -> "인코더 + 디코더" (입력/출력 길이를 분리)

#[모델 훈련 연습]
#1. 데이터 준비
#2. 데이터셋 클래스 만들기
#3. seq2seq 모델 만들기
#4. 훈련-검증

import re, unicodedata

#텍스트 정제
def normalize(text, lang='en'):
    if lang=='fr':
        #프랑스어 악센트 기호 처리
        text = unicodedata.normalize('NFD', text)
        #아스키 코드 범위 넘으면 'ignore' 한 후, 다시 ascii 변환
        text = text.encode('ascii', 'ignore').decode('ascii')

    text = text.lower().strip()
    text = re.sub(r'([.!?])', r' \1', text) #., !, ? 앞을 한 칸 띄우세요
    text = re.sub(r'[^a-z.!? ]+', ' ', text) #a-z.!?공백 이 아닌 것을 ' '공백으로 변환
    return text.split()


import os, shutil, urllib.request , zipfile
#인터넷에서 데이터를 다운로드
def load_data(max_pairs=20000):
    data_file = './data/data/eng-fra.txt'
    if not os.path.exists(data_file):
        #다운로드가 안됐다면? -> 다운로드 하세요!
        os.makedirs('data', exist_ok=True)

        urllib.request.urlretrieve(
            'https://download.pytorch.org/tutorial/data.zip', 'data/data.zip'
        )
        with zipfile.ZipFile('data/data.zip') as z :
            z.extractall('data')

    pairs = []
    with open('./data/data/eng-fra.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            print(parts)
            if len(parts) < 2 :
                continue

            en = normalize(parts[0], 'en')
            fr = normalize(parts[1], 'fr')

            pairs.append((en, fr))

    import random
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    return pairs

from collections import Counter
#Vocab 만들기
def build_vocab(pairs):
    en_cnt, fr_cnt = Counter(), Counter()

    for en, fr in pairs:
        en_cnt.update(en)
        fr_cnt.update(fr)

    #영어 Vocab
    #프랑스어 Vocab
    en_vocab, fr_vocab = Vocab(), Vocab()





if __name__ == '__main__':
    pairs = load_data()
    print(f'페어 길이 : {len(pairs)}, 페어[0] {pairs[0]}')
# BPE (Byte Pair Encoding) Deep Dive

## 목차
1. [BPE란 무엇인가](#1-bpe란-무엇인가)
2. [BPE 알고리즘 상세](#2-bpe-알고리즘-상세)
3. [Character-level과의 차이](#3-character-level과의-차이)
4. [실제 구현 예제](#4-실제-구현-예제)
5. [GPT에서의 BPE 활용](#5-gpt에서의-bpe-활용)

---

## 1. BPE란 무엇인가?

### 1.1 정의
**BPE (Byte Pair Encoding)**는 데이터 압축 알고리즘에서 시작되어, 현대 NLP에서 **subword 토크나이저**로 널리 사용되는 기법입니다.

**핵심 아이디어:**
> 자주 등장하는 문자 쌍(pair)을 반복적으로 병합(merge)하여, Character와 Word의 중간 수준인 Subword 단위의 어휘를 만든다.

### 1.2 역사
- **1994년**: 데이터 압축 알고리즘으로 등장
- **2016년**: Neural Machine Translation에 처음 적용 (Sennrich et al.)
- **2018년**: GPT-1에서 채택
- **현재**: GPT-2/3/4, RoBERTa 등 주요 LLM의 표준

### 1.3 왜 BPE가 필요한가?

**Word-level의 문제:**
```python
# 어휘에 없는 단어는 처리 불가
vocab = ["hello", "world", "good"]
text = "goodbye"  # ❌ OOV (Out-of-Vocabulary)
```

**Character-level의 문제:**
```python
# 시퀀스가 너무 길어짐
text = "internationalization"
tokens = ['i','n','t','e','r','n','a','t','i','o','n','a','l','i','z','a','t','i','o','n']
# 20개 토큰! (단어 하나에)
```

**BPE의 해결책:**
```python
# Subword 단위로 분해
text = "internationalization"
tokens = ['intern', 'ation', 'al', 'iz', 'ation']
# 5개 토큰 (효율적!)
# 모든 단어를 처리 가능 (OOV 해결)
```

---

## 2. BPE 알고리즘 상세

### 2.1 학습 단계 (Training)

**목표:** 어휘 사전(vocabulary) 생성

**입력:**
- 코퍼스 (학습 데이터)
- 목표 어휘 크기 (예: 50,000)

**알고리즘:**

```
1. 초기화:
   - 모든 문자를 개별 토큰으로 분리
   - 어휘 = 고유 문자 집합

2. 반복 (목표 어휘 크기에 도달할 때까지):
   a. 모든 인접한 토큰 쌍의 빈도 계산
   b. 가장 빈번한 쌍 선택
   c. 해당 쌍을 하나의 새 토큰으로 병합
   d. 새 토큰을 어휘에 추가
   e. 코퍼스에서 해당 쌍을 모두 새 토큰으로 교체

3. 출력:
   - 최종 어휘 사전
   - 병합 규칙 (merge rules)
```

### 2.2 단계별 예제

**입력 코퍼스:**
```
"low" (빈도: 5)
"lower" (빈도: 2)
"newest" (빈도: 6)
"widest" (빈도: 3)
```

**Step 0: 초기화**
```
어휘: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd']

토큰화된 코퍼스:
l o w </w>        (5회)
l o w e r </w>    (2회)
n e w e s t </w>  (6회)
w i d e s t </w>  (3회)

※ </w>는 단어 끝(End of Word) 마커
```

**Step 1: 가장 빈번한 쌍 찾기**
```
쌍 빈도 계산:
e s: 6+3 = 9회  ← 최다!
s t: 6+3 = 9회  ← 최다!
l o: 5+2 = 7회
o w: 5+2 = 7회
...

선택: "e" + "s" → "es" (알파벳 순으로 먼저)
```

**Step 2: 병합 및 어휘 추가**
```
어휘: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', 'es']

업데이트된 코퍼스:
l o w </w>        (5회)
l o w e r </w>    (2회)
n e w es t </w>   (6회)  ← "es" 병합
w i d es t </w>   (3회)  ← "es" 병합
```

**Step 3: 다음 쌍 찾기**
```
쌍 빈도 계산:
es t: 6+3 = 9회  ← 최다!
l o: 5+2 = 7회
o w: 5+2 = 7회

선택: "es" + "t" → "est"
```

**Step 4: 계속 반복...**
```
반복 3:
어휘에 "est" 추가
n e w est </w>
w i d est </w>

반복 4:
어휘에 "lo" 추가
lo w </w>
lo w e r </w>

반복 5:
어휘에 "low" 추가
low </w>
low e r </w>

...
```

**최종 어휘 (예시):**
```python
vocab = [
    # 원본 문자
    'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd',
    # 병합된 subword
    'es', 'est', 'lo', 'low', 'newest', 'lowest', 'widest',
    '</w>'
]
```

### 2.3 인코딩 단계 (Encoding)

학습된 병합 규칙을 사용하여 새 텍스트를 토큰화:

```python
# 새 단어: "lowest"
# 초기: ['l', 'o', 'w', 'e', 's', 't', '</w>']

# 병합 규칙 순서대로 적용:
# 1. 'e' + 's' → 'es'
#    ['l', 'o', 'w', 'es', 't', '</w>']
# 2. 'es' + 't' → 'est'
#    ['l', 'o', 'w', 'est', '</w>']
# 3. 'l' + 'o' → 'lo'
#    ['lo', 'w', 'est', '</w>']
# 4. 'lo' + 'w' → 'low'
#    ['low', 'est', '</w>']

# 최종: ['low', 'est', '</w>']
```

### 2.4 의사코드 (Pseudocode)

```python
def train_bpe(corpus, num_merges):
    """
    BPE 학습 알고리즘

    Args:
        corpus: 학습 데이터 (단어와 빈도의 딕셔너리)
        num_merges: 수행할 병합 횟수

    Returns:
        vocab: 최종 어휘
        merges: 병합 규칙 리스트
    """
    # 1. 초기화: 문자 단위로 분리
    vocab = get_all_characters(corpus)
    merges = []

    for i in range(num_merges):
        # 2. 모든 쌍의 빈도 계산
        pairs = defaultdict(int)
        for word, freq in corpus.items():
            symbols = word.split()
            for j in range(len(symbols) - 1):
                pairs[symbols[j], symbols[j+1]] += freq

        # 3. 가장 빈번한 쌍 선택
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)

        # 4. 병합 수행
        new_symbol = ''.join(best_pair)
        vocab.add(new_symbol)
        merges.append(best_pair)

        # 5. 코퍼스 업데이트
        corpus = merge_pair(corpus, best_pair, new_symbol)

    return vocab, merges


def encode(text, merges):
    """
    학습된 BPE로 텍스트 인코딩
    """
    # 문자 단위로 분리
    tokens = list(text) + ['</w>']

    # 병합 규칙 순서대로 적용
    for pair in merges:
        tokens = apply_merge(tokens, pair)

    return tokens
```

---

## 3. Character-level과의 차이

### 3.1 비교표

| 특징 | Character-level | BPE | Word-level |
|------|----------------|-----|-----------|
| **어휘 크기** | ~100 | 10K~50K | 100K~1M |
| **학습 필요** | ❌ | ✅ | ❌ |
| **OOV 처리** | 완벽 | 거의 완벽 | 취약 |
| **시퀀스 길이** | 매우 길다 | 적당 | 짧다 |
| **의미 단위** | 없음 | 일부 있음 | 완전 |
| **사용 예** | microgpt | GPT-2/3/4 | 초기 NLP |

### 3.2 구체적 예시

**텍스트:** "internationalization"

```python
# Character-level (microgpt 방식)
tokens_char = [
    'i', 'n', 't', 'e', 'r', 'n', 'a', 't', 'i', 'o',
    'n', 'a', 'l', 'i', 'z', 'a', 't', 'i', 'o', 'n'
]
# 토큰 수: 20
# 어휘 크기: ~26 (알파벳)

# BPE
tokens_bpe = [
    'intern', 'ation', 'al', 'ization'
]
# 토큰 수: 4
# 어휘 크기: ~30,000 (학습된 subword)

# Word-level
tokens_word = ['internationalization']
# 토큰 수: 1
# 어휘 크기: ~500,000 (모든 단어)
# 문제: 처음 보는 단어는? ❌
```

### 3.3 장단점 심화 분석

**Character-level 장점:**
```python
# 1. 완벽한 커버리지
text = "supercalifragilisticexpialidocious"  # 새 단어
tokens = list(text)  # ✅ 문제없이 처리

# 2. 작은 어휘
vocab_size = 26  # 알파벳만

# 3. 구현 간단
vocab = sorted(set(text))  # 끝!
```

**Character-level 단점:**
```python
# 1. 긴 시퀀스
text = "Hello world"
tokens = list(text)  # 11개 토큰
# → Attention 계산량: O(11²)

# 2. 의미 단위 없음
# 'h', 'e', 'l', 'l', 'o'를 모델이 "hello"로 학습해야 함
```

**BPE 장점:**
```python
# 1. 균형잡힌 시퀀스 길이
text = "Hello world"
tokens = ['Hello', ' world']  # 2개 토큰
# → Attention 계산량: O(2²) - 훨씬 적음!

# 2. 의미 있는 단위
# 'ation', 'ing' 등 자주 쓰이는 접사를 자동 학습

# 3. OOV 거의 없음
text = "supercalifragilistic"
tokens = ['super', 'cal', 'ifrag', 'ilistic']  # 분해 가능
```

**BPE 단점:**
```python
# 1. 학습 필요
# 대규모 코퍼스에서 수천~수만 번 병합

# 2. 큰 어휘
vocab_size = 50257  # GPT-2
# → 임베딩 테이블 메모리 증가

# 3. 구현 복잡
# Character-level: 3줄
# BPE: 수백 줄
```

---

## 4. 실제 구현 예제

### 4.1 간단한 BPE 구현

```python
from collections import defaultdict
import re

def get_vocab(corpus):
    """코퍼스에서 단어 빈도 추출"""
    vocab = defaultdict(int)
    for word in corpus:
        vocab[' '.join(word) + ' </w>'] += 1
    return vocab

def get_pairs(vocab):
    """모든 인접 쌍과 빈도 계산"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """선택된 쌍을 병합"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab

# 학습 예제
corpus = ['low', 'low', 'low', 'low', 'low',
          'lower', 'lower',
          'newest', 'newest', 'newest', 'newest', 'newest', 'newest',
          'widest', 'widest', 'widest']

vocab = get_vocab(corpus)
print("초기 어휘:")
print(vocab)

num_merges = 10
for i in range(num_merges):
    pairs = get_pairs(vocab)
    if not pairs:
        break

    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

    print(f"\n병합 {i+1}: {best}")
    print(f"어휘: {list(vocab.keys())}")
```

### 4.2 실행 결과 예측

```
초기 어휘:
{
    'l o w </w>': 5,
    'l o w e r </w>': 2,
    'n e w e s t </w>': 6,
    'w i d e s t </w>': 3
}

병합 1: ('e', 's')
어휘: ['l o w </w>', 'l o w e r </w>', 'n e w es t </w>', 'w i d es t </w>']

병합 2: ('es', 't')
어휘: ['l o w </w>', 'l o w e r </w>', 'n e w est </w>', 'w i d est </w>']

병합 3: ('l', 'o')
어휘: ['lo w </w>', 'lo w e r </w>', 'n e w est </w>', 'w i d est </w>']

병합 4: ('lo', 'w')
어휘: ['low </w>', 'low e r </w>', 'n e w est </w>', 'w i d est </w>']

...
```

---

## 5. GPT에서의 BPE 활용

### 5.1 GPT-2의 BPE 설정

```python
# GPT-2 토크나이저 스펙
vocab_size = 50257
# = 256 (bytes) + 50,000 (merges) + 1 (<|endoftext|>)

# 특수 토큰
special_tokens = ['<|endoftext|>']  # 문서 구분자
```

### 5.2 실제 토큰화 예제

```python
# Hugging Face Transformers 사용
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Hello, world!"
tokens = tokenizer.encode(text)
print(tokens)
# [15496, 11, 995, 0]

# 디코딩
decoded = tokenizer.decode(tokens)
print(decoded)
# "Hello, world!"

# 토큰별 확인
for token_id in tokens:
    print(f"{token_id}: '{tokenizer.decode([token_id])}'")
# 15496: 'Hello'
# 11: ','
# 995: ' world'
# 0: '!'
```

### 5.3 microgpt vs GPT-2 비교

```python
text = "internationalization"

# microgpt (Character-level)
tokens_micro = list(text)
print(len(tokens_micro))  # 20

# GPT-2 (BPE)
tokens_gpt2 = tokenizer.encode(text)
print(len(tokens_gpt2))   # 4~6 (subword 단위)
print(tokenizer.tokenize(text))
# ['international', 'ization'] 또는
# ['intern', 'ation', 'al', 'ization']
```

---

## 6. 심화 주제

### 6.1 Byte-level BPE (GPT-2의 방식)

GPT-2는 **문자 대신 바이트**를 기본 단위로 사용:

```python
# 기존 BPE: 문자 기반
vocab_base = ['a', 'b', 'c', ..., 'z', 'A', ..., 'Z']  # ~100개

# Byte-level BPE: 바이트 기반
vocab_base = [0, 1, 2, ..., 255]  # 256개 (모든 바이트)
# 장점: 모든 유니코드 문자 처리 가능!
```

### 6.2 WordPiece vs BPE

| 특징 | BPE | WordPiece |
|------|-----|-----------|
| **사용처** | GPT 시리즈 | BERT |
| **병합 기준** | 빈도 | Likelihood |
| **구현** | 간단 | 복잡 |

### 6.3 SentencePiece

- 언어 독립적
- 공백도 토큰으로 처리
- 사용: T5, mT5

---

## 7. 실습 과제

### Exercise 1: BPE 손계산
다음 코퍼스를 손으로 BPE 학습하세요 (3번 병합):

```
"aaabdaaabac"
```

### Exercise 2: BPE 구현
위의 간단한 BPE 코드를 완성하고 실행하세요.

### Exercise 3: GPT-2 토크나이저 실험
```python
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 다음 텍스트들의 토큰 수를 비교하세요:
texts = [
    "Hello world",
    "안녕하세요",
    "internationalization",
    "123456789"
]
```

---

## 8. 참고 자료

### 논문
- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al., 2016)](https://arxiv.org/abs/1508.07909)
- [Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

### 코드
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
- [Andrej Karpathy - minbpe](https://github.com/karpathy/minbpe)

### 영상
- [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)

---

**다음:** [notes.md로 돌아가기](./notes.md) | [Exercises](./exercises/tokenizer_practice.ipynb)

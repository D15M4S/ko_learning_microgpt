# 01. Tokenizer - 토크나이저

## 학습 목표
- 토크나이저의 역할과 중요성 이해
- Character-level tokenization 구현 방식 파악
- microgpt.py의 토크나이저 코드 분석
- 어휘 사전(vocabulary)과 특수 토큰(BOS) 개념 이해

---

## 1. 토크나이저란 무엇인가?

### 1.1 정의
**토크나이저(Tokenizer)**는 텍스트(문자열)를 **이산적인 심볼(토큰)**로 변환하는 시스템입니다.

```
입력 (텍스트):  "Hello"
         ↓ (토크나이징)
출력 (토큰):    [72, 101, 108, 108, 111]
```

### 1.2 왜 필요한가?
신경망은 숫자만 처리할 수 있습니다. 텍스트를 그대로 입력할 수 없기 때문에, 문자를 숫자(토큰 ID)로 변환해야 합니다.

**핵심 개념:**
- 텍스트 → 토큰 변환: **인코딩(Encoding)**
- 토큰 → 텍스트 변환: **디코딩(Decoding)**

### 1.3 토크나이저의 종류

| 종류 | 단위 | 예시 | 특징 |
|------|------|------|------|
| **Character-level** | 문자 | "Hi" → [H, i] | 어휘 크기 작음, 시퀀스 길어짐 |
| **Word-level** | 단어 | "Hi there" → [Hi, there] | 어휘 크기 큼, 미등록 단어 문제 |
| **Subword-level** | 하위 단어 | "Hello" → [Hel, lo] | BPE, WordPiece, SentencePiece |

**microgpt.py는 Character-level 토크나이저를 사용합니다.**

---

## 2. microgpt.py의 토크나이저 분석

### 2.1 코드 위치
`microgpt.py`의 23-27번째 줄:

```python
# Let there be a Tokenizer to translate strings to discrete symbols and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
BOS = len(uchars) # token id for the special Beginning of Sequence (BOS) token
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")
```

### 2.2 단계별 분석

#### Step 1: 고유 문자 추출
```python
uchars = sorted(set(''.join(docs)))
```

**과정:**
1. `''.join(docs)`: 모든 문서를 하나의 긴 문자열로 결합
   - 예: `["alice", "bob"]` → `"alicebob"`
2. `set(...)`: 중복 제거, 고유 문자만 남김
   - 예: `"alicebob"` → `{'a', 'b', 'c', 'e', 'i', 'l', 'o'}`
3. `sorted(...)`: 알파벳 순으로 정렬
   - 예: `['a', 'b', 'c', 'e', 'i', 'l', 'o']`

**결과:** `uchars`는 데이터셋에 등장하는 모든 고유 문자의 정렬된 리스트

#### Step 2: 특수 토큰 BOS 정의
```python
BOS = len(uchars)  # token id for the special Beginning of Sequence (BOS) token
```

**BOS (Beginning of Sequence):**
- 시퀀스의 **시작**과 **끝**을 나타내는 특수 토큰
- 토큰 ID는 `len(uchars)` (마지막 인덱스 다음)
- 예: `uchars`가 27개 문자라면, BOS는 27번 ID

**왜 BOS가 필요한가?**
- 모델에게 "문장이 시작되었다" 또는 "문장이 끝났다"는 신호 제공
- 생성 시 언제 멈춰야 하는지 알려줌

#### Step 3: 어휘 사전 크기 계산
```python
vocab_size = len(uchars) + 1  # total number of unique tokens, +1 is for BOS
```

**계산:**
- 고유 문자 개수 + BOS 토큰 1개
- 예: 27개 문자 + 1 (BOS) = 28

---

## 3. 인코딩과 디코딩 과정

### 3.1 인코딩: 문자열 → 토큰
`microgpt.py` 157번째 줄:

```python
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
```

**예시:**
```python
doc = "alice"
uchars = ['a', 'b', 'c', 'e', 'i', 'l', 'o']  # 인덱스: 0, 1, 2, 3, 4, 5, 6
BOS = 7

# 인코딩 과정:
# 'a' → uchars.index('a') → 0
# 'l' → uchars.index('l') → 5
# 'i' → uchars.index('i') → 4
# 'c' → uchars.index('c') → 2
# 'e' → uchars.index('e') → 3

tokens = [7, 0, 5, 4, 2, 3, 7]
#         ↑  ← "alice" →  ↑
#        BOS              BOS
```

### 3.2 디코딩: 토큰 → 문자열
`microgpt.py` 199번째 줄:

```python
sample.append(uchars[token_id])
```

**예시:**
```python
token_ids = [0, 5, 4, 2, 3]
chars = [uchars[tid] for tid in token_ids]
# [uchars[0], uchars[5], uchars[4], uchars[2], uchars[3]]
# ['a', 'l', 'i', 'c', 'e']

result = ''.join(chars)  # "alice"
```

---

## 4. 핵심 개념 정리

### 4.1 어휘 사전 (Vocabulary)
- **정의:** 모델이 인식할 수 있는 모든 토큰의 집합
- **microgpt.py:** `uchars` 리스트가 어휘 사전 역할
- **크기:** `vocab_size` = 고유 문자 수 + 특수 토큰 수

### 4.2 토큰 ID (Token ID)
- **정의:** 각 토큰을 나타내는 고유한 정수
- **범위:** 0 ~ (vocab_size - 1)
- **매핑:** `uchars` 리스트의 인덱스가 토큰 ID

### 4.3 특수 토큰 (Special Token)
- **BOS (Beginning of Sequence):** 시퀀스 시작/끝
- **다른 모델의 예:**
  - PAD: 패딩 (시퀀스 길이 맞추기)
  - UNK: Unknown (미등록 단어)
  - EOS: End of Sequence (BOS와 유사)

### 4.4 Character-level의 장단점

**장점:**
- ✅ 어휘 크기가 작다 (보통 50~100개)
- ✅ OOV (Out-of-Vocabulary) 문제 없음
- ✅ 구현이 간단함

**단점:**
- ❌ 시퀀스 길이가 길어짐 (단어 → 여러 문자)
- ❌ 문맥 이해에 더 많은 레이어 필요
- ❌ 계산 비용 증가

---

## 5. 실습 준비

### 5.1 다음 단계
`test/test.md`에서 AI Engineer 면접 질문으로 학습을 점검합니다:
1. 기본 개념 (토크나이저 정의, 분류, BOS 토큰)
2. 코드 구현 (인코딩/디코딩 함수)
3. 손으로 계산 (어휘 생성, 토큰화)
4. BPE 알고리즘 이해

### 5.2 학습 체크리스트
- [ ] 토크나이저의 역할을 1분 안에 설명할 수 있는가?
- [ ] `uchars`, `BOS`, `vocab_size`를 코드 없이 계산할 수 있는가?
- [ ] 인코딩/디코딩 과정을 손으로 재현할 수 있는가?
- [ ] Character-level vs BPE vs Word-level을 비교할 수 있는가?
- [ ] `test/test.md` 문제를 80점 이상 맞출 수 있는가?

---

## 6. 참고 자료

### 6.1 관련 파일
- [`ko_microgpt.py`](../ko_microgpt.py) - 23-27번 줄 (토크나이저 정의)
- [`ko_microgpt.py`](../ko_microgpt.py) - 157번 줄 (인코딩)
- [`ko_microgpt.py`](../ko_microgpt.py) - 199번 줄 (디코딩)

### 6.2 더 알아보기
- [Hugging Face Tokenizers 문서](https://huggingface.co/docs/tokenizers/)
- [Andrej Karpathy - Let's build GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- BPE (Byte Pair Encoding) 알고리즘

---

## 다음 단원
[02. Embedding - 임베딩](../02_embedding/notes.md)

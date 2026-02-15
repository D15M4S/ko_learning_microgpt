# 01. Tokenizer - 정답 및 해설

> `test.md`의 모범 답안입니다.
> 본인의 답변과 비교하여 부족한 부분을 체크하세요.

---

## Section 1: 기본 개념

### A1.1 토크나이저 정의
**답변:**
```
토크나이저(Tokenizer)는 텍스트(문자열)를 이산적인 심볼(토큰)로 변환하는 시스템입니다.

필요한 이유:
1. 신경망은 숫자만 처리할 수 있음
2. 텍스트를 그대로 입력할 수 없음
3. 문자를 정수(토큰 ID)로 변환하여 모델이 처리 가능하게 함

핵심 기능:
- 인코딩 (Encoding): 텍스트 → 토큰 ID
- 디코딩 (Decoding): 토큰 ID → 텍스트
```

**채점 기준:**
- 정의 (1점)
- 필요 이유 (2점)
- 인코딩/디코딩 언급 (2점)

---

### A1.2 토크나이저 분류
**답변:**
```
| 종류 | 어휘 크기 | 시퀀스 길이 | OOV 문제 | 예시 모델 |
|------|----------|------------|---------|----------|
| Character-level | 50~100 | 매우 길다 | 없음 | microgpt |
| Subword (BPE) | 10K~50K | 적당 | 거의 없음 | GPT-2/3/4 |
| Word-level | 100K~1M | 짧다 | 심각 | 초기 NLP 모델 |
```

**채점 기준:**
- 3가지 분류 (2점)
- 각 항목 정확성 (3점)

---

### A1.3 BOS 토큰
**답변:**
```
BOS (Beginning of Sequence)는 시퀀스의 시작/끝을 나타내는 특수 토큰입니다.

필요한 이유:
1. 문장 경계 표시: 모델에게 "문장이 시작되었다" 또는 "끝났다"는 신호 제공
2. 생성 중단 시점: 추론(inference) 시 BOS 토큰이 생성되면 텍스트 생성 중단
3. 문서 구분: 여러 문서를 학습할 때 각 문서의 경계를 명확히 구분
```

**채점 기준:**
- BOS 정의 (1점)
- 3가지 이유 각각 (1.5점씩)

---

## Section 2: 코드 구현

### A2.1 어휘 생성 코드 분석
**답변:**
```
Line 1: uchars = sorted(set(''.join(docs)))
- ''.join(docs): 모든 문서를 하나의 문자열로 결합
- set(...): 중복 제거, 고유 문자만 추출
- sorted(...): 알파벳 순으로 정렬
→ 결과: 데이터셋의 고유 문자 리스트 (정렬됨)

Line 2: BOS = len(uchars)
- uchars의 길이를 BOS 토큰 ID로 사용
- 예: uchars가 27개 문자라면, BOS = 27

Line 3: vocab_size = len(uchars) + 1
- 전체 어휘 크기 = 고유 문자 개수 + BOS 토큰 1개
- 예: 27 + 1 = 28
```

**채점 기준:**
- 각 줄 설명 정확성 (1.5점씩)

---

### A2.2 인코딩 함수 구현
**답변:**
```python
def encode(text, uchars, BOS):
    """문자열을 토큰 ID 리스트로 변환"""
    return [BOS] + [uchars.index(ch) for ch in text] + [BOS]

# 또는 명시적 버전:
def encode(text, uchars, BOS):
    """문자열을 토큰 ID 리스트로 변환"""
    tokens = [BOS]  # 시작 BOS
    for ch in text:
        tokens.append(uchars.index(ch))  # 각 문자의 인덱스
    tokens.append(BOS)  # 끝 BOS
    return tokens
```

**채점 기준:**
- BOS 앞뒤 추가 (2점)
- uchars.index() 사용 (2점)
- 정확한 구현 (1점)

---

### A2.3 디코딩 함수 구현
**답변:**
```python
def decode(tokens, uchars, BOS):
    """토큰 ID 리스트를 문자열로 변환"""
    return ''.join([uchars[t] for t in tokens if t != BOS])

# 또는 명시적 버전:
def decode(tokens, uchars, BOS):
    """토큰 ID 리스트를 문자열로 변환"""
    chars = []
    for token_id in tokens:
        if token_id != BOS:  # BOS 제외
            chars.append(uchars[token_id])
    return ''.join(chars)
```

**채점 기준:**
- BOS 제외 (2점)
- uchars[token_id] 사용 (2점)
- ''.join() 사용 (1점)

---

## Section 3: 손으로 계산

### A3.1 어휘 생성
**답변:**
```
docs = ["cat", "dog", "bat"]

Step 1: ''.join(docs) = "catdogbat"
Step 2: set(...) = {'c', 'a', 't', 'd', 'o', 'g', 'b'}
Step 3: sorted(...) = ['a', 'b', 'c', 'd', 'g', 'o', 't']

uchars = ['a', 'b', 'c', 'd', 'g', 'o', 't']
BOS = 7
vocab_size = 8
```

**채점 기준:**
- uchars 정확성 (2점)
- BOS 정확성 (1.5점)
- vocab_size 정확성 (1.5점)

---

### A3.2 인코딩 계산
**답변:**
```
uchars = ['a', 'b', 'c', 'd', 'g', 'o', 't']
         ( 0,   1,   2,   3,   4,   5,   6 )
BOS = 7

"dog" 인코딩:
'd' → uchars.index('d') → 3
'o' → uchars.index('o') → 5
'g' → uchars.index('g') → 4

tokens = [BOS, 3, 5, 4, BOS]
       = [7, 3, 5, 4, 7]
```

**채점 기준:**
- 각 문자 인덱스 (2점)
- BOS 추가 (2점)
- 최종 정답 (1점)

---

### A3.3 디코딩 계산
**답변:**
```
uchars = ['a', 'b', 'c', 'd', 'g', 'o', 't']
BOS = 7

tokens = [6, 1, 2, 6]

6 → uchars[6] → 't'
1 → uchars[1] → 'b'
2 → uchars[2] → 'c'
6 → uchars[6] → 't'

result = "tbct"

(만약 [7, 1, 2, 6, 7]이었다면 → "bc")
```

**채점 기준:**
- 각 인덱스 변환 (3점)
- 최종 결과 (2점)

---

## Section 4: BPE 알고리즘

### A4.1 BPE 정의
**답변:**
```
BPE (Byte Pair Encoding)란?
자주 등장하는 문자 쌍(pair)을 반복적으로 병합(merge)하여,
Character와 Word의 중간 수준인 Subword 단위의 어휘를 만드는 알고리즘.

Character-level과의 차이:
1. 학습 과정:
   - Character: 학습 불필요, 데이터셋의 문자를 그대로 사용
   - BPE: 학습 필요, 통계적으로 자주 등장하는 쌍을 병합

2. 어휘 크기:
   - Character: 매우 작음 (~100)
   - BPE: 중간 크기 (10K~50K)

3. 토큰 단위:
   - Character: 문자 ('h', 'e', 'l', 'l', 'o')
   - BPE: Subword ('hel', 'lo')
```

**채점 기준:**
- BPE 정의 (4점)
- 차이점 3가지 (2점씩)

---

### A4.2 BPE 학습 과정
**답변:**
```
1. 초기화:
   - 모든 문자를 개별 토큰으로 분리
   - 초기 어휘 = 고유 문자 집합
   - 예: "hello" → ['h', 'e', 'l', 'l', 'o']

2. 쌍 빈도 계산:
   - 코퍼스에서 모든 인접한 토큰 쌍의 빈도 계산
   - 예: ('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')

3. 선택:
   - 가장 빈번하게 등장하는 쌍 선택
   - 예: ('l', 'l')이 100번 등장 → 선택

4. 병합 및 반복:
   - 선택된 쌍을 하나의 새 토큰으로 병합
   - 어휘에 새 토큰 추가
   - 코퍼스 업데이트
   - 목표 어휘 크기에 도달할 때까지 2-4 반복
```

**채점 기준:**
- 각 단계 설명 (2.5점씩)

---

### A4.3 BPE 손계산
**답변:**
```
초기 상태:
l o w </w> (5)
n e w e s t </w> (6)

=== 병합 1 ===
쌍 빈도 계산:
- e-s: 6회
- s-t: 6회
- l-o: 5회
- o-w: 5회
- w-</w>: 5+6=11회  ← 최다! (하지만 보통 단어 끝 병합은 나중에)
- e-w: 6회

실제 선택: e-s (6회) - 알파벳 순 우선

병합 후:
l o w </w> (5)
n e w es t </w> (6)

=== 병합 2 ===
쌍 빈도 계산:
- es-t: 6회  ← 최다!
- l-o: 5회
- o-w: 5회

선택한 쌍: es-t

병합 후:
l o w </w> (5)
n e w est </w> (6)

=== 병합 3 ===
쌍 빈도 계산:
- l-o: 5회
- o-w: 5회
- w-</w>: 5+6=11회  ← 최다!

선택한 쌍: w-</w>

병합 후:
l o w</w> (5)
n e w est</w> (6)

최종 어휘: ['l', 'o', 'w', 'n', 'e', 's', 't', '</w>', 'es', 'est', 'w</w>']
```

**채점 기준:**
- 각 병합의 쌍 계산 (3점)
- 올바른 선택 (4점)
- 최종 결과 (3점)

---

## Section 5: 실전 시나리오

### A5.1 토크나이저 선택
**답변:**

**시나리오 1: 교육용 미니 GPT (200줄 이내)**
```
선택: Character-level

이유:
1. 구현 간단: 3줄로 어휘 생성 가능
2. 코드 이해 용이: 학습자가 알고리즘 원리 파악하기 쉬움
3. 의존성 없음: 외부 라이브러리 불필요
4. 200줄 제약: BPE 구현은 수백 줄 필요
```

**시나리오 2: 프로덕션 다국어 번역**
```
선택: Subword (BPE 또는 SentencePiece)

이유:
1. 다국어 지원: Byte-level BPE는 모든 유니코드 처리 가능
2. OOV 해결: 새로운 단어도 subword로 분해하여 처리
3. 효율성: Character보다 짧은 시퀀스, Word보다 작은 어휘
4. 산업 표준: GPT, BERT 등 주요 모델에서 사용
```

**시나리오 3: 영어 단어 자동완성**
```
선택: Word-level

이유:
1. 단어 단위 예측: 자동완성은 완전한 단어가 목표
2. 짧은 시퀀스: 입력이 짧아 OOV 문제 덜함
3. 어휘 제한 가능: 자주 쓰는 단어로 어휘 제한하면 효과적
4. 사용자 경험: Subword 분해는 자동완성에 부자연스러움
```

**채점 기준:**
- 각 시나리오 선택 (1점)
- 이유 2개 이상 (4점)

---

### A5.2 성능 비교
**답변:**
```
1. Character-level: 20개 토큰
   i-n-t-e-r-n-a-t-i-o-n-a-l-i-z-a-t-i-o-n
   (각 문자가 1토큰)

2. BPE (GPT-2 기준): 약 4~6개 토큰
   예상 분할:
   - ['intern', 'ation', 'al', 'ization']
   - ['international', 'ization']
   - ['intern', 'ational', 'ization']

   이유: BPE는 자주 쓰이는 접미사(-ation, -ization)를 학습했을 가능성 높음

3. Word-level:
   토큰 수: 1개 (단어 전체)
   문제점:
   - 어휘에 없으면 OOV (Unknown Token) 처리
   - 긴 단어일수록 어휘에 없을 확률 증가
   - 새로운 합성어 처리 불가
```

**채점 기준:**
- Character-level 정답 (1점)
- BPE 예상 (2점)
- Word-level 문제점 (2점)

---

### A5.3 메모리 계산
**답변:**
```
계산:
vocab_size = 50,000
embedding_dim = 768
dtype = float32 = 4 bytes

메모리 = vocab_size × embedding_dim × dtype
       = 50,000 × 768 × 4 bytes
       = 153,600,000 bytes
       = 153.6 MB
       ≈ 154 MB

결과: 약 154 MB (임베딩 테이블만)
```

**채점 기준:**
- 올바른 공식 (2점)
- 정확한 계산 (2점)
- 단위 변환 (1점)

---

## Section 6: 디버깅 & 최적화

### A6.1 버그 찾기
**답변:**
```
버그 1: BOS 토큰 누락
- 인코딩 결과에 BOS 토큰이 앞뒤로 추가되어야 함
- 현재: [id1, id2, ...]
- 정답: [BOS, id1, id2, ..., BOS]

버그 2: 에러 처리 없음
- 어휘에 없는 문자 입력 시 ValueError 발생
- uchars.index()는 값이 없으면 에러

수정된 코드:
def encode(text, uchars, BOS):
    tokens = [BOS]
    for ch in text:
        if ch in uchars:  # 에러 방지
            tokens.append(uchars.index(ch))
        else:
            # UNK 토큰 처리 또는 무시
            pass
    tokens.append(BOS)
    return tokens
```

**채점 기준:**
- 버그 1 발견 (2점)
- 버그 2 발견 (1.5점)
- 올바른 수정 (1.5점)

---

### A6.2 성능 문제
**답변:**
```
문제점:
1. 중복 검사의 시간 복잡도: O(n)
   - "char not in all_chars"는 리스트 전체 탐색
   - 1억 문서 × 평균 문자 수 × O(n) = 매우 느림

2. 불필요한 반복:
   - 어차피 set()으로 중복 제거할 것이므로 처음부터 set 사용

개선 방법:
- set()을 사용하여 O(1) 멤버십 테스트
- ''.join() + set()으로 한 번에 처리

개선된 코드:
# 방법 1: microgpt 방식 (최적)
uchars = sorted(set(''.join(docs)))

# 방법 2: set 사용
all_chars = set()
for doc in docs:
    all_chars.update(doc)  # set.update()는 빠름
uchars = sorted(all_chars)
```

**채점 기준:**
- 시간 복잡도 문제 지적 (2점)
- 개선 방법 (2점)
- 올바른 코드 (1점)

---

### A6.3 엣지 케이스
**답변:**

**케이스 1: 빈 문자열 ""**
```
기대 동작:
- 인코딩: [BOS, BOS] (BOS만 2개)
- 디코딩: ""  (빈 문자열 반환)
- 에러 발생하지 않아야 함
```

**케이스 2: 어휘에 없는 문자 (😀)**
```
기대 동작:
1. 무시: 해당 문자를 건너뜀
2. UNK 토큰: 특수 Unknown 토큰으로 대체
3. 에러: ValueError 발생 (권장하지 않음)

권장: UNK 토큰 사용
vocab에 UNK 토큰 추가 후, 없는 문자는 UNK로 매핑
```

**케이스 3: 매우 긴 단어 (1000자 이상)**
```
기대 동작:
- Character-level: 1000개 토큰 생성 (가능)
- 문제: 시퀀스 길이 제한 (block_size)에 걸림
  - microgpt: block_size=16이므로 16자까지만 처리
  - 나머지는 잘림 (truncate)

해결책:
- block_size 증가 (메모리/계산량 증가)
- 문서 분할 (chunking)
```

**채점 기준:**
- 각 케이스 분석 (1.5점씩)

---

## Section 7: 심화 질문

### A7.1 GPT-2 vs GPT-3 토크나이저
**답변:**
```
GPT-2 토크나이저:
- 어휘 크기: 50,257
- 방식: Byte-level BPE
- 특수 토큰: <|endoftext|>
- 학습 데이터: 영어 중심

GPT-3 토크나이저:
- 어휘 크기: 50,257 (GPT-2와 동일!)
- 방식: Byte-level BPE (동일)
- 특수 토큰: <|endoftext|> (동일)
- 학습 데이터: 동일

주요 차이점:
실제로 GPT-2와 GPT-3는 **같은 토크나이저**를 사용합니다!
- 이유: 하위 호환성 유지, 재학습 비용 절감
- 차이: 모델 크기와 학습 데이터만 다름

GPT-4의 경우:
- 어휘 크기: ~100,000 (확장됨)
- 다국어 지원 강화
```

**채점 기준:**
- GPT-2 설명 (2점)
- GPT-3 설명 (2점)
- 차이점 또는 동일함 지적 (1점)

---

### A7.2 Byte-level BPE
**답변:**
```
Byte-level BPE의 장점:

1. 완벽한 커버리지:
   - 모든 유니코드 문자 처리 가능
   - 256개 바이트만으로 모든 텍스트 표현
   - 어떤 언어든 OOV 없음

2. 고정된 기본 어휘:
   - Character-level BPE: 각 언어마다 다른 문자 집합
     (영어 26자, 한글 수천자, 중국어 수만자)
   - Byte-level: 항상 256개 바이트로 시작
   - 언어 독립적

3. 효율성:
   - 기본 어휘가 작아서 학습/추론 속도 향상
   - 이모지, 특수 기호 등도 자연스럽게 처리

예시:
- "안녕" (한글):
  Character-level: ['안', '녕'] - 문자 추가 필요
  Byte-level: [EC, 95, 88, EB, 85, 95] - 바이트로 분해, 어휘 확장 불필요
```

**채점 기준:**
- 3가지 장점 (각 1.5점)
- 예시 제시 (0.5점)

---

### A7.3 토크나이저와 모델 성능
**답변:**
```
어휘가 너무 작을 때 (예: 100):
장점:
- 임베딩 테이블 메모리 작음 (100 × 768 × 4 = 300KB)
- 학습 속도 빠름
- 구현 간단

단점:
- 시퀀스 길이 매우 김 (Attention 계산량 O(n²) 증가)
- 문맥 파악 어려움 (문자 단위로는 의미 이해 힘듦)
- 긴 시퀀스로 인한 메모리 부족
- 예: "Hello world" → 11 토큰 (Character-level)

어휘가 너무 클 때 (예: 1,000,000):
장점:
- 시퀀스 길이 짧음 (효율적 Attention)
- 단어 단위 의미 파악 용이
- 예: "Hello world" → 2 토큰 (Word-level)

단점:
- 임베딩 테이블 거대 (1M × 768 × 4 = 3GB)
- OOV 문제 심각 (새 단어 처리 불가)
- 학습 데이터 부족 (희귀 단어는 학습 어려움)
- Overfitting 위험

최적 크기:
- 실무: 30K~50K (GPT-2: 50,257)
- 균형점: 적당한 시퀀스 길이 + 관리 가능한 어휘
- BPE/WordPiece가 이 범위 달성
```

**채점 기준:**
- 작을 때 분석 (2점)
- 클 때 분석 (2점)
- 최적 크기 제시 (1점)

---

## Section 8: Python 지식

### A8.1 `set()` vs `list`
**답변:**
```
1. 중복:
   - set: 중복 불가 (자동 제거)
   - list: 중복 허용

2. 순서:
   - set: 순서 없음 (unordered)
   - list: 순서 유지 (ordered)

3. 인덱싱:
   - set: 불가능 (my_set[0] ❌)
   - list: 가능 (my_list[0] ✅)

4. 멤버십 테스트:
   - set: O(1) - 해시 테이블
   - list: O(n) - 순차 검색

5. 변경 가능성:
   - set: mutable (add, remove 가능)
   - list: mutable (append, remove 가능)

6. 저장 가능 요소:
   - set: immutable 요소만 (int, str, tuple)
   - list: 모든 타입 (list, dict도 가능)

7. 사용 예:
   - set: 고유값 추출, 빠른 검색
   - list: 순서 중요, 중복 허용 필요
```

**채점 기준:**
- 5가지 이상 정확 (5점)

---

### A8.2 `''.join()` 성능
**답변:**
```
+ 연산자:
- 시간 복잡도: O(n²)
- 이유: 문자열은 immutable
  - "a" + "b": 새 문자열 "ab" 생성
  - "ab" + "c": 또 새 문자열 "abc" 생성
  - n번 반복 시: 1+2+3+...+n = O(n²)
- 매번 새 메모리 할당

''.join():
- 시간 복잡도: O(n)
- 이유:
  1. 먼저 전체 길이 계산
  2. 필요한 메모리를 한 번에 할당
  3. 각 요소를 복사하여 결합
  - n번 복사: O(n)
- 메모리 할당 1회

예시:
# + 연산자 (느림)
result = ""
for s in strings:  # 1000개
    result += s  # 매번 새 문자열 생성
# O(1000²) = 1,000,000 연산

# join (빠름)
result = ''.join(strings)
# O(1000) = 1,000 연산
```

**채점 기준:**
- + 연산자 시간 복잡도 (2점)
- join 시간 복잡도 (2점)
- 이유 설명 (1점)

---

### A8.3 리스트 컴프리헨션
**답변:**
```python
# 원본 리스트 컴프리헨션:
tokens = [uchars.index(ch) for ch in text if ch in uchars]

# for 루프로 변환:
tokens = []
for ch in text:
    if ch in uchars:
        tokens.append(uchars.index(ch))

# 더 명시적 버전:
tokens = []
for ch in text:
    if ch in uchars:
        idx = uchars.index(ch)
        tokens.append(idx)
```

**채점 기준:**
- 올바른 변환 (3점)
- if 조건 포함 (2점)

---

## Section 9: 시스템 디자인

### A9.1 대규모 토크나이저 설계
**답변:**
```
1. 데이터 전처리:
   - Streaming 방식: 모든 데이터를 메모리에 올리지 않음
   - 청크 단위 처리 (예: 10만 문서씩)
   - 중간 결과 디스크 저장

2. 병렬 처리:
   - 멀티프로세싱: Python multiprocessing 사용
   - 각 프로세스가 독립적으로 청크 처리
   - Map-Reduce 패러다임:
     * Map: 각 청크에서 문자 추출
     * Reduce: 모든 결과 병합 (set union)
   - 예: 10 프로세스 → 10배 빠름

3. 메모리 관리:
   - 제너레이터 사용: yield로 lazy evaluation
   - 메모리 맵 파일: 큰 파일은 mmap으로 처리
   - 가비지 컬렉션: 사용 후 즉시 del

4. 성능 최적화:
   - C 확장: Cython으로 핵심 루프 최적화
   - 벡터화: NumPy 배열 연산 활용
   - 캐싱: 자주 쓰는 결과 LRU 캐시
   - 프로파일링: cProfile로 병목 찾기

코드 예시:
from multiprocessing import Pool
import itertools

def process_chunk(chunk):
    return set(''.join(chunk))

# 병렬 처리
with Pool(10) as pool:
    chunks = [docs[i:i+100000] for i in range(0, len(docs), 100000)]
    results = pool.map(process_chunk, chunks)

# 병합
all_chars = set().union(*results)
uchars = sorted(all_chars)
```

**채점 기준:**
- 각 항목 구체적 방안 (3점씩)
- 코드 예시 (3점)

---

### A9.2 다국어 지원
**답변:**
```
접근 방법:
1. Byte-level BPE 선택
   - 이유: 모든 유니코드 문자 자동 처리
   - 언어별 문자 집합 관리 불필요

2. 다국어 코퍼스 준비
   - 각 언어별 균형잡힌 데이터 수집
   - 예: 영어 40%, 한글 20%, 일본어 20%, 중국어 20%
   - 비율 조정으로 언어별 성능 제어

기술 선택:
1. SentencePiece 라이브러리
   - Google의 다국어 토크나이저
   - 언어 독립적 (공백도 토큰화)
   - 예:
     from sentencepiece import SentencePieceTrainer
     SentencePieceTrainer.train(
         input='multilingual_corpus.txt',
         model_prefix='m',
         vocab_size=50000,
         character_coverage=0.9995  # 다국어 커버리지
     )

2. 또는 Hugging Face Tokenizers
   - 빠른 Rust 구현
   - 다양한 알고리즘 지원

주의사항:
1. 문자 커버리지:
   - 한자 등 문자 수가 많은 언어는 character_coverage 높게
   - 예: 일본어/중국어 → 0.9995 이상

2. 정규화:
   - NFKC 정규화로 유니코드 변형 통일
   - 예: "é" vs "e+́" → 하나로 통일

3. 언어별 특수 처리:
   - 한국어: 조사 분리 여부 결정
   - 중국어: 간체/번체 통합 여부
   - 일본어: 히라가나/가타카나/한자 처리

4. 성능 검증:
   - 각 언어별로 tokenize/detokenize 테스트
   - 희귀 문자 처리 확인
```

**채점 기준:**
- 접근 방법 (2점)
- 기술 선택 (2점)
- 주의사항 (1점)

---

## 🎯 점수 환산표

### 섹션별 만점
- Section 1: 15점
- Section 2: 15점
- Section 3: 15점
- Section 4: 30점
- Section 5: 15점
- Section 6: 15점
- Section 7: 15점
- Section 8: 15점
- Section 9: 15점
**총 150점**

### 등급 기준 (재확인)
- **140-150점 (93%+)**: Senior - 면접관에게 설명 가능
- **120-139점 (80-93%)**: Mid - 실무 투입 가능
- **100-119점 (67-79%)**: Junior - 기본기 충분
- **80-99점 (53-66%)**: Entry - 추가 학습 필요
- **80점 미만**: 복습 필수

---

## 📚 추가 학습 방향

### 80점 미만
1. [notes.md](../notes.md) 전체 재학습
2. Python 기초 복습 (`set`, `join`, 리스트 컴프리헨션)
3. 손으로 계산 반복 연습

### 80-119점
1. [deepdive_BPE.md](../deepdive_BPE.md) 정독
2. BPE 알고리즘 손계산 반복
3. Section 3, 4 복습 (손계산 집중)

### 120점 이상
1. GPT-2 토크나이저 코드 읽기
2. SentencePiece 라이브러리 실습
3. 다음 단원으로 진행

---

**다음:** [02. Embedding 테스트](../../02_embedding/test/test.md)

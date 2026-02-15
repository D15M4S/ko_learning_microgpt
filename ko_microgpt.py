"""
순수 Python만으로 GPT를 학습하고 추론하는 가장 원자적인 방법.
이 파일이 완전한 알고리즘입니다.
나머지는 모두 효율성을 위한 것일 뿐입니다.

@karpathy
"""

# =============================================================================
# [섹션 1] 라이브러리 임포트 및 초기 설정
# =============================================================================
# 설명: 필요한 최소한의 표준 라이브러리만 사용합니다.
#       외부 의존성(PyTorch, NumPy 등) 없이 GPT의 핵심 원리를 구현합니다.
# =============================================================================

import os       # os.path.exists: 파일 존재 여부 확인
import math     # math.log, math.exp: 수학 연산 (로그, 지수)
import random   # random.seed, random.choices, random.gauss, random.shuffle: 난수 생성

random.seed(42) # 재현 가능한 실험을 위한 시드 고정 (혼돈 속의 질서)


# =============================================================================
# [섹션 2] 데이터셋 로드 및 준비
# =============================================================================
# 설명: 문서(docs) 리스트를 준비합니다.
#       - 예제: 이름 데이터셋 (names.txt)
#       - 각 문서는 문자열(str)이며, 모델은 이를 학습하여 비슷한 패턴을 생성합니다.
# =============================================================================

# input.txt가 없으면 자동으로 다운로드
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# 문서 리스트 생성: 각 줄을 하나의 문서로 취급
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)  # 학습 데이터 순서 섞기 (일반화 성능 향상)
print(f"num docs: {len(docs)}")


# =============================================================================
# [섹션 3] 토크나이저 (Tokenizer)
# =============================================================================
# 설명: 문자열을 이산적인 심볼(토큰)로 변환하고, 그 반대도 가능하게 합니다.
#       - Character-level tokenizer: 각 문자가 하나의 토큰
#       - BOS (Beginning of Sequence): 시퀀스의 시작/끝을 나타내는 특수 토큰
# 핵심 개념:
#   - vocab_size: 전체 토큰 종류의 개수
#   - uchars: 데이터셋에 등장하는 고유 문자들 (정렬된 상태)
# =============================================================================

uchars = sorted(set(''.join(docs)))  # 데이터셋의 고유 문자들 → 토큰 ID 0..n-1
BOS = len(uchars)  # 특수 토큰 BOS의 ID (시퀀스 시작/끝 표시)
vocab_size = len(uchars) + 1  # 총 토큰 개수 (+1은 BOS 토큰)
print(f"vocab size: {vocab_size}")


# =============================================================================
# [섹션 4] Autograd - 자동 미분 시스템
# =============================================================================
# 설명: 연산 그래프를 통해 역전파를 자동으로 수행하는 Value 클래스.
#       PyTorch의 Tensor, TensorFlow의 Variable과 유사한 역할입니다.
# 핵심 개념:
#   - data: 순전파(forward pass)에서 계산된 스칼라 값
#   - grad: 역전파(backward pass)에서 계산된 이 노드에 대한 손실의 미분값
#   - _children: 연산 그래프에서 이 노드의 자식들
#   - _local_grads: 자식에 대한 이 노드의 로컬 미분값 (체인 룰 적용 시 사용)
#
# 작동 원리:
#   1. 순전파: 연산을 수행하며 계산 그래프 생성
#   2. 역전파: loss.backward() 호출 시 체인 룰을 재귀적으로 적용
#   3. 각 파라미터의 grad에 기울기 누적
# =============================================================================

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # 메모리 최적화

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 순전파에서 계산된 스칼라 값
        self.grad = 0                   # 역전파에서 계산될 손실에 대한 미분값
        self._children = children       # 연산 그래프의 자식 노드들
        self._local_grads = local_grads # 자식에 대한 로컬 기울기

    def __add__(self, other):
        # 덧셈: d(a+b)/da = 1, d(a+b)/db = 1
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        # 곱셈: d(a*b)/da = b, d(a*b)/db = a
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        # 거듭제곱: d(a^n)/da = n * a^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def log(self):
        # 로그: d(log(a))/da = 1/a
        return Value(math.log(self.data), (self,), (1/self.data,))

    def exp(self):
        # 지수: d(e^a)/da = e^a
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        # ReLU: d(ReLU(a))/da = 1 if a > 0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    # 편의 연산자들
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """
        역전파 수행: 위상 정렬(topological sort)을 통해 연산 그래프를 역순으로 순회하며
        체인 룰(chain rule)을 적용하여 모든 노드의 기울기를 계산합니다.

        과정:
        1. build_topo: 연산 그래프의 위상 정렬 순서 생성
        2. self.grad = 1: 손실 노드 자기 자신에 대한 미분은 1
        3. 역순 순회: 각 노드에서 자식들에게 기울기 전파
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1  # 손실의 손실에 대한 미분 = 1

        # 위상 정렬의 역순으로 순회하며 기울기 전파
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad  # 체인 룰 적용


# =============================================================================
# [섹션 5] 모델 파라미터 초기화
# =============================================================================
# 설명: 모델의 지식을 저장할 파라미터들을 초기화합니다.
#       - state_dict: PyTorch와 유사한 파라미터 딕셔너리
#       - 각 파라미터는 가우시안 분포(평균 0, 표준편차 0.08)로 초기화
#
# 하이퍼파라미터:
#   - n_embd: 임베딩 차원 (각 토큰을 표현하는 벡터의 크기)
#   - n_head: 어텐션 헤드 개수 (Multi-head Attention)
#   - n_layer: 트랜스포머 레이어 개수
#   - block_size: 최대 시퀀스 길이 (컨텍스트 윈도우)
#   - head_dim: 각 헤드의 차원 (n_embd를 n_head로 나눈 값)
# =============================================================================

n_embd = 16     # 임베딩 차원
n_head = 4      # 어텐션 헤드 개수
n_layer = 1     # 레이어 개수
block_size = 16 # 최대 시퀀스 길이
head_dim = n_embd // n_head  # 각 헤드의 차원

# 행렬 초기화 헬퍼 함수: nout x nin 크기의 행렬 생성
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# 파라미터 딕셔너리 초기화
state_dict = {
    'wte': matrix(vocab_size, n_embd),  # Token Embedding: 토큰 → 벡터
    'wpe': matrix(block_size, n_embd),  # Position Embedding: 위치 → 벡터
    'lm_head': matrix(vocab_size, n_embd)  # Language Model Head: 벡터 → 다음 토큰 확률
}

# 각 레이어별 파라미터 생성
for i in range(n_layer):
    # Multi-head Attention 파라미터
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)  # Query 프로젝션
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)  # Key 프로젝션
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)  # Value 프로젝션
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)  # Output 프로젝션

    # MLP (Feed-Forward Network) 파라미터
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)  # 첫 번째 레이어 (확장)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)  # 두 번째 레이어 (축소)

# 모든 파라미터를 1차원 리스트로 평탄화 (optimizer 업데이트 용이)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")


# =============================================================================
# [섹션 6] 모델 아키텍처 정의
# =============================================================================
# 설명: GPT-2 기반 아키텍처 (일부 차이점 존재)
#       - LayerNorm → RMSNorm
#       - Bias 제거
#       - GeLU → ReLU
#
# 주요 함수:
#   - linear: 선형 변환 (행렬-벡터 곱)
#   - softmax: 소프트맥스 함수 (확률 분포 생성)
#   - rmsnorm: RMS Normalization (평균 없는 정규화)
#   - gpt: 메인 모델 함수 (토큰 → 다음 토큰 로짓)
# =============================================================================

def linear(x, w):
    """
    선형 변환: y = W @ x

    Args:
        x: 입력 벡터 [n_embd]
        w: 가중치 행렬 [nout, nin]

    Returns:
        출력 벡터 [nout]
    """
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    """
    소프트맥스 함수: 로짓을 확률 분포로 변환

    수치 안정성을 위해 max 값을 빼줍니다 (exp 오버플로우 방지)

    Args:
        logits: 로짓 벡터 [vocab_size]

    Returns:
        확률 분포 [vocab_size] (합이 1)
    """
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """
    RMS Normalization: Root Mean Square를 이용한 정규화

    LayerNorm과 달리 평균을 빼지 않고 RMS로만 나눕니다.
    계산이 더 간단하면서도 효과적입니다.

    공식: x / sqrt(mean(x^2) + eps)

    Args:
        x: 입력 벡터 [n_embd]

    Returns:
        정규화된 벡터 [n_embd]
    """
    ms = sum(xi * xi for xi in x) / len(x)  # mean of squares
    scale = (ms + 1e-5) ** -0.5  # 1 / sqrt(ms + eps)
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    """
    GPT 모델의 순전파 함수

    데이터 흐름:
    1. 토큰 임베딩 + 위치 임베딩
    2. 정규화 (RMSNorm)
    3. 레이어별 처리:
       a. Multi-head Attention (자기 주의)
       b. 잔차 연결 (Residual Connection)
       c. MLP (Feed-Forward)
       d. 잔차 연결
    4. 최종 선형 변환 (임베딩 → 로짓)

    Args:
        token_id: 현재 토큰 ID
        pos_id: 현재 위치 ID
        keys: 각 레이어의 과거 Key 벡터들 (KV 캐시)
        values: 각 레이어의 과거 Value 벡터들 (KV 캐시)

    Returns:
        logits: 다음 토큰에 대한 로짓 벡터 [vocab_size]
    """
    # === 1. 임베딩 ===
    tok_emb = state_dict['wte'][token_id]  # 토큰 임베딩
    pos_emb = state_dict['wpe'][pos_id]    # 위치 임베딩
    x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 토큰 + 위치 임베딩 합산
    x = rmsnorm(x)  # 입력 정규화

    # === 2. 트랜스포머 레이어 ===
    for li in range(n_layer):
        # --- (A) Multi-head Attention 블록 ---
        x_residual = x  # 잔차 연결을 위해 입력 저장
        x = rmsnorm(x)  # Pre-normalization

        # Query, Key, Value 생성
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # KV 캐시에 현재 Key, Value 추가 (다음 토큰 생성 시 사용)
        keys[li].append(k)
        values[li].append(v)

        # 각 헤드별로 어텐션 계산
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim  # 헤드 시작 인덱스

            # 현재 헤드의 Q, K, V 추출
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]  # 과거 + 현재 모든 Key
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]  # 과거 + 현재 모든 Value

            # 어텐션 스코어 계산: Q @ K^T / sqrt(d_k)
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            # 어텐션 가중치 계산 (softmax)
            attn_weights = softmax(attn_logits)

            # 가중합: Σ(attention_weight * Value)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]

            x_attn.extend(head_out)  # 모든 헤드 결과 연결

        # Output 프로젝션 및 잔차 연결
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        # --- (B) MLP 블록 ---
        x_residual = x  # 잔차 연결을 위해 입력 저장
        x = rmsnorm(x)  # Pre-normalization

        # Feed-Forward Network: W2 @ ReLU(W1 @ x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 확장 (n_embd → 4*n_embd)
        x = [xi.relu() for xi in x]  # 비선형 활성화
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 축소 (4*n_embd → n_embd)

        # 잔차 연결
        x = [a + b for a, b in zip(x, x_residual)]

    # === 3. 최종 출력 레이어 ===
    logits = linear(x, state_dict['lm_head'])  # 임베딩 → 다음 토큰 로짓
    return logits


# =============================================================================
# [섹션 7] Adam 옵티마이저 초기화
# =============================================================================
# 설명: Adam (Adaptive Moment Estimation) 옵티마이저
#       - 1차 모멘트 (m): 기울기의 지수이동평균
#       - 2차 모멘트 (v): 기울기 제곱의 지수이동평균
#       - Bias correction을 통해 초기 학습 안정화
# =============================================================================

learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)  # 1차 모멘트 버퍼 (기울기 평균)
v = [0.0] * len(params)  # 2차 모멘트 버퍼 (기울기 분산)


# =============================================================================
# [섹션 8] 학습 루프
# =============================================================================
# 설명: 반복적으로 문서를 샘플링하고, 순전파 → 역전파 → 파라미터 업데이트를 수행
#
# 과정:
#   1. 문서 샘플링 및 토큰화
#   2. 순전파: 각 위치에서 다음 토큰 예측
#   3. 손실 계산: Cross-Entropy Loss
#   4. 역전파: 모든 파라미터의 기울기 계산
#   5. Adam 업데이트: 파라미터 갱신
# =============================================================================

num_steps = 1000  # 학습 스텝 수

for step in range(num_steps):

    # --- 1. 데이터 준비 ---
    doc = docs[step % len(docs)]  # 문서 샘플링 (순환)
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]  # 토큰화 + BOS 추가
    n = min(block_size, len(tokens) - 1)  # 실제 처리할 시퀀스 길이

    # --- 2. 순전파 (Forward Pass) ---
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]  # KV 캐시 초기화
    losses = []

    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]  # 현재 토큰, 다음 토큰

        # 모델 순전파: 현재 토큰 → 다음 토큰 로짓
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)

        # Cross-Entropy Loss: -log(p(정답 토큰))
        loss_t = -probs[target_id].log()
        losses.append(loss_t)

    # 전체 시퀀스에 대한 평균 손실
    loss = (1 / n) * sum(losses)

    # --- 3. 역전파 (Backward Pass) ---
    loss.backward()  # 모든 파라미터의 기울기 계산

    # --- 4. Adam 옵티마이저 업데이트 ---
    lr_t = learning_rate * (1 - step / num_steps)  # 선형 학습률 감소

    for i, p in enumerate(params):
        # 1차, 2차 모멘트 업데이트
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2

        # Bias correction (초기 스텝에서 모멘트 보정)
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))

        # 파라미터 업데이트
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)

        # 기울기 초기화 (다음 스텝을 위해)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")


# =============================================================================
# [섹션 9] 추론 (Inference)
# =============================================================================
# 설명: 학습된 모델로 새로운 샘플 생성
#
# 과정:
#   1. BOS 토큰으로 시작
#   2. 반복적으로 다음 토큰 예측 및 샘플링
#   3. BOS 토큰이 나오거나 block_size 도달 시 종료
#
# Temperature:
#   - 낮을수록 (0에 가까울수록): 결정론적, 안전한 출력
#   - 높을수록 (1에 가까울수록): 창의적, 다양한 출력
# =============================================================================

temperature = 0.5  # (0, 1] 범위, 생성 텍스트의 "창의성" 조절

print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    # KV 캐시 초기화
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS  # BOS 토큰으로 시작
    sample = []

    # 자동회귀적 생성 (Autoregressive Generation)
    for pos_id in range(block_size):
        # 다음 토큰 로짓 예측
        logits = gpt(token_id, pos_id, keys, values)

        # Temperature scaling (확률 분포 조절)
        probs = softmax([l / temperature for l in logits])

        # 확률에 따라 다음 토큰 샘플링
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]

        # BOS 토큰 생성 시 종료 (시퀀스 끝)
        if token_id == BOS:
            break

        sample.append(uchars[token_id])

    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

# BC7 Mode 7 및 공통 Subset 처리

## 1. 구현한 범위

이번 작업에서 구현한 내용은 다음과 같다.

1. BC7의 1/2/3-subset 공간 배치를 공통 구조로 표현
2. 모든 Mode 0~7의 subset 수와 partition bit 위치 정의
3. 공통 SIMD partition 추출기 구현
4. Mode 7 endpoint와 P-bit 복원
5. Mode 1/7의 partitioned index 추출 공통화
6. Mode 1/7의 2-subset 사분면 평균 계산 공통화
7. Mode 7을 부모 블록 dispatcher에 연결
8. 입력 검증에서 Mode 7 허용

## 2. 공통 Subset 공간 구조

BC7 mode마다 subset 수와 partition 정보가 다르다.

| Mode | Subset 수 | Partition 시작 bit | Partition bit 수 | Shape 수 |
|---:|---:|---:|---:|---:|
| 0 | 3 | 1 | 4 | 16 |
| 1 | 2 | 2 | 6 | 64 |
| 2 | 3 | 3 | 6 | 64 |
| 3 | 2 | 4 | 6 | 64 |
| 4 | 1 | 없음 | 0 | 1 |
| 5 | 1 | 없음 | 0 | 1 |
| 6 | 1 | 없음 | 0 | 1 |
| 7 | 2 | 8 | 6 | 64 |

이 차이를 다음 공통 메타데이터로 표현했다.

```cpp
struct BC7ModeSpatialInfo
{
    uint8_t subsetCount;
    uint8_t partitionBitOffset;
    uint8_t partitionBitCount;
};
```

실제 texel의 공간 배치는 다음 구조로 표현한다.

```cpp
struct BC7PartitionLayout
{
    const uint8_t* subsetByTexel;
    std::array<uint8_t, 3> anchorTexel;
    size_t subsetCount;
};
```

- `subsetByTexel[t]`는 texel `t`가 속한 subset 번호이다.
- `anchorTexel[s]`는 subset `s`의 anchor texel 번호이다.
- `subsetCount`는 현재 mode가 사용하는 subset 수이다.

1-subset mode는 모든 texel을 subset 0으로 지정한다. 2-subset과 3-subset mode는 `BC6HBC7.cpp`의 partition 및 fixup lookup table을 사용한다.

예를 들어 다음 2-subset 배치는:

```text
0 0 1 1
0 0 1 1
0 0 1 1
0 0 1 1
```

다음 배열로 표현된다.

```text
{ 0,0,1,1, 0,0,1,1, 0,0,1,1, 0,0,1,1 }
```

## 3. 공통 Partition 추출

Mode를 `m`, 첫 번째 32-bit word를 `W`, partition 시작 bit를 `o_m`, partition bit 수를 `b_m`이라고 하자.

Partition 번호는 다음 식으로 구한다.

\[
P(W,m)=\left(W \gg o_m\right)\mathbin{\&}\left(2^{b_m}-1\right)
\]

Mode 4, 5, 6은 partition field가 없으므로 partition 0을 반환한다.

`ExtractBC7Partition()`은 위 계산을 네 SIMD lane에 동시에 적용한다. `GetBC7ModePartitionLayout()`은 추출된 partition과 mode를 받아 공통 `BC7PartitionLayout`을 반환한다.

## 4. Mode 7 Endpoint 복원

Mode 7은 다음 endpoint 형식을 사용한다.

- RGBA channel
- 2 subsets
- subset마다 endpoint 2개
- channel마다 5-bit 값
- endpoint마다 고유한 P-bit 1개

공통 endpoint 배열의 순서는 다음과 같다.

```cpp
value[subset][endpoint][channel]
```

Channel 번호는 `R=0`, `G=1`, `B=2`, `A=3`이다.

Mode 7의 관련 bit 배치는 다음과 같다.

| 값 | Bit 범위 |
|---|---:|
| Partition | 8–13 |
| R endpoint 4개 | 14–33 |
| G endpoint 4개 | 34–53 |
| B endpoint 4개 | 54–73 |
| A endpoint 4개 | 74–93 |
| Endpoint P-bit 4개 | 94–97 |
| Indices | 98–127 |

저장된 5-bit 값을 `v_5`, P-bit를 `p`라고 하면 먼저 6-bit 값을 만든다.

\[
v_6=(v_5\ll1)\mathbin{|}p
\]

그다음 Direct3D bit replication 규칙으로 8-bit 값까지 확장한다.

\[
v_8=(v_6\ll2)\mathbin{|}(v_6\gg4)
\]

`UnquantizeBC7_5Bit()`가 이 계산을 수행한다. `ExtractBC7Mode7Endpoints()`는 네 endpoint의 RGBA 값을 복원하여 `BC7TwoSubsetEndpointBatch`에 저장한다.

## 5. 공통 Index 표현과 Anchor 처리

Mode 1과 Mode 7의 index는 다음 공통 구조를 사용한다.

```cpp
struct BC7IndexBatch
{
    XMVECTOR indices[16];
};
```

각 `XMVECTOR` lane은 서로 다른 BC7 블록의 같은 texel index를 가진다.

BC7은 각 subset의 anchor texel에서 index의 최상위 bit를 저장하지 않는다. 생략된 bit는 0으로 복원한다.

Index precision을 `I`, texel을 `t`라고 하면 읽어야 하는 bit 수는 다음과 같다.

\[
B(t)=
\begin{cases}
I-1,&t=anchor(subset(t))\\
I,&\text{그 외}
\end{cases}
\]

`ExtractBC7PartitionedIndices()`는 다음 순서로 index를 읽는다.

1. `subsetByTexel[t]`에서 texel의 subset을 찾는다.
2. `anchorTexel[subset]`에서 해당 subset의 anchor를 찾는다.
3. Anchor이면 `I-1` bit, 아니면 `I` bit를 읽는다.
4. 생략된 최상위 bit는 0으로 유지한다.

현재 Mode 1과 Mode 7은 이 함수를 함께 사용한다.

| Mode | Index 시작 bit | Index precision | Anchor 수 |
|---:|---:|---:|---:|
| 1 | 82 | 3-bit | 2 |
| 7 | 98 | 2-bit | 2 |

Mode 7에서는 전체 index가 차지하는 bit 수가 다음과 같이 30-bit가 된다.

\[
16\times2-2=30\text{ bits}
\]

두 subset의 anchor에서 각각 1-bit씩 생략되기 때문이다.

## 6. Mode 7 Symbol 구조

Mode 7에서 추출한 값은 다음 구조에 모인다.

```cpp
struct BC7Mode7SymbolBatch
{
    XMVECTOR activeMask;
    XMVECTOR partition;
    BC7TwoSubsetEndpointBatch endpoints;
    BC7IndexBatch indices;
};
```

- `activeMask`: Mode 7인 SIMD lane
- `partition`: 각 lane의 partition 번호
- `endpoints`: RGBA8로 복원된 두 subset의 endpoint
- `indices`: anchor bit가 복원된 16개 index

`ExtractBC7Mode7Symbols()`가 압축된 `BC7BlockBatch`를 이 구조로 변환한다.

## 7. BC7 Palette 복원

두 endpoint를 `e_0`, `e_1`, index에 해당하는 정수 weight를 `w_i`라고 하면 palette 값은 다음과 같다.

\[
c_i=\left\lfloor
\frac{(64-w_i)e_0+w_i e_1+32}{64}
\right\rfloor
\]

Mode 1의 3-bit weight는 다음과 같다.

```text
{ 0, 9, 18, 27, 37, 46, 55, 64 }
```

Mode 7의 2-bit weight는 다음과 같다.

```text
{ 0, 21, 43, 64 }
```

`LookupBC7Weight<IndexBits>()`가 index를 실제 BC7 weight로 바꾼다. `InterpolateBC7PaletteVector()`는 정수 반올림을 포함한 palette 복원식을 적용한다.

## 8. Mode 1/7 공통 사분면 평균

Mode 1과 Mode 7은 endpoint 정밀도와 index precision이 다르지만, endpoint와 index를 공통 구조로 복원한 이후의 계산은 같다.

두 mode는 다음 공통 함수를 사용한다.

```cpp
ComputeBC7TwoSubsetQuadrantMeans<IndexBits>()
```

각 texel `t`에 대해 먼저 subset과 weight를 찾는다.

\[
s_t=subsetByTexel[t]
\]

\[
w_t=WeightTable[index_t]
\]

그다음 해당 subset의 두 endpoint로 RGBA8 texel을 복원한다.

\[
C_t=BC7Interpolate(E_{s_t,0},E_{s_t,1},w_t)
\]

4×4 블록의 네 사분면은 다음 texel로 구성된다.

```text
Q0 = { 0,  1,  4,  5 }
Q1 = { 2,  3,  6,  7 }
Q2 = { 8,  9, 12, 13 }
Q3 = {10, 11, 14, 15 }
```

사분면 `q`의 channel `c` 평균은 다음과 같다.

\[
Q_{q,c}=\frac{1}{4\times255}\sum_{t\in q}C_{t,c}
\]

결과는 RGBA `[0,1]` 범위의 `BC7QuadrantMeanBatch`에 저장된다.

Mode 1은 3-bit weight 버전을 사용한다.

```cpp
ComputeBC7TwoSubsetQuadrantMeans<3>(...)
```

Mode 7은 2-bit weight 버전을 사용한다.

```cpp
ComputeBC7TwoSubsetQuadrantMeans<2>(...)
```

## 9. Mode 7 Dispatcher 연결

`ComputeBC7ParentQuadrantMeans()`에 Mode 7 분기를 추가했다.

처리 순서는 다음과 같다.

```text
Mode 7 mask 생성
    → partition 추출
    → endpoint 복원
    → index 복원
    → 사분면 평균 계산
    → 공통 BC7QuadrantMeanBatch에 병합
```

Mode별 `activeMask`는 서로 겹치지 않는다. 따라서 Mode 1, Mode 6, Mode 7의 결과를 bitwise OR로 병합할 수 있다.

```text
Q = Q1 | Q6 | Q7
```

이 OR 연산은 실수의 수치 덧셈이 아니다. 각 SIMD lane에서 해당 mode의 결과만 0이 아니므로 올바른 lane 값을 선택하는 역할을 한다.

## 10. 입력 검증 변경

`IsSupportedBC7Image()`가 허용하는 부모 mode를 다음과 같이 확장했다.

```text
기존: Mode 1, Mode 6, Mode 7
1단계: Mode 0, Mode 1, Mode 2, Mode 3, Mode 6, Mode 7
2단계(완료): Mode 0, 1, 2, 3, 4, 5, 6, 7 (BC7 8대 전 모드 100% 지원)
```

## 11. 다중 서브셋 모드(Mode 0, 2, 3) 통합 구조

3-subset 및 2-subset 분할 모드를 효율적으로 지원하기 위해 endpoint 및 사분면 평균 계산기를 다중 서브셋 구조로 일반화했다.

### 11.1 다중 서브셋 Endpoint 구조 (`BC7MultiSubsetEndpointBatch`)

최대 3개 서브셋(Mode 0, 2)까지 보관할 수 있도록 endpoint 버퍼를 확장했다.

```cpp
struct BC7MultiSubsetEndpointBatch
{
    XMVECTOR value[3][2][4]; // [subsetIndex 0..2][endpoint 0..1][channel 0..3]
};
using BC7TwoSubsetEndpointBatch = BC7MultiSubsetEndpointBatch;
```

### 11.2 모드별 비트 레이아웃 및 역양자화(Unquantization)

| Mode | Subsets | 색상 비트 | P-bits | Index 비트 / 시작 비트 | 역양자화(Unquantize) 공식 |
|---:|---:|---:|---:|---:|:---|
| **Mode 0** | 3 | RGB 444 | 6개 (EP당 1개) | 3-bit / 83 | $v_5=(v_4\ll1)\mid p$, $v_8=(v_5\ll3)\mid(v_5\gg2)$ |
| **Mode 1** | 2 | RGB 666 | 2개 (Subset당 1개) | 3-bit / 82 | $v_7=(v_6\ll1)\mid p$, $v_8=(v_7\ll1)\mid(v_7\gg6)$ |
| **Mode 2** | 3 | RGB 555 | 0개 | 2-bit / 99 | $v_8=(v_5\ll3)\mid(v_5\gg2)$ |
| **Mode 3** | 2 | RGB 777 | 4개 (EP당 1개) | 2-bit / 98 | $v_8=(v_7\ll1)\mid p$ |
| **Mode 7** | 2 | RGBA 5555 | 4개 (EP당 1개) | 2-bit / 98 | $v_6=(v_5\ll1)\mid p$, $v_8=(v_6\ll2)\mid(v_6\gg4)$ |

### 11.3 일반화된 다중 서브셋 사분면 평균 (`ComputeBC7MultiSubsetQuadrantMeans`)

서브셋 인덱스 $s_t \in \{0, 1, 2\}$에 따라 SIMD 레지스터에서 endpoint를 동적으로 선택하고 하드웨어 weight 테이블로 색상을 복원한다.

```text
ep0 = Select(subset0, subset1, isSubset1)
ep0 = Select(ep0,     subset2, isSubset2)
ep1 = Select(subset0, subset1, isSubset1)
ep1 = Select(ep1,     subset2, isSubset2)
```

## 12. 1-Subset 듀얼 인덱스 및 채널 로테이션 모드(Mode 4, Mode 5) 통합 구조

Mode 4와 Mode 5는 1개의 서브셋을 가지며, 색상과 알파(또는 스왑된 채널)에 대해 서로 다른 독립된 인덱스 스트림을 적용한다.

### 12.1 모드별 비트 레이아웃 및 정밀도

| Mode | 회전 비트 | 인덱스 모드 | 색상 정밀도 | 알파 정밀도 | 인덱스 1 (시작/비트) | 인덱스 2 (시작/비트) |
|---:|---:|---:|:---|:---|:---|:---|
| **Mode 4** | 2-bit (5..6) | 1-bit (7) | RGB 555 | A 6 | 50 / 2-bit (31b) | 81 / 3-bit (47b) |
| **Mode 5** | 2-bit (6..7) | 없음 | RGB 777 | A 8 | 66 / 2-bit (31b) | 97 / 2-bit (31b) |

- **역양자화 공식**:
  - Mode 4 A (6-bit): $v_8 = (v_6 \ll 2) \mid (v_6 \gg 4)$
  - Mode 5 RGB (7-bit): $v_8 = (v_7 \ll 1) \mid (v_7 \gg 6)$
  - Mode 5 A (8-bit): 그대로 사용

### 12.2 인덱스 셀렉터(`IndexMode`) 해석 (Mode 4)
- `IndexMode == 0`: 색상 가중치 = 2-bit(`idx1`), 알파 가중치 = 3-bit(`idx2`)
- `IndexMode == 1`: 색상 가중치 = 3-bit(`idx2`), 알파 가중치 = 2-bit(`idx1`)

### 12.3 SIMD 채널 로테이션(Channel Rotation) 벡터화
보간된 $(R, G, B, A)$ 채널에 대해 회전 필드 $R \in \{0, 1, 2, 3\}$에 따라 알파 채널과 주 채널을 조건부 교환한다:
- $R=0$: 회전 없음 $(R, G, B, A)$
- $R=1$: $R \leftrightarrow A$ 스왑 $(A, G, B, R)$
- $R=2$: $G \leftrightarrow A$ 스왑 $(R, A, B, G)$
- $R=3$: $B \leftrightarrow A$ 스왑 $(R, G, A, B)$

이 로직은 분기문 없이 `XMVectorSelect`를 통해 4개의 SIMD 레인에서 100% 벡터화되어 동작한다.

### 12.4 최종 부모 블록 디스패처 병합 (`ComputeBC7ParentQuadrantMeans`)

8개 모드의 `activeMask`는 서로 배타적이므로, SIMD lane별 최종 사분면 평균은 다음 8모드 bitwise OR로 병합된다:

\[
Q = Q_0 \mid Q_1 \mid Q_2 \mid Q_3 \mid Q_4 \mid Q_5 \mid Q_6 \mid Q_7
\]

## 13. FasTC 해밍 거리 기반 2-서브셋 파티션 추정 (`EstimateBC7Partition2Subsets`)

2-서브셋 모드(Mode 1)의 64개 파티션을 전수조사하는 대신, FasTC 논문(Krajcevski et al., 2013)의 해밍 거리 투영 기법을 채택하여 **임의의 Threshold 없이** 단 하나의 최적 파티션 $S^*$를 $O(1)$로 결정한다.

### 13.1 16비트 투영 마스크 및 최소 해밍 거리 ($\arg\min$)

1. **주축 투영**:
   각 자식 텍셀 $C_t$를 캔버스의 PCA 주축 $V$에 투영한다:
   \[
   p_t = (C_t - \mu) \cdot V
   \]
2. **16비트 바이너리 마스크 생성**:
   중심값(0)을 기준으로 16비트 마스크 $M_{\text{proj}}$를 구성한다:
   \[
   M_{\text{proj}} = \sum_{t=0}^{15} \left( (p_t \ge 0) \ll t \right)
   \]
3. **해밍 거리 최소화**:
   사전 계산된 64개 파티션 마스크 $M_s$와 XOR 후 하드웨어 팝카운트(`_mm_popcnt_u32`)를 수행한다. 서브셋 극성(Polarity) 불변성을 위해 반전 마스크와의 거리도 함께 계산한다:
   \[
   d(s) = \min\Big(\text{popcnt}(M_{\text{proj}} \oplus M_s), \; \text{popcnt}((\sim M_{\text{proj}}) \oplus M_s)\Big)
   \]
   \[
   S^* = \arg\min_{s \in [0, 63]} d(s)
   \]
   임의의 매직 넘버나 Threshold 없이 순수 $\arg\min$ 최소 거리로 최적 파티션을 확정한다.

### 13.2 3-서브셋 파티션 추정기 (`EstimateBC7Partition3Subsets`)

Mode 0 (16개 셰이프) 및 Mode 2 (64개 셰이프)를 위해, 주축 투영 값을 3분위(Tertile)로 분할하여 최적 셰이프를 결정한다:
1. 투영 범위 $\Delta = p_{\max} - p_{\min}$를 3등분하여 각 텍셀에 이상적인 레이블 $L_t \in \{0, 1, 2\}$ 할당.
2. 텍셀 0은 하드웨어상 항상 서브셋 0이므로, $L_0 = 0$이 되도록 레이블 정규화.
3. 서브셋 1과 2의 치환(Permutation) 불변성을 고려하여 64개 셰이프와의 최대 일치도($\arg\max$) 계산:
   \[
   \text{match}(s) = \max\Big(\sum [g\_bc7PartitionTable3Subsets[s][t] == L_t], \; \sum [g\_bc7PartitionTable3Subsets[s][t] == \text{swap}_{1,2}(L_t)]\Big)
   \]
   - Mode 0 최적 셰이프: $S_{3, M0}^* = \arg\max_{s \in [0, 15]} \text{match}(s)$
   - Mode 2 최적 셰이프: $S_{3, M2}^* = \arg\max_{s \in [0, 63]} \text{match}(s)$

## 14. 무(無)Threshold Rate-Distortion 모드 선택기

### 14.1 불투명 블록 5대 모드 완전경쟁 ($\min(E_0, E_1, E_2, E_3, E_6)$)

1. **Mode 6 피팅 (1-subset, 4-bit)**:
   - 1개 선분, RGBA 7777 + P-bit, 4-bit index (16단계 팔레트)
   - 복원 왜곡 오차: $E_6 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(6)}\|^2$
2. **Mode 1 피팅 (2-subsets, 3-bit)**:
   - FasTC 추출 파티션 $S_2^*$, RGB 666 + 2 P-bits, 3-bit index (8단계 팔레트)
   - 복원 왜곡 오차: $E_1 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(1)}\|^2$
3. **Mode 3 피팅 (2-subsets, 2-bit, 고정밀 7-bit 엔드포인트)**:
   - 동일 파티션 $S_2^*$ 공유, RGB 777 + 4 P-bits, 2-bit index (4단계 팔레트)
   - 복원 왜곡 오차: $E_3 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(3)}\|^2$
4. **Mode 0 피팅 (3-subsets, 3-bit, 4-bit 엔드포인트 + 6 P-bits)**:
   - FasTC 추출 파티션 $S_{3, M0}^*$, RGB 444 + 6 P-bits, 3-bit index (8단계 팔레트)
   - 복원 왜곡 오차: $E_0 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(0)}\|^2$
5. **Mode 2 피팅 (3-subsets, 2-bit, 5-bit 엔드포인트)**:
   - FasTC 추출 파티션 $S_{3, M2}^*$, RGB 555, 2-bit index (4단계 팔레트)
   - 복원 왜곡 오차: $E_2 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(2)}\|^2$
6. **불투명 최종 결정**:
   \[
   \text{Winning Opaque Block} = \arg\min(E_0, E_1, E_2, E_3, E_6)
   \]

### 14.2 투명 블록 4대 모드 완전경쟁 ($\min(E_4, E_5, E_6, E_7)$)

알파 채널 변화가 감지된 블록(`isOpaque == false`)은 4D RGBA 공간에서 4대 알파 모드가 직접 MSE 오차 대결을 펼친다:

1. **Mode 6 피팅 (1-subset, 4-bit single index)**:
   - 단일 4D 선분, RGBA 7777 + 2 P-bits, 4-bit index
   - 복원 왜곡 오차: $E_6 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(6)}\|^2$
2. **Mode 7 피팅 (2-subsets, 2-bit single index)**:
   - FasTC 2-서브셋 파티션 $S_2^*$ 공유, RGBA 5555 + 4 P-bits, 2-bit index
   - 색상과 알파가 동시에 2개 영역으로 분할된 경계면 텍스처에서 최적 성능
   - 복원 왜곡 오차: $E_7 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(7)}\|^2$
3. **Mode 4 피팅 (1-subset, 듀얼 인덱스: RGB 2b, Alpha 3b)**:
   - RGB 555 (4단계 팔레트) + Alpha 6-bit (8단계 정밀 팔레트)
   - 색상과 알파가 공간적으로 서로 다른 그래디언트를 가질 때 최적 분리 압축
   - 복원 왜곡 오차: $E_4 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(4)}\|^2$
4. **Mode 5 피팅 (1-subset, 듀얼 인덱스: RGB 2b, Alpha 2b)**:
   - RGB 777 (고정밀 색상) + Alpha 8-bit (풀 8비트 정밀 알파)
   - 부드러운 고화질 색상과 독립적 알파를 동시에 복원
   - 복원 왜곡 오차: $E_5 = \sum_{t=0}^{15} \|C_t - \hat{C}_t^{(5)}\|^2$
5. **투명 최종 결정**:
   \[
   \text{Winning Transparent Block} = \arg\min(E_4, E_5, E_6, E_7)
   \]

이로써 BC7의 8개 전 모드가 임의의 Threshold 0% 상태에서 **순수 Rate-Distortion 물리적 오차 최소화 원칙**에 의해 완벽하게 자동 선택 및 인코딩된다.

### 14.3 직관적 전담 함수: `FitAndStoreBC7ChildBlock`

모드 선택 및 대상 메모리 저장을 단일 전담 함수로 캡슐화하여 유지보수성과 가독성을 극대화하였다:
- **입력**: 대상 블록 포인터 `destinationBlock`, 자식 캔버스, FasTC 2S/3S 파티션 셰이프, Mode 6 후보 블록 및 오차, 불투명 여부 (`isOpaque`)
- **처리**: Rate-Distortion 오차 대결을 거쳐 최저 오차를 달성한 우승 모드 결정 (`selectedMode`)
- **출력**: 최저 왜곡 모드의 128비트 비트스트림을 `*destinationBlock`에 직접 기록하고, 결정된 `selectedMode` 번호를 반환


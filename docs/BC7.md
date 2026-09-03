# BC7 압축 상태 밉맵: 현재 구현된 내용

이 문서는 `DirectXTex/DirectXTexCompressedMips.cpp`에 **현재 실제로 구현된 BC7 경로만** 설명한다.
향후 계획, 미구현 기능, 성능 수치, 실험 결과는 포함하지 않는다.

## 1. BC7을 처음 볼 때 필요한 말

BC7은 4x4 RGBA texel 16개를 16 byte 블록 하나에 저장한다.
원래 색 16개를 그대로 적는 대신, 몇 개의 대표색과 각 texel이 대표색 사이 어디에 놓이는지를 기록한다.

- **Mode**: 128 bit를 어떤 규칙으로 읽을지 정하는 번호다.
- **Endpoint**: 색 선분의 양 끝점이다.
- **Index**: texel이 두 endpoint 사이 어느 위치를 사용할지 정한다.
- **Subset**: 한 블록을 둘 또는 셋의 공간 그룹으로 나눈 것이다. 각 subset은 자기 endpoint 쌍을 가진다.
- **Partition**: 16개 texel을 어느 subset에 넣을지 정한 BC7 표의 번호다.
- **P-bit**: endpoint 정밀도에 한 bit를 보태는 BC7의 보조 bit다.
- **Anchor texel**: subset마다 정해진 대표 texel이다. index의 최상위 bit 하나가 생략된다.
- **Rotation**: Mode 4와 5에서 alpha와 R, G, B 중 하나를 바꾸어 놓는 규칙이다.

각 Mode가 표현하는 공간은 다음과 같다.

| Mode | Subset | Endpoint 구성 | Index 구성 |
|---:|---:|---|---|
| 0 | 3 | RGB 4-bit와 endpoint별 P-bit | 3-bit |
| 1 | 2 | RGB 6-bit와 subset별 공유 P-bit | 3-bit |
| 2 | 3 | RGB 5-bit, P-bit 없음 | 2-bit |
| 3 | 2 | RGB 7-bit와 endpoint별 P-bit | 2-bit |
| 4 | 1 | RGB 5-bit, A 6-bit | 2-bit와 3-bit의 두 stream |
| 5 | 1 | RGB 7-bit, A 8-bit | 두 개의 2-bit stream |
| 6 | 1 | RGBA 7-bit와 endpoint별 P-bit | 4-bit |
| 7 | 2 | RGBA 5-bit와 endpoint별 P-bit | 2-bit |

Mode가 많다는 것은 같은 색을 여덟 번 저장한다는 뜻이 아니다.
블록마다 입력 특징에 잘 맞는 규칙 하나만 골라 128 bit에 저장한다.

## 2. 현재 구현 범위

현재 BC7 경로는 다음을 구현한다.

- 입력: `DXGI_FORMAT_BC7_UNORM`, `DXGI_FORMAT_BC7_UNORM_SRGB`
- 부모 블록: Mode 0부터 Mode 7까지 모두 해석
- 공통 중간 표현: `BC7ChildCanvas`
- 자식 후보 인코딩: Mode 0부터 Mode 7까지 구현
- 불투명 자식: Mode 0, 1, 2, 3, 6 후보 비교
- 반투명 자식: Mode 4, 5, 6, 7 후보 비교
- 선택 기준: 생성된 후보 블록의 실제 재구성 제곱 오차
- mip 0: 원본 압축 데이터를 그대로 복사
- mip 1: 압축된 부모 BC7 블록에서 직접 생성
- mip 2 이후: 선형 색 공간의 블록 평균 피라미드에서 생성

관련 코드는 다음 위치에 있다.

- BC7 공통 자료구조: 2188줄 이후
- 모든 부모 Mode의 사분면 평균 dispatcher: 3573줄
- `BC7ChildCanvas` 생성: 4036줄
- 2-subset partition 추정: 4154줄
- 3-subset partition 추정: 4212줄
- 자식 Mode별 fitting 및 bitstream 생성: 4309~5924줄
- 최종 Mode 선택: 5952줄
- 압축 부모에서 mip 1 생성: 6039줄
- 상위 mip 평균 피라미드: 6169~6414줄

## 3. 전체 처리 순서

```text
압축된 부모 BC7 블록 2x2
    ↓
각 부모의 Mode, partition, endpoint, index 복원
    ↓
각 부모를 네 개의 2x2 사분면 평균으로 축약
    ↓
16개 자식 texel로 배치하여 BC7ChildCanvas 생성
    ↓
평균, 4D 공분산, PCA 축, 불투명 여부 계산
    ↓
2-subset 및 3-subset partition 후보 추정
    ↓
허용된 자식 Mode 후보를 각각 인코딩하고 재구성 오차 계산
    ↓
가장 작은 오차의 128-bit BC7 블록 저장
```

하나의 자식 BC7 블록은 부모 BC7 블록 네 개가 덮던 8x8 texel을 4x4로 축소한 결과다.
각 부모 블록에서 네 개의 2x2 평균을 얻으면 총 16개의 값이 만들어지며, 이 값들이 자식 블록의 16개 texel이 된다.

## 4. 기존 DirectXTex 방식과 우리의 방식

압축 입력에서 밉맵을 만드는 일반적인 DirectXTex 작업 흐름은 다음과 같다.

```text
BC7 입력
    ↓ Decompress
비압축 RGBA texel
    ↓ GenerateMipMaps
비압축 mip chain
    ↓ Compress
BC7 mip chain
```

저장소의 `texconv` 기본 경로도 압축 입력을 먼저 풀고, 밉맵을 만든 뒤, 다시 목표 BC 형식으로 압축한다.
이 방법은 범용 filter와 format 변환에 유리하지만 전체 비압축 중간 이미지를 사용한다.

현재 구현은 전체 이미지를 풀지 않고 필요한 부모 블록 네 개의 endpoint, index, partition만 읽는다.
그 블록들이 실제로 복원하는 texel의 2x2 평균을 곧바로 구하고 공통 자식 표현으로 넘긴다.

```text
BC7 부모 블록 → 사분면 평균 → BC7ChildCanvas → 새 BC7 블록
```

두 방식 모두 결국 복원된 색을 평균내고 다시 손실 압축한다.
차이는 현재 구현이 전체 비압축 mip image 대신 블록 단위의 평균과 통계만 유지한다는 점이다.

## 5. 모든 부모 Mode를 공통 공간으로 변환

BC7 Mode마다 subset 수와 partition field 위치가 다르다.

| Mode | Subset 수 | Partition 시작 bit | Partition bit 수 |
|---:|---:|---:|---:|
| 0 | 3 | 1 | 4 |
| 1 | 2 | 2 | 6 |
| 2 | 3 | 3 | 6 |
| 3 | 2 | 4 | 6 |
| 4 | 1 | 없음 | 0 |
| 5 | 1 | 없음 | 0 |
| 6 | 1 | 없음 | 0 |
| 7 | 2 | 8 | 6 |

이 차이는 `BC7ModeSpatialInfo`와 `BC7PartitionLayout`으로 정리한다.

```cpp
struct BC7PartitionLayout
{
    const uint8_t* subsetByTexel;
    std::array<uint8_t, 3> anchorTexel;
    size_t subsetCount;
};
```

- `subsetByTexel[t]`: texel `t`가 속한 subset
- `anchorTexel[s]`: subset `s`의 anchor texel
- `subsetCount`: 현재 Mode의 subset 수

2-subset과 3-subset의 배치 및 fix-up 위치는 `BC6HBC7.cpp`와 같은 BC7 lookup 정보를 이 파일 안에 둔 표로 처리한다.
1-subset Mode 4, 5, 6은 모든 texel을 subset 0으로 본다.

Mode 0, 1, 2, 3, 7의 index는 공통 함수 `ExtractBC7PartitionedIndices()`가 읽는다.
anchor texel에는 index의 최상위 bit가 저장되지 않으므로 읽는 bit 수는 다음과 같다.

\[
B(t)=
\begin{cases}
I-1, & t=\operatorname{anchor}(s_t) \\
I, & \text{그 외}
\end{cases}
\]

여기서 `I`는 해당 Mode의 index bit 수이고 `s_t`는 texel `t`의 subset이다.
생략된 최상위 bit는 0으로 복원된다.

저장된 endpoint bit 수가 8보다 작으면 P-bit를 붙인 뒤 상위 bit를 아래쪽에 반복해 8-bit로 늘린다.
예를 들어 Mode 7의 5-bit 값 `v_5`와 P-bit `p`는 다음 순서로 복원된다.

\[
v_6=(v_5\ll1)\mid p
\]

\[
v_8=(v_6\ll2)\mid(v_6\gg4)
\]

Mode마다 원래 endpoint 정밀도와 P-bit 공유 방식은 다르지만, 복원이 끝난 뒤에는 모두 `BC7MultiSubsetEndpointBatch` 또는 `BC7EndpointPairBatch`의 RGBA8 값이 된다.

Mode 4와 5는 color와 alpha에 서로 다른 index stream을 적용하고 channel rotation도 복원한다.
Mode 6은 4-bit index를 packed 상태로 세어 사분면 평균을 계산한다.

## 6. 부모 texel 복원과 사분면 평균

BC7 endpoint `e_0`, `e_1`과 index에 대응하는 정수 weight `w`가 있을 때 복원 값은 BC7 반올림 규칙에 따라 계산한다.

\[
\hat C=\left\lfloor\frac{(64-w)e_0+w e_1+32}{64}\right\rfloor
\]

구현된 weight 표는 다음과 같다.

- 2-bit: `{ 0, 21, 43, 64 }`
- 3-bit: `{ 0, 9, 18, 27, 37, 46, 55, 64 }`
- 4-bit: `{ 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 }`

한 부모 블록의 사분면은 다음 texel들로 구성된다.

```text
Q0 = { 0,  1,  4,  5 }
Q1 = { 2,  3,  6,  7 }
Q2 = { 8,  9, 12, 13 }
Q3 = {10, 11, 14, 15 }
```

사분면 `q`와 channel `c`의 평균은 다음과 같다.

\[
Q_{q,c}=\frac14\sum_{t\in q}\hat C_{t,c}
\]

`ComputeBC7ParentQuadrantMeans()`는 Mode 0~7을 판별한 뒤 각 Mode의 endpoint, index, partition을 복원하고 최종적으로 같은 `BC7QuadrantMeanBatch` 형식으로 합친다.

BC7 sRGB 입력에서는 RGB만 sRGB에서 linear로 변환한 후 평균을 낸다. Alpha는 항상 linear 값으로 다룬다.

## 7. 공통 중간 표현 `BC7ChildCanvas`

```cpp
struct BC7ChildCanvas
{
    XMVECTOR texels[16][4];
    BC7BlockMeanBatch mean;
    BC7CovarianceMatrixBatch covariance;
    BC7BlockMeanBatch axis;
    XMVECTOR isOpaque;
};
```

- `texels`: 다운샘플된 자식 블록의 16개 RGBA 값
- `mean`: 16개 texel의 4D 평균
- `covariance`: RGBA 4D 공분산 행렬
- `axis`: 공분산 행렬의 주축을 power iteration으로 근사한 값
- `isOpaque`: 모든 자식 alpha가 `254/255` 이상인지 나타내는 SIMD mask

부모 네 개의 사분면 평균을 `p_{00}`, `p_{10}`, `p_{01}`, `p_{11}`이라 하면, 각 부모 평균은 다음과 같다.

\[
\mu_k=\frac14\sum_{q=0}^{3}p_{k,q}
\]

전체 자식 평균은 다음과 같다.

\[
\mu=\frac14\sum_{k=0}^{3}\mu_k
\]

공분산은 부모 내부 변화와 부모 사이 변화를 합치는 ANOVA 형태로 계산한다.

\[
\Sigma=\frac14\sum_{k=0}^{3}
\left(
\Sigma_k+(\mu_k-\mu)(\mu_k-\mu)^T
\right)
\]

이 식이 성립하는 이유는 한 표본 `x`의 중심 차이를 다음처럼 두 조각으로 나눌 수 있기 때문이다.

\[
x-\mu=(x-\mu_k)+(\mu_k-\mu)
\]

오른쪽을 외적으로 펼치면 부모 내부 변화, 부모 사이 변화, 두 교차항이 나온다.
같은 부모 안에서 `x-\mu_k`의 평균은 0이므로 교차항의 평균은 사라진다.
따라서 남은 두 변화량을 더한 것이 전체 공분산이다.

`axis`는 이 공분산에 power iteration을 세 번 적용해 얻는다.

\[
v_{n+1}=\frac{\Sigma v_n}{\|\Sigma v_n\|}
\]

sRGB 경로에서는 평균 계산까지 linear 공간을 유지하고, BC7 endpoint fitting 전에 RGB만 다시 sRGB code 공간으로 변환한다.

## 8. Partition 후보 추정

### 8.1 2-subset

16개 자식 texel을 PCA 축에 투영한다.

\[
p_t=(C_t-\mu)\cdot v
\]

`p_t >= 0`인지에 따라 16-bit mask를 만든 뒤, 64개 2-subset 표와 Hamming distance를 비교한다.
subset 번호의 반전은 같은 공간 분할이므로 양쪽 방향 중 작은 거리를 사용한다.

\[
d(s)=\min\left(
\operatorname{popcount}(M\oplus S_s),
\operatorname{popcount}(\neg M\oplus S_s)
\right)
\]

\[
s^*=\arg\min_{0\le s<64}d(s)
\]

이 partition 하나를 Mode 1, 3, 7 후보가 공유한다.

### 8.2 3-subset

PCA 투영 범위를 정확히 세 구간으로 나누어 각 texel에 label 0, 1, 2를 붙인다.
그 label과 BC7 3-subset 표의 일치 개수를 센다.

\[
s^*=\arg\max_s\operatorname{match}(L,S_s)
\]

- Mode 0: shape 0~15 중 하나 선택
- Mode 2: shape 0~63 중 하나 선택

subset 1과 2의 번호가 뒤바뀐 경우도 같은 분할로 취급해 함께 비교한다.

이 단계는 모든 partition을 endpoint fitting하는 전역 탐색이 아니다. PCA로 만든 공간 분할과 가장 가까운 **partition 후보 하나를 빠르게 고르는 현재 구현**이다.

## 9. 자식 Mode 후보와 최종 선택

각 Mode의 fitting 함수는 endpoint와 index를 구하고 실제 BC7 bitstream을 만든다.

후보 `m`이 복원한 texel을 `\hat C_t^{(m)}`이라 하면 해당 후보의 오차는 다음 제곱 오차다.

\[
E_m=\sum_{t=0}^{15}\sum_{c\in\{R,G,B,A\}}
\left(C_{t,c}-\hat C_{t,c}^{(m)}\right)^2
\]

현재 sRGB 경로의 자식 fitting과 이 오차 계산은 sRGB code 공간에서 이루어진다.
다운샘플 평균 자체는 linear light에서 계산한 뒤 code 공간으로 돌아온 값이다.

불투명 자식은 다음 후보를 비교한다.

\[
m^*=\arg\min_{m\in\{0,1,2,3,6\}}E_m
\]

반투명 자식은 다음 후보를 비교한다.

\[
m^*=\arg\min_{m\in\{4,5,6,7\}}E_m
\]

각 분기 안에서 후보를 고를 때는 `0.1`, `0.05` 같은 오차 차이 threshold를 사용하지 않고 SSE를 직접 비교한다.
다만 이 결과는 앞 단계에서 선택한 partition과 각 Mode fitting이 만든 후보들 사이의 최소 오차다. BC7의 모든 endpoint, index, partition 조합을 완전 탐색한 전역 최적해를 뜻하지는 않는다.

## 10. mip 단계별 데이터 흐름

`GenerateCompressedMipMapsBC7()`의 동작은 다음과 같다.

1. mip 0의 BC7 데이터는 bit 단위로 그대로 복사한다.
2. mip 1은 부모 압축 블록 네 개를 읽어 `BC7ChildCanvas`를 만들고 새 BC7 블록으로 인코딩한다.
3. 동시에 원본 부모 블록의 linear 평균을 별도 버퍼에 저장한다.
4. mip 2 이후는 생성된 BC7 블록을 다시 해석하지 않고 이 평균 버퍼에서 다음 입력을 만든다.
5. 더 작은 단계가 필요하면 평균 버퍼를 2x2 box filter로 축소한다.
6. 홀수 크기 경계는 마지막 행 또는 열을 반복하는 clamp-to-edge 방식으로 처리한다.

평균 피라미드를 사용하는 이유는 이전 단계에서 새로 양자화한 BC7 결과를 다음 단계의 입력으로 다시 사용하지 않기 위해서다. 이로써 생성 단계 사이에 BC7 재인코딩 오차가 연쇄적으로 누적되는 경로를 피한다.

## 11. 현재 구현의 정확한 한계

- 지원 형식은 BC7 UNORM과 BC7 UNORM SRGB다.
- 유효한 Mode 0~7 부모 블록은 모두 공통 사분면 표현으로 변환된다.
- partition 선택은 PCA 기반 후보 추정이며 모든 partition의 최종 재구성 오차를 비교하지 않는다.
- 최종 Mode 선택은 생성된 후보들에 대해서만 실제 제곱 오차를 비교한다.
- 후보 집합은 `alpha >= 254/255`인 texel만 있는지를 기준으로 두 갈래로 나뉜다. 따라서 전체 Mode 0~7을 한꺼번에 SSE 비교하는 구현은 아니다.
- 특히 `254/255` 불투명 판정은 실제로 Mode 후보 집합을 바꾸는 현재 구현의 threshold다. 엄밀히 모든 선택을 오차만으로 결정한다는 목표와는 구분해 검토해야 한다.
- Mode 0, 1, 2, 3 후보의 오차는 RGB만 합산한다. 위 불투명 판정에 들어온 alpha가 정확히 1이 아니라면 Mode 6과의 비교에서 alpha 항이 대칭적으로 반영되지 않는다.
- 이 문서 작성 과정에서는 사용자의 검증 원칙에 따라 빌드나 실행 실험을 수행하지 않았다.

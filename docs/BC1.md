# BC1 압축 상태 밉맵의 원리와 현재 구현

이 문서는 BC1을 처음 보는 사람도 `DirectXTex/DirectXTexCompressedMips.cpp`의 BC1 구현을 검토할 수 있도록 작성한다.
현재 코드에 없는 기능이나 확인되지 않은 성능 수치는 다루지 않는다.

## 1. 먼저 알아야 할 것

### 1.1 밉맵이란 무엇인가

큰 그림을 화면에서 아주 작게 표시할 때 원본의 모든 점을 읽을 필요는 없다.
그래서 원본의 가로와 세로를 절반씩 줄인 그림을 미리 여러 장 만든다.

```text
mip 0: 1024 x 1024
mip 1:  512 x  512
mip 2:  256 x  256
...
```

가로와 세로가 각각 절반이므로 texel 수는 보통 1/4이 된다.
가장 단순한 2x2 box filter는 네 값을 더해 4로 나눈다.

\[
y=\frac{x_{00}+x_{10}+x_{01}+x_{11}}4
\]

예를 들어 `2, 4, 6, 8`의 평균은 `(2+4+6+8)/4=5`다.

### 1.2 BC1 블록이란 무엇인가

BC1은 4x4, 즉 16개 texel을 8 byte에 저장한다.
16개 색을 모두 적는 대신 다음만 저장한다.

- RGB565 endpoint 두 개: `color0`, `color1`
- 각 texel이 네 palette 색 중 무엇을 사용할지 나타내는 2-bit selector 16개

RGB565는 R 5-bit, G 6-bit, B 5-bit로 이루어진 16-bit 색이다.
5-bit 값 `v_5`와 6-bit 값 `v_6`은 현재 코드에서 bit replication으로 8-bit로 확장된다.

\[
v_8=(v_5\ll3)\mid(v_5\gg2)
\]

\[
v_8=(v_6\ll2)\mid(v_6\gg4)
\]

불투명 4-color Mode에서 palette는 다음 네 색이다.

\[
C_0=E_0,\qquad C_1=E_1
\]

\[
C_2=\left\lfloor\frac{2E_0+E_1}{3}\right\rfloor,qquad
C_3=\left\lfloor\frac{E_0+2E_1}{3}\right\rfloor
\]

이 계산은 R, G, B channel마다 따로 적용된다.

## 2. 현재 지원 범위

현재 구현은 다음 형식을 지원한다.

- `DXGI_FORMAT_BC1_UNORM`
- `DXGI_FORMAT_BC1_UNORM_SRGB`

단, 실제로 투명 palette entry를 사용하는 BC1 블록은 거부한다.
현재 자식 encoder가 항상 `color0 > color1`인 불투명 4-color 블록을 만들기 때문이다.

`color0 <= color1`이어도 selector 3을 사용하지 않으면 실제 texel은 불투명하므로 허용한다.
검사는 `IsOpaqueBC1Image()`에서 수행한다.

주요 코드 위치는 다음과 같다.

- BC1 자료구조: 98~225줄
- RGB565 복원: 241줄
- palette 복원: 318줄
- selector 사분면 histogram: 371줄
- 사분면 평균: 458줄
- 평균과 공분산: 497~673줄
- PCA 초기 endpoint: 703줄
- least-squares endpoint 보정: 803줄
- RGB565 양자화와 selector 재배치: 1012줄
- 전체 BC1 자식 encoder: 1061줄
- 입력 불투명 검사: 1132줄
- 압축 부모에서 mip 1 생성: 1158줄
- 상위 mip 평균 피라미드: 1234~1424줄

## 3. 기존 DirectXTex 방식과 우리의 방식

### 3.1 일반적인 DirectXTex 처리 순서

압축된 입력으로 새 밉맵을 만드는 일반적인 경로는 개념적으로 다음과 같다.

```text
BC 압축 이미지
    ↓ Decompress
비압축 texel 이미지
    ↓ GenerateMipMaps
비압축 mip chain
    ↓ Compress
BC 압축 mip chain
```

저장소의 `texconv` 기본 경로에서도 압축 입력은 `Decompress()`하고, `GenerateMipMaps()`로 밉을 만든 뒤, 필요한 압축 형식으로 `Compress()`한다.

이 방식은 범용적이다. 여러 filter와 여러 pixel format을 같은 틀에서 처리하기 좋다.
대신 전체 비압축 이미지를 중간 결과로 준비하고 각 mip을 다시 압축해야 한다.

### 3.2 현재 압축 상태 밉맵 경로

현재 구현은 다음 순서를 사용한다.

```text
BC1 부모 블록 네 개
    ↓ endpoint와 selector 통계 복원
부모별 2x2 사분면 평균 네 개
    ↓ 공간 위치에 맞게 배치
자식 4x4의 16개 RGB 표본
    ↓ 평균·공분산·PCA·least squares
새 BC1 endpoint와 selector
```

핵심 차이는 전체 비압축 이미지를 만들지 않는다는 점이다.
필요한 작은 통계만 압축 블록에서 직접 꺼내 자식 블록을 만든다.

## 4. 핵심 아이디어 1: selector 개수만 알아도 평균을 구할 수 있다

한 2x2 사분면에는 texel이 네 개 있다.
각 texel은 palette `C_0`, `C_1`, `C_2`, `C_3` 중 하나다.

사분면 안에서 selector `i`가 나온 횟수를 `n_i`라고 하자.

\[
n_0+n_1+n_2+n_3=4
\]

그러면 사분면 평균은 다음과 같다.

\[
Q=\frac{n_0C_0+n_1C_1+n_2C_2+n_3C_3}{4}
\]

### 왜 이것이 texel을 하나씩 더한 것과 같은가

예를 들어 네 selector가 `0, 2, 2, 3`이라면 직접 더한 값은 다음과 같다.

\[
Q=\frac{C_0+C_2+C_2+C_3}{4}
\]

같은 항끼리 묶으면 다음과 같다.

\[
Q=\frac{1C_0+0C_1+2C_2+1C_3}{4}
\]

즉 selector 순서는 평균에 영향을 주지 않고 각 selector가 몇 번 나왔는지만 필요하다.
이것은 근사가 아니라 덧셈 항을 묶어 쓴 동일한 계산이다.

`Extract2x2SelectorHistograms()`는 SWAR bit 연산으로 네 사분면의 `n_0`부터 `n_3`까지 센다.
`ComputeParentQuadrantMeansBatch()`는 그 개수와 실제 BC1 palette를 곱해 평균을 만든다.

## 5. sRGB는 왜 먼저 linear로 바꾸는가

sRGB code 값은 빛의 세기와 직선 관계가 아니다.
따라서 sRGB 숫자 두 개를 그대로 평균내면 실제 빛의 평균과 달라진다.

현재 코드는 BC1의 정수 palette를 먼저 정확히 복원한 다음, sRGB 형식이면 각 palette 색을 linear 값으로 변환한다.

8-bit sRGB code를 `[0,1]`의 `s`로 정규화했을 때 linear 값 `L`은 다음 함수다.

\[
L(s)=
\begin{cases}
s/12.92, & s\le 0.04045 \\
\left(\frac{s+0.055}{1.055}\right)^{2.4}, & s>0.04045
\end{cases}
\]

그 뒤 linear 공간에서 평균, 공분산, endpoint fitting을 수행한다.
마지막 endpoint를 BC1에 저장하기 직전에만 반대 방향 sRGB 함수로 되돌린다.

중요한 순서는 다음과 같다.

```text
RGB565 복원 → BC1 정수 보간 → sRGB-to-linear → 평균
```

정수 보간 전에 sRGB를 풀어 버리면 실제 BC1 decoder가 보여 주는 palette와 다른 값을 분석하게 된다.

## 6. 부모 네 개가 자식 texel 16개가 되는 방법

하나의 부모 BC1 블록은 4x4다.
그 안의 각 2x2 영역을 평균내면 2x2 값 네 개가 된다.

부모 블록 네 개를 다음처럼 놓자.

```text
p00 p10
p01 p11
```

각 부모가 2x2 값으로 줄어들므로 네 부모를 합치면 자식의 4x4 값이 된다.

```text
p00의 2x2 | p10의 2x2
-----------+-----------
p01의 2x2 | p11의 2x2
```

따라서 원래 8x8 영역의 2x2 box-filter 결과 16개가 정확히 자식 4x4 표본으로 배치된다.

## 7. 핵심 아이디어 2: 평균과 공분산을 합쳐 계산한다

### 7.1 평균

부모 `k`의 네 사분면 평균을 `q_{k,0}`부터 `q_{k,3}`이라 하자.

\[
\mu_k=\frac14\sum_{j=0}^{3}q_{k,j}
\]

자식 블록 전체 평균은 부모 평균 네 개의 평균이다.

\[
\mu=\frac14\sum_{k=0}^{3}\mu_k
\]

이를 펼치면 다음과 같다.

\[
\mu=\frac1{16}\sum_{k=0}^{3}\sum_{j=0}^{3}q_{k,j}
\]

즉 자식 표본 16개의 평균과 정확히 같다.

### 7.2 공분산을 쉬운 말로 이해하기

평균이 색들의 중심이라면 공분산은 색들이 중심에서 어느 방향으로 퍼졌는지를 나타낸다.

- `rr`이 크다: 빨강 방향 변화가 크다.
- `gg`가 크다: 초록 방향 변화가 크다.
- `rg`가 양수로 크다: 빨강이 커질 때 초록도 함께 커지는 경향이 있다.

현재 구현은 3x3 대칭 행렬의 여섯 값 `rr, gg, bb, rg, rb, gb`만 저장한다.

### 7.3 ANOVA 형태의 결합

각 부모 안의 공분산을 `\Sigma_k`라고 하면 자식 공분산은 다음과 같이 계산할 수 있다.

\[
\Sigma=\frac14\sum_{k=0}^{3}
\left[
\Sigma_k+(\mu_k-\mu)(\mu_k-\mu)^T
\right]
\]

대괄호 안의 두 항은 다음 뜻이다.

- `\Sigma_k`: 한 부모 내부에서 퍼진 정도
- `(\mu_k-\mu)(\mu_k-\mu)^T`: 부모 중심들이 서로 떨어진 정도

### 왜 두 항을 더하면 전체 공분산인가

부모 `k`의 표본을 `x`라고 하면 다음처럼 나눌 수 있다.

\[
x-\mu=(x-\mu_k)+(\mu_k-\mu)
\]

이를 제곱 형태로 펼치면 내부 변화, 부모 사이 변화, 교차항이 나온다.
같은 부모 안에서 `x-\mu_k`의 평균은 0이므로 교차항의 평균은 사라진다.
그래서 내부 공분산과 부모 사이 공분산만 남는다.

이 식을 `ComputeChildBlockMoments()`가 구현한다.

## 8. PCA로 endpoint의 첫 방향을 찾는다

BC1은 네 palette 색이 두 endpoint를 잇는 선 위에 있다.
따라서 16개 RGB 표본이 가장 길게 늘어선 방향을 찾으면 좋은 endpoint 선을 만들 수 있다.

공분산 행렬을 `\Sigma`, 방향을 `v`라고 하면 principal axis는 다음 관계를 만족하는 방향이다.

\[
\Sigma v=\lambda v
\]

현재 BC1 코드는 시작 방향 `(1,1,1)/\sqrt3`에 공분산 행렬을 한 번 곱하고 정규화한다.

\[
v\leftarrow\frac{\Sigma v_0}{\|\Sigma v_0\|}
\]

즉 정확한 고유벡터를 끝까지 반복 계산하는 것이 아니라 빠른 1회 power-iteration 근사다.

각 표본을 이 축에 투영한다.

\[
t_i=v\cdot(x_i-\mu)
\]

가장 작은 투영값 `t_min`과 가장 큰 투영값 `t_max`를 찾아 초기 endpoint를 만든다.

\[
P_0=\mu+t_{min}v,\qquad P_1=\mu+t_{max}v
\]

## 9. Least squares로 endpoint를 한 번 보정한다

초기 선 위에서 각 표본이 가장 가까운 BC1 위치 `0, 1/3, 2/3, 1` 중 하나를 고른다.
선형 weight를 `w_i`라 하면 복원색은 다음과 같다.

\[
\hat x_i=(1-w_i)P_0+w_iP_1=P_0+w_iD
\]

여기서 `D=P_1-P_0`다.

고정된 weight에 대해 다음 오차가 가장 작아지는 `P_0`와 `D`를 구한다.

\[
E=\sum_{i=1}^{16}\|x_i-(P_0+w_iD)\|^2
\]

미분값을 0으로 두면 2x2 normal equation이 나온다.

\[
\begin{bmatrix}
N & \sum w_i\\
\sum w_i & \sum w_i^2
\end{bmatrix}
\begin{bmatrix}
P_0\\D
\end{bmatrix}
=
\begin{bmatrix}
\sum x_i\\
\sum w_i x_i
\end{bmatrix}
\]

`N=16`이다. R, G, B는 같은 2x2 행렬을 사용하고 오른쪽 값만 channel별로 다르다.
행렬식이 거의 0인 평평한 블록은 불안정한 나눗셈 대신 두 endpoint를 평균색으로 둔다.

이 단계는 고정된 네 weight에 대해서는 최소 제곱 해다.
하지만 selector와 양자화까지 포함한 모든 BC1 조합의 전역 최적해를 보장하지는 않는다.

## 10. RGB565 양자화와 selector 결정

보정한 endpoint가 linear 공간에 있다면 sRGB 형식에서 먼저 sRGB code 공간으로 변환한다.
그 뒤 각 channel을 반올림해 RGB565로 줄인다.

\[
r_5=\operatorname{round}(31r),\quad
g_6=\operatorname{round}(63g),\quad
b_5=\operatorname{round}(31b)
\]

불투명 4-color Mode를 보장하기 위해 packed endpoint가 `color0 > color1`이 되도록 순서를 맞춘다.
두 값이 같으면 한 RGB565 단계 벌린다.

양자화된 endpoint로 실제 hardware palette를 다시 만든 뒤, 각 자식 texel에 대해 네 palette 색과의 RGB 제곱 거리를 계산한다.

\[
d_i=(R-R_i)^2+(G-G_i)^2+(B-B_i)^2
\]

가장 작은 `d_i`의 selector를 저장한다.
양자화 전 palette가 아니라 실제 저장될 RGB565 palette를 사용한다는 점이 중요하다.

## 11. SIMD 배치

`XMVECTOR`의 네 lane은 RGBA를 뜻하지 않는다.
이 BC1 코드에서는 각 lane이 서로 다른 BC1 블록 하나를 뜻한다.

```text
lane 0 = block A
lane 1 = block B
lane 2 = block C
lane 3 = block D
```

따라서 endpoint 복원, histogram 계산, 평균, 공분산, endpoint fitting을 네 블록에 동시에 수행한다.
마지막 묶음에 실제 블록이 네 개보다 적으면 `validLanes`만 저장한다.

## 12. mip 0, mip 1, mip 2 이후

### mip 0

원본 압축 데이터를 그대로 복사한다. 입력 BC1 bitstream은 바뀌지 않는다.

### mip 1

`ProcessCompressedRowBC1()`이 압축된 부모 블록을 직접 읽는다.
부모별 사분면 평균을 구하고 새 자식 BC1 블록을 만든다.
동시에 각 원본 부모 블록의 linear 평균을 별도 배열에 저장한다.

### mip 2 이후

이전 단계에서 새로 압축한 BC1 블록을 다시 입력으로 사용하지 않는다.
원본 부모에서 얻어 둔 linear 평균 배열을 사용한다.

필요한 해상도에 맞춰 평균 배열을 다음 식으로 줄인다.

\[
m'=\frac{m_{00}+m_{10}+m_{01}+m_{11}}4
\]

두 개의 평균 버퍼를 번갈아 사용하는 ping-pong 구조이며, 홀수 크기 경계는 마지막 행이나 열을 반복하는 clamp-to-edge 방식이다.

### 왜 평균 피라미드를 쓰는가

새로 생성한 BC1 블록에는 endpoint와 selector 양자화 오차가 들어 있다.
그 블록을 다시 풀어 다음 mip을 만들면 그 오차가 다음 단계 입력에 섞인다.

현재 구현은 원본 BC1에서 얻은 linear 평균을 계속 축소하므로 생성된 BC1을 다시 읽는 연쇄 경로를 피한다.
다만 최초 입력 mip 0 자체가 BC1이므로, 원본 BC1 압축에 이미 들어 있던 손실까지 없어지는 것은 아니다.

## 13. 현재 연산 순서 요약

```text
1. 입력이 실제 불투명 BC1인지 검사
2. mip 0을 그대로 복사
3. 부모 BC1 블록 네 개를 SIMD lane에 로드
4. RGB565 endpoint와 실제 BC1 palette 복원
5. selector histogram으로 부모 사분면 평균 계산
6. 16개 자식 표본의 평균과 ANOVA 공분산 계산
7. 1회 power iteration으로 PCA 초기 endpoint 계산
8. 고정 weight least squares로 endpoint 보정
9. 필요하면 linear endpoint를 sRGB code로 변환
10. RGB565로 양자화하고 불투명 endpoint 순서 보장
11. 실제 양자화 palette에서 가장 가까운 selector 선택
12. 자식 BC1 블록 저장
13. mip 2 이후는 linear 평균 피라미드에서 같은 encoder 사용
```

## 14. 논리 검증과 정확한 한계

### 구현과 수학이 일치하는 부분

- selector histogram 평균은 texel별 palette 합과 대수적으로 동일하다.
- 네 부모의 사분면 평균 배치는 8x8 영역의 2x2 box filter 결과 16개와 대응한다.
- ANOVA 식은 16개 자식 표본의 평균과 공분산을 부모별 통계로 다시 묶은 것이다.
- sRGB 평균은 linear 공간에서 수행한다.
- selector는 최종 양자화 palette에 대한 실제 RGB 제곱 거리로 선택한다.
- mip 0은 bit-for-bit 보존된다.

### 현재 구현이 보장하지 않는 것

- 투명 BC1 texel은 지원하지 않는다.
- PCA는 1회 근사이며 모든 가능한 endpoint를 탐색하지 않는다.
- least squares는 먼저 정한 weight를 고정한 조건부 최소해다.
- 가능한 모든 RGB565 endpoint와 selector 조합의 전역 최소 오차를 보장하지 않는다.
- 전체 비압축 이미지를 만들지 않지만, endpoint fitting을 위해 자식 블록에 필요한 16개 평균 표본과 통계는 계산한다.
- 이 문서 작성 과정에서는 빌드와 실행 실험을 수행하지 않았다.

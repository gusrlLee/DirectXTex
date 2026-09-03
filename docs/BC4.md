# BC4 압축 상태 밉맵의 원리와 현재 구현

이 문서는 BC4를 처음 보는 사람도 `DirectXTex/DirectXTexCompressedMips.cpp`의 BC4 구현을 검토할 수 있도록 작성한다.
현재 코드에 없는 BC5·BC4 SNORM 확장이나 확인되지 않은 성능 수치는 다루지 않는다.

## 1. BC4는 무엇을 저장하는가

BC4는 4x4 영역의 숫자 16개를 8 byte에 저장한다.
BC1이 RGB 색을 다룬다면 BC4는 `[0,1]` 범위의 channel 하나만 다룬다.

예를 들어 다음과 같은 한 종류의 값을 저장할 수 있다.

- 흑백 높이
- 재질의 거칠기
- mask
- normal map의 한 channel

한 BC4 블록에는 다음이 들어 있다.

- 8-bit endpoint `red0`
- 8-bit endpoint `red1`
- texel마다 3-bit index 하나씩, 총 16개

```cpp
struct BC4Block
{
    uint8_t red0;
    uint8_t red1;
    uint8_t indices[6];
};
```

두 endpoint가 2 byte, index 16개가 `16 × 3 = 48 bit = 6 byte`이므로 모두 8 byte다.

## 2. 현재 지원 범위

현재 압축 상태 밉맵 경로가 지원하는 BC4 형식은 다음 하나다.

- `DXGI_FORMAT_BC4_UNORM`

두 BC4 palette 규칙은 모두 입력과 출력에서 처리한다.

- `red0 > red1`: endpoint 사이를 여섯 번 보간하는 8단계 ramp
- `red0 <= red1`: endpoint 사이의 네 보간값과 고정값 0, 1

주요 코드 위치는 다음과 같다.

- BC4 자료구조와 상수: 1432~1458줄
- six-interpolated 입력의 row lookup table: 1475줄
- four-interpolated palette: 1502줄
- 압축 부모의 사분면 평균: 1524줄
- least-squares 통계와 해: 1611~1687줄
- six-interpolated index와 오차: 1698~1785줄
- four-interpolated 후보: 1790~1866줄
- six-interpolated 후보: 1870~1930줄
- 실제 오차 기반 palette 규칙 선택: 1934줄
- 압축 부모에서 mip 1 생성: 1964줄
- 상위 mip 평균 피라미드: 2041~2183줄

## 3. 밉맵에서 필요한 계산

가로와 세로를 절반으로 줄이는 2x2 box filter는 네 값을 평균낸다.

\[
y=\frac{x_{00}+x_{10}+x_{01}+x_{11}}4
\]

BC4 부모 블록 네 개는 원래 8x8 영역을 덮는다.
각 부모의 네 2x2 사분면을 평균내면 부모 하나당 2x2 값이 생긴다.
네 부모의 결과를 이어 붙이면 새 4x4 자식 블록의 16개 값이 된다.

```text
부모 p00의 2x2 | 부모 p10의 2x2
----------------+----------------
부모 p01의 2x2 | 부모 p11의 2x2
```

## 4. BC4의 두 palette 규칙

endpoint를 `[0,1]` 값 `e_0=red0/255`, `e_1=red1/255`로 적겠다.

### 4.1 `red0 > red1`: six-interpolated Mode

이 경우 palette에는 두 endpoint와 그 사이의 여섯 값이 있다.

\[
P_0=e_0,\qquad P_1=e_1
\]

\[
P_k=\frac{(8-k)e_0+(k-1)e_1}{7},\qquad 2\le k\le7
\]

예를 들어 다음과 같다.

\[
P_2=\frac{6e_0+e_1}{7},\qquad
P_7=\frac{e_0+6e_1}{7}
\]

endpoint 사이를 같은 간격으로 여덟 위치로 나눈 ramp라고 생각하면 쉽다.

### 4.2 `red0 <= red1`: four-interpolated Mode

이 경우 endpoint 사이에는 네 보간값만 있고 마지막 두 palette entry는 항상 0과 1이다.

\[
P_0=e_0,\qquad P_1=e_1
\]

\[
P_k=\frac{(6-k)e_0+(k-1)e_1}{5},\qquad 2\le k\le5
\]

\[
P_6=0,\qquad P_7=1
\]

0이나 1이 자주 나오는 mask에서는 endpoint를 그 극값에 소비하지 않고도 정확히 표현할 수 있다는 장점이 있다.

## 5. 기존 DirectXTex 방식과 우리의 방식

### 5.1 일반적인 DirectXTex 경로

압축 입력에서 밉맵을 만드는 일반적인 작업 흐름은 다음과 같다.

```text
BC4 입력
    ↓ Decompress
비압축 단일-channel texel
    ↓ GenerateMipMaps
비압축 mip chain
    ↓ Compress
BC4 mip chain
```

이 방법은 범용 filter와 pixel format 변환에 유리하다.
대신 전체 비압축 이미지를 중간에 만들고 각 mip을 다시 압축한다.

### 5.2 현재 압축 상태 밉맵 경로

현재 구현은 전체 비압축 이미지를 만들지 않는다.

```text
압축된 BC4 부모 블록 네 개
    ↓ palette와 index에서 사분면 평균 계산
자식 4x4 값 16개
    ↓ 두 BC4 palette 규칙을 각각 fitting
두 후보의 실제 SSE 비교
    ↓
더 작은 오차의 BC4 블록 저장
```

## 6. Six-interpolated 입력의 사분면 평균을 빠르게 구하는 식

six-interpolated palette의 모든 값은 두 endpoint의 가중합이다.
palette index `i`에 대응하는 ramp 위치를 `j_i`라고 하자.

```text
index i: 0 1 2 3 4 5 6 7
weight j:0 7 1 2 3 4 5 6
```

그러면 한 texel 값은 다음과 같다.

\[
x_i=\frac{(7-j_i)red0+j_i red1}{7\cdot255}
\]

한 사분면의 네 texel에 대한 `j`의 합을 `J`라고 하자.

\[
J=j_0+j_1+j_2+j_3
\]

네 texel의 합은 다음과 같다.

\[
\sum_{n=0}^{3}x_n
=\sum_{n=0}^{3}\frac{(7-j_n)red0+j_n red1}{7\cdot255}
\]

같은 endpoint 항을 묶으면 다음과 같다.

\[
\sum_{n=0}^{3}x_n
=\frac{(28-J)red0+J red1}{7\cdot255}
\]

평균은 다시 4로 나눈 값이다.

\[
Q=\frac{(28-J)red0+J red1}{4\cdot7\cdot255}
\]

즉 네 texel을 각각 float로 복원할 필요 없이 `J` 하나만 알면 평균을 계산할 수 있다.

### row lookup table

한 BC4 행에는 3-bit index 네 개, 즉 12 bit가 있다.
12 bit가 가질 수 있는 경우는 `2^{12}=4096`개다.

`GetBC4RowWeightTable()`은 각 12-bit 행에 대해 다음 두 합을 미리 저장한다.

- 왼쪽 두 texel의 `j` 합
- 오른쪽 두 texel의 `j` 합

위쪽 두 행의 합으로 위 사분면 둘을 만들고, 아래쪽 두 행의 합으로 아래 사분면 둘을 만든다.
이 표는 근삿값이 아니라 반복되는 정수 합을 미리 계산해 둔 것이다.

## 7. Four-interpolated 입력은 왜 직접 복원하는가

four-interpolated Mode의 `P_6=0`, `P_7=1`은 endpoint의 가중합이 아니다.
따라서 모든 palette entry를 같은 `J` 하나로 합치는 six-interpolated 식을 그대로 쓸 수 없다.

현재 구현은 이 경우 palette 여덟 값을 만든 뒤, 각 사분면의 texel 네 개를 읽어 직접 평균낸다.

\[
Q=\frac14\sum_{t\in q}P_{index_t}
\]

두 입력 Mode 모두 최종 결과는 같은 네 개의 사분면 평균이다.
이후 단계는 부모가 어느 Mode였는지 알 필요가 없다.

## 8. 자식 블록의 공통 표현

`BC4TexelBlock`은 16개 scalar 값을 네 SIMD row로 저장한다.

```cpp
struct BC4TexelBlock
{
    XMVECTOR rows[4];
};
```

각 `XMVECTOR` lane은 한 행의 서로 다른 texel 네 개다.
BC1·BC7처럼 lane마다 서로 다른 블록을 담는 구조와 다르다.

```text
rows[0] = 자식 texel  0,  1,  2,  3
rows[1] = 자식 texel  4,  5,  6,  7
rows[2] = 자식 texel  8,  9, 10, 11
rows[3] = 자식 texel 12, 13, 14, 15
```

입력 BC4의 두 palette 규칙은 이 단계 전에 모두 `[0,1]`의 같은 scalar 공간으로 변환된다.

## 9. Six-interpolated 자식 후보 fitting

### 9.1 초기 endpoint

channel이 하나뿐이므로 RGB처럼 PCA 방향을 찾을 필요가 없다.
초깃값은 블록의 최대값과 최소값이다.

\[
e_0=\max_i x_i,\qquad e_1=\min_i x_i
\]

`e_0 > e_1` 순서는 six-interpolated Mode의 조건과도 맞는다.

### 9.2 가장 가까운 ramp 위치

현재 endpoint 사이에서 texel `x_i`의 연속적인 위치는 다음과 같다.

\[
r_i=7\frac{x_i-e_0}{e_1-e_0}
\]

이를 `[0,7]`로 제한하고 가장 가까운 정수로 반올림한다.

\[
k_i=\operatorname{round}(\operatorname{clamp}(r_i,0,7))
\]

least-squares에서 사용하는 weight는 다음과 같다.

\[
w_i=\frac{k_i}{7}
\]

### 9.3 endpoint 동시 해법

복원값을 다음 선형식으로 쓴다.

\[
\hat x_i=(1-w_i)e_0+w_i e_1=e_0+w_i d
\]

여기서 `d=e_1-e_0`이다.
고정된 `w_i`에 대해 다음 제곱 오차를 최소화한다.

\[
E=\sum_{i=1}^{16}(x_i-e_0-w_i d)^2
\]

미분값을 0으로 놓으면 다음 2x2 식이 된다.

\[
\begin{bmatrix}
N & \sum w_i\\
\sum w_i & \sum w_i^2
\end{bmatrix}
\begin{bmatrix}
e_0\\d
\end{bmatrix}
=
\begin{bmatrix}
\sum x_i\\
\sum w_i x_i
\end{bmatrix}
\]

`N=16`이며 `SolveBC4Endpoints()`가 이 식을 푼다.
모든 texel이 같은 ramp 위치에 놓여 행렬식이 사라지면 평균값을 두 endpoint에 사용한다.

현재 코드는 `index 배정 → endpoint 풀이`를 두 번 반복한 뒤 endpoint를 8-bit로 양자화한다.
반복 횟수 두 번은 현재 구현의 고정된 계산 예산이며 수렴할 때까지 반복하는 구조는 아니다.

## 10. Four-interpolated 자식 후보 fitting

four-interpolated Mode는 palette에 고정값 0과 1이 있다.
현재 구현은 0과 1에 매우 가까운 texel을 제외한 내부 값들의 최소와 최대를 endpoint 후보로 사용한다.

내부 값이 하나도 없으면 블록 전체 최소와 최대를 사용한다.
endpoint를 8-bit로 양자화하고 `red0 <= red1` 순서로 맞춘 뒤, 16개 texel 각각에 대해 palette 여덟 값을 모두 비교한다.

\[
index_i=\arg\min_{0\le k<8}(x_i-P_k)^2
\]

이 후보 생성에는 포화값과 내부값을 구분하기 위한 `1/512` 경계가 현재 코드에 존재한다.
이 값은 최종 Mode 선택을 위한 화질 임계값이 아니라 four-interpolated endpoint 초기 범위를 정하는 구현 상수다.

## 11. 두 후보는 실제 재구성 오차로 선택한다

six-interpolated 후보와 four-interpolated 후보를 각각 실제 palette로 복원해 제곱 오차를 합한다.

\[
E_6=\sum_{i=1}^{16}(x_i-\hat x_i^{(6)})^2
\]

\[
E_4=\sum_{i=1}^{16}(x_i-\hat x_i^{(4)})^2
\]

최종 블록은 다음처럼 선택한다.

\[
\text{result}=
\begin{cases}
\text{four-interpolated}, & E_4<E_6 \\
\text{six-interpolated}, & E_6\le E_4
\end{cases}
\]

따라서 “오차 차이가 0.05보다 크면 바꾼다” 같은 임의의 품질 threshold는 최종 선택에 없다.
동점이면 코드의 비교 순서에 따라 six-interpolated 후보가 선택된다.

## 12. mip 0, mip 1, mip 2 이후

### mip 0

원본 BC4 데이터를 그대로 복사한다.

### mip 1

`ProcessCompressedRowBC4()`가 부모 BC4 블록 네 개를 읽는다.

1. 각 부모의 palette와 index에서 사분면 평균 네 개를 구한다.
2. 네 부모의 결과를 자식 4x4 texel 16개에 배치한다.
3. 자식의 two palette Mode 후보를 만든다.
4. 실제 SSE가 더 작은 BC4 블록을 저장한다.
5. 부모별 전체 평균을 상위 mip용 배열에 보관한다.

### mip 2 이후

새로 만든 BC4 블록을 다시 해석하지 않고 원본 부모에서 얻은 scalar 평균 배열을 사용한다.
필요할 때 다음 2x2 평균으로 배열을 축소한다.

\[
m'=\frac{m_{00}+m_{10}+m_{01}+m_{11}}4
\]

홀수 크기 경계는 마지막 행이나 열을 반복하는 clamp-to-edge 방식이다.

이 구조는 생성된 BC4의 재양자화 오차가 다음 mip의 입력으로 계속 들어가는 경로를 피한다.
단, mip 0 자체의 BC4 압축 손실은 이미 입력값에 포함되어 있다.

## 13. 현재 연산 순서 요약

```text
1. BC4_UNORM 입력 확인
2. mip 0 압축 데이터 그대로 복사
3. 부모 BC4 블록 네 개 로드
4. red0 > red1이면 row weight table로 사분면 평균 계산
5. red0 <= red1이면 palette texel을 직접 복원해 사분면 평균 계산
6. 사분면 평균 16개를 자식 BC4TexelBlock에 배치
7. six-interpolated 후보 생성 및 실제 SSE 계산
8. four-interpolated 후보 생성 및 실제 SSE 계산
9. 더 작은 SSE의 8-byte BC4 블록 저장
10. mip 2 이후는 scalar 평균 피라미드에서 같은 encoder 사용
```

## 14. 논리 검증과 정확한 한계

### 구현과 수학이 일치하는 부분

- six-interpolated 사분면 식은 texel 네 개의 선형 palette 식을 묶은 정확한 대수 변환이다.
- four-interpolated 입력은 고정 palette 값 0과 1 때문에 별도로 직접 복원한다.
- 네 부모의 사분면 평균 배치는 8x8 영역에 대한 2x2 box-filter의 4x4 결과와 대응한다.
- 두 출력 palette 규칙의 선택은 각 후보의 실제 재구성 SSE를 비교한다.
- mip 0은 bit-for-bit 보존된다.
- BC4 UNORM은 이미 linear scalar 자료이므로 sRGB 전달 함수를 적용하지 않는다.

### 현재 구현이 보장하지 않는 것

- BC4 SNORM과 BC5는 지원하지 않는다.
- six-interpolated fitting은 두 번의 고정 refinement만 수행한다.
- four-interpolated endpoint는 내부 최소·최대로 정하며 모든 endpoint 조합을 탐색하지 않는다.
- 가능한 모든 8-bit endpoint와 3-bit index 조합의 전역 최소 오차를 보장하지 않는다.
- 최종 Mode 선택은 생성된 두 후보 중에서는 실제 SSE가 작은 후보를 고른다.
- 이 문서 작성 과정에서는 빌드와 실행 실험을 수행하지 않았다.

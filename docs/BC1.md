# BC1 Compression-Domain Mipmap Generation

이 문서는 `DirectXTex/DirectXTexCompressedMips.cpp` (CPU) 와
`DirectXTex/Shaders/BC1CompressedMips.hlsl` (GPU) 에 구현된 알고리즘을
수식 유도부터 함수 단위 데이터 흐름까지 서술한다.

모든 주장에는 근거를 붙였다. 근거는 세 종류다.

| 표기 | 의미 |
|---|---|
| `[코드]` | 해당 소스 파일의 라인 번호 |
| `[측정]` | 이 저장소의 실제 이미지로 실행해 얻은 수치 |
| `[유도]` | 이 문서 안에서 전개한 증명 |

---

## 목차

1. [문제 정의](#1-문제-정의)
2. [기존 방식: DirectXTex 파이프라인](#2-기존-방식-directxtex-파이프라인)
3. [우리 방식: 압축 도메인 파이프라인](#3-우리-방식-압축-도메인-파이프라인)
4. [수학적 기반](#4-수학적-기반)
5. [자료구조](#5-자료구조)
6. [함수별 상세 서술](#6-함수별-상세-서술)
7. [GPU 경로](#7-gpu-경로)
8. [정확성 검증](#8-정확성-검증)
9. [성능 특성](#9-성능-특성)
10. [정확한 부분과 근사인 부분](#10-정확한-부분과-근사인-부분)
11. [알려진 한계](#11-알려진-한계)

---

## 1. 문제 정의

### 1.1 입력과 출력

- **입력**: opaque BC1 로 압축된 mip 0 이미지 하나 (`BC1_UNORM` 또는 `BC1_UNORM_SRGB`)
- **출력**: mip 0 을 그대로 보존하고, mip 1 이하 전체를 BC1 로 채운 밉 체인

진입점은 두 개다.

```cpp
HRESULT DirectX::GenerateCompressedMipMaps(
    const Image& baseImage, size_t levels, ScratchImage& mipChain) noexcept;          // CPU

HRESULT DirectX::GenerateCompressedMipMaps(
    ID3D11Device* device, const Image& baseImage,
    size_t levels, ScratchImage& mipChain) noexcept;                                  // D3D11 DirectCompute
```

`[코드]` `DirectXTex/DirectXTex.h:849`, `DirectXTex/DirectXTex.h:856`

### 1.2 BC1 블록 형식

BC1 블록은 8바이트다.

```
struct D3DX_BC1 {
    uint16_t rgb[2];   // color0, color1  (RGB565)
    uint32_t bitmap;   // 4x4 = 16개의 2-bit selector
};
```

`[코드]` `DirectXTex/BC.h:160`

텍셀 $t \in \{0,\dots,15\}$ 는 행 우선(row-major)으로 배치되며,
$t = 4\,\text{row} + \text{col}$ 이고 selector 는 `bitmap` 의 비트 $2t,\,2t+1$ 에 있다.

디코딩 규칙은 `color0` 과 `color1` 의 대소 관계로 갈린다.

$$
c_0 = \mathrm{expand}_{565}(\text{color}_0), \qquad
c_1 = \mathrm{expand}_{565}(\text{color}_1)
$$

**4-color mode** ($\text{color}_0 > \text{color}_1$, opaque):

$$
c_2 = \left\lfloor \frac{2c_0 + c_1}{3} \right\rfloor, \qquad
c_3 = \left\lfloor \frac{c_0 + 2c_1}{3} \right\rfloor
$$

**3-color mode** ($\text{color}_0 \le \text{color}_1$):

$$
c_2 = \left\lfloor \frac{c_0 + c_1}{2} \right\rfloor, \qquad
c_3 = (0,0,0)\ \text{with}\ \alpha = 0\ \text{(투명)}
$$

`[코드]` `DirectXTexCompressedMips.cpp:314-366` (`BuildOpaqueLinearPaletteBC1Batch`)

RGB565 확장은 상위 비트를 하위로 복제한다.

$$
\mathrm{expand}_5(v) = (v \ll 3) \mathbin{|} (v \gg 2), \qquad
\mathrm{expand}_6(v) = (v \ll 2) \mathbin{|} (v \gg 4)
$$

`[코드]` `DirectXTexCompressedMips.cpp:238-257` (`DecodeRGB565Batch`)

이 나눗셈들은 **정수 나눗셈**이고 **코드 공간(code space)** 에서 수행된다.
이 점이 뒤에서 결정적이다 — 선형광에서 $1/3,\,2/3$ 보간을 하면 하드웨어와 다른 값이 나온다.

### 1.3 문제의 본질

압축된 mip 0 에서 압축된 mip 1..N 을 만들어야 한다.
자연스러운 방법은 "풀어서 → 축소해서 → 다시 압축"이지만,
그 경로에는 두 가지 비용이 있다.

1. **연산 비용**: 전체 픽셀을 재료화(materialize)하고 모든 레벨을 다시 BC1 인코딩
2. **품질 비용**: mip 0 이 이미 BC1 이었다면 재압축으로 2세대 양자화 오차

이 알고리즘의 목표는 **텍셀을 한 번도 재료화하지 않고**, 압축 데이터의
심볼(엔드포인트 + selector)만으로 필요한 통계량을 직접 계산하는 것이다.

---

## 2. 기존 방식: DirectXTex 파이프라인

비교 대상을 정확히 규정해야 우리 방식의 차이가 드러난다.
texconv 로 `-m 0 -if BOX -f BC1_UNORM_SRGB` 를 주었을 때의 실제 경로다.

### 2.1 단계

```
BC1 mip0 (.dds)
   │
   │  ① 압축 해제
   ▼
R8G8B8A8_UNORM_SRGB  mip0   (16 MB @ 4096²)
   │
   │  ② GenerateMipMaps — BOX 필터, 레벨마다 재귀
   ▼
R8G8B8A8_UNORM_SRGB  mip0..mip12   (약 21 MB)
   │
   │  ③ Compress — 모든 레벨을 BC1 로 인코딩 (mip 0 포함)
   ▼
BC1  mip0..mip12
```

`[코드]` `Texconv/texconv.cpp:3445` (② 호출), `Texconv/texconv.cpp:3780` (③ 호출)

### 2.2 ② 의 내부 — 재귀 축소

```cpp
for (size_t level = 1; level < levels; ++level)
{
    const Image* src  = mipChain.GetImage(level - 1, item, 0);   // ← 직전 레벨
    const Image* dest = mipChain.GetImage(level,     item, 0);
    ...
        LoadScanlineLinear(urow0, width, pSrc, rowPitch, src->format, filter);
        ...
        AVERAGE4(target[x], urow0[x2], urow1[x2], urow2[x2], urow3[x2]);
        ...
        StoreScanlineLinear(pDest, dest->rowPitch, dest->format, target, nwidth, filter);
}
```

`[코드]` `DirectXTex/DirectXTexMipmaps.cpp:1022-1073`

세 가지를 확인할 수 있다.

1. **재귀**: 레벨 $L$ 은 레벨 $L-1$ 에서 만들어진다. 원본에서 직접 만들지 않는다.
2. **선형광 평균**: `LoadScanlineLinear` / `StoreScanlineLinear` 는 sRGB ↔ 선형 변환을 포함한다.
3. **매 레벨 8비트 재양자화**: 중간 결과가 `R8G8B8A8_UNORM_SRGB` 로 저장된다.

3번의 영향은 실측할 수 있다. 원본 PNG 에서 부동소수 무한정밀도로 계산한
"이상적 체인"과 DirectXTex 가 만든 체인을 비교하면:

```
mip  1  (2048²)   58.93 dB
mip  3  ( 512²)   57.73 dB
mip  6  (  64²)   56.93 dB
mip  9  (   8²)   54.69 dB
mip 12  (   1²)   49.78 dB
```

`[측정]` `acg_asphalt_018_basecolor`, 참조 DDS vs float64 박스 체인

레벨이 깊어질수록 누적되어 **9 dB 열화**한다.
이는 뒤에서 논할 "무엇을 정답으로 볼 것인가" 문제와 직결된다.

### 2.3 ③ 의 내부 — BC1 인코더의 오차 함수

```cpp
HDRColorA X = (flags & BC_FLAGS_UNIFORM) ? HDRColorA(1.f, 1.f, 1.f, 1.f) : g_Luminance;
```

`[코드]` `DirectXTex/BC.cpp:82`

```cpp
const HDRColorA g_Luminance(0.2125f / 0.7154f, 1.0f, 0.0721f / 0.7154f, 1.0f);
```

`[코드]` `DirectXTex/BC.cpp:30`

`TEX_COMPRESS_UNIFORM` 을 주지 않으면 DirectXTex 는 색을
$(0.297,\ 1.0,\ 0.101)$ 로 스케일한 뒤 거리를 잰다.
즉 **휘도 가중 오차**를 최소화한다. 균등 채널 MSE(=일반적인 PSNR)를 최소화하지 않는다.

이 사실은 뒤의 §8.3 에서 품질 비교를 해석할 때 반드시 필요하다.

### 2.4 비용 구조

4096² 이미지 하나에 대해:

| 단계 | 처리량 |
|---|---|
| ① 압축 해제 | 16.8 M 텍셀 |
| ② 밉 생성 | 약 5.6 M 텍셀 (모든 레벨 합) |
| ③ 재인코딩 | 22.4 M 텍셀 = **1.4 M 개 BC1 블록** |

③ 이 지배적이다. 그리고 **mip 0 도 다시 인코딩된다** — 입력이 이미 BC1 이었는데도.

이 재인코딩 손실은 측정된다.

| | mip 0 PSNR |
|---|---|
| DirectXTex (DDS 입력) | 39.597 dB |
| 우리 방식 (DDS 입력) | **39.615 dB** |

`[측정]` 4096² 6장 평균. 차이 +0.018 dB 는 재압축 1세대분이다.

---

## 3. 우리 방식: 압축 도메인 파이프라인

### 3.1 전체 구조

```
BC1 mip0  (압축 상태 그대로, 절대 풀지 않음)
   │
   ├─────────────── 그대로 복사 ─────────────────────►  BC1 mip0  (비트 동일)
   │
   │  경로 A: 심볼 → 통계 → 인코딩
   ▼
BC1 mip1
   │
   │  경로 A 가 부산물로 뱉은 블록 평균
   ▼
M = 선형광 블록 평균 이미지  ( = mip 2 의 텍셀 격자)
   │
   ├── 경로 B ──►  BC1 mip2
   │
   ▼ 1/2 축소 (선형광, float)
  T₃ ── 경로 B ──►  BC1 mip3
   │
   ▼ 1/2
  T₄ ── 경로 B ──►  BC1 mip4  ...
```

`[코드]` `DirectXTexCompressedMips.cpp:1385-1440` (`GenerateCompressedMipMapsBC1`)

### 3.2 세 가지 설계 결정

**(a) mip 0 은 손대지 않는다.**

```cpp
memcpy_s(outputBase->pixels, outputBase->slicePitch, baseImage.pixels, baseImage.slicePitch);
```

`[코드]` `DirectXTexCompressedMips.cpp:1478`

`[측정]` 10/10 이미지에서 출력 mip 0 이 입력과 바이트 동일.

**(b) mip 1 은 압축 심볼에서 직접 만든다.**

부모 BC1 블록 4개의 엔드포인트와 selector 만으로 자식 블록의 4×4 텍셀 16개를
계산한다. 텍셀을 복원하지 않는다. 근거는 §4.2 의 정리 1.

**(c) mip 2 이상은 선형광 평균 피라미드에서 만든다.**

mip 1 을 만들면서 부산물로 얻은 블록 평균 이미지 $M$ 을 float32 로 유지하고
반복 축소한다. **BC1 재인코딩을 거치지 않으므로 레벨 간 오차 누적이 없다.**
근거는 §4.4 의 정리 3.

### 3.3 DirectXTex 와의 근본적 차이

| | DirectXTex | 우리 방식 |
|---|---|---|
| mip 0 | 재인코딩 | **비트 보존** |
| 중간 표현 | R8G8B8A8 (8비트) | 압축 심볼 → float32 |
| 레벨 간 관계 | 재귀 (레벨 $L-1$ → $L$) | mip 1 만 재귀, mip 2+ 는 $M$ 에서 |
| 레벨당 8비트 재양자화 | **있음** (누적) | **없음** |
| 텍셀 재료화 | 22.4 M 텍셀 | **0** |
| BC1 인코딩 대상 | 모든 레벨 (mip 0 포함) | mip 1 이하만 |
| BC1 오차 함수 | 휘도 가중 | 균등 (선형 RGB) |

---

## 4. 수학적 기반

### 4.1 표기

- $u_k \in [0,1]^3$ : BC1 팔레트 색 $k$ 를 **선형광**으로 변환한 값, $k \in \{0,1,2,3\}$
- $s(t) \in \{0,1,2,3\}$ : 텍셀 $t$ 의 selector
- 부모 블록 하나의 2×2 사분면 $Q \subset \{0,\dots,15\}$, $|Q| = 4$

### 4.2 정리 1 — Selector 히스토그램으로 2×2 평균을 정확히 계산

**주장.** 사분면 $Q$ 의 선형광 평균은 팔레트 색 4개와 히스토그램만으로 정확히 얻어진다.

$$
\bar{q} \;=\; \frac{1}{4}\sum_{t \in Q} u_{s(t)}
\;=\; \frac{1}{4}\sum_{k=0}^{3} n_k\, u_k,
\qquad n_k = \bigl|\{\,t \in Q : s(t) = k\,\}\bigr|
$$

**유도.** 합을 selector 값으로 분류하면 자명하다. 핵심은 $n_k$ 가
$\sum_k n_k = 4$ 를 만족하는 **네 개의 정수**뿐이라는 점이다.
즉 텍셀 4개를 복원해 더하는 대신, 개수만 세면 된다. $\square$

**의미.** 부모 블록 하나에서 16개 텍셀을 복원하는 대신
사분면 4개의 히스토그램(각 4개 정수)만 구하면 된다.
그리고 §6.3 에서 보듯 그 히스토그램은 **분기 없이 5개 명령**으로 얻어진다.

**검증.** `[측정]` 랜덤 BC1 블록 20,000 세트 × 4 lane × sRGB/UNORM 양쪽에 대해,
독립 스칼라 브루트포스 구현(16 텍셀 완전 복호 후 박스 평균)과 비교:

```
histogram mismatches : 0
max quadrant error   : 1.300e-07     (float32 정밀도 한계)
```

### 4.3 정리 2 — ANOVA 공분산 분해

자식 블록의 16개 텍셀을 크기 4인 그룹 $g \in \{p_{00}, p_{10}, p_{01}, p_{11}\}$ 넷으로 나눈다.
각 그룹의 평균을 $m_g$, 전체 평균을 $\mu$ 라 하자. 그룹 크기가 모두 같으므로

$$
\mu = \frac{1}{16}\sum_{\text{all}} x = \frac{1}{4}\sum_{g} m_g
$$

**주장.**

$$
\Sigma \;=\; \frac{1}{16}\sum_{\text{all}} (x-\mu)(x-\mu)^\top
\;=\; \underbrace{\frac{1}{4}\sum_g \Sigma^{\text{within}}_g}_{\text{그룹 내}}
\;+\; \underbrace{\frac{1}{4}\sum_g (m_g-\mu)(m_g-\mu)^\top}_{\text{그룹 간}}
$$

여기서 $\Sigma^{\text{within}}_g = \frac{1}{4}\sum_{i \in g}(x_i - m_g)(x_i - m_g)^\top$ 이다.

**유도.** $x_i - \mu = (x_i - m_g) + (m_g - \mu)$ 로 쪼개고 전개한다.

$$
\sum_{g}\sum_{i \in g}(x_i-\mu)(x_i-\mu)^\top
= \sum_g \Bigl[\underbrace{\sum_{i \in g}(x_i-m_g)(x_i-m_g)^\top}_{=\,4\Sigma^{\text{within}}_g}
+ 4(m_g-\mu)(m_g-\mu)^\top + \text{cross}\Bigr]
$$

교차항은

$$
\text{cross} = \Bigl(\sum_{i \in g}(x_i - m_g)\Bigr)(m_g-\mu)^\top + (m_g-\mu)\Bigl(\sum_{i \in g}(x_i-m_g)\Bigr)^\top = 0
$$

$\sum_{i \in g}(x_i - m_g) = 0$ 이기 때문이다. 양변을 16으로 나누면 주장이 나온다. $\square$

**왜 이렇게 하는가.** 그룹 평균 $m_g$ 는 **어차피 계산해야 하는 값**이다 —
그것이 곧 부모 블록의 평균이고, mip 2 이상을 위한 평균 이미지 $M$ 의 텍셀이다.
그룹 내 공분산도 사분면 평균 4개에서 바로 나온다.
따라서 이 분해는 추가 비용 없이 전체 공분산을 준다.

**검증.** `[측정]` 16개 텍셀로 직접 계산한 모집단 공분산과 비교:

```
max covariance error : 4.698e-08
max block-mean error : 1.477e-07
```

### 4.4 정리 3 — 평균 피라미드의 결합성 (skip-level)

블록 평균 이미지를 $M$ 이라 하자. $M$ 의 텍셀 하나는 mip 0 의 4×4 영역에 대응하므로,
$M$ 의 격자는 **mip 2 의 텍셀 격자와 정확히 일치**한다.

$$
M_{\text{width}} = \left\lceil \frac{W}{4} \right\rceil, \qquad
M_{\text{height}} = \left\lceil \frac{H}{4} \right\rceil
$$

레벨 $L \ge 2$ 의 목표 텍셀은 $M$ 을 $s_L = 2^{L-2}$ 크기 박스로 평균한 값으로 정의된다.

$$
T_L(x,y) = \frac{1}{s_L^2}\sum_{j=0}^{s_L-1}\sum_{i=0}^{s_L-1} M(s_L x + i,\; s_L y + j)
$$

**주장.** 이 값은 직전 레벨을 2×2 로 평균한 것과 같다.

$$
T_L(x,y) = \frac{1}{4}\sum_{j=0}^{1}\sum_{i=0}^{1} T_{L-1}(2x+i,\; 2y+j),
\qquad T_2 \equiv M
$$

**유도.** 선형 도메인의 박스 평균은 결합적이다.
$s_L \times s_L$ 영역은 $s_{L-1} \times s_{L-1}$ 영역 4개로 정확히 분할되고,
네 부분의 원소 수가 같으므로 부분 평균의 평균이 전체 평균과 같다. $\square$

**전제 조건.** 2의 거듭제곱 해상도. 홀수 폭이 생기면 clamp-to-edge 때문에
각 부분의 유효 원소 수가 달라져 등식이 미세하게 깨진다 (§11 참조).

**의미 — 이것이 가장 중요한 성질이다.**

재귀 파이프라인에서는 레벨마다 인코딩 오차가 누적된다.

$$
\hat y_1 \to \hat y_2 \to \hat y_3 \to \cdots
$$

우리 방식은 mip 1 만 압축 데이터에서 만들고,
mip 2 이상은 전부 float32 $M$ 에서 내려온다.
**중간에 BC1 인코딩이 끼어들지 않으므로 오차 누적 항이 0 이다.**

$$
\|T_L - \hat y_L\|^2 = (\text{레벨 } L \text{ 의 인코딩 오차만}), \qquad (\text{누적 항}) = 0
$$

부수적으로 비용도 등비수열로 줄어든다. 레벨당 상수 비용이 아니라
$1 + \frac14 + \frac1{16} + \cdots \to \frac43$ 이다.

### 4.5 왜 손상된 입력에서 출발해도 격차가 작은가

이 방법은 이미 BC1 로 양자화된 mip 0 에서 출발한다.
직관적으로는 반드시 불리해야 한다. 실제로 그렇고, 그 크기를 측정할 수 있다.

우리 인코더가 **실제로 받는 입력**(= 디코드된 mip 0 을 선형광에서 박스 축소한 것)이
참조 대비 갖는 PSNR:

```
mip  0   4096²   34.76 dB     ← mip 0 의 BC1 오차를 100% 물려받음
mip  1   2048²   39.52  (+4.76)
mip  2   1024²   43.26  (+3.74)
mip  3    512²   48.14  (+4.88)
mip  4    256²   51.25  (+3.11)   ← 핸디캡 소멸
mip  5    128²   52.18            ← 최고점
mip  9      8²   50.11            ← 더 좋아지지 않음
mip 12      1²   46.59            ← 오히려 나빠짐
```

`[측정]` `acg_asphalt_018_basecolor`. 같은 레벨의 인코딩 오차는 35–42 dB 수준으로 거의 일정.

**해석.** BC1 양자화 오차는 대부분 블록 간 상관이 낮은 고주파 성분이다.
2×2 박스 평균을 한 번 거칠 때마다 그 분산이 크게 줄어들어,
mip 4 쯤에는 약 16 dB(≈40배) 감쇠해 인코딩 오차에 비해 무시할 수준이 된다.

그러나 감쇠에는 바닥이 있다. 잡음 성분은 평균으로 사라지지만
**블록 평균의 계통 편향(DC bias)** 은 아무리 축소해도 남는다.
그래서 drift 는 mip 5 에서 정점을 찍고 다시 내려간다.
인코딩 오차가 작아지는 아주 작은 밉(2², 1²)에서는
그 잔여 편향이 다시 지배적이 된다.

**결론.** 구조적 핸디캡은 실재하며 **mip 1–2 에서 약 0.2 dB** 로 나타난다.
mip 4 쯤 소멸하고, 아주 작은 밉에서 DC 편향으로 재등장한다.

---

## 5. 자료구조

모든 구조체는 4개의 **서로 독립적인 BC1 작업**을 SIMD lane 에 담는다.
`XMVECTOR` 의 x/y/z/w 는 RGBA 채널이 아니라 **블록 인덱스**다.
채널은 별도 변수로 분리된 Structure-of-Arrays 배치다.

```
lane x → 블록 0        RGB 채널은 별도 멤버 (q0R, q0G, q0B)
lane y → 블록 1
lane z → 블록 2
lane w → 블록 3
```

| 구조체 | 내용 | 코드 |
|---|---|---|
| `BC1BlockBatch` | `color0`, `color1`, `selectors` — 4블록의 원시 비트 | `:95` |
| `RGB8Batch` | RGB8 코드값 3채널 | `:103` |
| `LinearPaletteBC1Batch` | 팔레트 4색 × 3채널 = 12개 `XMVECTOR` | `:111` |
| `QuadrantMeansBatch` | 사분면 평균 4개 × 3채널 | `:131` |
| `LinearRGBBatch` | 선형 RGB 1개 | `:151` |
| `CovarianceMatrixBatch` | 대칭 3×3 의 6성분 (`rr,gg,bb,rg,rb,gb`) | `:159` |
| `ParentStatisticsBatch` | 그룹 평균 + 그룹 내 공분산 | `:171` |
| `SourceBlockMeansBatch` | 소스 블록 4개의 평균 (출력 전용) | `:178` |
| `EndpointPairBatch` | 엔드포인트 2개 × 3채널 | `:187` |
| `LeastSquaresAccumulatorBatch` | $\Sigma w$, $\Sigma w^2$, $\Sigma w x$ | `:209` |
| `LinearBlockMean` | scalar float 3개 — 평균 이미지의 텍셀 (12 B) | `:217` |

공분산이 6성분인 이유: 대칭이므로 상삼각만 저장하면 된다.

$$
\begin{pmatrix} rr & rg & rb \\ rg & gg & gb \\ rb & gb & bb \end{pmatrix}
$$

---

## 6. 함수별 상세 서술

### 6.1 `LoadBC1BlockBatch` — 4블록을 lane 에 적재

`[코드]` `:225-236`

```cpp
batch.color0    = XMVectorSetInt(blocks[0].rgb[0], blocks[1].rgb[0], blocks[2].rgb[0], blocks[3].rgb[0]);
batch.color1    = XMVectorSetInt(blocks[0].rgb[1], ...);
batch.selectors = XMVectorSetInt(blocks[0].bitmap, ...);
```

스칼라 BC1 블록 4개를 3개의 `XMVECTOR` 로 전치한다.
이후 모든 연산이 lane 4개에 대해 동시에 진행된다.

### 6.2 `DecodeRGB565Batch` — RGB565 → RGB8

`[코드]` `:238-257`

$$
r_5 = (p \gg 11) \wedge 31, \quad
g_6 = (p \gg 5) \wedge 63, \quad
b_5 = p \wedge 31
$$
$$
R = (r_5 \ll 3) \mathbin{|} (r_5 \gg 2), \quad
G = (g_6 \ll 2) \mathbin{|} (g_6 \gg 4), \quad
B = (b_5 \ll 3) \mathbin{|} (b_5 \gg 2)
$$

비트 복제 확장은 $v \mapsto \lfloor 255v/31 \rfloor$ 의 정확한 근사이며
D3D 사양 및 하드웨어와 일치한다.

### 6.3 `Count2x2Regions` — SWAR 사분면 카운터

이 함수가 이 알고리즘에서 가장 밀도 높은 부분이다. 유도를 전부 적는다.

`[코드]` `:293-311`

**입력.** `flags` 는 32비트로, 텍셀 $t$ 가 특정 selector 값을 가지면
비트 $2t$ 가 1 인 비트마스크다. (홀수 비트는 항상 0)

**1단계 — 수평 합산**

```cpp
const XMVECTOR horizontalMask = XMVectorReplicateInt(0x11111111u);
horizontal = (flags & 0x11111111) + ((flags >> 2) & 0x11111111);
```

마스크 `0x11111111` 은 비트 $0,4,8,12,\dots$ 즉 **니블 경계**를 고른다.

- `flags & 0x11111111` 의 니블 $m$ (비트 $4m$) = 원래 비트 $4m$ = **텍셀 $2m$**
- `(flags >> 2) & 0x11111111` 의 니블 $m$ = 원래 비트 $4m+2$ = **텍셀 $2m+1$**

따라서

$$
\text{horizontal 의 니블 } m \;=\; [\,\text{텍셀 } 2m\,] + [\,\text{텍셀 } 2m{+}1\,] \;\in\; \{0,1,2\}
$$

텍셀은 행 우선이므로 $2m$ 과 $2m+1$ 은 **가로로 인접**하다.

**2단계 — 수직 합산**

```cpp
const XMVECTOR verticalMask = XMVectorReplicateInt(0x00FF00FFu);
return (horizontal & 0x00FF00FF) + ((horizontal >> 8) & 0x00FF00FF);
```

`0x00FF00FF` 는 비트 0–7 (니블 0,1) 과 비트 16–23 (니블 4,5) 을 남긴다.
`>> 8` 은 니블 2,3 을 0,1 자리로, 니블 6,7 을 4,5 자리로 옮긴다.

$$
\begin{aligned}
\text{결과 니블 0 (비트 0–3)} &= (t_0+t_1) + (t_4+t_5) &&\to \text{좌상단 2×2} \\
\text{결과 니블 1 (비트 4–7)} &= (t_2+t_3) + (t_6+t_7) &&\to \text{우상단 2×2} \\
\text{결과 니블 4 (비트 16–19)} &= (t_8+t_9) + (t_{12}+t_{13}) &&\to \text{좌하단 2×2} \\
\text{결과 니블 5 (비트 20–23)} &= (t_{10}+t_{11}) + (t_{14}+t_{15}) &&\to \text{우하단 2×2}
\end{aligned}
$$

**자리올림이 없는 이유.** 각 니블의 최댓값은 4 이고 $4 < 16$ 이다.
따라서 니블 간 오염이 발생하지 않는다.

**결과 비트 배치.**

```
bits  0..3  → 사분면 0 (좌상)
bits  4..7  → 사분면 1 (우상)
bits 16..19 → 사분면 2 (좌하)
bits 20..23 → 사분면 3 (우하)
```

이 오프셋 $\{0,4,16,20\}$ 이 `ExtractQuadrantWeight<BitOffset>` 의 템플릿 인자다.
`[코드]` `:396-406`

**총 비용.** 마스킹 4회 + 시프트 2회 + 덧셈 2회 = **8 명령**으로
4블록 × 사분면 4개 × selector 1개의 카운트를 동시에 얻는다.

### 6.4 `Extract2x2SelectorHistograms` — 4개 selector 값을 분리

`[코드]` `:368-394`

selector 는 2비트이므로 low/high 비트로 분해한다.

```cpp
const XMVECTOR lowBitMask = XMVectorReplicateInt(0x55555555u);
const XMVECTOR lowBits  = selectors & 0x55555555;
const XMVECTOR highBits = (selectors >> 1) & 0x55555555;
```

각 selector 값에 대한 지시 비트마스크는 논리 연산 하나씩이다.

| selector | low | high | flags 식 | 코드 |
|---|---|---|---|---|
| 0 | 0 | 0 | $\lnot(\text{low} \lor \text{high}) \wedge \text{mask}$ | `XMVectorNorInt(low, high) & mask` |
| 1 | 1 | 0 | $\text{low} \wedge \lnot\text{high}$ | `XMVectorAndCInt(low, high)` |
| 2 | 0 | 1 | $\text{high} \wedge \lnot\text{low}$ | `XMVectorAndCInt(high, low)` |
| 3 | 1 | 1 | $\text{low} \wedge \text{high}$ | `XMVectorAndInt(low, high)` |

그 다음 각각에 `Count2x2Regions` 를 적용한다.

**결과.** 부모 블록 하나에서 히스토그램 4개(`histogram0..3`)를 얻는다.
각 히스토그램의 니블 4개가 사분면별 개수다. 즉

$$
n_{q,k} = (\text{histogram}_k \gg \text{offset}_q) \wedge \text{0xF},
\qquad \sum_{k=0}^{3} n_{q,k} = 4 \ \ \forall q
$$

**16개 텍셀을 복원하지 않고 16개의 정수로 요약이 끝났다.**

### 6.5 `BuildOpaqueLinearPaletteBC1Batch` — 하드웨어 팔레트 재현

`[코드]` `:314-366`

두 엔드포인트를 RGB8 로 확장한 뒤, **코드 공간에서 정수 보간**한다.

```cpp
const XMVECTOR is4Color = GreaterInt32(packedColor0, packedColor1);

c4_2 = DividePaletteSumBy3(2*c0 + c1);      // 4-color mode
c4_3 = DividePaletteSumBy3(c0 + 2*c1);
c3_2 = (c0 + c1) >> 1;                       // 3-color mode
c3_3 = 0;

color2 = select(c3_2, c4_2, is4Color);
color3 = select(0,    c4_3, is4Color);
```

3-color 모드 지원은 분기 없이 `XMVectorSelect` 로 처리된다.

**`DividePaletteSumBy3` — 나눗셈 제거**

`[코드]` `:52-64`

SSE2 에는 32비트 정수 나눗셈도 곱셈도 없다. 그래서 시프트-덧셈으로 구현한다.

$$
\left\lfloor \frac{v}{3} \right\rfloor = \left\lfloor \frac{683\,v}{2048} \right\rfloor
\qquad \text{for } 0 \le v \le 765
$$

$683 = 2^9 + 2^7 + 2^5 + 2^3 + 2^1 + 2^0$ 이므로

```cpp
product = (v<<9) + (v<<7) + (v<<5) + (v<<3) + (v<<1) + v;
return product >> 11;
```

**유효 범위 확인.** 입력은 $2c_0 + c_1$ 또는 $c_0 + 2c_1$ 이고 $c \le 255$ 이므로
최댓값은 $3 \times 255 = 765$. 이 범위에서 등식이 정확히 성립한다.

**마지막에 선형 변환**

```cpp
palette.c0R = ConvertRGB8ToLinearBatch<IsSrgb>(color0.r);
...
```

`[코드]` `:347-364`

**순서가 중요하다.** 엔드포인트만 선형으로 바꾼 뒤 선형 공간에서 $1/3, 2/3$
보간을 하면 하드웨어와 다른 색이 나온다.
반드시 코드 공간에서 정수 보간을 끝낸 뒤 4색 각각을 선형으로 변환해야 한다.

- sRGB: 256엔트리 LUT (`GetSrgb8ToLinearTable`, `[코드]` `:66-88`)
- UNORM: $v/255$ (`NormalizeUNorm8Batch`, `[코드]` `:260-264`)

LUT 는 `XMColorSRGBToRGB` 의 기준 곡선으로 만들어지며,
함수 지역 static 이라 스레드 안전하게 한 번만 초기화된다.

### 6.6 `ComputeParentQuadrantMeansBatch` — 정리 1의 구현

`[코드]` `:454-492`

$$
\bar{q}_j = \sum_{k=0}^{3} \frac{n_{j,k}}{4}\, u_k
$$

`ExtractQuadrantWeight<BitOffset>` 이 $n_{j,k}/4$ 를 만든다.

```cpp
const XMVECTOR counts = (histogram >> BitOffset) & 0xF;
return XMVectorScale(XMConvertVectorUIntToFloat(counts, 0), 0.25f);
```

`ComputePaletteMean` 이 가중합을 FMA 로 누적한다. `[코드]` `:408-435`

```cpp
result.r = palette.c0R * weight0;
result.r = palette.c1R * weight1 + result.r;
result.r = palette.c2R * weight2 + result.r;
result.r = palette.c3R * weight3 + result.r;
```

결과인 사분면 평균 16개가 자식 블록의 4×4 텍셀에 다음과 같이 대응한다.

```
p00.q0  p00.q1  p10.q0  p10.q1
p00.q2  p00.q3  p10.q2  p10.q3
p01.q0  p01.q1  p11.q0  p11.q1
p01.q2  p01.q3  p11.q2  p11.q3
```

즉 texel 인덱스로는 `p00 → {0,1,4,5}`, `p10 → {2,3,6,7}`,
`p01 → {8,9,12,13}`, `p11 → {10,11,14,15}` 이다.
이 배치는 `PackAndReallocateSelectors` 의 `AssignNearestSelector<N>` 템플릿 인자와
정확히 일치한다. `[코드]` `:1029-1049`

**밉 다운샘플링이 여기서 이미 끝났다.** 그리고 sRGB 코드값 평균이 아니라
**선형광 평균**이다 — DirectXTex 의 `LoadScanlineLinear` 와 같은 색공간 처리를 한다.

### 6.7 `ComputeBlockMeansBatch` — 블록 평균

`[코드]` `:494-515`

$$
\bar{b} = \frac{\bar{q}_0 + \bar{q}_1 + \bar{q}_2 + \bar{q}_3}{4}
$$

각 $\bar q$ 가 2×2 텍셀의 평균이므로 이 값은 **16개 텍셀 전체의 선형광 평균과 정확히 같다.**
$M$ 의 텍셀이 되는 값이며, 이미 계산된 중간값의 평균이라 추가 비용이 사실상 없다.

### 6.8 `ComputeParentStatisticsBatch` — 그룹 통계

`[코드]` `:517-579`

$$
m_g = \frac{1}{4}\sum_{i=0}^{3} \bar q_i, \qquad
\Sigma^{\text{within}}_g = \frac{1}{4}\sum_{i=0}^{3} (\bar q_i - m_g)(\bar q_i - m_g)^\top
$$

6성분을 각각 FMA 4회 + 스케일 1회로 누적한다.

```cpp
result.withinCovariance.rr  = d0R * d0R;
result.withinCovariance.rr  = d1R * d1R + result.withinCovariance.rr;
result.withinCovariance.rr  = d2R * d2R + result.withinCovariance.rr;
result.withinCovariance.rr  = d3R * d3R + result.withinCovariance.rr;
result.withinCovariance.rr *= 0.25f;
```

### 6.9 `AccumulateBetweenParentCovariance` — 그룹 간 공분산

`[코드]` `:594-611`

$$
\Sigma^{\text{between}} \mathrel{+}= \frac{1}{4}(m_g - \mu)(m_g - \mu)^\top
$$

4번 호출해 누적하면 정리 2의 두 번째 항이 된다.
$1/4$ 스케일이 항마다 곱해져 있어 별도 정규화가 필요 없다.

### 6.10 `ComputeChildBlockMoments` — 정리 2의 조립

`[코드]` `:613-671`

```
① ComputeParentStatisticsBatch × 4              → m_g, Σ^within_g
② sourceMeans 에 m_g 4개를 내보냄               → 평균 이미지 M 용
③ μ = MeanFourVectors(m_00, m_10, m_01, m_11)
④ within = MeanFourVectors(Σ^within × 4)
⑤ between = Σ_g AccumulateBetweenParentCovariance
⑥ Σ = between + within
```

②가 §4.4 의 skip-level 을 가능하게 하는 부산물이다.
공분산 계산 도중 어차피 나오는 값을 밖으로 빼는 것뿐이라 추가 연산이 없다.

### 6.11 `ComputeInitialEndpointsPCA` — 초기 엔드포인트

`[코드]` `:700-759`

**1) 거듭제곱 반복 1회**

$$
v_0 = \tfrac{1}{\sqrt3}(1,1,1), \qquad
v_1 = \Sigma v_0, \qquad
\hat a = \frac{v_1}{\sqrt{\|v_1\|^2 + \varepsilon}}, \quad \varepsilon = 10^{-20}
$$

전개하면

$$
v_{1,r} = \Sigma_{rr} a_r + \Sigma_{rg} a_g + \Sigma_{rb} a_b, \quad \text{(g, b 도 동일)}
$$

$\varepsilon$ 은 완전 평탄한 블록($\Sigma = 0$)에서 0 벡터를 정규화하는 것을 막는다.
그 경우 축은 0 이 되고, 모든 투영이 0 이라 두 엔드포인트가 평균으로 수렴한다 —
정의된 동작이다.

**2) 투영 범위**

$$
t_i = \hat a \cdot (x_i - \mu), \qquad
t_{\min} = \min_i t_i, \quad t_{\max} = \max_i t_i
$$

16개 사분면 평균 전부에 대해 계산한다. `[코드]` `:673-698`

**3) 초기 엔드포인트**

$$
p_0 = \mu + t_{\min}\hat a, \qquad p_1 = \mu + t_{\max}\hat a
$$

이 단계는 **최종 답이 아니라 다음 단계의 초기값**이다.
거듭제곱 반복이 1회뿐인 것도 그래서 허용된다.

### 6.12 `OptimizeEndpointsLeastSquares` — 이산 가중치 최소제곱

`[코드]` `:800-880`

**모델.** BC1 은 팔레트를 선분 위 4점으로 만든다.

$$
x_i \approx a + w_i\, d, \qquad w_i \in \left\{0,\ \tfrac13,\ \tfrac23,\ 1\right\}
$$

**1) 가중치 배정 (고정)**

현재 선분에 각 샘플을 투영하고 가장 가까운 BC1 보간 위치로 스냅한다.

$$
w_i = \frac{1}{3}\left\lfloor 3 \cdot \mathrm{clamp}\!\left(\frac{(x_i - p_0)\cdot d}{\|d\|^2 + 10^{-12}},\, 0,\, 1\right) \right\rceil
$$

`[코드]` `:761-786` (`AccumulateLeastSquaresSample`)

**2) 정규방정식**

$w_i$ 를 고정하고 $\sum_i \|x_i - a - w_i d\|^2$ 를 $a, d$ 에 대해 최소화한다.
$a, d$ 각각으로 편미분해 0 으로 두면

$$
\begin{pmatrix} n & S_1 \\ S_1 & S_2 \end{pmatrix}
\begin{pmatrix} a \\ d \end{pmatrix}
=
\begin{pmatrix} \sum_i x_i \\ \sum_i w_i x_i \end{pmatrix},
\qquad
n = 16,\quad S_1 = \sum_i w_i,\quad S_2 = \sum_i w_i^2
$$

행렬식과 해는

$$
\det = 16 S_2 - S_1^2
$$

$$
a = \frac{S_2 \sum_i x_i - S_1 \sum_i w_i x_i}{\det}, \qquad
d = \frac{16 \sum_i w_i x_i - S_1 \sum_i x_i}{\det}
$$

코드에서 $\sum_i x_i$ 는 $16\mu$ 로 계산된다 (`total`).

```cpp
result.p0.r = (weightSquaredSum * total.r - weightSum * weighted.r) * inverseDeterminant;
direction.r = (16.0f * weighted.r - weightSum * total.r) * inverseDeterminant;
result.p1.r = result.p0.r + direction.r;
```

**3) 특이 케이스**

$\det < 10^{-6}$ 이면 (모든 $w_i$ 가 같은 경우 — 완전 평탄 블록)
두 엔드포인트를 모두 $\mu$ 로 설정한다. 분기 없이 `XMVectorSelect` 로 처리한다.

```cpp
const XMVECTOR singularMask = XMVectorLess(determinant, XMVectorReplicate(1e-6f));
result.p0.r = XMVectorSelect(result.p0.r, mean.r, singularMask);
```

**왜 이 방법인가.** PCA 는 분산이 최대인 축을 주지만, BC1 은 그 축 위의
**4개 이산점**만 쓸 수 있다. 실제로 쓰일 4개 위치를 먼저 정해놓고
그 배치에 최적인 선분을 다시 푸는 것이 PCA 범위를 그대로 쓰는 것보다 낫다.

### 6.13 `ConvertEndpointsToCodeSpace` — 코드 공간 변환

`[코드]` `:910-929`

$$
q_i = \begin{cases}
\mathrm{OETF}_{\text{sRGB}}(p_i) & \text{sRGB} \\
p_i & \text{UNORM}
\end{cases}
$$

$$
\mathrm{OETF}_{\text{sRGB}}(u) =
\begin{cases}
12.92\,u & u \le 0.0031308 \\
1.055\,u^{1/2.4} - 0.055 & \text{otherwise}
\end{cases}
$$

`[코드]` `:895-908` (`LinearToSrgbBatch`)

**왜 필요한가.** 엔드포인트 적합은 선형광에서 했지만
BC1 이 저장하는 것은 **코드값**이다. 다음 단계의 RGB565 양자화가
$[0,1]$ 코드값을 가정하므로 반드시 전달 곡선 위로 옮겨야 한다.
UNORM 팔레트는 코드값이 곧 선형값이므로 변환이 필요 없다.

> **이력.** 이전 버전에는 고정 $4/9$ chord-curve 보정이 있었다.
> BC1 이 코드 공간에서 보간하므로 선형광에서 적합한 직선이 실제 디코드 현(chord)과
> 어긋나는 것을 보정하려는 heuristic 이었다.
> 계수 $4/9$ 가 selector 분포가 균등하다는 가정에서 나왔는데 실제 블록은 그렇지 않고,
> 실측에서 품질이 오히려 나빠져 제거했다.
> 재도입하려면 블록별 selector 가중치로 보정량을 유도해야 한다.

### 6.14 `QuantizeUNormBatch` / `EnforceOpaqueEndpointOrder`

`[코드]` `:980-1006`

**양자화**

$$
Q_m(u) = \left\lfloor m \cdot \mathrm{clamp}(u, 0, 1) \right\rceil,
\qquad m_R = m_B = 31,\ m_G = 63
$$

```cpp
color0 = (r0 << 11) | (g0 << 5) | b0;
```

**opaque 순서 강제**

BC1 은 $\text{color}_0 > \text{color}_1$ 일 때만 4-color 모드다.
같으면 최소 이동으로 분리한다.

$$
\text{if } \text{color}_0 = \text{color}_1:
\begin{cases}
\text{color}_1 \mathrel{-}= 1 & \text{if } (\text{color}_1 \wedge 31) > 0 \\
\text{color}_0 \mathrel{+}= 1 & \text{otherwise}
\end{cases}
$$

$\pm 1$ 은 blue 필드의 LSB 하나만 바꾸므로 색 이동이 최소다.
(blue 가 0 이면 감소 대신 증가시켜 언더플로를 피한다.)

그 다음 필요하면 교환한다.

$$
\text{if } \text{color}_1 > \text{color}_0: \ \mathrm{swap}
$$

전부 `XMVectorSelect` 기반 branchless 다.

### 6.15 `PackAndReallocateSelectors` — selector 재배정

`[코드]` `:1008-1055`

**핵심.** 양자화 **전**의 이상적 엔드포인트나 이상적 선형 보간으로
selector 를 고르면 안 된다. 양자화된 엔드포인트로 팔레트를 **다시 만들어야** 한다.

```cpp
EnforceOpaqueEndpointOrder(result.color0, result.color1);
const LinearPaletteBC1Batch palette = BuildOpaqueLinearPaletteBC1Batch<IsSrgb>(result.color0, result.color1);
```

이유: RGB565 양자화가 팔레트 4색 전체를 이동시키고, 중간 두 색은
정수 나눗셈까지 거치므로 이동량이 균일하지 않다.
실제 디코더가 만들 색에 대해 최근접을 다시 계산해야 오차가 최소가 된다.

$$
s(t) = \arg\min_{k \in \{0,1,2,3\}} \bigl\| x_t - u_k \bigr\|^2
$$

`AssignNearestSelector<TexelIndex>` 가 텍셀 하나를 처리한다. `[코드]` `:964-978`

```cpp
packedSelectors |= selector << (TexelIndex * 2);
```

`FindBestSelector` 는 토너먼트 방식으로 분기 없이 argmin 을 구한다. `[코드]` `:932-946`

```cpp
index01  = select(0, 1, d1 < d0);
index23  = select(2, 3, d3 < d2);
min01    = min(d0, d1);
min23    = min(d2, d3);
return select(index01, index23, min23 < min01);
```

### 6.16 `ProcessCompressedRowBC1` — 경로 A (mip 1)

`[코드]` `:1154-1256`

목적지 블록 $(x, y)$ 에 대해 부모 4개를 clamp 하여 읽는다.

$$
\begin{aligned}
x_0 &= \min(2x,\ W_b - 1), & x_1 &= \min(x_0+1,\ W_b - 1) \\
y_0 &= \min(2y,\ H_b - 1), & y_1 &= \min(y_0+1,\ H_b - 1)
\end{aligned}
$$

```cpp
p00Blocks[lane] = sourceRow0[sourceX0];
p10Blocks[lane] = sourceRow0[sourceX1];
p01Blocks[lane] = sourceRow1[sourceX0];
p11Blocks[lane] = sourceRow1[sourceX1];
```

목적지 블록 4개를 lane 에 모은다.
마지막 묶음에서 부족하면 마지막 유효 작업을 복제하고
(`std::min(lane, validLanes - 1)`), 저장할 때 유효 lane 만 기록한다.

블록 평균 저장은 clamp 로 인한 중복 기록을 피한다.

```cpp
StoreBlockMean(means.p00, lane, sourceBlockMeans[sourceY0 * sourceBlockWidth + sourceX0]);
if (sourceX1 != sourceX0) StoreBlockMean(means.p10, ...);
if (sourceY1 != sourceY0) { StoreBlockMean(means.p01, ...); ... }
```

### 6.17 `DownsampleLinearMeanRow` — 평균 피라미드 축소

`[코드]` `:1289-1319`

$$
T_{L}(x,y) = \frac{1}{4}\bigl[T_{L-1}(2x,2y) + T_{L-1}(2x{+}1,2y) + T_{L-1}(2x,2y{+}1) + T_{L-1}(2x{+}1,2y{+}1)\bigr]
$$

폭/높이가 홀수면 마지막 행·열을 반복한다 (clamp-to-edge).

```cpp
const size_t sourceX0 = destinationX * 2;
const size_t sourceX1 = std::min(sourceX0 + 1, sourceWidth - 1);
```

$x_0$ 에 clamp 가 없는 이유: 목적지 폭이 $\lceil W/2 \rceil$ 이므로
$2x \le 2(\lceil W/2\rceil - 1) < W$ 가 항상 성립한다. 짝수·홀수 모두 안전하다.

### 6.18 `ProcessLinearRowBC1` — 경로 B (mip 2 이상)

`[코드]` `:1345-1385`

압축 블록을 전혀 읽지 않는다. 평균 이미지에서 자식 텍셀을 직접 가져온다.

$$
\begin{aligned}
\text{target}_x &= 4\,\text{block}_x + \text{local}_x \\
\text{target}_y &= 4\,\text{row} + \text{local}_y
\end{aligned}
$$

읽은 값을 경로 A 와 **동일한 슬롯 배치**로 넣는다.

$$
\text{region} = 2\left\lfloor \frac{\text{local}_y}{2} \right\rfloor + \left\lfloor \frac{\text{local}_x}{2} \right\rfloor,
\qquad
\text{sample} = 2\,(\text{local}_y \bmod 2) + (\text{local}_x \bmod 2)
$$

```cpp
const size_t region = ((localY >> 1) << 1) + (localX >> 1);
const size_t sample = ((localY & 1) << 1) + (localX & 1);
SetQuadrantSample(regions[region], sample, lane, color);
```

`region` 은 `p00/p10/p01/p11` 에, `sample` 은 `q0/q1/q2/q3` 에 대응한다.
이 덕분에 **`EncodeLinearBlocksBC1Batch` 는 두 경로를 구분할 필요가 없다.**

경로 B 에서 `QuadrantMeansBatch` 의 값은 "평균"이 아니라 텍셀 값 자체다.
그래도 §4.3 의 ANOVA 분해는 그대로 유효하다 —
16개를 4개씩 묶어 계산하는 것과 직접 계산하는 것이 같기 때문이다.

### 6.19 `GenerateCompressedMipMapsBC1` — 전체 스케줄링

`[코드]` `:1385-1440`

```cpp
if (mipLevels > 2) {
    meanImage.reset(new (std::nothrow) LinearBlockMean[meanCount]{});
    meanScratch.reset(new (std::nothrow) LinearBlockMean[scratchWidth * scratchHeight]{});
}

for (size_t mipLevel = 1; mipLevel < mipLevels; ++mipLevel)
{
    if (mipLevel >= 3) {
        #pragma omp parallel for
        for (row ...) DownsampleLinearMeanRow(meanFront, ..., meanBack, ...);
        std::swap(meanFront, meanBack);
    }

    #pragma omp parallel for
    for (row ...) {
        if (mipLevel == 1) ProcessCompressedRowBC1<IsSrgb>(baseImage, *destination, row, meanFront);
        else               ProcessLinearRowBC1<IsSrgb>(meanFront, meanWidth, meanHeight, *destination, row);
    }
}
```

**메모리.** scratch 는 $M$ 의 1/4 크기면 이후 모든 레벨을 커버한다.
ping-pong 두 버퍼를 번갈아 쓴다.

$$
\text{scratch} = \left\lceil \frac{W_b}{2} \right\rceil \times \left\lceil \frac{H_b}{2} \right\rceil
$$

**병렬화.** 목적지 블록 행이 작업 단위다. 레벨마다 배리어가 있다.

**컴파일 타임 분기 제거.** `template<bool IsSrgb>` 로 sRGB/UNORM 두 벌을 만들어
내부 루프에서 색공간 분기가 사라진다. `[코드]` `:1495-1497`

### 6.20 `IsOpaqueBC1Image` — 입력 게이트

`[코드]` `:1129-1152`

```cpp
const uint32_t selector3Bits = row[x].bitmap & (row[x].bitmap >> 1) & 0x55555555u;
if (row[x].rgb[0] <= row[x].rgb[1] && selector3Bits != 0)
    return false;
```

3-color 모드 블록이라도 **투명 인덱스(selector 3)를 실제로 쓰지 않으면 통과**시킨다.
§6.5 의 3-color 팔레트 fallback 과 짝을 이룬다.
selector 3 이 쓰인 블록만 거부하면 되므로 커버리지가 넓다.

`selector3Bits` 계산은 §6.4 와 같은 SWAR 기법이다 — `low & high` 가 selector 3 이다.

---

## 7. GPU 경로

`[코드]` `DirectXTex/Shaders/BC1CompressedMips.hlsl`,
`DirectXTex/DirectXTexCompressedMipsGPU.cpp`

### 7.1 커널 3개

| 커널 | 역할 |
|---|---|
| `GenerateMip1CS` | 경로 A. BC1 부모 4개 → BC1 자식 1개 + 블록 평균 4개 |
| `GenerateMipFromMeansCS` | 경로 B. 평균 이미지 → BC1 |
| `DownsampleMeansCS` | 평균 이미지 2×2 축소 |

스레드 하나가 목적지 블록 하나 또는 평균 텍셀 하나를 만든다.
스레드 그룹은 8×8 이다.

### 7.2 CPU 와 다른 점

- CPU 는 ANOVA 분해로 공분산을 구하지만 GPU 는 16 샘플로 직접 구한다.
  §4.3 에 의해 수학적으로 동일하다.
- `float3 samples[16]` 을 레지스터에 두고 `[unroll]` 로 처리한다.
- 평균은 `float4` 로 저장한다 (CPU 는 12 B `LinearBlockMean`).
  정렬을 위해 alpha 를 낭비한다.

### 7.3 디스패치 전략

레벨 전체를 먼저 디스패치하고 **체인이 끝난 뒤 readback 을 한 번만** 한다.

```cpp
for (layoutIndex ...) { ... DispatchMipShader(...); }
// 모든 디스패치 후
cache.context->CopySubresourceRegion(stagingBuffer, ..., outputBuffer, ...);
cache.context->Map(stagingBuffer, 0, D3D11_MAP_READ, 0, &mapped);
```

`[코드]` `DirectXTexCompressedMipsGPU.cpp:370-431`

모든 레벨이 하나의 출력 버퍼를 공유하고, 레벨마다 `DestinationOffset` 으로 구간을 잡는다.
셰이더와 상수 버퍼는 `thread_local` 캐시에 유지된다.

### 7.4 CPU/GPU 일치도

`[측정]`

```
CPU 출력 vs GPU 출력 :  66 – 83 dB PSNR
차이 나는 블록       :  0.14 – 11 %
그중 selector 만 다른 것 : 92 – 100 %
```

대부분은 두 팔레트 색과 정확히 등거리인 텍셀의 tie 를 다르게 깨는 것이다.
엔드포인트까지 갈리는 소수 블록은 최소제곱 행렬식이 $10^{-6}$ 임계 부근인
거의 평탄한 블록으로, 두 해 모두 유효하다.

원인은 `rcp`/`normalize` 근사와 `pow` 구현 차이다.

---

## 8. 정확성 검증

### 8.1 단위 수준 — 브루트포스 대조

랜덤 opaque BC1 블록 20,000 세트 × 4 lane, sRGB/UNORM 양쪽.
독립 스칼라 참조 구현(16 텍셀 완전 복호 → 박스 평균 → 직접 공분산)과 비교.

```
histogram mismatches : 0
max quadrant error   : 1.300e-07
max block-mean error : 1.477e-07
max covariance error : 4.698e-08
```

정리 1과 정리 2가 float32 정밀도 한계까지 성립함을 확인.

### 8.2 시스템 수준 — 재현성

```
HEAD 재빌드 → 게시된 출력       : 20/20 바이트 동일 (CPU 10, GPU 10)
DDS 입력 시 mip 0 보존          : 10/10 비트 동일
PNG(-srgb) 출력 ≡ DDS 입력 출력 : 10/10 완전 동일
```

세 번째 항목이 중요하다. PNG 를 texconv 가 압축해 얻은 mip 0 과
미리 압축해 둔 DDS 의 mip 0 이 같으면 이후 전체 체인이 같아야 한다.
실제로 그렇다 — 파이프라인이 입력 경로에 의존하지 않음을 보인다.

### 8.3 품질 — 무엇을 정답으로 볼 것인가

참조는 원본 PNG 를 `R8G8B8A8_UNORM_SRGB` 로 BOX 밉 체인을 만든 것이다.
`[측정]` 참조 파일의 1×1 밉이 원본의 선형광 평균과 일치함을 확인
(`[109,127,128]` vs 참값 `[109.46,127.14,128.53]`).

**측정 지표에 따라 결론이 달라진다.**

| 지표 | DirectXTex 기본 | DirectXTex `-bc u` | 우리 |
|---|---|---|---|
| texdiag 등가중 평균 | 40.847 | **41.217** | 41.160 |
| 픽셀 수 가중 | 41.354 | — | 41.429 |
| 밉별 평균 (4096², mip 1–12) | 40.256 | **40.677** | 40.503 |

`[측정]` 10장 / 6장

세 가지를 반드시 함께 말해야 한다.

**(가) texdiag 의 `Average MSE` 는 밉별 MSE 의 단순 산술평균이다.**

`[코드]` `Texdiag/texdiag.cpp:3653`

픽셀 수 가중이 없다. 4096² 13레벨 체인에서 가장 작은 4개 레벨
(8²·4²·2²·1² = 총 85 픽셀)이 지표의 31 % 를 차지하고,
16,777,216 픽셀짜리 mip 0 은 7.7 % 만 차지한다.

**(나) DirectXTex 는 다른 오차를 최소화한다.**

§2.3 에서 본 대로 기본값은 휘도 가중이다.
`-bc u` 로 목적함수를 맞추면 DirectXTex 가 +0.37 dB 개선되어 역전한다.

**(다) 참조 자체도 완벽하지 않다.**

§2.2 에서 측정한 대로 참조 체인은 레벨마다 8비트 재양자화를 누적해
mip 12 에서 49.78 dB 까지 열화한다.
깊은 밉에서는 방법 간 차이와 같은 크기다.

**정직한 결론.**
목적함수를 맞춘 비교에서 우리 방식은 **−0.17 dB** 로 근소하게 뒤진다.
방어 가능한 주장은 "화질 우위"가 아니라
**"거의 동등한 화질을 4.3배 속도와 mip 0 무손실 보존으로 달성"** 이다.

---

## 9. 성능 특성

### 9.1 End-to-end (10장 배치, 워밍업 후 3회 중앙값)

| 입력 | DirectXTex | 우리 CPU | 우리 GPU | GPU 속도향상 |
|---|---|---|---|---|
| DDS (BC1) | 7.464 s | 1.935 s | 1.733 s | **4.31×** |
| PNG | 4.007 s | 2.584 s | 2.587 s | **1.55×** |

`[측정]`

PNG 에서 배율이 작은 이유: PNG 디코딩과 mip 0 압축은 두 방식이 똑같이 부담하는
고정 비용이고, 우리가 줄이는 것은 그 뒤 단계이기 때문이다.

### 9.2 밉 생성 커널만 (이미지 1장당)

| 해상도 | 우리 CPU | 우리 GPU |
|---|---|---|
| 2048² (12 레벨) | 6.4 – 8.7 ms | 2.9 – 3.2 ms |
| 4096² (13 레벨) | 26.6 – 33.1 ms | 8.6 – 10.9 ms |

`[측정]` texconv `--timing` 의 `[mip generation: ...]`

10장 합계는 CPU 약 0.21 s (E2E 의 11 %), GPU 약 0.067 s (E2E 의 3.9 %) 다.

**즉 4.3배는 밉 생성 커널이 4.3배 빨라서가 아니다.**
압축 해제 → 12레벨 밉 생성 → 13레벨 전체 BC1 재인코딩을 통째로 건너뛰기 때문이다.
커널 자체만 비교하면 배율은 훨씬 크다.

### 9.3 남은 최적화 여지

`[측정]` 마이크로벤치마크 (clang -O2, SSE2 베이스라인)

| 항목 | 현재 | 개선 가능 | 배율 |
|---|---|---|---|
| `SetQuadrantSample` (경로 B lane 채우기, 배치당 192회 스칼라 삽입) | 1086 ns | `_MM_TRANSPOSE4_PS` 56 ns | **20×** |
| `LinearToSrgbBatch` 의 `XMVectorPow` | — | SSE2 log2/exp2 다항식 | 1.5–2× (sRGB 경로) |

`XMVectorPow` 는 SSE2 빌드에서 스택 저장 → 스칼라 `powf` 4회 → 재로드 경로를 탄다.
`[코드]` Windows SDK `DirectXMathVector.inl:4094`

---

## 10. 정확한 부분과 근사인 부분

### 10.1 정확 (연산 모델 안에서)

- 부모 selector 의 2×2 사분면 히스토그램 — §4.2, `[측정]` 불일치 0
- RGB565 확장과 코드 공간 정수 팔레트 생성 — 하드웨어 사양과 동일
- 팔레트별 sRGB → 선형 변환
- 히스토그램을 이용한 선형광 2×2 박스 평균 — `[측정]` 오차 1.3e-7
- 사분면 평균 4개에서 얻는 블록 평균
- ANOVA 로 계산한 평균과 공분산 — `[측정]` 오차 4.7e-8
- 2의 거듭제곱 해상도에서 평균 피라미드와 단일 박스 평균의 동등성 — §4.4
- 최종 양자화 엔드포인트에 대한 하드웨어 팔레트 재생성
- 고정된 엔드포인트에 대한 최근접 selector 선택

### 10.2 근사 또는 heuristic

- **거듭제곱 반복 1회로 구하는 PCA 방향.** 수렴 보장 없음.
  다만 최소제곱 단계의 초기값일 뿐이라 영향이 제한적이다.
- **이산 가중치를 먼저 배정한 뒤 한 번 푸는 최소제곱.**
  가중치 재배정 → 재적합을 반복하면 더 좋아질 수 있으나 하지 않는다.
- **연속 엔드포인트를 RGB565 격자로 반올림하는 양자화.**
  격자 근방을 탐색하지 않는다.
- **엔드포인트가 같을 때 blue LSB 하나를 움직이는 opaque 보정.**
- **$M$ 이 원본 텍셀이 아니라 mip 0 BC1 양자화 결과에서 계산된 평균이라는 점.**
  §4.5 에서 정량화했다 — mip 1–2 에서 약 0.2 dB.
- **경계에서 clamp-to-edge.** 유효 텍셀 수에 따른 가중 평균이 아니다.

---

## 11. 알려진 한계

| 항목 | 내용 |
|---|---|
| **3-color 모드 출력** | 입력의 3-color 블록은 읽을 수 있으나(§6.20) 출력은 항상 4-color 다. 투명 텍셀이 있는 블록은 아예 거부된다. |
| **비-2의 거듭제곱** | §4.4 의 동등성이 깨진다. 중간 레벨에 홀수 폭이 생기면 clamp 가중치가 달라진다. |
| **8n+1 크기의 미초기화 평균 텍셀** | 가로 또는 세로가 8로 나눈 나머지 1인 크기(9, 17, 25, …)에서 mip 1 이 $M$ 의 일부를 채우지 못한다. CPU 는 `LinearBlockMean[n]{}` 로 0 초기화되어 UB 는 없으나 값은 부정확하다. GPU 는 `DirectXTexCompressedMipsGPU.cpp:353` 의 버퍼가 초기화되지 않는다. `[측정]` 4–129 전 조합에서 3,776 건 발생. |
| **array / cubemap / volume** | texconv 통합이 `GetImage(0,0,0)` 한 장만 처리하고 ScratchImage 전체를 덮어쓴다. `info.arraySize` 가 갱신되지 않는다. |
| **비-BC1 포맷** | `--compression-domain-mips` 를 BC3/BC7 등에 주면 일반 밉 생성이 건너뛰어져 `HRESULT_E_NOT_SUPPORTED` 로 실패한다. fallback 이 없다. |
| **BC3/4/5/7** | 이 경로에 구현되어 있지 않다. BC7 확장은 `docs/BC7.md` 참조. |

---

## 부록 A. 기호표

| 기호 | 의미 |
|---|---|
| $c_k$ | BC1 팔레트 색 $k$ 의 RGB8 코드값 |
| $u_k$ | $c_k$ 를 선형광으로 변환한 값 |
| $s(t)$ | 텍셀 $t$ 의 2-bit selector |
| $n_{q,k}$ | 사분면 $q$ 에서 selector $k$ 의 개수 |
| $\bar q$ | 사분면 평균 (선형광) |
| $m_g$ | 그룹(부모 블록) 평균 |
| $\mu$ | 자식 블록 16 텍셀의 전체 평균 |
| $\Sigma$ | 3×3 공분산 |
| $\hat a$ | PCA 주축 |
| $w_i$ | BC1 보간 가중치, $\{0, 1/3, 2/3, 1\}$ |
| $M$ | 블록 평균 이미지 ($= T_2$) |
| $T_L$ | 레벨 $L$ 의 평균 피라미드 |
| $W_b, H_b$ | 블록 단위 폭·높이 $\lceil W/4 \rceil, \lceil H/4 \rceil$ |

## 부록 B. 관련 문서

- `docs/BC7.md` — BC7 로의 확장. 통일 심볼 형식.
- `note.md` — 개발 노트 및 초기 측정 기록.

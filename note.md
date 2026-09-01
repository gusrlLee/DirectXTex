# Compression-domain mipmap notes

## What we completed

- Added BC1 compression-domain mip generation for CPU and D3D11/HLSL GPU.
- Reused GPU shaders and grow-only buffers between images.
- Reduced GPU readback to one operation per complete mip chain.
- Added precompressed BC1 mip 0 reuse.
- Added `[mip generation: ...]` timing that excludes loading, mip 0 encoding, and saving.
- Verified that optimized and previous GPU outputs are bit-identical.

## Build

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' `
  .\Texconv\Texconv_Desktop_2022.vcxproj `
  /t:Build /p:Configuration=Release /p:Platform=x64
```

## Test image

```text
.\data\quality\input\0557_brick_uneven_stones_basecolor.dds
```

```powershell
New-Item -ItemType Directory -Force .\data\test-standard, .\data\test-cpu, .\data\test-gpu | Out-Null
```

## DirectXTex baseline

```powershell
.\Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -y --timing -m 0 -if BOX -f BC1_UNORM_SRGB -o .\data\test-standard .\data\quality\input\0557_brick_uneven_stones_basecolor.dds
```

## CPU (our method)

```powershell
.\Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -y --timing -m 0 -f BC1_UNORM_SRGB --compression-domain-mips -nogpu -o .\data\test-cpu .\data\quality\input\0557_brick_uneven_stones_basecolor.dds
```

## GPU (our method)

```powershell
.\Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -y --timing -m 0 -f BC1_UNORM_SRGB --compression-domain-mips -gpu 0 -o .\data\test-gpu .\data\quality\input\0557_brick_uneven_stones_basecolor.dds
```

## Timing rules

- `Processing time`: includes load, mip 0 encoding, mip generation, readback, and save.
- `mip generation`: lower mip generation only; mip 0 encoding is excluded.
- For warm E2E, process multiple images in one command and calculate the per-image slope.
- Discard the first GPU `mip generation` result as warm-up.
- Apply the same input, output format, mip count, and timing range to every method.

## Current results (2048 x 2048, BC1, 12 mips)

| Method | Warm E2E | Mip generation only |
|---|---:|---:|
| DirectXTex | 187.19 ms | - |
| CPU (our) | 40.65 ms | 3.372 ms |
| GPU (our) | 34.50 ms | 1.322 ms |

## PSNR by mip level

Reference: uncompressed `R8G8B8A8_UNORM_SRGB` BOX mip chain.

| Mip | Size | DirectXTex | CPU (our) | GPU (our) |
|---:|---:|---:|---:|---:|
| 0 | 2048 x 2048 | 45.139 dB | 45.139 dB | 45.139 dB |
| 1 | 1024 x 1024 | 44.454 dB | 44.092 dB | 44.091 dB |
| 2 | 512 x 512 | 43.985 dB | 43.985 dB | 43.985 dB |
| 3 | 256 x 256 | 43.968 dB | 44.479 dB | 44.479 dB |
| 4 | 128 x 128 | 43.897 dB | 44.379 dB | 44.379 dB |
| 5 | 64 x 64 | 43.445 dB | 43.898 dB | 43.898 dB |
| 6 | 32 x 32 | 43.117 dB | 43.187 dB | 43.187 dB |
| 7 | 16 x 16 | 41.974 dB | 42.206 dB | 42.206 dB |
| 8 | 8 x 8 | 41.726 dB | 42.341 dB | 42.341 dB |
| 9 | 4 x 4 | 43.788 dB | 44.071 dB | 44.071 dB |
| 10 | 2 x 2 | 45.508 dB | 44.854 dB | 44.854 dB |
| 11 | 1 x 1 | 48.433 dB | 48.514 dB | 48.514 dB |
| **Average** | - | **43.834 dB** | **44.026 dB** | **44.026 dB** |

## Quality comparison

```powershell
.\Texdiag\Bin\Desktop_2022\x64\Release\texdiag.exe compare .\data\quality\reference\0557_brick_uneven_stones_basecolor.dds .\data\test-cpu\0557_brick_uneven_stones_basecolor.dds

.\Texdiag\Bin\Desktop_2022\x64\Release\texdiag.exe compare .\data\quality\reference\0557_brick_uneven_stones_basecolor.dds .\data\test-gpu\0557_brick_uneven_stones_basecolor.dds
```

### 📁 [0단계] 폴더 준비 (PowerShell)
기존 결과를 깔끔하게 비우고 새 `_SRGB` 실험용 폴더들을 준비합니다:
```powershell
New-Item -ItemType Directory -Force -Path `
    "output\reference", `
    "output\input_mip0_bc1", `
    "output\test_dxtex_bc1_cpu_dds", `
    "output\test_our_bc1_cpu_dds", `
    "output\test_our_bc1_gpu_dds"
```

---

### 🎯 [1단계] Reference 비압축 sRGB 밉맵 생성 (Ground Truth 정답지)
원본 PNG 10장으로부터 감마 보정된 **비압축 무손실 RGBA8 sRGB 풀 밉맵**을 생성합니다:
```cmd
Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -f R8G8B8A8_UNORM_SRGB -y -o output\reference data\*.png
```

---

### 📦 [2단계] Input용 BC1 sRGB Mip 0 DDS 생성 (`-m 1`)
원본 PNG로부터 **Mip 0만 가진 단일 레벨 BC1 sRGB DDS**를 추출합니다:
```cmd
Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -f BC1_UNORM_SRGB -m 1 -y -o output\input_mip0_bc1 data\*.png
```

---

### ⚡ [3단계] 밉맵 생성 벤치마크 (Pure Compute Time 측정)

#### ① 원본 DirectXTex Baseline CPU (디코딩 $\to$ 다운샘플링 $\to$ 인코딩)
```cmd
Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -f BC1_UNORM_SRGB -nogpu -timing -y -o output\test_dxtex_bc1_cpu_dds output\input_mip0_bc1\*.dds
```

#### ② 우리 방식 (Our Method) CPU 압축 도메인 밉맵 생성
```cmd
Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -f BC1_UNORM_SRGB --compression-domain-mips -nogpu -timing -y -o output\test_our_bc1_cpu_dds output\input_mip0_bc1\*.dds
```

#### ③ 우리 방식 (Our Method) GPU (DirectCompute) 압축 도메인 밉맵 생성
```cmd
Texconv\Bin\Desktop_2022\x64\Release\texconv.exe -f BC1_UNORM_SRGB --compression-domain-mips -gpu 0 -timing -y -o output\test_our_bc1_gpu_dds output\input_mip0_bc1\*.dds
```

---

### 📊 [4단계] 화질(Average PSNR) 일괄 측정 & `benchmark.md` 저장 (PowerShell)

#### ① DirectXTex Baseline CPU 화질 측정:
```powershell
$res = Get-ChildItem "output\test_dxtex_bc1_cpu_dds\*.dds" | ForEach-Object { $ref = "output\reference\" + $_.Name; $raw = & "Texdiag\Bin\Desktop_2022\x64\Release\texdiag.exe" compare $_.FullName $ref; "[$($_.Name)]     " + ($raw | Where-Object { $_ -match "Average MSE" }).Trim() }; Add-Content "output\test_dxtex_bc1_cpu_dds\benchmark.md" "`nQuality"; $res | Add-Content "output\test_dxtex_bc1_cpu_dds\benchmark.md"; $res
```

#### ② 우리 방식 CPU 화질 측정:
```powershell
$res = Get-ChildItem "output\test_our_bc1_cpu_dds\*.dds" | ForEach-Object { $ref = "output\reference\" + $_.Name; $raw = & "Texdiag\Bin\Desktop_2022\x64\Release\texdiag.exe" compare $_.FullName $ref; "[$($_.Name)]     " + ($raw | Where-Object { $_ -match "Average MSE" }).Trim() }; Add-Content "output\test_our_bc1_cpu_dds\benchmark.md" "`nQuality"; $res | Add-Content "output\test_our_bc1_cpu_dds\benchmark.md"; $res
```

#### ③ 우리 방식 GPU 화질 측정:
```powershell
$res = Get-ChildItem "output\test_our_bc1_gpu_dds\*.dds" | ForEach-Object { $ref = "output\reference\" + $_.Name; $raw = & "Texdiag\Bin\Desktop_2022\x64\Release\texdiag.exe" compare $_.FullName $ref; "[$($_.Name)]     " + ($raw | Where-Object { $_ -match "Average MSE" }).Trim() }; Add-Content "output\test_our_bc1_gpu_dds\benchmark.md" "`nQuality"; $res | Add-Content "output\test_our_bc1_gpu_dds\benchmark.md"; $res
```

---
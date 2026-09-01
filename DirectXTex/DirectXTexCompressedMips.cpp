#include "DirectXTexP.h"
#include "BC.h"

#include <immintrin.h>

using namespace DirectX;
static_assert(sizeof(D3DX_BC1) == 8, "D3DX_BC1 must be 8 bytes");

namespace
{
    // Shift every 32-bit SIMD lane to the right.
    template<int Shift>
    inline XMVECTOR ShiftRight32(FXMVECTOR value) noexcept
    {
        return _mm_castsi128_ps(_mm_srli_epi32(_mm_castps_si128(value), Shift));
    }

    // Shift every 32-bit SIMD lane to the left.
    template<int Shift>
    inline XMVECTOR ShiftLeft32(FXMVECTOR value) noexcept
    {
        return _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(value), Shift));
    }

    // Add every 32-bit integer SIMD lane.
    inline XMVECTOR AddInt32(FXMVECTOR value0, FXMVECTOR value1) noexcept
    {
        return _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(value0), _mm_castps_si128(value1)));
    }
}

// Lee: Four independent BC1 blocks stored in SIMD lane.
struct BC1BlockBatch
{
    XMVECTOR color0;
    XMVECTOR color1;
    XMVECTOR selectors;
};

// Lee: RGB8 channels for four SIMD lanes.
struct RGB8Batch
{
    XMVECTOR r;
    XMVECTOR g;
    XMVECTOR b;
};

// Lee: Four decoded BC1 palette colors for four SIMD lanes.
struct BC1PaletteBatch
{
    RGB8Batch color[4];
};

// Lee: Four SIMD batches forming one downsampled 4x4 BC1 block
struct DownsampledBC1Block
{
    RGB8Batch texelGroup[4];
};

// Lee: Load four BC1 blocks into SIMD lane
inline BC1BlockBatch LoadBC1BlockBatch(
    const D3DX_BC1* blocks) noexcept
{
    BC1BlockBatch batch;
    batch.color0 = XMVectorSetInt(blocks[0].rgb[0], blocks[1].rgb[0], blocks[2].rgb[0], blocks[3].rgb[0]);
    batch.color1 = XMVectorSetInt(blocks[0].rgb[1], blocks[1].rgb[1], blocks[2].rgb[1], blocks[3].rgb[1]);
    batch.selectors = XMVectorSetInt(blocks[0].bitmap, blocks[1].bitmap, blocks[2].bitmap, blocks[3].bitmap);

    return batch;
}

// Lee: Decode four RGB565 endpoints into RGB8 channels.
inline RGB8Batch DecodeRGB565Batch(XMVECTOR packed) noexcept
{
    const XMVECTOR mask5 = XMVectorReplicateInt(0x1Fu);
    const XMVECTOR mask6 = XMVectorReplicateInt(0x3Fu);

    const XMVECTOR r5 = XMVectorAndInt(ShiftRight32<11>(packed), mask5);
    const XMVECTOR g6 = XMVectorAndInt(ShiftRight32<5>(packed), mask6);
    const XMVECTOR b5 = XMVectorAndInt(packed, mask5);

    RGB8Batch result;

    // Expand RGB565 channels to RGB8.
    result.r = XMVectorOrInt(ShiftLeft32<3>(r5), ShiftRight32<2>(r5));
    result.g = XMVectorOrInt(ShiftLeft32<2>(g6), ShiftRight32<4>(g6));
    result.b = XMVectorOrInt(ShiftLeft32<3>(b5), ShiftRight32<2>(b5));

    return result;
}

// Lee: Return an all-one mask for BC1 blocks using four-color mode.
inline XMVECTOR GetBC1FourColorMask(FXMVECTOR color0, FXMVECTOR color1) noexcept
{
    return _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_castps_si128(color0), _mm_castps_si128(color1)));
}

// Lee: Extract one 2-bit selector from four BC1 blocks in parallel.
template<uint32_t TexelIndex>
inline XMVECTOR ExtractBC1Selector(FXMVECTOR selectors) noexcept
{
    static_assert(TexelIndex < 16, "BC1 texel index must be below 16");
    const XMVECTOR selectorMask = XMVectorReplicateInt(0x3u);

    return XMVectorAndInt(ShiftRight32<TexelIndex * 2>(selectors), selectorMask);
}

// Lee: Select one palette channel using four BC1 selector in parallel.
inline XMVECTOR SelectBC1PaletteChannel(
    FXMVECTOR selectors,
    FXMVECTOR color0,
    FXMVECTOR color1,
    GXMVECTOR color2,
    HXMVECTOR color3) noexcept
{
    const XMVECTOR one = XMVectorReplicateInt(1u);
    const XMVECTOR two = XMVectorReplicateInt(2u);

    const XMVECTOR lowBit = XMVectorAndInt(selectors, one);
    const XMVECTOR highBit = XMVectorAndInt(selectors, two);

    const XMVECTOR lowMask = XMVectorEqualInt(lowBit, one);
    const XMVECTOR highMask = XMVectorEqualInt(highBit, two);

    // Select color 0/1 and color 2/3 using the low selector bit.
    const XMVECTOR lowPair = XMVectorSelect(color0, color1, lowMask);
    const XMVECTOR highPair = XMVectorSelect(color2, color3, lowMask);

    // Select between the two pairs using the high selector bit.
    return XMVectorSelect(lowPair, highPair, highMask);
}

// Lee:  Interpolate (color0 + color1) / 2 in four SIMD lanes.
inline XMVECTOR InterpolateOneToOne(FXMVECTOR color0, FXMVECTOR color1) noexcept
{
    const XMVECTOR color0Float = XMConvertVectorUIntToFloat(color0, 0);
    const XMVECTOR color1Float = XMConvertVectorUIntToFloat(color1, 0);
    const XMVECTOR interpolated = XMVectorLerp(color0Float, color1Float, 0.5f);

    return XMConvertVectorFloatToUInt(XMVectorRound(interpolated), 0);
}

// Lee: Interpolate (2 * c0 + c1) / 3 in four SIMD lanes.
inline XMVECTOR InterpolateTwoToOne(FXMVECTOR color0, FXMVECTOR color1) noexcept
{
    const XMVECTOR color0Float = XMConvertVectorUIntToFloat(color0, 0);
    const XMVECTOR color1Float = XMConvertVectorUIntToFloat(color1, 0);
    const XMVECTOR interpolated = XMVectorLerp(color0Float, color1Float, 1.0f / 3.0f);

    return XMConvertVectorFloatToUInt(XMVectorRound(interpolated), 0);
}

// Lee: Decode the BC1 palette for four blocks in parallel.
inline BC1PaletteBatch DecodeBC1PaletteBatch(const BC1BlockBatch &blocks) noexcept
{
    BC1PaletteBatch palette{};

    // endpoint 0, 1
    palette.color[0] = DecodeRGB565Batch(blocks.color0);
    palette.color[1] = DecodeRGB565Batch(blocks.color1);

    const XMVECTOR fourColorMask = GetBC1FourColorMask(blocks.color0, blocks.color1);

    // Color 2 depends on the BC1 palette mode.
    palette.color[2].r = XMVectorSelect(
        InterpolateOneToOne(palette.color[0].r, palette.color[1].r),
        InterpolateTwoToOne(palette.color[0].r, palette.color[1].r),
        fourColorMask);

    palette.color[2].g = XMVectorSelect(
        InterpolateOneToOne(palette.color[0].g, palette.color[1].g),
        InterpolateTwoToOne(palette.color[0].g, palette.color[1].g),
        fourColorMask);

    palette.color[2].b = XMVectorSelect(
        InterpolateOneToOne(palette.color[0].b, palette.color[1].b),
        InterpolateTwoToOne(palette.color[0].b, palette.color[1].b),
        fourColorMask);

    // Color 3 is transparent black in three-color mode.
    const XMVECTOR zero = XMVectorZero();

    palette.color[3].r = XMVectorSelect(
        zero,
        InterpolateTwoToOne(palette.color[1].r, palette.color[0].r),
        fourColorMask);

    palette.color[3].g = XMVectorSelect(
        zero,
        InterpolateTwoToOne(palette.color[1].g, palette.color[0].g),
        fourColorMask);

    palette.color[3].b = XMVectorSelect(
        zero,
        InterpolateTwoToOne(palette.color[1].b, palette.color[0].b),
        fourColorMask);

    return palette;
}

// Lee: Decode one texel position from four BC1 blocks in parallel.
template<uint32_t TexelIndex>
inline RGB8Batch DecodeBC1TexelBatch(const BC1BlockBatch& blocks, const BC1PaletteBatch& palette) noexcept
{
    const XMVECTOR selectors = ExtractBC1Selector<TexelIndex>(blocks.selectors);

    RGB8Batch result{};

    result.r = SelectBC1PaletteChannel(
        selectors,
        palette.color[0].r,
        palette.color[1].r,
        palette.color[2].r,
        palette.color[3].r);

    result.g = SelectBC1PaletteChannel(
        selectors,
        palette.color[0].g,
        palette.color[1].g,
        palette.color[2].g,
        palette.color[3].g);

    result.b = SelectBC1PaletteChannel(
        selectors,
        palette.color[0].b,
        palette.color[1].b,
        palette.color[2].b,
        palette.color[3].b);

    return result;
}

// Lee: Average four unsigned 8-bit channel values in each SIMD lane.
inline XMVECTOR AverageFourUInt8(
    FXMVECTOR value0,
    FXMVECTOR value1,
    FXMVECTOR value2,
    GXMVECTOR value3) noexcept
{
    XMVECTOR sum = AddInt32(value0, value1);
    sum = AddInt32(sum, value2);
    sum = AddInt32(sum, value3);

    sum = AddInt32(sum, XMVectorReplicateInt(2u));
    return ShiftRight32<2>(sum);
}

// Lee: Average four RGB texel batches in parallel
inline RGB8Batch AverageFourRGB8(
    const RGB8Batch& texel0,
    const RGB8Batch& texel1,
    const RGB8Batch& texel2,
    const RGB8Batch& texel3) noexcept
{
    RGB8Batch result{};
    result.r = AverageFourUInt8(texel0.r, texel1.r, texel2.r, texel3.r);
    result.g = AverageFourUInt8(texel0.g, texel1.g, texel2.g, texel3.g);
    result.b = AverageFourUInt8(texel0.b, texel1.b, texel2.b, texel3.b);

    return result;
}

// Lee: Downsample one 2x2 texel region from four BC1 blocks in parallel.
template<uint32_t OutputX, uint32_t OutputY>
inline RGB8Batch DownsampleBC1TexelBatch(const BC1BlockBatch& blocks, const BC1PaletteBatch& palette) noexcept
{
    static_assert(OutputX < 2, "OutputX must be 0 or 1");
    static_assert(OutputY < 2, "OutputY must be 0 or 1");

    constexpr uint32_t sourceX = OutputX * 2;
    constexpr uint32_t sourceY = OutputY * 2;
    constexpr uint32_t topLeft = sourceY * 4 + sourceX;

    const RGB8Batch texel0 = DecodeBC1TexelBatch<topLeft>(blocks, palette);
    const RGB8Batch texel1 = DecodeBC1TexelBatch<topLeft + 1>(blocks, palette);
    const RGB8Batch texel2 = DecodeBC1TexelBatch<topLeft + 4>(blocks, palette);
    const RGB8Batch texel3 = DecodeBC1TexelBatch<topLeft + 5>(blocks, palette);

    return AverageFourRGB8(texel0, texel1, texel2, texel3);
}

// Lee: Downsample four source BC1 blocks into one 4x4 output block.
inline DownsampledBC1Block DownsampleBC1BlockGroup(const BC1BlockBatch& blocks, const BC1PaletteBatch& palette) noexcept
{
    DownsampledBC1Block result{};
    result.texelGroup[0] = DownsampleBC1TexelBatch<0, 0>(blocks, palette);
    result.texelGroup[1] = DownsampleBC1TexelBatch<1, 0>(blocks, palette);
    result.texelGroup[2] = DownsampleBC1TexelBatch<0, 1>(blocks, palette);
    result.texelGroup[3] = DownsampleBC1TexelBatch<1, 1>(blocks, palette);

    return result;
}

// Lee: Our main function 
HRESULT DirectX::GenerateCompressedMipMaps(const Image& baseImage, size_t levels, ScratchImage& mipChain) noexcept
{
    if (!baseImage.pixels)
    {
        return E_POINTER;
    }

    if (baseImage.format != DXGI_FORMAT_BC1_UNORM && baseImage.format != DXGI_FORMAT_BC1_UNORM_SRGB)
    {
        return HRESULT_E_NOT_SUPPORTED;
    }

    // Lee: allocate the compressed output mip chain
    HRESULT hr = mipChain.Initialize2D(baseImage.format, baseImage.width, baseImage.height, 1, levels);
    if (FAILED(hr))
    {
        return hr;
    }

    // Lee: Get hte base level of the output mip chain.
    const Image* outputBase = mipChain.GetImage(0, 0, 0);
    if (!outputBase || !outputBase->pixels)
    {
        return E_POINTER;
    }

    // Lee: The source and destination BC1 layout must match.
    if (baseImage.rowPitch != outputBase->rowPitch || baseImage.slicePitch != outputBase->slicePitch)
    {
        return E_FAIL;
    }

    // Lee: Copy data 
    memcpy_s(outputBase->pixels, outputBase->slicePitch, baseImage.pixels, baseImage.slicePitch);



    return E_NOTIMPL;
}

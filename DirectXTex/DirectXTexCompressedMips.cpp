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

// Lee: Load four BC1 blocks into SIMD lane
inline BC1BlockBatch LoadBC1BlockBatch(
    const D3DX_BC1* blocks) noexcept 
{
    BC1BlockBatch batch;

    batch.color0 = XMVectorSetInt(
        blocks[0].rgb[0],
        blocks[1].rgb[0],
        blocks[2].rgb[0],
        blocks[3].rgb[0]);

    batch.color1 = XMVectorSetInt(
        blocks[0].rgb[1],
        blocks[1].rgb[1],
        blocks[2].rgb[1],
        blocks[3].rgb[1]);

    batch.selectors = XMVectorSetInt(
        blocks[0].bitmap,
        blocks[1].bitmap,
        blocks[2].bitmap,
        blocks[3].bitmap);

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

// Lee: Our main function 
HRESULT DirectX::GenerateCompressedMipMaps(
    const Image& baseImage,
    size_t levels,
    ScratchImage& mipChain) noexcept
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

#include "DirectXTexP.h"

using namespace DirectX;

HRESULT DirectX::GenerateCompressedMipMaps(
    const Image& baseImage,
    size_t levels,
    ScratchImage& mipChain) noexcept
{
    (void)baseImage;
    (void)levels;
    (void)mipChain;

    const XMVECTOR values = XMVectorSet(1.f, 2.f, 3.f, 4.f);
    const XMVECTOR doubled = XMVectorAdd(values, values);
    (void)doubled;
    
    return E_NOTIMPL;
}

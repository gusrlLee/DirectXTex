//-------------------------------------------------------------------------------------
// DirectXTexCompressedMipsGPU.cpp
// D3D11 DirectCompute implementation of BC1 compression-domain mip generation.
//-------------------------------------------------------------------------------------

#include "DirectXTexP.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

using namespace DirectX;
using Microsoft::WRL::ComPtr;

namespace
{
    namespace cs5
    {
    #include "BC1CompressedMips_GenerateMip1CS.inc"
    #include "BC1CompressedMips_GenerateMipFromMeansCS.inc"
    #include "BC1CompressedMips_DownsampleMeansCS.inc"
    }

    struct BC1RawBlock
    {
        uint32_t endpoints;
        uint32_t selectors;
    };

    struct MipConstants
    {
        uint32_t sourceWidth;
        uint32_t sourceHeight;
        uint32_t destinationWidth;
        uint32_t destinationHeight;
        uint32_t isSrgb;
        uint32_t destinationOffset;
        uint32_t padding[2];
    };

    struct MipReadbackLayout
    {
        Image* image;
        uint32_t blockWidth;
        uint32_t blockHeight;
        uint32_t blockOffset;
    };

    // Keep immutable shaders and grow-only working buffers alive between images.
    // The immediate context is not intended for concurrent submission, so a
    // thread-local cache also prevents one caller from rebinding another's state.
    struct GpuMipCache
    {
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        ComPtr<ID3D11ComputeShader> mip1Shader;
        ComPtr<ID3D11ComputeShader> mipShader;
        ComPtr<ID3D11ComputeShader> downsampleShader;
        ComPtr<ID3D11Buffer> constantBuffer;

        ComPtr<ID3D11Buffer> baseBuffer;
        ComPtr<ID3D11ShaderResourceView> baseSRV;
        uint32_t baseCapacity = 0;

        ComPtr<ID3D11Buffer> outputBuffer;
        ComPtr<ID3D11UnorderedAccessView> outputUAV;
        ComPtr<ID3D11Buffer> stagingBuffer;
        uint32_t outputCapacity = 0;

        ComPtr<ID3D11Buffer> meanBuffers[2];
        ComPtr<ID3D11ShaderResourceView> meanSRVs[2];
        ComPtr<ID3D11UnorderedAccessView> meanUAVs[2];
        uint32_t meanCapacities[2] = {};
    };

    static_assert(sizeof(BC1RawBlock) == 8, "BC1 raw block size mismatch");
    static_assert(sizeof(MipConstants) == 32, "HLSL constant buffer size mismatch");

    HRESULT CreateStructuredBuffer(
        ID3D11Device* device,
        uint32_t stride,
        uint32_t count,
        const void* initialData,
        ComPtr<ID3D11Buffer>& buffer,
        ComPtr<ID3D11ShaderResourceView>* srv,
        ComPtr<ID3D11UnorderedAccessView>* uav) noexcept
    {
        if (!device || !stride || !count)
            return E_INVALIDARG;

        const uint64_t byteWidth = uint64_t(stride) * uint64_t(count);
        if (byteWidth > std::numeric_limits<uint32_t>::max())
            return HRESULT_E_ARITHMETIC_OVERFLOW;

        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = static_cast<uint32_t>(byteWidth);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = (srv ? D3D11_BIND_SHADER_RESOURCE : 0u) | (uav ? D3D11_BIND_UNORDERED_ACCESS : 0u);
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = stride;

        D3D11_SUBRESOURCE_DATA data{};
        data.pSysMem = initialData;
        HRESULT hr = device->CreateBuffer(&desc, initialData ? &data : nullptr, buffer.ReleaseAndGetAddressOf());
        if (FAILED(hr))
            return hr;

        if (srv)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_UNKNOWN;
            view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            view.Buffer.NumElements = count;
            hr = device->CreateShaderResourceView(buffer.Get(), &view, srv->ReleaseAndGetAddressOf());
            if (FAILED(hr))
                return hr;
        }

        if (uav)
        {
            D3D11_UNORDERED_ACCESS_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_UNKNOWN;
            view.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            view.Buffer.NumElements = count;
            hr = device->CreateUnorderedAccessView(buffer.Get(), &view, uav->ReleaseAndGetAddressOf());
            if (FAILED(hr))
                return hr;
        }

        return S_OK;
    }

    HRESULT InitializeCache(ID3D11Device* device, GpuMipCache& cache) noexcept
    {
        if (cache.device.Get() == device)
            return S_OK;

        // A cache belongs to exactly one device because D3D11 resources cannot
        // be shared implicitly between devices.
        cache = GpuMipCache{};
        cache.device = device;
        device->GetImmediateContext(cache.context.GetAddressOf());
        if (!cache.context)
            return E_FAIL;

        HRESULT hr = device->CreateComputeShader(cs5::BC1CompressedMips_GenerateMip1CS,
            sizeof(cs5::BC1CompressedMips_GenerateMip1CS), nullptr, cache.mip1Shader.GetAddressOf());
        if (FAILED(hr)) return hr;
        hr = device->CreateComputeShader(cs5::BC1CompressedMips_GenerateMipFromMeansCS,
            sizeof(cs5::BC1CompressedMips_GenerateMipFromMeansCS), nullptr, cache.mipShader.GetAddressOf());
        if (FAILED(hr)) return hr;
        hr = device->CreateComputeShader(cs5::BC1CompressedMips_DownsampleMeansCS,
            sizeof(cs5::BC1CompressedMips_DownsampleMeansCS), nullptr, cache.downsampleShader.GetAddressOf());
        if (FAILED(hr)) return hr;

        D3D11_BUFFER_DESC cbDesc{};
        cbDesc.ByteWidth = sizeof(MipConstants);
        cbDesc.Usage = D3D11_USAGE_DEFAULT;
        cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        return device->CreateBuffer(&cbDesc, nullptr, cache.constantBuffer.GetAddressOf());
    }

    HRESULT EnsureStructuredBuffer(
        ID3D11Device* device,
        uint32_t stride,
        uint32_t requiredCount,
        ComPtr<ID3D11Buffer>& buffer,
        ComPtr<ID3D11ShaderResourceView>* srv,
        ComPtr<ID3D11UnorderedAccessView>* uav,
        uint32_t& capacity) noexcept
    {
        // Force buffer recreation by always proceeding.
        HRESULT hr = CreateStructuredBuffer(device, stride, requiredCount, nullptr, buffer, srv, uav);
        if (SUCCEEDED(hr))
            capacity = requiredCount;
        return hr;
    }

    HRESULT EnsureOutputBuffers(ID3D11Device* device, uint32_t requiredCount, GpuMipCache& cache) noexcept
    {
        // Force buffer recreation by always proceeding.
        ComPtr<ID3D11Buffer> outputBuffer;
        ComPtr<ID3D11UnorderedAccessView> outputUAV;
        HRESULT hr = CreateStructuredBuffer(device, sizeof(BC1RawBlock), requiredCount, nullptr,
            outputBuffer, nullptr, &outputUAV);
        if (FAILED(hr))
            return hr;

        const uint64_t byteWidth = uint64_t(requiredCount) * sizeof(BC1RawBlock);
        if (byteWidth > std::numeric_limits<uint32_t>::max())
            return HRESULT_E_ARITHMETIC_OVERFLOW;

        D3D11_BUFFER_DESC stagingDesc{};
        stagingDesc.ByteWidth = static_cast<uint32_t>(byteWidth);
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        ComPtr<ID3D11Buffer> stagingBuffer;
        hr = device->CreateBuffer(&stagingDesc, nullptr, stagingBuffer.GetAddressOf());
        if (FAILED(hr))
            return hr;

        cache.outputBuffer = std::move(outputBuffer);
        cache.outputUAV = std::move(outputUAV);
        cache.stagingBuffer = std::move(stagingBuffer);
        cache.outputCapacity = requiredCount;
        return S_OK;
    }

    void ResetComputeBindings(ID3D11DeviceContext* context) noexcept
    {
        ID3D11ShaderResourceView* nullSRVs[2] = {};
        ID3D11UnorderedAccessView* nullUAVs[2] = {};
        ID3D11Buffer* nullCB = nullptr;
        context->CSSetShaderResources(0, 2, nullSRVs);
        context->CSSetUnorderedAccessViews(0, 2, nullUAVs, nullptr);
        context->CSSetConstantBuffers(0, 1, &nullCB);
        context->CSSetShader(nullptr, nullptr, 0);
    }

    void DispatchMipShader(
        ID3D11DeviceContext* context,
        ID3D11ComputeShader* shader,
        ID3D11Buffer* constants,
        ID3D11ShaderResourceView* sourceBC,
        ID3D11ShaderResourceView* sourceMeans,
        ID3D11UnorderedAccessView* destinationBC,
        ID3D11UnorderedAccessView* destinationMeans,
        uint32_t width,
        uint32_t height) noexcept
    {
        ID3D11ShaderResourceView* srvs[2] = { sourceBC, sourceMeans };
        ID3D11UnorderedAccessView* uavs[2] = { destinationBC, destinationMeans };
        context->CSSetShader(shader, nullptr, 0);
        context->CSSetShaderResources(0, 2, srvs);
        context->CSSetUnorderedAccessViews(0, 2, uavs, nullptr);
        context->CSSetConstantBuffers(0, 1, &constants);
        context->Dispatch((width + 7u) / 8u, (height + 7u) / 8u, 1);
        ResetComputeBindings(context);
    }

    bool IsOpaqueBC1(const Image& image) noexcept
    {
        const size_t width = std::max<size_t>(1, (image.width + 3) / 4);
        const size_t height = std::max<size_t>(1, (image.height + 3) / 4);
        for (size_t y = 0; y < height; ++y)
        {
            const auto* row = reinterpret_cast<const BC1RawBlock*>(image.pixels + y * image.rowPitch);
            for (size_t x = 0; x < width; ++x)
            {
                const uint32_t endpoint0 = row[x].endpoints & 0xffffu;
                const uint32_t endpoint1 = row[x].endpoints >> 16;
                const uint32_t selector3Bits = row[x].selectors & (row[x].selectors >> 1) & 0x55555555u;
                if (endpoint0 <= endpoint1 && selector3Bits != 0)
                    return false;
            }
        }
        return true;
    }
}

_Use_decl_annotations_
HRESULT DirectX::GenerateCompressedMipMaps(
    ID3D11Device* device,
    const Image& baseImage,
    size_t levels,
    ScratchImage& mipChain) noexcept
{
    if (!device || !baseImage.pixels)
        return E_POINTER;
    if (device->GetFeatureLevel() < D3D_FEATURE_LEVEL_11_0)
        return HRESULT_E_NOT_SUPPORTED;
    if (baseImage.format != DXGI_FORMAT_BC1_UNORM && baseImage.format != DXGI_FORMAT_BC1_UNORM_SRGB)
        return HRESULT_E_NOT_SUPPORTED;
    if (!IsOpaqueBC1(baseImage))
        return HRESULT_E_NOT_SUPPORTED;
    if (baseImage.width > UINT32_MAX || baseImage.height > UINT32_MAX)
        return E_INVALIDARG;

    HRESULT hr = mipChain.Initialize2D(baseImage.format, baseImage.width, baseImage.height, 1, levels);
    if (FAILED(hr))
        return hr;

    const Image* outputBase = mipChain.GetImage(0, 0, 0);
    if (!outputBase || outputBase->rowPitch != baseImage.rowPitch || outputBase->slicePitch != baseImage.slicePitch)
        return E_FAIL;
    memcpy_s(outputBase->pixels, outputBase->slicePitch, baseImage.pixels, baseImage.slicePitch);

    const size_t mipLevels = mipChain.GetMetadata().mipLevels;
    if (mipLevels <= 1)
        return S_OK;

    const uint32_t baseWidth = static_cast<uint32_t>(std::max<size_t>(1, (baseImage.width + 3) / 4));
    const uint32_t baseHeight = static_cast<uint32_t>(std::max<size_t>(1, (baseImage.height + 3) / 4));
    const uint64_t baseCount64 = uint64_t(baseWidth) * uint64_t(baseHeight);
    if (baseCount64 > std::numeric_limits<uint32_t>::max())
        return HRESULT_E_ARITHMETIC_OVERFLOW;
    const uint32_t baseCount = static_cast<uint32_t>(baseCount64);

    // Pack rows tightly before uploading because a DDS row can contain padding.
    std::vector<BC1RawBlock> baseBlocks(baseCount);
    for (uint32_t y = 0; y < baseHeight; ++y)
    {
        memcpy_s(baseBlocks.data() + size_t(y) * baseWidth, size_t(baseWidth) * sizeof(BC1RawBlock),
            baseImage.pixels + size_t(y) * baseImage.rowPitch, size_t(baseWidth) * sizeof(BC1RawBlock));
    }

    // Record each mip's position inside one shared output buffer. No readback is
    // performed until every dispatch in the chain has been submitted.
    std::vector<MipReadbackLayout> layouts;
    layouts.reserve(mipLevels - 1);
    uint64_t totalOutputBlocks = 0;
    for (size_t mipLevel = 1; mipLevel < mipLevels; ++mipLevel)
    {
        Image* destination = const_cast<Image*>(mipChain.GetImage(mipLevel, 0, 0));
        if (!destination || !destination->pixels)
            return E_FAIL;

        const size_t blockWidthSize = std::max<size_t>(1, (destination->width + 3) / 4);
        const size_t blockHeightSize = std::max<size_t>(1, (destination->height + 3) / 4);
        if (blockWidthSize > UINT32_MAX || blockHeightSize > UINT32_MAX)
            return HRESULT_E_ARITHMETIC_OVERFLOW;

        const uint64_t blockCount = uint64_t(blockWidthSize) * uint64_t(blockHeightSize);
        if (totalOutputBlocks + blockCount > UINT32_MAX)
            return HRESULT_E_ARITHMETIC_OVERFLOW;

        layouts.push_back({ destination, static_cast<uint32_t>(blockWidthSize),
            static_cast<uint32_t>(blockHeightSize), static_cast<uint32_t>(totalOutputBlocks) });
        totalOutputBlocks += blockCount;
    }

    thread_local GpuMipCache cache;
    hr = InitializeCache(device, cache);
    if (FAILED(hr)) return hr;

    // Grow cached buffers only when a larger image requires more capacity.
    hr = EnsureStructuredBuffer(device, sizeof(BC1RawBlock), baseCount,
        cache.baseBuffer, &cache.baseSRV, nullptr, cache.baseCapacity);
    if (FAILED(hr)) return hr;
    hr = EnsureOutputBuffers(device, static_cast<uint32_t>(totalOutputBlocks), cache);
    if (FAILED(hr)) return hr;

    const uint64_t halfWidth = (uint64_t(baseWidth) + 1u) / 2u;
    const uint64_t halfHeight = (uint64_t(baseHeight) + 1u) / 2u;
    const uint64_t secondMeanCount64 = std::max<uint64_t>(1, halfWidth * halfHeight);
    if (secondMeanCount64 > UINT32_MAX)
        return HRESULT_E_ARITHMETIC_OVERFLOW;
    const uint32_t meanCounts[2] = { baseCount, static_cast<uint32_t>(secondMeanCount64) };
    for (size_t i = 0; i < 2; ++i)
    {
        hr = EnsureStructuredBuffer(device, sizeof(float) * 4, meanCounts[i],
            cache.meanBuffers[i], &cache.meanSRVs[i], &cache.meanUAVs[i], cache.meanCapacities[i]);
        if (FAILED(hr)) return hr;
    }

    // Since we now force buffer recreation, we can update the entire buffer at once.
    cache.context->UpdateSubresource(cache.baseBuffer.Get(), 0, nullptr, baseBlocks.data(), 0, 0);

    size_t meanFront = 0;
    size_t meanBack = 1;
    uint32_t meanWidth = baseWidth;
    uint32_t meanHeight = baseHeight;
    for (size_t layoutIndex = 0; layoutIndex < layouts.size(); ++layoutIndex)
    {
        const size_t mipLevel = layoutIndex + 1;
        const MipReadbackLayout& layout = layouts[layoutIndex];

        if (mipLevel >= 3)
        {
            const uint32_t nextWidth = (meanWidth + 1u) / 2u;
            const uint32_t nextHeight = (meanHeight + 1u) / 2u;
            const MipConstants constants{ meanWidth, meanHeight, nextWidth, nextHeight,
                IsSRGB(baseImage.format) ? 1u : 0u, 0u, {} };
            cache.context->UpdateSubresource(cache.constantBuffer.Get(), 0, nullptr, &constants, 0, 0);
            DispatchMipShader(cache.context.Get(), cache.downsampleShader.Get(), cache.constantBuffer.Get(), nullptr,
                cache.meanSRVs[meanFront].Get(), nullptr, cache.meanUAVs[meanBack].Get(), nextWidth, nextHeight);
            std::swap(meanFront, meanBack);
            meanWidth = nextWidth;
            meanHeight = nextHeight;
        }

        const MipConstants constants{ mipLevel == 1 ? baseWidth : meanWidth,
            mipLevel == 1 ? baseHeight : meanHeight, layout.blockWidth, layout.blockHeight,
            IsSRGB(baseImage.format) ? 1u : 0u, layout.blockOffset, {} };
        cache.context->UpdateSubresource(cache.constantBuffer.Get(), 0, nullptr, &constants, 0, 0);
        if (mipLevel == 1)
        {
            DispatchMipShader(cache.context.Get(), cache.mip1Shader.Get(), cache.constantBuffer.Get(),
                cache.baseSRV.Get(), nullptr, cache.outputUAV.Get(), cache.meanUAVs[meanFront].Get(),
                layout.blockWidth, layout.blockHeight);
        }
        else
        {
            DispatchMipShader(cache.context.Get(), cache.mipShader.Get(), cache.constantBuffer.Get(), nullptr,
                cache.meanSRVs[meanFront].Get(), cache.outputUAV.Get(), nullptr,
                layout.blockWidth, layout.blockHeight);
        }
    }

    // Synchronize once after the full chain. CopySubresourceRegion limits the
    // transfer when cached buffers are larger than the current image requires.
    D3D11_BOX readbackBox{};
    readbackBox.right = static_cast<uint32_t>(totalOutputBlocks * sizeof(BC1RawBlock));
    readbackBox.bottom = 1;
    readbackBox.back = 1;
    cache.context->CopySubresourceRegion(cache.stagingBuffer.Get(), 0, 0, 0, 0,
        cache.outputBuffer.Get(), 0, &readbackBox);

    D3D11_MAPPED_SUBRESOURCE mapped{};
    hr = cache.context->Map(cache.stagingBuffer.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) return hr;

    const auto* source = static_cast<const uint8_t*>(mapped.pData);
    for (const MipReadbackLayout& layout : layouts)
    {
        const uint8_t* mipSource = source + size_t(layout.blockOffset) * sizeof(BC1RawBlock);
        for (uint32_t y = 0; y < layout.blockHeight; ++y)
        {
            memcpy_s(layout.image->pixels + size_t(y) * layout.image->rowPitch, layout.image->rowPitch,
                mipSource + size_t(y) * layout.blockWidth * sizeof(BC1RawBlock),
                size_t(layout.blockWidth) * sizeof(BC1RawBlock));
        }
    }
    cache.context->Unmap(cache.stagingBuffer.Get(), 0);

    return S_OK;
}

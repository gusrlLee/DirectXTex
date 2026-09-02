// BC1 compression-domain mip generation for DirectCompute 5.0.
//
// The shader never reconstructs a full uncompressed source image. Mip 1 reads
// raw BC1 blocks, recovers compact color statistics, and writes new BC1 blocks.
// Later levels operate on a linear-RGB block-mean pyramid generated alongside
// mip 1. Every compute thread produces one destination block or one mean texel.

// Dimensions are measured in BC1 blocks for BC kernels and in mean texels for
// the downsampling kernel. DestinationOffset selects one mip inside the shared
// output buffer so every GPU dispatch can finish before a single CPU readback.
cbuffer MipConstants : register(b0)
{
    uint SourceWidth;
    uint SourceHeight;
    uint DestinationWidth;
    uint DestinationHeight;
    uint IsSrgb;
    uint DestinationOffset;
    uint2 Padding;
};

// One uint2 stores an entire BC1 block: packed endpoints in x and selectors in y.
StructuredBuffer<uint2> SourceBC1 : register(t0);
// Each float4 stores a linear RGB block mean; alpha is unused and kept for alignment.
StructuredBuffer<float4> SourceMeans : register(t1);
// All generated BC1 mip levels are appended to this common output buffer.
RWStructuredBuffer<uint2> DestinationBC1 : register(u0);
// Mip 1 and the downsampling kernel write the ping-pong mean pyramid here.
RWStructuredBuffer<float4> DestinationMeans : register(u1);

// Convert one normalized sRGB component to linear light using the standard curve.
float SrgbToLinear(float value)
{
    value = saturate(value);
    return value <= 0.04045f ? value / 12.92f : pow((value + 0.055f) / 1.055f, 2.4f);
}

// Convert one linear-light component back to normalized sRGB code space.
float LinearToSrgb(float value)
{
    value = saturate(value);
    return value <= 0.0031308f ? value * 12.92f : 1.055f * pow(value, 1.0f / 2.4f) - 0.055f;
}

// Expand one packed RGB565 endpoint to the exact RGB8 values used by BC1 hardware.
float3 DecodeRGB565(uint packed)
{
    // Extract the stored five-, six-, and five-bit channel values.
    const uint r5 = (packed >> 11) & 31u;
    const uint g6 = (packed >> 5) & 63u;
    const uint b5 = packed & 31u;
    // Bit replication matches DirectXTex and GPU BC1 endpoint expansion exactly.
    const uint3 rgb8 = uint3((r5 << 3) | (r5 >> 2), (g6 << 2) | (g6 >> 4), (b5 << 3) | (b5 >> 2));
    float3 result = float3(rgb8) / 255.0f;
    // All endpoint fitting is performed in linear RGB, even for an sRGB texture.
    if (IsSrgb != 0u)
    {
        result = float3(SrgbToLinear(result.r), SrgbToLinear(result.g), SrgbToLinear(result.b));
    }
    return result;
}

// Reconstruct the four opaque BC1 palette entries in linear RGB.
void BuildPalette(uint2 block, out float3 palette[4])
{
    // Endpoint 0 occupies the low 16 bits and endpoint 1 occupies the high 16 bits.
    const uint color0 = block.x & 0xffffu;
    const uint color1 = block.x >> 16;
    // Unpack and expand both endpoints before applying BC1 integer interpolation.
    const uint3 raw0 = uint3((color0 >> 11) & 31u, (color0 >> 5) & 63u, color0 & 31u);
    const uint3 raw1 = uint3((color1 >> 11) & 31u, (color1 >> 5) & 63u, color1 & 31u);
    const uint3 c0 = uint3((raw0.r << 3) | (raw0.r >> 2), (raw0.g << 2) | (raw0.g >> 4), (raw0.b << 3) | (raw0.b >> 2));
    const uint3 c1 = uint3((raw1.r << 3) | (raw1.r >> 2), (raw1.g << 2) | (raw1.g >> 4), (raw1.b << 3) | (raw1.b >> 2));
    
    // Opaque BC1 defines palette entries 2 and 3 as integer-weighted thirds.
    // Added 3-color mode fallback for endpoints where c0 <= c1
    const bool is4Color = color0 > color1;
    
    const uint3 c4_2 = (2u * c0 + c1) / 3u;
    const uint3 c4_3 = (c0 + 2u * c1) / 3u;
    
    const uint3 c3_2 = (c0 + c1) / 2u;
    const uint3 c3_3 = uint3(0u, 0u, 0u);
    
    const uint mask = is4Color ? 0xffffffffu : 0u; const uint3 c2 = (c4_2 & mask) | (c3_2 & ~mask);
    const uint3 c3 = (c4_3 & mask) | (c3_3 & ~mask);
    
    float3 p0 = float3(c0) / 255.0f;
    float3 p1 = float3(c1) / 255.0f;
    float3 p2 = float3(c2) / 255.0f;
    float3 p3 = float3(c3) / 255.0f;
    // Hardware interpolation happens in code space; linearization happens afterward.
    if (IsSrgb != 0u)
    {
        p0 = float3(SrgbToLinear(p0.r), SrgbToLinear(p0.g), SrgbToLinear(p0.b));
        p1 = float3(SrgbToLinear(p1.r), SrgbToLinear(p1.g), SrgbToLinear(p1.b));
        p2 = float3(SrgbToLinear(p2.r), SrgbToLinear(p2.g), SrgbToLinear(p2.b));
        p3 = float3(SrgbToLinear(p3.r), SrgbToLinear(p3.g), SrgbToLinear(p3.b));
    }
    palette[0] = p0;
    palette[1] = p1;
    palette[2] = p2;
    palette[3] = p3;
}

// Recover four 2x2 quadrant means and the full mean of one compressed parent block.
void DecodeQuadrants(uint2 block, out float3 quadrants[4], out float3 blockMean)
{
    // The palette plus the selector bitmap is sufficient to compute means directly.
    float3 palette[4];
    BuildPalette(block, palette);
    [unroll]
    for (uint quadrant = 0u; quadrant < 4u; ++quadrant)
    {
        // Convert quadrant number 0..3 into its top-left texel coordinate.
        const uint baseX = (quadrant & 1u) * 2u;
        const uint baseY = (quadrant >> 1u) * 2u;
        float3 sum = 0.0f;
        [unroll]
        for (uint y = 0u; y < 2u; ++y)
        {
            [unroll]
            for (uint x = 0u; x < 2u; ++x)
            {
                // Extract this texel's two-bit palette selector from the BC1 bitmap.
                const uint texel = (baseY + y) * 4u + baseX + x;
                sum += palette[(block.y >> (texel * 2u)) & 3u];
            }
        }
        quadrants[quadrant] = sum * 0.25f;
    }
    // Every quadrant has the same area, so the block mean is their simple average.
    blockMean = (quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3]) * 0.25f;
}

// Quantize a normalized endpoint and pack it into RGB565 layout.
uint PackRGB565(float3 color)
{
    const uint r = (uint)round(saturate(color.r) * 31.0f);
    const uint g = (uint)round(saturate(color.g) * 63.0f);
    const uint b = (uint)round(saturate(color.b) * 31.0f);
    return (r << 11) | (g << 5) | b;
}

// Decode a newly quantized endpoint using the same path as an existing BC1 endpoint.
float3 DecodePackedEndpoint(uint packed)
{
    return DecodeRGB565(packed);
}

// Fit and emit one opaque BC1 block from sixteen linear-RGB representative samples.
uint2 EncodeSamples(float3 samples[16])
{
    // Compute the centroid of the reconstructed sample distribution.
    float3 mean = 0.0f;
    [unroll]
    for (uint i = 0u; i < 16u; ++i)
    {
        mean += samples[i];
    }
    mean *= 1.0f / 16.0f;

    // Build the symmetric RGB covariance matrix around that centroid.
    float3x3 covariance = (float3x3)0.0f;
    [unroll]
    for (uint c = 0u; c < 16u; ++c)
    {
        const float3 delta = samples[c] - mean;
        // Each row accumulates one component of the outer product delta * delta^T.
        covariance[0] += delta.r * delta;
        covariance[1] += delta.g * delta;
        covariance[2] += delta.b * delta;
    }
    covariance *= 1.0f / 16.0f;

    // One power-iteration step estimates the dominant color direction. The epsilon
    // prevents normalize from receiving a zero vector for a perfectly flat block.
    float3 axis = normalize(mul(covariance, float3(0.57735f, 0.57735f, 0.57735f)) + 1e-20f);
    float minimum = 10000.0f;
    float maximum = -10000.0f;
    [unroll]
    // Find the minimum and maximum sample projections along the principal axis.
    for (uint p = 0u; p < 16u; ++p)
    {
        const float projection = dot(axis, samples[p] - mean);
        minimum = min(minimum, projection);
        maximum = max(maximum, projection);
    }

    // The projection extremes form the initial endpoint pair.
    float3 endpoint0 = mean + axis * minimum;
    float3 endpoint1 = mean + axis * maximum;
    const float3 direction0 = endpoint1 - endpoint0;
    const float inverseLengthSquared = rcp(dot(direction0, direction0) + 1e-12f);
    float weightSum = 0.0f;
    float weightSquaredSum = 0.0f;
    float3 weighted = 0.0f;
    [unroll]
    // Temporarily assign each sample to the nearest continuous position on the
    // endpoint line, then snap it to one of BC1's four selector weights.
    for (uint s = 0u; s < 16u; ++s)
    {
        float weight = saturate(dot(samples[s] - endpoint0, direction0) * inverseLengthSquared);
        weight = round(weight * 3.0f) / 3.0f;
        weightSum += weight;
        weightSquaredSum += weight * weight;
        weighted += weight * samples[s];
    }

    // Solve the two-endpoint least-squares normal equation with selectors fixed.
    const float determinant = 16.0f * weightSquaredSum - weightSum * weightSum;
    // A nearly singular system represents a flat block and safely collapses to its mean.
    if (determinant < 1e-6f)
    {
        endpoint0 = mean;
        endpoint1 = mean;
    }
    else
    {
        const float3 total = mean * 16.0f;
        endpoint0 = (weightSquaredSum * total - weightSum * weighted) / determinant;
        const float3 direction = (16.0f * weighted - weightSum * total) / determinant;
        endpoint1 = endpoint0 + direction;
    }

    // Endpoints were fitted in linear light, but BC1 stores code values, so move
    // them onto the transfer curve before quantization. A UNORM palette already
    // holds linear code values and needs no conversion.
    if (IsSrgb != 0u)
    {
        endpoint0 = float3(LinearToSrgb(endpoint0.r), LinearToSrgb(endpoint0.g), LinearToSrgb(endpoint0.b));
        endpoint1 = float3(LinearToSrgb(endpoint1.r), LinearToSrgb(endpoint1.g), LinearToSrgb(endpoint1.b));
    }

    // Quantize the optimized endpoint pair into the actual stored BC1 representation.
    uint packed0 = PackRGB565(endpoint0);
    uint packed1 = PackRGB565(endpoint1);
    // Separate equal endpoints so the block cannot accidentally enter three-color mode.
    if (packed0 == packed1)
    {
        if ((packed1 & 31u) != 0u)
            --packed1;
        else
            ++packed0;
    }
    // Opaque BC1 requires endpoint 0 to be numerically greater than endpoint 1.
    if (packed1 > packed0)
    {
        const uint temporary = packed0;
        packed0 = packed1;
        packed1 = temporary;
    }

    // Rebuild the palette after quantization before assigning final selectors.
    float3 palette[4];
    BuildPalette(uint2(packed0 | (packed1 << 16), 0u), palette);
    uint selectors = 0u;
    [unroll]
    for (uint t = 0u; t < 16u; ++t)
    {
        // Choose the palette color with the smallest squared linear-RGB error.
        uint best = 0u;
        float bestDistance = dot(samples[t] - palette[0], samples[t] - palette[0]);
        [unroll]
        for (uint candidate = 1u; candidate < 4u; ++candidate)
        {
            const float3 delta = samples[t] - palette[candidate];
            const float distance = dot(delta, delta);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = candidate;
            }
        }
        // Store the winning two-bit selector at its BC1 texel position.
        selectors |= best << (t * 2u);
    }
    return uint2(packed0 | (packed1 << 16), selectors);
}

// Convert a possibly out-of-range 2D coordinate into a clamp-to-edge linear index.
uint ClampIndex(uint x, uint y, uint width, uint height)
{
    return min(y, height - 1u) * width + min(x, width - 1u);
}

[numthreads(8, 8, 1)]
void GenerateMip1CS(uint3 id : SV_DispatchThreadID)
{
    // Dispatch dimensions are rounded to thread-group size, so discard excess threads.
    if (id.x >= DestinationWidth || id.y >= DestinationHeight)
        return;

    // One destination block represents a clamp-to-edge 2x2 group of source BC1 blocks.
    const uint x0 = min(id.x * 2u, SourceWidth - 1u);
    const uint x1 = min(x0 + 1u, SourceWidth - 1u);
    const uint y0 = min(id.y * 2u, SourceHeight - 1u);
    const uint y1 = min(y0 + 1u, SourceHeight - 1u);
    const uint indices[4] = { y0 * SourceWidth + x0, y0 * SourceWidth + x1, y1 * SourceWidth + x0, y1 * SourceWidth + x1 };
    // Recover four quadrant colors and one block mean from every compressed parent.
    float3 quadrants[4][4];
    float3 means[4];
    [unroll]
    for (uint parent = 0u; parent < 4u; ++parent)
    {
        DecodeQuadrants(SourceBC1[indices[parent]], quadrants[parent], means[parent]);
    }

    // Arrange the sixteen parent quadrants in the spatial order of the new 4x4 block.
    float3 samples[16];
    [unroll]
    for (uint q = 0u; q < 4u; ++q)
    {
        const uint ox = (q & 1u) * 2u;
        const uint oy = (q >> 1u) * 2u;
        [unroll]
        for (uint k = 0u; k < 4u; ++k)
        {
            const uint tx = ox + (k & 1u);
            const uint ty = oy + (k >> 1u);
            samples[ty * 4u + tx] = quadrants[q][k];
        }
    }
    // Fit and append the completed destination BC1 block to the shared output buffer.
    DestinationBC1[DestinationOffset + id.y * DestinationWidth + id.x] = EncodeSamples(samples);

    // Seed the base-resolution mean image. Conditional edge writes avoid two threads
    // targeting the same source block when a block dimension is odd.
    DestinationMeans[indices[0]] = float4(means[0], 0.0f);
    if (x1 != x0) DestinationMeans[indices[1]] = float4(means[1], 0.0f);
    if (y1 != y0)
    {
        DestinationMeans[indices[2]] = float4(means[2], 0.0f);
        if (x1 != x0) DestinationMeans[indices[3]] = float4(means[3], 0.0f);
    }
}

[numthreads(8, 8, 1)]
void GenerateMipFromMeansCS(uint3 id : SV_DispatchThreadID)
{
    // Later mip levels use the linear mean pyramid instead of reading BC1 parents.
    if (id.x >= DestinationWidth || id.y >= DestinationHeight)
        return;
    // Gather one clamp-to-edge 4x4 mean footprint for this destination block.
    float3 samples[16];
    [unroll]
    for (uint y = 0u; y < 4u; ++y)
    {
        [unroll]
        for (uint x = 0u; x < 4u; ++x)
        {
            samples[y * 4u + x] = SourceMeans[ClampIndex(id.x * 4u + x, id.y * 4u + y, SourceWidth, SourceHeight)].rgb;
        }
    }
    DestinationBC1[DestinationOffset + id.y * DestinationWidth + id.x] = EncodeSamples(samples);
}

[numthreads(8, 8, 1)]
void DownsampleMeansCS(uint3 id : SV_DispatchThreadID)
{
    // This kernel halves the mean image before mip 3 and every subsequent level.
    if (id.x >= DestinationWidth || id.y >= DestinationHeight)
        return;
    // Clamp the 2x2 source footprint so odd dimensions repeat their final row or column.
    const uint x0 = min(id.x * 2u, SourceWidth - 1u);
    const uint x1 = min(x0 + 1u, SourceWidth - 1u);
    const uint y0 = min(id.y * 2u, SourceHeight - 1u);
    const uint y1 = min(y0 + 1u, SourceHeight - 1u);
    // Equal-area averaging preserves the linear mean represented by each texel.
    const float3 mean = (SourceMeans[y0 * SourceWidth + x0].rgb + SourceMeans[y0 * SourceWidth + x1].rgb
        + SourceMeans[y1 * SourceWidth + x0].rgb + SourceMeans[y1 * SourceWidth + x1].rgb) * 0.25f;
    DestinationMeans[id.y * DestinationWidth + id.x] = float4(mean, 0.0f);
}

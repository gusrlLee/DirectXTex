#include "DirectXTexP.h"
#include "BC.h"

// Standard containers and allocation helpers used by the mean-image pyramid.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <new>

using namespace DirectX;
// A BC1 block must contain two 16-bit endpoints and one 32-bit selector bitmap.
static_assert(sizeof(D3DX_BC1) == 8, "D3DX_BC1 must be 8 bytes");

namespace // Common SIMD wrappers
{
    // These wrappers perform integer SSE operations while keeping values in XMVECTOR registers.
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

    // Subtract every 32-bit integer SIMD lane.
    inline XMVECTOR SubtractInt32(FXMVECTOR value0, FXMVECTOR value1) noexcept
    {
        return _mm_castsi128_ps(_mm_sub_epi32(_mm_castps_si128(value0), _mm_castps_si128(value1)));
    }

    // Compare signed 32-bit lanes. BC1 packed values are always below INT32_MAX.
    inline XMVECTOR GreaterInt32(FXMVECTOR value0, FXMVECTOR value1) noexcept
    {
        return _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_castps_si128(value0), _mm_castps_si128(value1)));
    }

    // Divide palette sums in [0, 765] by 3 using integer SIMD.
    inline XMVECTOR DividePaletteSumBy3(FXMVECTOR value) noexcept
    {
        // For values up to 765, (value * 683) >> 11
        // is exactly equal to integer division by 3.
        XMVECTOR product = AddInt32(ShiftLeft32<9>(value), ShiftLeft32<7>(value));
        product = AddInt32(product, ShiftLeft32<5>(value));
        product = AddInt32(product, ShiftLeft32<3>(value));
        product = AddInt32(product, ShiftLeft32<1>(value));
        product = AddInt32(product, value);

        return ShiftRight32<11>(product);
    }

    // Build the sRGB8 lookup table with DirectXMath conversion rules.
    inline const std::array<float, 256>& GetSrgb8ToLinearTable() noexcept
    {
        // Function-local static initialization builds this table only once and is thread-safe.
        static const std::array<float, 256> table = []
            {
                std::array<float, 256> values{};

                constexpr float scale = 1.0f / 255.0f;

                // Convert every possible 8-bit sRGB code through DirectXMath's reference curve.
                for (size_t index = 0; index < values.size(); ++index)
                {
                    const float srgb = static_cast<float>(index) * scale;
                    const XMVECTOR replicated = XMVectorReplicate(srgb);
                    const XMVECTOR linear = XMColorSRGBToRGB(replicated);

                    values[index] = XMVectorGetX(linear);
                }

                return values;
            }();

            // Later block operations use indexed lookup instead of evaluating the transfer curve.
        return table;
    }
} // namespace Common SIMD wrappers

namespace // for bc1
{
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

    // Lee: Linear RGB palette colors for four opaque BC1 blocks.
    struct LinearPaletteBC1Batch
    {
        XMVECTOR c0R;
        XMVECTOR c0G;
        XMVECTOR c0B;

        XMVECTOR c1R;
        XMVECTOR c1G;
        XMVECTOR c1B;

        XMVECTOR c2R;
        XMVECTOR c2G;
        XMVECTOR c2B;

        XMVECTOR c3R;
        XMVECTOR c3G;
        XMVECTOR c3B;
    };

    // Linear RGB means of four 2x2 quadrants for four BC1 blocks.
    struct QuadrantMeansBatch
    {
        XMVECTOR q0R;
        XMVECTOR q0G;
        XMVECTOR q0B;

        XMVECTOR q1R;
        XMVECTOR q1G;
        XMVECTOR q1B;

        XMVECTOR q2R;
        XMVECTOR q2G;
        XMVECTOR q2B;

        XMVECTOR q3R;
        XMVECTOR q3G;
        XMVECTOR q3B;
    };

    // Linear RGB values for four independent SIMD lanes.
    struct LinearRGBBatch
    {
        XMVECTOR r;
        XMVECTOR g;
        XMVECTOR b;
    };

    // Symmetric 3x3 covariance matrices for four SIMD lanes.
    struct CovarianceMatrixBatch
    {
        XMVECTOR rr;
        XMVECTOR gg;
        XMVECTOR bb;

        XMVECTOR rg;
        XMVECTOR rb;
        XMVECTOR gb;
    };

    // Mean and within-block covariance for four parent BC1 blocks.
    struct ParentStatisticsBatch
    {
        LinearRGBBatch mean;
        CovarianceMatrixBatch withinCovariance;
    };

    // Linear means of the four source blocks used by one destination block.
    struct SourceBlockMeansBatch
    {
        LinearRGBBatch p00;
        LinearRGBBatch p10;
        LinearRGBBatch p01;
        LinearRGBBatch p11;
    };

    // Two RGB endpoints for four independent blocks.
    struct EndpointPairBatch
    {
        LinearRGBBatch p0;
        LinearRGBBatch p1;
    };

    // Projection axis and center used by the PCA range search.
    struct ProjectionContextBatch
    {
        LinearRGBBatch axis;
        LinearRGBBatch mean;
    };

    // State used while assigning least-squares palette weights.
    struct LeastSquaresContextBatch
    {
        LinearRGBBatch p0;
        LinearRGBBatch direction;
        XMVECTOR inverseLengthSquared;
    };

    // Sufficient statistics for the endpoint least-squares solve.
    struct LeastSquaresAccumulatorBatch
    {
        XMVECTOR weightSum;
        XMVECTOR weightSquaredSum;
        LinearRGBBatch weighted;
    };

    // Scalar linear block mean used by the higher mip mean pyramid.
    struct LinearBlockMean
    {
        float r;
        float g;
        float b;
    };

    // Lee: Load four BC1 blocks into SIMD lane
    inline BC1BlockBatch LoadBC1BlockBatch(
        const D3DX_BC1* blocks) noexcept
    {
        // One SIMD lane represents one independent BC1 block throughout the encoder.
        BC1BlockBatch batch;
        batch.color0 = XMVectorSetInt(blocks[0].rgb[0], blocks[1].rgb[0], blocks[2].rgb[0], blocks[3].rgb[0]);
        batch.color1 = XMVectorSetInt(blocks[0].rgb[1], blocks[1].rgb[1], blocks[2].rgb[1], blocks[3].rgb[1]);
        batch.selectors = XMVectorSetInt(blocks[0].bitmap, blocks[1].bitmap, blocks[2].bitmap, blocks[3].bitmap);

        return batch;
    }

    // Lee: Decode four RGB565 endpoints into RGB8 channels.
    inline RGB8Batch DecodeRGB565Batch(XMVECTOR packed) noexcept
    {
        // RGB565 stores red and blue in five bits and green in six bits.
        const XMVECTOR mask5 = XMVectorReplicateInt(0x1Fu);
        const XMVECTOR mask6 = XMVectorReplicateInt(0x3Fu);

        // Isolate each packed endpoint component without converting lanes to scalar values.
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

    // Normalize four unsigned 8-bit values into the [0, 1] range.
    inline XMVECTOR NormalizeUNorm8Batch(FXMVECTOR values) noexcept
    {
        const XMVECTOR floatValues = XMConvertVectorUIntToFloat(values, 0);
        return XMVectorScale(floatValues, 1.0f / 255.0f);
    }

    // Convert four sRGB8 code values to linear values.
    inline XMVECTOR Srgb8ToLinearBatch(FXMVECTOR values) noexcept
    {
        const auto& table = GetSrgb8ToLinearTable();

        return XMVectorSet(
            table[XMVectorGetIntX(values)],
            table[XMVectorGetIntY(values)],
            table[XMVectorGetIntZ(values)],
            table[XMVectorGetIntW(values)]);
    }

    // Convert RGB8 code values according to the texture color space.
    template<bool IsSrgb>
    inline XMVECTOR ConvertRGB8ToLinearBatch(FXMVECTOR values) noexcept
    {
        if (IsSrgb)
        {
            return Srgb8ToLinearBatch(values);
        }
        else
        {
            return NormalizeUNorm8Batch(values);
        }
    }

    // Lee: Count selector flags in four 2x2 regions using SWAR 
    inline XMVECTOR Count2x2Regions(FXMVECTOR flags) noexcept
    {
        // Each nibble will accumulate the number of matching selectors in one 2x2 quadrant.
        const XMVECTOR horizontalMask = XMVectorReplicateInt(0x11111111u);

        // Sum horizontally adjacent flag bits.
        const XMVECTOR horizontal = AddInt32(
            XMVectorAndInt(flags, horizontalMask),
            XMVectorAndInt(ShiftRight32<2>(flags),
                horizontalMask));

        const XMVECTOR verticalMask = XMVectorReplicateInt(0x00FF00FFu);

        // Sum vertically adjacent pairs to complete each 2x2 region.
        return AddInt32(
            XMVectorAndInt(horizontal, verticalMask),
            XMVectorAndInt(ShiftRight32<8>(horizontal),
                verticalMask));
    }

    // Build linear hardware palettes for four opaque BC1 blocks.
    template<bool IsSrgb>
    inline LinearPaletteBC1Batch BuildOpaqueLinearPaletteBC1Batch(FXMVECTOR packedColor0, FXMVECTOR packedColor1) noexcept
    {
        // Decode the two stored endpoints before reconstructing the hardware interpolation colors.
        const RGB8Batch color0 = DecodeRGB565Batch(packedColor0);
        const RGB8Batch color1 = DecodeRGB565Batch(packedColor1);

        RGB8Batch color2{};
        RGB8Batch color3{};

        const XMVECTOR is4Color = GreaterInt32(packedColor0, packedColor1);

        const XMVECTOR c4_2r = DividePaletteSumBy3(AddInt32(ShiftLeft32<1>(color0.r), color1.r));
        const XMVECTOR c4_2g = DividePaletteSumBy3(AddInt32(ShiftLeft32<1>(color0.g), color1.g));
        const XMVECTOR c4_2b = DividePaletteSumBy3(AddInt32(ShiftLeft32<1>(color0.b), color1.b));
        const XMVECTOR c4_3r = DividePaletteSumBy3(AddInt32(color0.r, ShiftLeft32<1>(color1.r)));
        const XMVECTOR c4_3g = DividePaletteSumBy3(AddInt32(color0.g, ShiftLeft32<1>(color1.g)));
        const XMVECTOR c4_3b = DividePaletteSumBy3(AddInt32(color0.b, ShiftLeft32<1>(color1.b)));

        const XMVECTOR c3_2r = ShiftRight32<1>(AddInt32(color0.r, color1.r));
        const XMVECTOR c3_2g = ShiftRight32<1>(AddInt32(color0.g, color1.g));
        const XMVECTOR c3_2b = ShiftRight32<1>(AddInt32(color0.b, color1.b));

        color2.r = XMVectorSelect(c3_2r, c4_2r, is4Color);
        color2.g = XMVectorSelect(c3_2g, c4_2g, is4Color);
        color2.b = XMVectorSelect(c3_2b, c4_2b, is4Color);

        color3.r = XMVectorSelect(XMVectorZero(), c4_3r, is4Color);
        color3.g = XMVectorSelect(XMVectorZero(), c4_3g, is4Color);
        color3.b = XMVectorSelect(XMVectorZero(), c4_3b, is4Color);

        LinearPaletteBC1Batch palette{};

        // Convert only after integer interpolation so the values match actual BC1 decoding hardware.
        palette.c0R = ConvertRGB8ToLinearBatch<IsSrgb>(color0.r);
        palette.c0G = ConvertRGB8ToLinearBatch<IsSrgb>(color0.g);
        palette.c0B = ConvertRGB8ToLinearBatch<IsSrgb>(color0.b);

        palette.c1R = ConvertRGB8ToLinearBatch<IsSrgb>(color1.r);
        palette.c1G = ConvertRGB8ToLinearBatch<IsSrgb>(color1.g);
        palette.c1B = ConvertRGB8ToLinearBatch<IsSrgb>(color1.b);

        palette.c2R = ConvertRGB8ToLinearBatch<IsSrgb>(color2.r);
        palette.c2G = ConvertRGB8ToLinearBatch<IsSrgb>(color2.g);
        palette.c2B = ConvertRGB8ToLinearBatch<IsSrgb>(color2.b);

        palette.c3R = ConvertRGB8ToLinearBatch<IsSrgb>(color3.r);
        palette.c3G = ConvertRGB8ToLinearBatch<IsSrgb>(color3.g);
        palette.c3B = ConvertRGB8ToLinearBatch<IsSrgb>(color3.b);

        return palette;
    }

    // Lee: Extract 2x2 selector histograms from four BC1 blocks in parallel.
    inline void Extract2x2SelectorHistograms(
        FXMVECTOR packedSelectors,
        XMVECTOR& histogram0,
        XMVECTOR& histogram1,
        XMVECTOR& histogram2,
        XMVECTOR& histogram3) noexcept
    {
        // BC1 stores one two-bit selector for each texel in a 32-bit bitmap.
        const XMVECTOR lowBitMask = XMVectorReplicateInt(0x55555555u);

        // Separate the low and high bits of every 2-bit selector.
        const XMVECTOR lowBits = XMVectorAndInt(packedSelectors, lowBitMask);
        const XMVECTOR highBits = XMVectorAndInt(ShiftRight32<1>(packedSelectors), lowBitMask);

        // Build one flag bit for each selector value.
        const XMVECTOR flags0 = XMVectorAndInt(XMVectorNorInt(lowBits, highBits), lowBitMask);
        const XMVECTOR flags1 = XMVectorAndCInt(lowBits, highBits);
        const XMVECTOR flags2 = XMVectorAndCInt(highBits, lowBits);
        const XMVECTOR flags3 = XMVectorAndInt(lowBits, highBits);

        // Count how often each palette index appears in every 2x2 quadrant.
        histogram0 = Count2x2Regions(flags0);
        histogram1 = Count2x2Regions(flags1);
        histogram2 = Count2x2Regions(flags2);
        histogram3 = Count2x2Regions(flags3);
    }

    // Extract one quadrant count and convert it to a normalized weight.
    template<int BitOffset>
    inline XMVECTOR ExtractQuadrantWeight(FXMVECTOR histogram) noexcept
    {
        static_assert(BitOffset == 0 || BitOffset == 4 || BitOffset == 16 || BitOffset == 20, "Invalid BC1 quadrant histogram offset");

        const XMVECTOR countMask = XMVectorReplicateInt(0xFu);
        const XMVECTOR counts = XMVectorAndInt(ShiftRight32<BitOffset>(histogram), countMask);

        return XMVectorScale(XMConvertVectorUIntToFloat(counts, 0), 0.25f);
    }

    // Compute one weighted linear RGB quadrant mean.
    inline LinearRGBBatch ComputePaletteMean(
        const LinearPaletteBC1Batch& palette,
        FXMVECTOR weight0,
        FXMVECTOR weight1,
        FXMVECTOR weight2,
        GXMVECTOR weight3) noexcept
    {
        // The four weights are normalized selector counts, so their weighted sum is the quadrant mean.
        LinearRGBBatch result{};

        result.r = XMVectorMultiply(palette.c0R, weight0);
        result.r = XMVectorMultiplyAdd(palette.c1R, weight1, result.r);
        result.r = XMVectorMultiplyAdd(palette.c2R, weight2, result.r);
        result.r = XMVectorMultiplyAdd(palette.c3R, weight3, result.r);

        result.g = XMVectorMultiply(palette.c0G, weight0);
        result.g = XMVectorMultiplyAdd(palette.c1G, weight1, result.g);
        result.g = XMVectorMultiplyAdd(palette.c2G, weight2, result.g);
        result.g = XMVectorMultiplyAdd(palette.c3G, weight3, result.g);

        result.b = XMVectorMultiply(palette.c0B, weight0);
        result.b = XMVectorMultiplyAdd(palette.c1B, weight1, result.b);
        result.b = XMVectorMultiplyAdd(palette.c2B, weight2, result.b);
        result.b = XMVectorMultiplyAdd(palette.c3B, weight3, result.b);

        return result;
    }

    // Compute one quadrant mean from four selector histograms.
    template<int BitOffset>
    inline LinearRGBBatch ComputeQuadrantMean(
        const LinearPaletteBC1Batch& palette,
        FXMVECTOR histogram0,
        FXMVECTOR histogram1,
        FXMVECTOR histogram2,
        GXMVECTOR histogram3) noexcept
    {
        const XMVECTOR weight0 = ExtractQuadrantWeight<BitOffset>(histogram0);
        const XMVECTOR weight1 = ExtractQuadrantWeight<BitOffset>(histogram1);
        const XMVECTOR weight2 = ExtractQuadrantWeight<BitOffset>(histogram2);
        const XMVECTOR weight3 = ExtractQuadrantWeight<BitOffset>(histogram3);

        return ComputePaletteMean(palette, weight0, weight1, weight2, weight3);
    }

    // Compute four linear quadrant means for four parent BC1 blocks.
    template<bool IsSrgb>
    inline QuadrantMeansBatch ComputeParentQuadrantMeansBatch(
        FXMVECTOR packedColor0,
        FXMVECTOR packedColor1,
        FXMVECTOR histogram0,
        GXMVECTOR histogram1,
        HXMVECTOR histogram2,
        HXMVECTOR histogram3) noexcept
    {
        // Reconstruct one linear palette per SIMD lane from the compressed endpoints.
        const LinearPaletteBC1Batch palette = BuildOpaqueLinearPaletteBC1Batch<IsSrgb>(packedColor0, packedColor1);

        // Selector histograms recover the average color of each 2x2 quadrant without decoding 16 texels.
        const LinearRGBBatch q0 = ComputeQuadrantMean<0>(palette, histogram0, histogram1, histogram2, histogram3);
        const LinearRGBBatch q1 = ComputeQuadrantMean<4>(palette, histogram0, histogram1, histogram2, histogram3);
        const LinearRGBBatch q2 = ComputeQuadrantMean<16>(palette, histogram0, histogram1, histogram2, histogram3);
        const LinearRGBBatch q3 = ComputeQuadrantMean<20>(palette, histogram0, histogram1, histogram2, histogram3);

        QuadrantMeansBatch result{};

        result.q0R = q0.r;
        result.q0G = q0.g;
        result.q0B = q0.b;

        result.q1R = q1.r;
        result.q1G = q1.g;
        result.q1B = q1.b;

        result.q2R = q2.r;
        result.q2G = q2.g;
        result.q2B = q2.b;

        result.q3R = q3.r;
        result.q3G = q3.g;
        result.q3B = q3.b;

        return result;
    }

    // Compute the linear mean of each parent BC1 block.
    inline LinearRGBBatch ComputeBlockMeansBatch(const QuadrantMeansBatch& quadrants) noexcept
    {
        LinearRGBBatch result{};

        result.r = XMVectorAdd(quadrants.q0R, quadrants.q1R);
        result.r = XMVectorAdd(result.r, quadrants.q2R);
        result.r = XMVectorAdd(result.r, quadrants.q3R);
        result.r = XMVectorScale(result.r, 0.25f);

        result.g = XMVectorAdd(quadrants.q0G, quadrants.q1G);
        result.g = XMVectorAdd(result.g, quadrants.q2G);
        result.g = XMVectorAdd(result.g, quadrants.q3G);
        result.g = XMVectorScale(result.g, 0.25f);

        result.b = XMVectorAdd(quadrants.q0B, quadrants.q1B);
        result.b = XMVectorAdd(result.b, quadrants.q2B);
        result.b = XMVectorAdd(result.b, quadrants.q3B);
        result.b = XMVectorScale(result.b, 0.25f);

        return result;
    }

    // Compute mean and within-block covariance for four parent blocks.
    inline ParentStatisticsBatch ComputeParentStatisticsBatch(const QuadrantMeansBatch& parent) noexcept
    {
        // The mean describes the parent block center; covariance describes its local color variation.
        ParentStatisticsBatch result{};
        result.mean = ComputeBlockMeansBatch(parent);

        // Measure each quadrant's displacement from the parent mean.
        const XMVECTOR d0R = XMVectorSubtract(parent.q0R, result.mean.r);
        const XMVECTOR d0G = XMVectorSubtract(parent.q0G, result.mean.g);
        const XMVECTOR d0B = XMVectorSubtract(parent.q0B, result.mean.b);

        const XMVECTOR d1R = XMVectorSubtract(parent.q1R, result.mean.r);
        const XMVECTOR d1G = XMVectorSubtract(parent.q1G, result.mean.g);
        const XMVECTOR d1B = XMVectorSubtract(parent.q1B, result.mean.b);

        const XMVECTOR d2R = XMVectorSubtract(parent.q2R, result.mean.r);
        const XMVECTOR d2G = XMVectorSubtract(parent.q2G, result.mean.g);
        const XMVECTOR d2B = XMVectorSubtract(parent.q2B, result.mean.b);

        const XMVECTOR d3R = XMVectorSubtract(parent.q3R, result.mean.r);
        const XMVECTOR d3G = XMVectorSubtract(parent.q3G, result.mean.g);
        const XMVECTOR d3B = XMVectorSubtract(parent.q3B, result.mean.b);

        // Accumulate the six unique entries of the symmetric 3x3 covariance matrix.
        result.withinCovariance.rr = XMVectorMultiply(d0R, d0R);
        result.withinCovariance.rr = XMVectorMultiplyAdd(d1R, d1R, result.withinCovariance.rr);
        result.withinCovariance.rr = XMVectorMultiplyAdd(d2R, d2R, result.withinCovariance.rr);
        result.withinCovariance.rr = XMVectorMultiplyAdd(d3R, d3R, result.withinCovariance.rr);
        result.withinCovariance.rr = XMVectorScale(result.withinCovariance.rr, 0.25f);

        result.withinCovariance.gg = XMVectorMultiply(d0G, d0G);
        result.withinCovariance.gg = XMVectorMultiplyAdd(d1G, d1G, result.withinCovariance.gg);
        result.withinCovariance.gg = XMVectorMultiplyAdd(d2G, d2G, result.withinCovariance.gg);
        result.withinCovariance.gg = XMVectorMultiplyAdd(d3G, d3G, result.withinCovariance.gg);
        result.withinCovariance.gg = XMVectorScale(result.withinCovariance.gg, 0.25f);

        result.withinCovariance.bb = XMVectorMultiply(d0B, d0B);
        result.withinCovariance.bb = XMVectorMultiplyAdd(d1B, d1B, result.withinCovariance.bb);
        result.withinCovariance.bb = XMVectorMultiplyAdd(d2B, d2B, result.withinCovariance.bb);
        result.withinCovariance.bb = XMVectorMultiplyAdd(d3B, d3B, result.withinCovariance.bb);
        result.withinCovariance.bb = XMVectorScale(result.withinCovariance.bb, 0.25f);

        result.withinCovariance.rg = XMVectorMultiply(d0R, d0G);
        result.withinCovariance.rg = XMVectorMultiplyAdd(d1R, d1G, result.withinCovariance.rg);
        result.withinCovariance.rg = XMVectorMultiplyAdd(d2R, d2G, result.withinCovariance.rg);
        result.withinCovariance.rg = XMVectorMultiplyAdd(d3R, d3G, result.withinCovariance.rg);
        result.withinCovariance.rg = XMVectorScale(result.withinCovariance.rg, 0.25f);

        result.withinCovariance.rb = XMVectorMultiply(d0R, d0B);
        result.withinCovariance.rb = XMVectorMultiplyAdd(d1R, d1B, result.withinCovariance.rb);
        result.withinCovariance.rb = XMVectorMultiplyAdd(d2R, d2B, result.withinCovariance.rb);
        result.withinCovariance.rb = XMVectorMultiplyAdd(d3R, d3B, result.withinCovariance.rb);
        result.withinCovariance.rb = XMVectorScale(result.withinCovariance.rb, 0.25f);

        result.withinCovariance.gb = XMVectorMultiply(d0G, d0B);
        result.withinCovariance.gb = XMVectorMultiplyAdd(d1G, d1B, result.withinCovariance.gb);
        result.withinCovariance.gb = XMVectorMultiplyAdd(d2G, d2B, result.withinCovariance.gb);
        result.withinCovariance.gb = XMVectorMultiplyAdd(d3G, d3B, result.withinCovariance.gb);
        result.withinCovariance.gb = XMVectorScale(result.withinCovariance.gb, 0.25f);

        return result;
    }

    // Average four floating-point SIMD vectors.
    inline XMVECTOR MeanFourVectors(
        FXMVECTOR value0,
        FXMVECTOR value1,
        FXMVECTOR value2,
        GXMVECTOR value3) noexcept
    {
        XMVECTOR result = XMVectorAdd(value0, value1);
        result = XMVectorAdd(result, value2);
        result = XMVectorAdd(result, value3);
        return XMVectorScale(result, 0.25f);
    }

    // Add the between-parent covariance contribution.
    inline void AccumulateBetweenParentCovariance(
        const ParentStatisticsBatch& parent,
        const LinearRGBBatch& mean,
        CovarianceMatrixBatch& covariance) noexcept
    {
        // This is the between-group term of the law of total covariance.
        const XMVECTOR deltaR = XMVectorSubtract(parent.mean.r, mean.r);
        const XMVECTOR deltaG = XMVectorSubtract(parent.mean.g, mean.g);
        const XMVECTOR deltaB = XMVectorSubtract(parent.mean.b, mean.b);

        covariance.rr = XMVectorMultiplyAdd(deltaR, XMVectorScale(deltaR, 0.25f), covariance.rr);
        covariance.gg = XMVectorMultiplyAdd(deltaG, XMVectorScale(deltaG, 0.25f), covariance.gg);
        covariance.bb = XMVectorMultiplyAdd(deltaB, XMVectorScale(deltaB, 0.25f), covariance.bb);
        covariance.rg = XMVectorMultiplyAdd(deltaR, XMVectorScale(deltaG, 0.25f), covariance.rg);
        covariance.rb = XMVectorMultiplyAdd(deltaR, XMVectorScale(deltaB, 0.25f), covariance.rb);
        covariance.gb = XMVectorMultiplyAdd(deltaG, XMVectorScale(deltaB, 0.25f), covariance.gb);
    }

    // Combine four parent distributions with the ANOVA covariance identity.
    inline void ComputeChildBlockMoments(
        const QuadrantMeansBatch& p00,
        const QuadrantMeansBatch& p10,
        const QuadrantMeansBatch& p01,
        const QuadrantMeansBatch& p11,
        SourceBlockMeansBatch& sourceMeans,
        LinearRGBBatch& mean,
        CovarianceMatrixBatch& covariance) noexcept
    {
        // First recover the mean and internal covariance of each of the four parent blocks.
        const ParentStatisticsBatch stats00 = ComputeParentStatisticsBatch(p00);
        const ParentStatisticsBatch stats10 = ComputeParentStatisticsBatch(p10);
        const ParentStatisticsBatch stats01 = ComputeParentStatisticsBatch(p01);
        const ParentStatisticsBatch stats11 = ComputeParentStatisticsBatch(p11);

        // Preserve parent means because higher mip levels operate on the mean pyramid.
        sourceMeans.p00 = stats00.mean;
        sourceMeans.p10 = stats10.mean;
        sourceMeans.p01 = stats01.mean;
        sourceMeans.p11 = stats11.mean;

        // The child block mean is the equally weighted mean of its four compressed parents.
        mean.r = MeanFourVectors(stats00.mean.r, stats10.mean.r, stats01.mean.r, stats11.mean.r);
        mean.g = MeanFourVectors(stats00.mean.g, stats10.mean.g, stats01.mean.g, stats11.mean.g);
        mean.b = MeanFourVectors(stats00.mean.b, stats10.mean.b, stats01.mean.b, stats11.mean.b);

        // Average color variation that already exists inside the parent blocks.
        CovarianceMatrixBatch within{};
        within.rr = MeanFourVectors(stats00.withinCovariance.rr, stats10.withinCovariance.rr, stats01.withinCovariance.rr, stats11.withinCovariance.rr);
        within.gg = MeanFourVectors(stats00.withinCovariance.gg, stats10.withinCovariance.gg, stats01.withinCovariance.gg, stats11.withinCovariance.gg);
        within.bb = MeanFourVectors(stats00.withinCovariance.bb, stats10.withinCovariance.bb, stats01.withinCovariance.bb, stats11.withinCovariance.bb);
        within.rg = MeanFourVectors(stats00.withinCovariance.rg, stats10.withinCovariance.rg, stats01.withinCovariance.rg, stats11.withinCovariance.rg);
        within.rb = MeanFourVectors(stats00.withinCovariance.rb, stats10.withinCovariance.rb, stats01.withinCovariance.rb, stats11.withinCovariance.rb);
        within.gb = MeanFourVectors(stats00.withinCovariance.gb, stats10.withinCovariance.gb, stats01.withinCovariance.gb, stats11.withinCovariance.gb);

        // Add variation caused by differences between the four parent means.
        CovarianceMatrixBatch between{};
        const XMVECTOR zero = XMVectorZero();
        between.rr = zero;
        between.gg = zero;
        between.bb = zero;
        between.rg = zero;
        between.rb = zero;
        between.gb = zero;

        AccumulateBetweenParentCovariance(stats00, mean, between);
        AccumulateBetweenParentCovariance(stats10, mean, between);
        AccumulateBetweenParentCovariance(stats01, mean, between);
        AccumulateBetweenParentCovariance(stats11, mean, between);

        // ANOVA identity: total covariance equals within-parent plus between-parent covariance.
        covariance.rr = XMVectorAdd(between.rr, within.rr);
        covariance.gg = XMVectorAdd(between.gg, within.gg);
        covariance.bb = XMVectorAdd(between.bb, within.bb);
        covariance.rg = XMVectorAdd(between.rg, within.rg);
        covariance.rb = XMVectorAdd(between.rb, within.rb);
        covariance.gb = XMVectorAdd(between.gb, within.gb);
    }

    // Expand the projection range with one linear RGB sample.
    inline void ExpandProjectionRange(
        const LinearRGBBatch& color,
        const ProjectionContextBatch& context,
        XMVECTOR& minimum,
        XMVECTOR& maximum) noexcept
    {
        XMVECTOR projection = XMVectorMultiply(context.axis.r, XMVectorSubtract(color.r, context.mean.r));
        projection = XMVectorMultiplyAdd(context.axis.g, XMVectorSubtract(color.g, context.mean.g), projection);
        projection = XMVectorMultiplyAdd(context.axis.b, XMVectorSubtract(color.b, context.mean.b), projection);
        minimum = XMVectorMin(minimum, projection);
        maximum = XMVectorMax(maximum, projection);
    }

    // Expand the projection range with all four quadrants of one parent.
    inline void ExpandParentProjectionRange(
        const QuadrantMeansBatch& parent,
        const ProjectionContextBatch& context,
        XMVECTOR& minimum,
        XMVECTOR& maximum) noexcept
    {
        ExpandProjectionRange({ parent.q0R, parent.q0G, parent.q0B }, context, minimum, maximum);
        ExpandProjectionRange({ parent.q1R, parent.q1G, parent.q1B }, context, minimum, maximum);
        ExpandProjectionRange({ parent.q2R, parent.q2G, parent.q2B }, context, minimum, maximum);
        ExpandProjectionRange({ parent.q3R, parent.q3G, parent.q3B }, context, minimum, maximum);
    }

    // Estimate the principal axis and initial endpoint range.
    inline EndpointPairBatch ComputeInitialEndpointsPCA(
        const CovarianceMatrixBatch& covariance,
        const LinearRGBBatch& mean,
        const QuadrantMeansBatch& p00,
        const QuadrantMeansBatch& p10,
        const QuadrantMeansBatch& p01,
        const QuadrantMeansBatch& p11) noexcept
    {
        // Begin power iteration with a normalized diagonal direction in RGB space.
        LinearRGBBatch axis{};
        axis.r = XMVectorReplicate(0.57735f);
        axis.g = XMVectorReplicate(0.57735f);
        axis.b = XMVectorReplicate(0.57735f);

        // Multiply the initial direction by the covariance matrix once to approach its principal axis.
        LinearRGBBatch next{};
        next.r = XMVectorMultiply(covariance.rr, axis.r);
        next.r = XMVectorMultiplyAdd(covariance.rg, axis.g, next.r);
        next.r = XMVectorMultiplyAdd(covariance.rb, axis.b, next.r);

        next.g = XMVectorMultiply(covariance.rg, axis.r);
        next.g = XMVectorMultiplyAdd(covariance.gg, axis.g, next.g);
        next.g = XMVectorMultiplyAdd(covariance.gb, axis.b, next.g);

        next.b = XMVectorMultiply(covariance.rb, axis.r);
        next.b = XMVectorMultiplyAdd(covariance.gb, axis.g, next.b);
        next.b = XMVectorMultiplyAdd(covariance.bb, axis.b, next.b);

        // Normalize the estimated axis; epsilon keeps flat-color blocks finite.
        XMVECTOR lengthSquared = XMVectorMultiply(next.r, next.r);
        lengthSquared = XMVectorMultiplyAdd(next.g, next.g, lengthSquared);
        lengthSquared = XMVectorMultiplyAdd(next.b, next.b, lengthSquared);
        lengthSquared = XMVectorAdd(lengthSquared, XMVectorReplicate(1e-20f));

        const XMVECTOR inverseLength = XMVectorReciprocalSqrt(lengthSquared);
        axis.r = XMVectorMultiply(next.r, inverseLength);
        axis.g = XMVectorMultiply(next.g, inverseLength);
        axis.b = XMVectorMultiply(next.b, inverseLength);

        // Project all sixteen quadrant means and retain the minimum and maximum positions.
        const ProjectionContextBatch context{ axis, mean };
        XMVECTOR minimum = XMVectorReplicate(10000.0f);
        XMVECTOR maximum = XMVectorReplicate(-10000.0f);

        ExpandParentProjectionRange(p00, context, minimum, maximum);
        ExpandParentProjectionRange(p10, context, minimum, maximum);
        ExpandParentProjectionRange(p01, context, minimum, maximum);
        ExpandParentProjectionRange(p11, context, minimum, maximum);

        // Convert the two extreme scalar projections back into RGB endpoint candidates.
        EndpointPairBatch result{};
        result.p0.r = XMVectorMultiplyAdd(axis.r, minimum, mean.r);
        result.p0.g = XMVectorMultiplyAdd(axis.g, minimum, mean.g);
        result.p0.b = XMVectorMultiplyAdd(axis.b, minimum, mean.b);
        result.p1.r = XMVectorMultiplyAdd(axis.r, maximum, mean.r);
        result.p1.g = XMVectorMultiplyAdd(axis.g, maximum, mean.g);
        result.p1.b = XMVectorMultiplyAdd(axis.b, maximum, mean.b);
        return result;
    }

    // Accumulate one sample for the fixed-selector least-squares solve.
    inline void AccumulateLeastSquaresSample(
        const LinearRGBBatch& color,
        const LeastSquaresContextBatch& context,
        LeastSquaresAccumulatorBatch& accumulator) noexcept
    {
        // Project the sample onto the current endpoint line.
        XMVECTOR projection = XMVectorMultiply(XMVectorSubtract(color.r, context.p0.r), context.direction.r);
        projection = XMVectorMultiplyAdd(XMVectorSubtract(color.g, context.p0.g), context.direction.g, projection);
        projection = XMVectorMultiplyAdd(XMVectorSubtract(color.b, context.p0.b), context.direction.b, projection);
        projection = XMVectorMultiply(projection, context.inverseLengthSquared);

        const XMVECTOR zero = XMVectorZero();
        const XMVECTOR one = XMVectorReplicate(1.0f);
        // Snap the continuous projection to one of the four BC1 palette weights: 0, 1/3, 2/3, or 1.
        XMVECTOR weight = XMVectorClamp(projection, zero, one);
        weight = XMVectorRound(XMVectorScale(weight, 3.0f));
        weight = XMVectorScale(weight, 1.0f / 3.0f);

        // Gather the sufficient statistics needed by the two-endpoint normal equation.
        accumulator.weightSum = XMVectorAdd(accumulator.weightSum, weight);
        accumulator.weightSquaredSum = XMVectorMultiplyAdd(weight, weight, accumulator.weightSquaredSum);
        accumulator.weighted.r = XMVectorMultiplyAdd(weight, color.r, accumulator.weighted.r);
        accumulator.weighted.g = XMVectorMultiplyAdd(weight, color.g, accumulator.weighted.g);
        accumulator.weighted.b = XMVectorMultiplyAdd(weight, color.b, accumulator.weighted.b);
    }

    // Accumulate all four quadrant samples from one parent block.
    inline void AccumulateParentLeastSquares(
        const QuadrantMeansBatch& parent,
        const LeastSquaresContextBatch& context,
        LeastSquaresAccumulatorBatch& accumulator) noexcept
    {
        AccumulateLeastSquaresSample({ parent.q0R, parent.q0G, parent.q0B }, context, accumulator);
        AccumulateLeastSquaresSample({ parent.q1R, parent.q1G, parent.q1B }, context, accumulator);
        AccumulateLeastSquaresSample({ parent.q2R, parent.q2G, parent.q2B }, context, accumulator);
        AccumulateLeastSquaresSample({ parent.q3R, parent.q3G, parent.q3B }, context, accumulator);
    }

    // Refine the two endpoints while holding four BC1 selector weights fixed.
    inline EndpointPairBatch OptimizeEndpointsLeastSquares(
        const QuadrantMeansBatch& p00,
        const QuadrantMeansBatch& p10,
        const QuadrantMeansBatch& p01,
        const QuadrantMeansBatch& p11,
        const LinearRGBBatch& mean,
        const EndpointPairBatch& endpoints) noexcept
    {
        // Use the PCA endpoints to define the initial line for selector assignment.
        LeastSquaresContextBatch context{};
        context.p0 = endpoints.p0;
        context.direction.r = XMVectorSubtract(endpoints.p1.r, endpoints.p0.r);
        context.direction.g = XMVectorSubtract(endpoints.p1.g, endpoints.p0.g);
        context.direction.b = XMVectorSubtract(endpoints.p1.b, endpoints.p0.b);

        XMVECTOR lengthSquared = XMVectorMultiply(context.direction.r, context.direction.r);
        lengthSquared = XMVectorMultiplyAdd(context.direction.g, context.direction.g, lengthSquared);
        lengthSquared = XMVectorMultiplyAdd(context.direction.b, context.direction.b, lengthSquared);
        lengthSquared = XMVectorAdd(lengthSquared, XMVectorReplicate(1e-12f));
        context.inverseLengthSquared = XMVectorReciprocal(lengthSquared);

        // Initialize and accumulate statistics from all sixteen reconstructed quadrant samples.
        LeastSquaresAccumulatorBatch accumulator{};
        const XMVECTOR zero = XMVectorZero();
        accumulator.weightSum = zero;
        accumulator.weightSquaredSum = zero;
        accumulator.weighted.r = zero;
        accumulator.weighted.g = zero;
        accumulator.weighted.b = zero;

        AccumulateParentLeastSquares(p00, context, accumulator);
        AccumulateParentLeastSquares(p10, context, accumulator);
        AccumulateParentLeastSquares(p01, context, accumulator);
        AccumulateParentLeastSquares(p11, context, accumulator);

        // Compute the determinant of the 2x2 endpoint least-squares system.
        const XMVECTOR sixteen = XMVectorReplicate(16.0f);
        XMVECTOR determinant = XMVectorMultiply(sixteen, accumulator.weightSquaredSum);
        determinant = XMVectorNegativeMultiplySubtract(accumulator.weightSum, accumulator.weightSum, determinant);

        // Flat or nearly flat blocks use their mean instead of dividing by an unstable determinant.
        const XMVECTOR singularMask = XMVectorLess(determinant, XMVectorReplicate(1e-6f));
        const XMVECTOR safeDeterminant = XMVectorSelect(determinant, XMVectorReplicate(1.0f), singularMask);
        const XMVECTOR inverseDeterminant = XMVectorReciprocal(safeDeterminant);

        const LinearRGBBatch total
        {
            XMVectorScale(mean.r, 16.0f),
            XMVectorScale(mean.g, 16.0f),
            XMVectorScale(mean.b, 16.0f)
        };

        EndpointPairBatch result{};
        LinearRGBBatch direction{};

        // Solve the normal equations for the first endpoint and the endpoint direction.
        result.p0.r = XMVectorMultiply(accumulator.weightSquaredSum, total.r);
        result.p0.r = XMVectorNegativeMultiplySubtract(accumulator.weightSum, accumulator.weighted.r, result.p0.r);
        result.p0.r = XMVectorMultiply(result.p0.r, inverseDeterminant);

        result.p0.g = XMVectorMultiply(accumulator.weightSquaredSum, total.g);
        result.p0.g = XMVectorNegativeMultiplySubtract(accumulator.weightSum, accumulator.weighted.g, result.p0.g);
        result.p0.g = XMVectorMultiply(result.p0.g, inverseDeterminant);

        result.p0.b = XMVectorMultiply(accumulator.weightSquaredSum, total.b);
        result.p0.b = XMVectorNegativeMultiplySubtract(accumulator.weightSum, accumulator.weighted.b, result.p0.b);
        result.p0.b = XMVectorMultiply(result.p0.b, inverseDeterminant);

        direction.r = XMVectorMultiply(sixteen, accumulator.weighted.r);
        direction.r = XMVectorNegativeMultiplySubtract(accumulator.weightSum, total.r, direction.r);
        direction.r = XMVectorMultiply(direction.r, inverseDeterminant);

        direction.g = XMVectorMultiply(sixteen, accumulator.weighted.g);
        direction.g = XMVectorNegativeMultiplySubtract(accumulator.weightSum, total.g, direction.g);
        direction.g = XMVectorMultiply(direction.g, inverseDeterminant);

        direction.b = XMVectorMultiply(sixteen, accumulator.weighted.b);
        direction.b = XMVectorNegativeMultiplySubtract(accumulator.weightSum, total.b, direction.b);
        direction.b = XMVectorMultiply(direction.b, inverseDeterminant);

        result.p1.r = XMVectorAdd(result.p0.r, direction.r);
        result.p1.g = XMVectorAdd(result.p0.g, direction.g);
        result.p1.b = XMVectorAdd(result.p0.b, direction.b);

        // Replace both endpoints with the block mean in singular SIMD lanes.
        result.p0.r = XMVectorSelect(result.p0.r, mean.r, singularMask);
        result.p0.g = XMVectorSelect(result.p0.g, mean.g, singularMask);
        result.p0.b = XMVectorSelect(result.p0.b, mean.b, singularMask);
        result.p1.r = XMVectorSelect(result.p1.r, mean.r, singularMask);
        result.p1.g = XMVectorSelect(result.p1.g, mean.g, singularMask);
        result.p1.b = XMVectorSelect(result.p1.b, mean.b, singularMask);
        return result;
    }

    // Apply the sRGB transfer curve independently to all four SIMD lanes.
    inline XMVECTOR LinearToSrgbBatch(FXMVECTOR linear) noexcept
    {
        // Clamp first because the sRGB transfer function is defined only for display-range values.
        const XMVECTOR zero = XMVectorZero();
        const XMVECTOR one = XMVectorReplicate(1.0f);
        const XMVECTOR value = XMVectorClamp(linear, zero, one);
        const XMVECTOR low = XMVectorScale(value, 12.92f);
        const XMVECTOR gamma = XMVectorReplicate(1.0f / 2.4f);
        XMVECTOR high = XMVectorPow(value, gamma);
        high = XMVectorSubtract(XMVectorScale(high, 1.055f), XMVectorReplicate(0.055f));
        const XMVECTOR highMask = XMVectorGreater(value, XMVectorReplicate(0.0031308f));
        return XMVectorSelect(low, high, highMask);
    }

    // Convert linear endpoints into the code space that BC1 actually stores.
    template<bool IsSrgb>
    inline EndpointPairBatch ConvertEndpointsToCodeSpace(const EndpointPairBatch& endpoints) noexcept
    {
        // A UNORM palette stores linear code values, so the endpoints are already there.
        if (!IsSrgb)
        {
            return endpoints;
        }
        else
        {
            EndpointPairBatch result{};
            result.p0.r = LinearToSrgbBatch(endpoints.p0.r);
            result.p0.g = LinearToSrgbBatch(endpoints.p0.g);
            result.p0.b = LinearToSrgbBatch(endpoints.p0.b);
            result.p1.r = LinearToSrgbBatch(endpoints.p1.r);
            result.p1.g = LinearToSrgbBatch(endpoints.p1.g);
            result.p1.b = LinearToSrgbBatch(endpoints.p1.b);
            return result;
        }
    }

    // Return the closest BC1 hardware palette selector in each SIMD lane.
    inline XMVECTOR FindBestSelector(
        FXMVECTOR distance0,
        FXMVECTOR distance1,
        FXMVECTOR distance2,
        GXMVECTOR distance3) noexcept
    {
        const XMVECTOR select1 = XMVectorLess(distance1, distance0);
        const XMVECTOR select3 = XMVectorLess(distance3, distance2);
        const XMVECTOR index01 = XMVectorSelect(XMVectorReplicateInt(0u), XMVectorReplicateInt(1u), select1);
        const XMVECTOR index23 = XMVectorSelect(XMVectorReplicateInt(2u), XMVectorReplicateInt(3u), select3);
        const XMVECTOR minimum01 = XMVectorMin(distance0, distance1);
        const XMVECTOR minimum23 = XMVectorMin(distance2, distance3);
        return XMVectorSelect(index01, index23, XMVectorLess(minimum23, minimum01));
    }

    // Compute squared RGB distance to one palette color.
    inline XMVECTOR ComputeColorDistance(
        const LinearRGBBatch& color,
        FXMVECTOR paletteR,
        FXMVECTOR paletteG,
        FXMVECTOR paletteB) noexcept
    {
        const XMVECTOR deltaR = XMVectorSubtract(color.r, paletteR);
        const XMVECTOR deltaG = XMVectorSubtract(color.g, paletteG);
        const XMVECTOR deltaB = XMVectorSubtract(color.b, paletteB);
        XMVECTOR distance = XMVectorMultiply(deltaR, deltaR);
        distance = XMVectorMultiplyAdd(deltaG, deltaG, distance);
        distance = XMVectorMultiplyAdd(deltaB, deltaB, distance);
        return distance;
    }

    // Assign one output texel selector for four blocks in parallel.
    template<int TexelIndex>
    inline void AssignNearestSelector(
        const LinearRGBBatch& color,
        const LinearPaletteBC1Batch& palette,
        XMVECTOR& packedSelectors) noexcept
    {
        static_assert(TexelIndex >= 0 && TexelIndex < 16, "Invalid BC1 texel index");
        const XMVECTOR distance0 = ComputeColorDistance(color, palette.c0R, palette.c0G, palette.c0B);
        const XMVECTOR distance1 = ComputeColorDistance(color, palette.c1R, palette.c1G, palette.c1B);
        const XMVECTOR distance2 = ComputeColorDistance(color, palette.c2R, palette.c2G, palette.c2B);
        const XMVECTOR distance3 = ComputeColorDistance(color, palette.c3R, palette.c3G, palette.c3B);
        const XMVECTOR selector = FindBestSelector(distance0, distance1, distance2, distance3);
        packedSelectors = XMVectorOrInt(packedSelectors, ShiftLeft32<TexelIndex * 2>(selector));
    }

    // Quantize four normalized endpoint components to an unsigned integer range.
    inline XMVECTOR QuantizeUNormBatch(FXMVECTOR values, float maximum) noexcept
    {
        const XMVECTOR clamped = XMVectorClamp(values, XMVectorZero(), XMVectorReplicate(1.0f));
        const XMVECTOR rounded = XMVectorRound(XMVectorScale(clamped, maximum));
        return XMConvertVectorFloatToUInt(rounded, 0);
    }

    // Guarantee opaque BC1 endpoint ordering in all four lanes.
    inline void EnforceOpaqueEndpointOrder(XMVECTOR& packed0, XMVECTOR& packed1) noexcept
    {
        // Equal endpoints are separated by one RGB565 step so the block remains in opaque mode.
        const XMVECTOR equalMask = XMVectorEqualInt(packed0, packed1);
        const XMVECTOR blue = XMVectorAndInt(packed1, XMVectorReplicateInt(0x1Fu));
        const XMVECTOR canDecreaseMask = GreaterInt32(blue, XMVectorZero());
        const XMVECTOR decreaseMask = XMVectorAndInt(equalMask, canDecreaseMask);
        const XMVECTOR increaseMask = XMVectorAndCInt(equalMask, canDecreaseMask);

        packed1 = XMVectorSelect(packed1, SubtractInt32(packed1, XMVectorReplicateInt(1u)), decreaseMask);
        packed0 = XMVectorSelect(packed0, AddInt32(packed0, XMVectorReplicateInt(1u)), increaseMask);

        // Opaque BC1 requires color0 to be numerically greater than color1.
        const XMVECTOR swapMask = GreaterInt32(packed1, packed0);
        const XMVECTOR original0 = packed0;
        packed0 = XMVectorSelect(packed0, packed1, swapMask);
        packed1 = XMVectorSelect(packed1, original0, swapMask);
    }

    // Quantize endpoints and assign the sixteen destination selectors.
    template<bool IsSrgb>
    inline BC1BlockBatch PackAndReallocateSelectors(
        const EndpointPairBatch& endpoints,
        const QuadrantMeansBatch& p00,
        const QuadrantMeansBatch& p10,
        const QuadrantMeansBatch& p01,
        const QuadrantMeansBatch& p11) noexcept
    {
        // Quantize endpoint components to the RGB565 bit widths used by BC1.
        const XMVECTOR r0 = QuantizeUNormBatch(endpoints.p0.r, 31.0f);
        const XMVECTOR g0 = QuantizeUNormBatch(endpoints.p0.g, 63.0f);
        const XMVECTOR b0 = QuantizeUNormBatch(endpoints.p0.b, 31.0f);
        const XMVECTOR r1 = QuantizeUNormBatch(endpoints.p1.r, 31.0f);
        const XMVECTOR g1 = QuantizeUNormBatch(endpoints.p1.g, 63.0f);
        const XMVECTOR b1 = QuantizeUNormBatch(endpoints.p1.b, 31.0f);

        // Pack R, G, and B fields and force the opaque endpoint ordering rule.
        BC1BlockBatch result{};
        result.color0 = XMVectorOrInt(XMVectorOrInt(ShiftLeft32<11>(r0), ShiftLeft32<5>(g0)), b0);
        result.color1 = XMVectorOrInt(XMVectorOrInt(ShiftLeft32<11>(r1), ShiftLeft32<5>(g1)), b1);
        EnforceOpaqueEndpointOrder(result.color0, result.color1);

        // Rebuild the quantized hardware palette before assigning selectors.
        const LinearPaletteBC1Batch palette = BuildOpaqueLinearPaletteBC1Batch<IsSrgb>(result.color0, result.color1);
        result.selectors = XMVectorZero();

        AssignNearestSelector<0>({ p00.q0R, p00.q0G, p00.q0B }, palette, result.selectors);
        AssignNearestSelector<1>({ p00.q1R, p00.q1G, p00.q1B }, palette, result.selectors);
        AssignNearestSelector<4>({ p00.q2R, p00.q2G, p00.q2B }, palette, result.selectors);
        AssignNearestSelector<5>({ p00.q3R, p00.q3G, p00.q3B }, palette, result.selectors);

        AssignNearestSelector<2>({ p10.q0R, p10.q0G, p10.q0B }, palette, result.selectors);
        AssignNearestSelector<3>({ p10.q1R, p10.q1G, p10.q1B }, palette, result.selectors);
        AssignNearestSelector<6>({ p10.q2R, p10.q2G, p10.q2B }, palette, result.selectors);
        AssignNearestSelector<7>({ p10.q3R, p10.q3G, p10.q3B }, palette, result.selectors);

        AssignNearestSelector<8>({ p01.q0R, p01.q0G, p01.q0B }, palette, result.selectors);
        AssignNearestSelector<9>({ p01.q1R, p01.q1G, p01.q1B }, palette, result.selectors);
        AssignNearestSelector<12>({ p01.q2R, p01.q2G, p01.q2B }, palette, result.selectors);
        AssignNearestSelector<13>({ p01.q3R, p01.q3G, p01.q3B }, palette, result.selectors);

        AssignNearestSelector<10>({ p11.q0R, p11.q0G, p11.q0B }, palette, result.selectors);
        AssignNearestSelector<11>({ p11.q1R, p11.q1G, p11.q1B }, palette, result.selectors);
        AssignNearestSelector<14>({ p11.q2R, p11.q2G, p11.q2B }, palette, result.selectors);
        AssignNearestSelector<15>({ p11.q3R, p11.q3G, p11.q3B }, palette, result.selectors);
        return result;
    }

    // Encode four 4x4 linear RGB sample groups into four BC1 blocks.
    template<bool IsSrgb>
    inline BC1BlockBatch EncodeLinearBlocksBC1Batch(
        const QuadrantMeansBatch& p00,
        const QuadrantMeansBatch& p10,
        const QuadrantMeansBatch& p01,
        const QuadrantMeansBatch& p11,
        SourceBlockMeansBatch& sourceMeans) noexcept
    {
        LinearRGBBatch mean{};
        CovarianceMatrixBatch covariance{};
        ComputeChildBlockMoments(p00, p10, p01, p11, sourceMeans, mean, covariance);
        const EndpointPairBatch initial = ComputeInitialEndpointsPCA(covariance, mean, p00, p10, p01, p11);
        const EndpointPairBatch optimized = OptimizeEndpointsLeastSquares(p00, p10, p01, p11, mean, initial);
        const EndpointPairBatch encoded = ConvertEndpointsToCodeSpace<IsSrgb>(optimized);
        return PackAndReallocateSelectors<IsSrgb>(encoded, p00, p10, p01, p11);
    }

    // Generate four destination blocks directly from sixteen compressed parents.
    template<bool IsSrgb>
    inline BC1BlockBatch EncodeMipBlocksBC1Batch(
        const BC1BlockBatch& p00,
        const BC1BlockBatch& p10,
        const BC1BlockBatch& p01,
        const BC1BlockBatch& p11,
        SourceBlockMeansBatch& sourceMeans) noexcept
    {
        XMVECTOR p00Hist0, p00Hist1, p00Hist2, p00Hist3;
        XMVECTOR p10Hist0, p10Hist1, p10Hist2, p10Hist3;
        XMVECTOR p01Hist0, p01Hist1, p01Hist2, p01Hist3;
        XMVECTOR p11Hist0, p11Hist1, p11Hist2, p11Hist3;

        Extract2x2SelectorHistograms(p00.selectors, p00Hist0, p00Hist1, p00Hist2, p00Hist3);
        Extract2x2SelectorHistograms(p10.selectors, p10Hist0, p10Hist1, p10Hist2, p10Hist3);
        Extract2x2SelectorHistograms(p01.selectors, p01Hist0, p01Hist1, p01Hist2, p01Hist3);
        Extract2x2SelectorHistograms(p11.selectors, p11Hist0, p11Hist1, p11Hist2, p11Hist3);

        const QuadrantMeansBatch p00Means = ComputeParentQuadrantMeansBatch<IsSrgb>(p00.color0, p00.color1, p00Hist0, p00Hist1, p00Hist2, p00Hist3);
        const QuadrantMeansBatch p10Means = ComputeParentQuadrantMeansBatch<IsSrgb>(p10.color0, p10.color1, p10Hist0, p10Hist1, p10Hist2, p10Hist3);
        const QuadrantMeansBatch p01Means = ComputeParentQuadrantMeansBatch<IsSrgb>(p01.color0, p01.color1, p01Hist0, p01Hist1, p01Hist2, p01Hist3);
        const QuadrantMeansBatch p11Means = ComputeParentQuadrantMeansBatch<IsSrgb>(p11.color0, p11.color1, p11Hist0, p11Hist1, p11Hist2, p11Hist3);
        return EncodeLinearBlocksBC1Batch<IsSrgb>(p00Means, p10Means, p01Means, p11Means, sourceMeans);
    }

    // Store valid SIMD lanes as ordinary BC1 blocks.
    inline void StoreBC1BlockBatch(const BC1BlockBatch& batch, D3DX_BC1* blocks, size_t validLanes) noexcept
    {
        // Store raw bits because XMStoreUInt4 would numerically convert these integer bit patterns.
        alignas(16) uint32_t packedColor0[4];
        alignas(16) uint32_t packedColor1[4];
        alignas(16) uint32_t packedSelectors[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(packedColor0), _mm_castps_si128(batch.color0));
        _mm_store_si128(reinterpret_cast<__m128i*>(packedColor1), _mm_castps_si128(batch.color1));
        _mm_store_si128(reinterpret_cast<__m128i*>(packedSelectors), _mm_castps_si128(batch.selectors));

        // The last SIMD batch may contain fewer than four real destination blocks.
        for (size_t lane = 0; lane < validLanes; ++lane)
        {
            blocks[lane].rgb[0] = static_cast<uint16_t>(packedColor0[lane]);
            blocks[lane].rgb[1] = static_cast<uint16_t>(packedColor1[lane]);
            blocks[lane].bitmap = packedSelectors[lane];
        }
    }

    // Store one source-block mean from a SIMD lane.
    inline void StoreBlockMean(const LinearRGBBatch& means, size_t lane, LinearBlockMean& destination) noexcept
    {
        destination.r = XMVectorGetByIndex(means.r, lane);
        destination.g = XMVectorGetByIndex(means.g, lane);
        destination.b = XMVectorGetByIndex(means.b, lane);
    }

    // Reject BC1 blocks that use transparent three-color mode.
    inline bool IsOpaqueBC1Image(const Image& image) noexcept
    {
        // BC storage dimensions are rounded up to complete 4x4 blocks.
        const size_t blockWidth = std::max<size_t>(1, (image.width + 3) / 4);
        const size_t blockHeight = std::max<size_t>(1, (image.height + 3) / 4);

        for (size_t y = 0; y < blockHeight; ++y)
        {
            const auto* row = reinterpret_cast<const D3DX_BC1*>(image.pixels + y * image.rowPitch);
            for (size_t x = 0; x < blockWidth; ++x)
            {
                // color0 < color1 selects BC1's transparent three-color mode.
                // Equal endpoints are opaque as long as no texel selects palette entry 3.
                const uint32_t selector3Bits = row[x].bitmap & (row[x].bitmap >> 1) & 0x55555555u;
                if (row[x].rgb[0] <= row[x].rgb[1] && selector3Bits != 0)
                {
                    return false;
                }
            }
        }

        return true;
    }

    // Process one mip-1 block row directly from compressed parent blocks.
    template<bool IsSrgb>
    inline void ProcessCompressedRowBC1(
        const Image& source,
        Image& destination,
        size_t destinationRow,
        LinearBlockMean* sourceBlockMeans) noexcept
    {
        // Four output blocks are encoded together, one in each SIMD lane.
        constexpr size_t laneCount = 4;
        const size_t sourceBlockWidth = std::max<size_t>(1, (source.width + 3) / 4);
        const size_t sourceBlockHeight = std::max<size_t>(1, (source.height + 3) / 4);
        const size_t destinationBlockWidth = std::max<size_t>(1, (destination.width + 3) / 4);

        // One destination block covers a 2x2 group of compressed source blocks.
        const size_t sourceY0 = std::min(destinationRow * 2, sourceBlockHeight - 1);
        const size_t sourceY1 = std::min(sourceY0 + 1, sourceBlockHeight - 1);
        const auto* sourceRow0 = reinterpret_cast<const D3DX_BC1*>(source.pixels + sourceY0 * source.rowPitch);
        const auto* sourceRow1 = reinterpret_cast<const D3DX_BC1*>(source.pixels + sourceY1 * source.rowPitch);
        auto* destinationBlocks = reinterpret_cast<D3DX_BC1*>(destination.pixels + destinationRow * destination.rowPitch);

        for (size_t destinationX = 0; destinationX < destinationBlockWidth; destinationX += laneCount)
        {
            D3DX_BC1 p00Blocks[laneCount]{};
            D3DX_BC1 p10Blocks[laneCount]{};
            D3DX_BC1 p01Blocks[laneCount]{};
            D3DX_BC1 p11Blocks[laneCount]{};
            const size_t validLanes = std::min(laneCount, destinationBlockWidth - destinationX);

            for (size_t lane = 0; lane < laneCount; ++lane)
            {
                const size_t childX = destinationX + std::min(lane, validLanes - 1);
                const size_t sourceX0 = std::min(childX * 2, sourceBlockWidth - 1);
                const size_t sourceX1 = std::min(sourceX0 + 1, sourceBlockWidth - 1);
                p00Blocks[lane] = sourceRow0[sourceX0];
                p10Blocks[lane] = sourceRow0[sourceX1];
                p01Blocks[lane] = sourceRow1[sourceX0];
                p11Blocks[lane] = sourceRow1[sourceX1];
            }

            // Encode mip 1 directly and retain recovered means for the higher-level pyramid.
            SourceBlockMeansBatch means{};
            const BC1BlockBatch encoded = EncodeMipBlocksBC1Batch<IsSrgb>(
                LoadBC1BlockBatch(p00Blocks),
                LoadBC1BlockBatch(p10Blocks),
                LoadBC1BlockBatch(p01Blocks),
                LoadBC1BlockBatch(p11Blocks),
                means);
            StoreBC1BlockBatch(encoded, destinationBlocks + destinationX, validLanes);

            if (sourceBlockMeans)
            {
                for (size_t lane = 0; lane < validLanes; ++lane)
                {
                    const size_t childX = destinationX + lane;
                    const size_t sourceX0 = std::min(childX * 2, sourceBlockWidth - 1);
                    const size_t sourceX1 = std::min(sourceX0 + 1, sourceBlockWidth - 1);
                    StoreBlockMean(means.p00, lane, sourceBlockMeans[sourceY0 * sourceBlockWidth + sourceX0]);

                    if (sourceX1 != sourceX0)
                    {
                        StoreBlockMean(means.p10, lane, sourceBlockMeans[sourceY0 * sourceBlockWidth + sourceX1]);
                    }

                    if (sourceY1 != sourceY0)
                    {
                        StoreBlockMean(means.p01, lane, sourceBlockMeans[sourceY1 * sourceBlockWidth + sourceX0]);
                        if (sourceX1 != sourceX0)
                        {
                            StoreBlockMean(means.p11, lane, sourceBlockMeans[sourceY1 * sourceBlockWidth + sourceX1]);
                        }
                    }
                }
            }
        }
    }

    // Halve one row of the linear block-mean image with edge clamping.
    inline void DownsampleLinearMeanRow(
        const LinearBlockMean* source,
        size_t sourceWidth,
        size_t sourceHeight,
        LinearBlockMean* destination,
        size_t destinationWidth,
        size_t destinationRow) noexcept
    {
        // Select two source rows and repeat the final row when the height is odd.
        const size_t sourceY0 = destinationRow * 2;
        const size_t sourceY1 = std::min(sourceY0 + 1, sourceHeight - 1);
        const auto* sourceRow0 = source + sourceY0 * sourceWidth;
        const auto* sourceRow1 = source + sourceY1 * sourceWidth;
        auto* destinationRowPtr = destination + destinationRow * destinationWidth;

        for (size_t destinationX = 0; destinationX < destinationWidth; ++destinationX)
        {
            const size_t sourceX0 = destinationX * 2;
            const size_t sourceX1 = std::min(sourceX0 + 1, sourceWidth - 1);

            const auto& s00 = sourceRow0[sourceX0];
            const auto& s10 = sourceRow0[sourceX1];
            const auto& s01 = sourceRow1[sourceX0];
            const auto& s11 = sourceRow1[sourceX1];

            auto& output = destinationRowPtr[destinationX];
            output.r = (s00.r + s10.r + s01.r + s11.r) * 0.25f;
            output.g = (s00.g + s10.g + s01.g + s11.g) * 0.25f;
            output.b = (s00.b + s10.b + s01.b + s11.b) * 0.25f;
        }
    }

    // Insert one scalar mean into one lane of a quadrant sample.
    inline void SetQuadrantSample(
        QuadrantMeansBatch& quadrants,
        size_t sample,
        size_t lane,
        const LinearBlockMean& color) noexcept
    {
        XMVECTOR* red[] = { &quadrants.q0R, &quadrants.q1R, &quadrants.q2R, &quadrants.q3R };
        XMVECTOR* green[] = { &quadrants.q0G, &quadrants.q1G, &quadrants.q2G, &quadrants.q3G };
        XMVECTOR* blue[] = { &quadrants.q0B, &quadrants.q1B, &quadrants.q2B, &quadrants.q3B };
        *red[sample] = XMVectorSetByIndex(*red[sample], color.r, lane);
        *green[sample] = XMVectorSetByIndex(*green[sample], color.g, lane);
        *blue[sample] = XMVectorSetByIndex(*blue[sample], color.b, lane);
    }

    // Fetch one mean texel with clamp-to-edge addressing.
    inline const LinearBlockMean& FetchLinearMean(
        const LinearBlockMean* source,
        size_t width,
        size_t height,
        size_t x,
        size_t y) noexcept
    {
        return source[std::min(y, height - 1) * width + std::min(x, width - 1)];
    }

    // Encode one destination row from the linear block-mean image.
    template<bool IsSrgb>
    inline void ProcessLinearRowBC1(
        const LinearBlockMean* source,
        size_t sourceWidth,
        size_t sourceHeight,
        Image& destination,
        size_t destinationRow) noexcept
    {
        // Higher mip levels also encode four independent destination blocks per SIMD batch.
        constexpr size_t laneCount = 4;
        const size_t destinationBlockWidth = std::max<size_t>(1, (destination.width + 3) / 4);
        auto* destinationBlocks = reinterpret_cast<D3DX_BC1*>(destination.pixels + destinationRow * destination.rowPitch);

        for (size_t destinationX = 0; destinationX < destinationBlockWidth; destinationX += laneCount)
        {
            // Each region holds the four samples belonging to one 2x2 output quadrant.
            QuadrantMeansBatch regions[4]{};
            const size_t validLanes = std::min(laneCount, destinationBlockWidth - destinationX);

            for (size_t lane = 0; lane < validLanes; ++lane)
            {
                const size_t blockX = destinationX + lane;
                const size_t texelBaseX = blockX * 4;
                const size_t texelBaseY = destinationRow * 4;

                for (size_t localY = 0; localY < 4; ++localY)
                {
                    for (size_t localX = 0; localX < 4; ++localX)
                    {
                        const size_t region = ((localY >> 1) << 1) + (localX >> 1);
                        const size_t sample = ((localY & 1) << 1) + (localX & 1);
                        const LinearBlockMean& color = FetchLinearMean(source, sourceWidth, sourceHeight, texelBaseX + localX, texelBaseY + localY);
                        SetQuadrantSample(regions[region], sample, lane, color);
                    }
                }
            }

            SourceBlockMeansBatch unusedMeans{};
            const BC1BlockBatch encoded = EncodeLinearBlocksBC1Batch<IsSrgb>(regions[0], regions[1], regions[2], regions[3], unusedMeans);
            StoreBC1BlockBatch(encoded, destinationBlocks + destinationX, validLanes);
        }
    }

    // Generate all compressed mip levels after level zero.
    template<bool IsSrgb>
    HRESULT GenerateCompressedMipMapsBC1(const Image& baseImage, ScratchImage& mipChain) noexcept
    {
        // Level 0 is already present, so a one-level chain requires no generation work.
        const size_t mipLevels = mipChain.GetMetadata().mipLevels;
        if (mipLevels <= 1)
        {
            return S_OK;
        }

        // Initialize the shared lookup table before entering an OpenMP region.
        if (IsSrgb)
        {
            (void)GetSrgb8ToLinearTable();
        }

        const size_t baseBlockWidth = std::max<size_t>(1, (baseImage.width + 3) / 4);
        const size_t baseBlockHeight = std::max<size_t>(1, (baseImage.height + 3) / 4);
        std::unique_ptr<LinearBlockMean[]> meanImage;
        std::unique_ptr<LinearBlockMean[]> meanScratch;

        // Mip 1 is direct; later levels require two ping-pong linear-mean buffers.
        if (mipLevels > 2)
        {
            const size_t meanCount = baseBlockWidth * baseBlockHeight;
            const size_t scratchWidth = (baseBlockWidth + 1) / 2;
            const size_t scratchHeight = (baseBlockHeight + 1) / 2;
            meanImage.reset(new (std::nothrow) LinearBlockMean[meanCount]{});
            meanScratch.reset(new (std::nothrow) LinearBlockMean[scratchWidth * scratchHeight]{});
            if (!meanImage || !meanScratch)
            {
                return E_OUTOFMEMORY;
            }
        }

        LinearBlockMean* meanFront = meanImage.get();
        LinearBlockMean* meanBack = meanScratch.get();
        size_t meanWidth = baseBlockWidth;
        size_t meanHeight = baseBlockHeight;

        // Generate lower levels while keeping all intermediate statistics in linear RGB.
        for (size_t mipLevel = 1; mipLevel < mipLevels; ++mipLevel)
        {
            Image* destination = const_cast<Image*>(mipChain.GetImage(mipLevel, 0, 0));
            if (!destination || !destination->pixels)
            {
                return E_FAIL;
            }

            const size_t destinationBlockHeight = std::max<size_t>(1, (destination->height + 3) / 4);

            // Before mip 3 and later, halve the mean pyramid to the required scale.
            if (mipLevel >= 3)
            {
                const size_t nextWidth = (meanWidth + 1) / 2;
                const size_t nextHeight = (meanHeight + 1) / 2;

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
                for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(nextHeight); ++row)
                {
                    DownsampleLinearMeanRow(meanFront, meanWidth, meanHeight, meanBack, nextWidth, static_cast<size_t>(row));
                }

                std::swap(meanFront, meanBack);
                meanWidth = nextWidth;
                meanHeight = nextHeight;
            }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
            for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(destinationBlockHeight); ++row)
            {
                // Only mip 1 reads BC1 parents; later levels read the linear mean pyramid.
                if (mipLevel == 1)
                {
                    ProcessCompressedRowBC1<IsSrgb>(baseImage, *destination, static_cast<size_t>(row), meanFront);
                }
                else
                {
                    ProcessLinearRowBC1<IsSrgb>(meanFront, meanWidth, meanHeight, *destination, static_cast<size_t>(row));
                }
            }
        }

        return S_OK;
    }

} // namespace for bc1

namespace // for bc4
{
    // Lee: A BC4 block stores two 8-bit endpoints followed by sixteen 3-bit indices.
    struct BC4Block
    {
        uint8_t red0;
        uint8_t red1;
        uint8_t indices[6];
    };

    static_assert(sizeof(BC4Block) == 8, "BC4Block must be 8 bytes");

    // Lee: One scalar mean per source block, the material of the mean pyramid.
    using BC4BlockMean = float;

    // Lee: Sixteen texel values of one destination block, four per SIMD register.
    struct BC4TexelBlock
    {
        XMVECTOR rows[4];
    };

    // Integer palette weights of the six-interpolated mode, j = [0, 7, 1, 2, 3, 4, 5, 6].
    // Palette entry k decodes to ((7 - j[k]) * red0 + j[k] * red1) / (7 * 255).
    constexpr uint8_t g_bc4SixInterpWeights[8] = { 0, 7, 1, 2, 3, 4, 5, 6 };

    // Ramp position p counts steps away from red1, so index k = g_bc4RampToIndex[p].
    constexpr uint8_t g_bc4RampToIndex[8] = { 1, 7, 6, 5, 4, 3, 2, 0 };

    // A quadrant mean divides the palette sum by four, and the palette itself by 7 * 255.
    constexpr float g_bc4QuadrantScale = 1.0f / (4.0f * 7.0f * 255.0f);

    // Read one BC4 block as a single 64-bit word without assuming source alignment.
    inline uint64_t LoadBC4Block(const BC4Block& block) noexcept
    {
        uint64_t data;
        memcpy(&data, &block, sizeof(data));
        return data;
    }

    // Write one BC4 block from its packed 64-bit representation.
    inline void StoreBC4Block(uint64_t data, BC4Block& block) noexcept
    {
        memcpy(&block, &data, sizeof(data));
    }

    // Build the twelve-bit row table that folds four texel indices into two pair sums.
    inline const std::array<uint16_t, 4096>& GetBC4RowWeightTable() noexcept
    {
        // Function-local static initialization builds this table only once and is thread-safe.
        static const std::array<uint16_t, 4096> table = []
            {
                std::array<uint16_t, 4096> values{};

                // A BC4 index row occupies exactly twelve bits, so one entry covers four texels.
                for (size_t group = 0; group < values.size(); ++group)
                {
                    const uint32_t j0 = g_bc4SixInterpWeights[(group >> 0) & 0x07];
                    const uint32_t j1 = g_bc4SixInterpWeights[(group >> 3) & 0x07];
                    const uint32_t j2 = g_bc4SixInterpWeights[(group >> 6) & 0x07];
                    const uint32_t j3 = g_bc4SixInterpWeights[(group >> 9) & 0x07];

                    // The low byte holds the left quadrant pair and the high byte the right one.
                    values[group] = static_cast<uint16_t>((j0 + j1) | ((j2 + j3) << 8));
                }

                return values;
            }();

            // Quadrant symbols then cost four table reads plus four additions per block.
        return table;
    }

    // Rebuild the eight palette values of a block that uses the four-interpolated mode.
    inline void BuildBC4FourInterpPalette(uint32_t red0, uint32_t red1, float palette[8]) noexcept
    {
        constexpr float scale = 1.0f / 255.0f;
        const float fred0 = static_cast<float>(red0) * scale;
        const float fred1 = static_cast<float>(red1) * scale;

        palette[0] = fred0;
        palette[1] = fred1;

        // Entries two through five interpolate on a five-step ramp.
        for (size_t index = 2; index < 6; ++index)
        {
            const float weight = static_cast<float>(index - 1);
            palette[index] = (fred0 * (5.0f - weight) + fred1 * weight) * (1.0f / 5.0f);
        }

        // Entries six and seven are absolute values that do not depend on the endpoints.
        palette[6] = 0.0f;
        palette[7] = 1.0f;
    }

    // Recover the four quadrant means of one compressed BC4 block.
    inline void DecodeBC4QuadrantMeans(uint64_t data, float quadrants[4]) noexcept
    {
        const uint32_t red0 = static_cast<uint32_t>(data & 0xFFu);
        const uint32_t red1 = static_cast<uint32_t>((data >> 8) & 0xFFu);
        const uint64_t indices = data >> 16;

        if (red0 > red1)
        {
            // Six-interpolated mode keeps every palette entry affine in the endpoints,
            // so one integer symbol per quadrant carries all of the information.
            const auto& table = GetBC4RowWeightTable();
            const uint32_t row0 = table[static_cast<size_t>((indices >> 0) & 0xFFFu)];
            const uint32_t row1 = table[static_cast<size_t>((indices >> 12) & 0xFFFu)];
            const uint32_t row2 = table[static_cast<size_t>((indices >> 24) & 0xFFFu)];
            const uint32_t row3 = table[static_cast<size_t>((indices >> 36) & 0xFFFu)];

            // Quadrants zero and one live in the first two rows, two and three in the last two.
            const uint32_t topPairs = row0 + row1;
            const uint32_t bottomPairs = row2 + row3;
            const uint32_t symbols[4] =
            {
                topPairs & 0xFFu,
                (topPairs >> 8) & 0xFFu,
                bottomPairs & 0xFFu,
                (bottomPairs >> 8) & 0xFFu,
            };

            // Theorem 1': four texels sum to ((28 - J) * red0 + J * red1) / (7 * 255).
            for (size_t quadrant = 0; quadrant < 4; ++quadrant)
            {
                const uint32_t symbol = symbols[quadrant];
                const uint32_t weighted = (28u - symbol) * red0 + symbol * red1;
                quadrants[quadrant] = static_cast<float>(weighted) * g_bc4QuadrantScale;
            }
        }
        else
        {
            // Four-interpolated mode breaks the affine form, so decode its texels directly.
            float palette[8];
            BuildBC4FourInterpPalette(red0, red1, palette);

            for (size_t quadrant = 0; quadrant < 4; ++quadrant)
            {
                // Quadrant q covers two texel columns on two consecutive texel rows.
                const size_t baseTexel = ((quadrant >> 1) << 3) + ((quadrant & 1) << 1);
                float sum = 0.0f;

                for (size_t localY = 0; localY < 2; ++localY)
                {
                    for (size_t localX = 0; localX < 2; ++localX)
                    {
                        const size_t texel = baseTexel + localY * 4 + localX;
                        const size_t index = static_cast<size_t>((indices >> (3 * texel)) & 0x07u);
                        sum += palette[index];
                    }
                }

                quadrants[quadrant] = sum * 0.25f;
            }
        }
    }

    // Add the four lanes of a SIMD register.
    inline float HorizontalSumBC4(FXMVECTOR value) noexcept
    {
        XMVECTOR folded = _mm_add_ps(value, _mm_movehl_ps(value, value));
        folded = _mm_add_ss(folded, _mm_shuffle_ps(folded, folded, _MM_SHUFFLE(1, 1, 1, 1)));
        return _mm_cvtss_f32(folded);
    }

    // Take the smallest of the four lanes of a SIMD register.
    inline float HorizontalMinBC4(FXMVECTOR value) noexcept
    {
        XMVECTOR folded = _mm_min_ps(value, _mm_movehl_ps(value, value));
        folded = _mm_min_ss(folded, _mm_shuffle_ps(folded, folded, _MM_SHUFFLE(1, 1, 1, 1)));
        return _mm_cvtss_f32(folded);
    }

    // Take the largest of the four lanes of a SIMD register.
    inline float HorizontalMaxBC4(FXMVECTOR value) noexcept
    {
        XMVECTOR folded = _mm_max_ps(value, _mm_movehl_ps(value, value));
        folded = _mm_max_ss(folded, _mm_shuffle_ps(folded, folded, _MM_SHUFFLE(1, 1, 1, 1)));
        return _mm_cvtss_f32(folded);
    }

    // Accumulated one-dimensional least-squares moments of one block.
    struct BC4LeastSquaresMoments
    {
        float sumY;
        float sumW;
        float sumW2;
        float sumWY;
    };

    // Snap every texel to the closest ramp position and accumulate the least-squares moments.
    inline BC4LeastSquaresMoments AccumulateBC4Moments(
        const BC4TexelBlock& texels,
        float endpoint0,
        float endpoint1) noexcept
    {
        // Ramp position zero sits on endpoint0 and position seven on endpoint1.
        const float inverseSpan = 7.0f / (endpoint1 - endpoint0);
        const XMVECTOR multiplier = XMVectorReplicate(inverseSpan);
        const XMVECTOR offset = XMVectorReplicate(-endpoint0 * inverseSpan);
        const XMVECTOR lowerBound = XMVectorZero();
        const XMVECTOR upperBound = XMVectorReplicate(7.0f);
        const XMVECTOR weightScale = XMVectorReplicate(1.0f / 7.0f);

        XMVECTOR sumY = XMVectorZero();
        XMVECTOR sumW = XMVectorZero();
        XMVECTOR sumW2 = XMVectorZero();
        XMVECTOR sumWY = XMVectorZero();

        for (size_t row = 0; row < 4; ++row)
        {
            const XMVECTOR values = texels.rows[row];

            // Quantized ramp position of every texel, clamped into the representable range.
            XMVECTOR position = XMVectorMultiplyAdd(values, multiplier, offset);
            position = XMVectorClamp(position, lowerBound, upperBound);
            position = XMVectorRound(position);

            const XMVECTOR weights = XMVectorMultiply(position, weightScale);
            sumY = XMVectorAdd(sumY, values);
            sumW = XMVectorAdd(sumW, weights);
            sumW2 = XMVectorMultiplyAdd(weights, weights, sumW2);
            sumWY = XMVectorMultiplyAdd(weights, values, sumWY);
        }

        BC4LeastSquaresMoments moments;
        moments.sumY = HorizontalSumBC4(sumY);
        moments.sumW = HorizontalSumBC4(sumW);
        moments.sumW2 = HorizontalSumBC4(sumW2);
        moments.sumWY = HorizontalSumBC4(sumWY);
        return moments;
    }

    // Solve the two by two normal equations of the segment that fits the block.
    inline void SolveBC4Endpoints(
        const BC4LeastSquaresMoments& moments,
        float& endpoint0,
        float& endpoint1) noexcept
    {
        constexpr float texelCount = 16.0f;
        const float determinant = texelCount * moments.sumW2 - moments.sumW * moments.sumW;

        // A vanishing determinant means every texel landed on the same ramp position.
        if (std::fabs(determinant) < 1e-9f)
        {
            const float mean = moments.sumY * (1.0f / texelCount);
            endpoint0 = mean;
            endpoint1 = mean;
            return;
        }

        // Unlike coordinate descent, this solves both endpoints at once and needs no iteration.
        const float inverseDeterminant = 1.0f / determinant;
        const float intercept = (moments.sumW2 * moments.sumY - moments.sumW * moments.sumWY) * inverseDeterminant;
        const float direction = (texelCount * moments.sumWY - moments.sumW * moments.sumY) * inverseDeterminant;

        endpoint0 = std::min(std::max(intercept, 0.0f), 1.0f);
        endpoint1 = std::min(std::max(intercept + direction, 0.0f), 1.0f);
    }

    // Convert one endpoint from normalized form to its eight-bit code with rounding.
    inline uint32_t QuantizeBC4Endpoint(float value) noexcept
    {
        // Rounding removes the half-LSB bias that truncation would introduce.
        const float scaled = std::min(std::max(value, 0.0f), 1.0f) * 255.0f + 0.5f;
        return static_cast<uint32_t>(scaled);
    }

    // Assign the closest palette index to every texel of a six-interpolated block.
    inline uint64_t AssignBC4Indices(const BC4TexelBlock& texels, uint32_t red0, uint32_t red1) noexcept
    {
        // Equal endpoints leave a single palette value, which index zero already reproduces.
        if (red0 == red1)
        {
            return 0;
        }

        // Closed form of the nearest ramp position on an evenly spaced eight-entry palette.
        const float inverseSpan = 1.0f / static_cast<float>(red0 - red1);
        const XMVECTOR multiplier = XMVectorReplicate(255.0f * 7.0f * inverseSpan);
        const XMVECTOR offset = XMVectorReplicate(-7.0f * static_cast<float>(red1) * inverseSpan);
        const XMVECTOR lowerBound = XMVectorZero();
        const XMVECTOR upperBound = XMVectorReplicate(7.0f);

        uint64_t packed = 0;

        for (size_t row = 0; row < 4; ++row)
        {
            XMVECTOR position = XMVectorMultiplyAdd(texels.rows[row], multiplier, offset);
            position = XMVectorClamp(position, lowerBound, upperBound);

            // Converting to integer rounds to nearest, which selects the closest ramp position.
            alignas(16) int32_t positions[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(positions), _mm_cvtps_epi32(position));

            // One row of four texels occupies exactly twelve bits of the index field.
            uint32_t group = 0;
            for (size_t lane = 0; lane < 4; ++lane)
            {
                const uint32_t index = g_bc4RampToIndex[static_cast<size_t>(positions[lane]) & 0x07u];
                group |= index << (3 * lane);
            }

            packed |= static_cast<uint64_t>(group) << (12 * row);
        }

        return packed;
    }

    // Sum of squared errors between the block and the six-interpolated palette it decodes to.
    inline float EvaluateBC4SixInterpError(const BC4TexelBlock& texels, uint32_t red0, uint32_t red1) noexcept
    {
        // Equal endpoints degenerate into the four-interpolated mode, which is scored separately.
        if (red0 <= red1)
        {
            return FLT_MAX;
        }

        float palette[8];
        constexpr float scale = 1.0f / 255.0f;
        const float fred0 = static_cast<float>(red0) * scale;
        const float fred1 = static_cast<float>(red1) * scale;

        palette[0] = fred0;
        palette[1] = fred1;
        for (size_t index = 2; index < 8; ++index)
        {
            const float weight = static_cast<float>(g_bc4SixInterpWeights[index]);
            palette[index] = (fred0 * (7.0f - weight) + fred1 * weight) * (1.0f / 7.0f);
        }

        // The index assignment is a closed form, so scoring only needs the chosen ramp position.
        const float inverseSpan = 1.0f / static_cast<float>(red0 - red1);
        const XMVECTOR multiplier = XMVectorReplicate(255.0f * 7.0f * inverseSpan);
        const XMVECTOR offset = XMVectorReplicate(-7.0f * static_cast<float>(red1) * inverseSpan);
        const XMVECTOR lowerBound = XMVectorZero();
        const XMVECTOR upperBound = XMVectorReplicate(7.0f);

        float total = 0.0f;

        for (size_t row = 0; row < 4; ++row)
        {
            XMVECTOR position = XMVectorMultiplyAdd(texels.rows[row], multiplier, offset);
            position = XMVectorClamp(position, lowerBound, upperBound);

            alignas(16) int32_t positions[4];
            _mm_store_si128(reinterpret_cast<__m128i*>(positions), _mm_cvtps_epi32(position));

            for (size_t lane = 0; lane < 4; ++lane)
            {
                const uint32_t index = g_bc4RampToIndex[static_cast<size_t>(positions[lane]) & 0x07u];
                const float difference = XMVectorGetByIndex(texels.rows[row], lane) - palette[index];
                total += difference * difference;
            }
        }

        return total;
    }

    // Assign the closest palette index to every texel of a four-interpolated block.
    // Palette entries six and seven are the absolute values 0.0 and 1.0, so no closed form exists.
    inline uint64_t AssignBC4FourInterpIndices(const BC4TexelBlock& texels, uint32_t red0, uint32_t red1, float& outError) noexcept
    {
        float palette[8];
        BuildBC4FourInterpPalette(red0, red1, palette);

        uint64_t packed = 0;
        outError = 0.0f;

        for (size_t texel = 0; texel < 16; ++texel)
        {
            const float value = XMVectorGetByIndex(texels.rows[texel >> 2], texel & 3);

            float bestDistance = FLT_MAX;
            uint32_t bestIndex = 0;

            for (uint32_t index = 0; index < 8; ++index)
            {
                const float difference = value - palette[index];
                const float distance = difference * difference;
                if (distance < bestDistance)
                {
                    bestDistance = distance;
                    bestIndex = index;
                }
            }

            packed |= static_cast<uint64_t>(bestIndex) << (3 * texel);
            outError += bestDistance;
        }

        return packed;
    }

    // Fit the four-interpolated mode, whose palette holds the absolute values 0.0 and 1.0.
    // Blocks that saturate at either extreme spend no ramp entry reproducing it.
    inline uint64_t EncodeBC4FourInterpBlock(const BC4TexelBlock& texels, float& outError) noexcept
    {
        // The absolute entries cover the extremes, so the ramp only has to span the interior.
        const XMVECTOR minimum = XMVectorMin(XMVectorMin(texels.rows[0], texels.rows[1]), XMVectorMin(texels.rows[2], texels.rows[3]));
        const XMVECTOR maximum = XMVectorMax(XMVectorMax(texels.rows[0], texels.rows[1]), XMVectorMax(texels.rows[2], texels.rows[3]));

        const float blockMinimum = std::min(std::max(HorizontalMinBC4(minimum), 0.0f), 1.0f);
        const float blockMaximum = std::min(std::max(HorizontalMaxBC4(maximum), 0.0f), 1.0f);

        // Interior extremes ignore texels that the absolute palette entries already reproduce.
        constexpr float saturationEpsilon = 1.0f / 512.0f;
        float interiorMinimum = 1.0f;
        float interiorMaximum = 0.0f;
        bool hasInterior = false;

        for (size_t texel = 0; texel < 16; ++texel)
        {
            const float value = XMVectorGetByIndex(texels.rows[texel >> 2], texel & 3);
            if (value > saturationEpsilon && value < 1.0f - saturationEpsilon)
            {
                hasInterior = true;
                interiorMinimum = std::min(interiorMinimum, value);
                interiorMaximum = std::max(interiorMaximum, value);
            }
        }

        if (!hasInterior)
        {
            interiorMinimum = blockMinimum;
            interiorMaximum = blockMaximum;
        }

        // The four-interpolated mode requires red0 <= red1, so the smaller code comes first.
        uint32_t red0 = QuantizeBC4Endpoint(interiorMinimum);
        uint32_t red1 = QuantizeBC4Endpoint(interiorMaximum);
        if (red0 > red1)
        {
            std::swap(red0, red1);
        }

        const uint64_t indices = AssignBC4FourInterpIndices(texels, red0, red1, outError);
        return static_cast<uint64_t>(red0) | (static_cast<uint64_t>(red1) << 8) | (indices << 16);
    }

    // Encode sixteen texel values into one BC4 block using the six-interpolated mode.
    inline uint64_t EncodeBC4SixInterpBlock(const BC4TexelBlock& texels, float& outError) noexcept
    {
        // Two passes suffice because the min and max seed already lies close to the optimum.
        constexpr size_t refinementPasses = 2;
        constexpr float degenerateSpan = 1.0f / 512.0f;

        const XMVECTOR minimum = XMVectorMin(XMVectorMin(texels.rows[0], texels.rows[1]), XMVectorMin(texels.rows[2], texels.rows[3]));
        const XMVECTOR maximum = XMVectorMax(XMVectorMax(texels.rows[0], texels.rows[1]), XMVectorMax(texels.rows[2], texels.rows[3]));

        // A single channel has no principal axis, so the extremes seed the segment directly.
        float endpoint0 = std::min(std::max(HorizontalMaxBC4(maximum), 0.0f), 1.0f);
        float endpoint1 = std::min(std::max(HorizontalMinBC4(minimum), 0.0f), 1.0f);

        for (size_t pass = 0; pass < refinementPasses; ++pass)
        {
            // A flat block defines no direction, so keep the value it already carries.
            if (endpoint0 - endpoint1 < degenerateSpan)
            {
                break;
            }

            const BC4LeastSquaresMoments moments = AccumulateBC4Moments(texels, endpoint0, endpoint1);
            float refined0 = endpoint0;
            float refined1 = endpoint1;
            SolveBC4Endpoints(moments, refined0, refined1);

            // The six-interpolated mode requires red0 > red1, so keep the larger value first.
            if (refined1 > refined0)
            {
                std::swap(refined0, refined1);
            }

            endpoint0 = refined0;
            endpoint1 = refined1;
        }

        uint32_t red0 = QuantizeBC4Endpoint(endpoint0);
        uint32_t red1 = QuantizeBC4Endpoint(endpoint1);

        // Rounding can invert the order of two endpoints that share one code.
        if (red0 < red1)
        {
            std::swap(red0, red1);
        }

        // Equal endpoints decode through the four-interpolated mode, where index zero is exact.
        outError = (red0 > red1) ? EvaluateBC4SixInterpError(texels, red0, red1) : 0.0f;

        if (red0 == red1)
        {
            // A constant block reproduces its single value exactly through palette entry zero.
            const float constantValue = static_cast<float>(red0) * (1.0f / 255.0f);
            for (size_t texel = 0; texel < 16; ++texel)
            {
                const float difference = XMVectorGetByIndex(texels.rows[texel >> 2], texel & 3) - constantValue;
                outError += difference * difference;
            }
        }

        const uint64_t indices = AssignBC4Indices(texels, red0, red1);
        return static_cast<uint64_t>(red0) | (static_cast<uint64_t>(red1) << 8) | (indices << 16);
    }

    // Encode one BC4 block, choosing whichever interpolation mode reproduces it more closely.
    inline uint64_t EncodeBC4Block(const BC4TexelBlock& texels) noexcept
    {
        float sixInterpError = FLT_MAX;
        const uint64_t sixInterp = EncodeBC4SixInterpBlock(texels, sixInterpError);

        float fourInterpError = FLT_MAX;
        const uint64_t fourInterp = EncodeBC4FourInterpBlock(texels, fourInterpError);

        return (fourInterpError < sixInterpError) ? fourInterp : sixInterp;
    }

    // Place one recovered quadrant mean into the child texel grid.
    inline void SetBC4Texel(BC4TexelBlock& texels, size_t texel, float value) noexcept
    {
        XMVECTOR& row = texels.rows[texel >> 2];
        row = XMVectorSetByIndex(row, value, texel & 3);
    }

    // Fetch one mean texel with clamp-to-edge addressing.
    inline float FetchBC4Mean(
        const BC4BlockMean* source,
        size_t width,
        size_t height,
        size_t x,
        size_t y) noexcept
    {
        return source[std::min(y, height - 1) * width + std::min(x, width - 1)];
    }

    // Process one mip-1 block row directly from compressed parent blocks.
    inline void ProcessCompressedRowBC4(
        const Image& source,
        Image& destination,
        size_t destinationRow,
        BC4BlockMean* sourceBlockMeans) noexcept
    {
        const size_t sourceBlockWidth = std::max<size_t>(1, (source.width + 3) / 4);
        const size_t sourceBlockHeight = std::max<size_t>(1, (source.height + 3) / 4);
        const size_t destinationBlockWidth = std::max<size_t>(1, (destination.width + 3) / 4);

        // One destination block covers a 2x2 group of compressed source blocks.
        const size_t sourceY0 = std::min(destinationRow * 2, sourceBlockHeight - 1);
        const size_t sourceY1 = std::min(sourceY0 + 1, sourceBlockHeight - 1);
        const auto* sourceRow0 = reinterpret_cast<const BC4Block*>(source.pixels + sourceY0 * source.rowPitch);
        const auto* sourceRow1 = reinterpret_cast<const BC4Block*>(source.pixels + sourceY1 * source.rowPitch);
        auto* destinationBlocks = reinterpret_cast<BC4Block*>(destination.pixels + destinationRow * destination.rowPitch);

        // Each parent block collapses into one 2x2 texel corner of the destination block.
        constexpr size_t parentBaseTexel[4] = { 0, 2, 8, 10 };
        constexpr size_t quadrantTexelOffset[4] = { 0, 1, 4, 5 };

        for (size_t destinationX = 0; destinationX < destinationBlockWidth; ++destinationX)
        {
            const size_t sourceX0 = std::min(destinationX * 2, sourceBlockWidth - 1);
            const size_t sourceX1 = std::min(sourceX0 + 1, sourceBlockWidth - 1);

            const uint64_t parents[4] =
            {
                LoadBC4Block(sourceRow0[sourceX0]),
                LoadBC4Block(sourceRow0[sourceX1]),
                LoadBC4Block(sourceRow1[sourceX0]),
                LoadBC4Block(sourceRow1[sourceX1]),
            };

            // Every parent quadrant contributes exactly one child texel.
            BC4TexelBlock texels{};
            float parentMeans[4];

            for (size_t parent = 0; parent < 4; ++parent)
            {
                float quadrants[4];
                DecodeBC4QuadrantMeans(parents[parent], quadrants);

                for (size_t quadrant = 0; quadrant < 4; ++quadrant)
                {
                    SetBC4Texel(texels, parentBaseTexel[parent] + quadrantTexelOffset[quadrant], quadrants[quadrant]);
                }

                // The block mean is the material of the higher-level mean pyramid.
                parentMeans[parent] = (quadrants[0] + quadrants[1] + quadrants[2] + quadrants[3]) * 0.25f;
            }

            StoreBC4Block(EncodeBC4Block(texels), destinationBlocks[destinationX]);

            if (sourceBlockMeans)
            {
                // Clamped addressing can make two parents refer to the same source block.
                sourceBlockMeans[sourceY0 * sourceBlockWidth + sourceX0] = parentMeans[0];

                if (sourceX1 != sourceX0)
                {
                    sourceBlockMeans[sourceY0 * sourceBlockWidth + sourceX1] = parentMeans[1];
                }

                if (sourceY1 != sourceY0)
                {
                    sourceBlockMeans[sourceY1 * sourceBlockWidth + sourceX0] = parentMeans[2];
                    if (sourceX1 != sourceX0)
                    {
                        sourceBlockMeans[sourceY1 * sourceBlockWidth + sourceX1] = parentMeans[3];
                    }
                }
            }
        }
    }

    // Halve one row of the scalar block-mean image with edge clamping.
    inline void DownsampleBC4MeanRow(
        const BC4BlockMean* source,
        size_t sourceWidth,
        size_t sourceHeight,
        BC4BlockMean* destination,
        size_t destinationWidth,
        size_t destinationRow) noexcept
    {
        // Select two source rows and repeat the final row when the height is odd.
        const size_t sourceY0 = destinationRow * 2;
        const size_t sourceY1 = std::min(sourceY0 + 1, sourceHeight - 1);
        const auto* sourceRow0 = source + sourceY0 * sourceWidth;
        const auto* sourceRow1 = source + sourceY1 * sourceWidth;
        auto* destinationRowPtr = destination + destinationRow * destinationWidth;

        for (size_t destinationX = 0; destinationX < destinationWidth; ++destinationX)
        {
            const size_t sourceX0 = destinationX * 2;
            const size_t sourceX1 = std::min(sourceX0 + 1, sourceWidth - 1);

            destinationRowPtr[destinationX] =
                (sourceRow0[sourceX0] + sourceRow0[sourceX1] + sourceRow1[sourceX0] + sourceRow1[sourceX1]) * 0.25f;
        }
    }

    // Encode one destination row from the scalar block-mean image.
    inline void ProcessLinearRowBC4(
        const BC4BlockMean* source,
        size_t sourceWidth,
        size_t sourceHeight,
        Image& destination,
        size_t destinationRow) noexcept
    {
        const size_t destinationBlockWidth = std::max<size_t>(1, (destination.width + 3) / 4);
        auto* destinationBlocks = reinterpret_cast<BC4Block*>(destination.pixels + destinationRow * destination.rowPitch);

        for (size_t destinationX = 0; destinationX < destinationBlockWidth; ++destinationX)
        {
            const size_t texelBaseX = destinationX * 4;
            const size_t texelBaseY = destinationRow * 4;

            // The mean pyramid grid already matches this level's texel grid.
            BC4TexelBlock texels{};
            for (size_t localY = 0; localY < 4; ++localY)
            {
                alignas(16) float values[4];
                for (size_t localX = 0; localX < 4; ++localX)
                {
                    values[localX] = FetchBC4Mean(source, sourceWidth, sourceHeight, texelBaseX + localX, texelBaseY + localY);
                }

                texels.rows[localY] = _mm_load_ps(values);
            }

            StoreBC4Block(EncodeBC4Block(texels), destinationBlocks[destinationX]);
        }
    }

    // Generate all compressed mip levels after level zero.
    HRESULT GenerateCompressedMipMapsBC4(const Image& baseImage, ScratchImage& mipChain) noexcept
    {
        // Level 0 is already present, so a one-level chain requires no generation work.
        const size_t mipLevels = mipChain.GetMetadata().mipLevels;
        if (mipLevels <= 1)
        {
            return S_OK;
        }

        // Initialize the shared lookup table before entering an OpenMP region.
        (void)GetBC4RowWeightTable();

        const size_t baseBlockWidth = std::max<size_t>(1, (baseImage.width + 3) / 4);
        const size_t baseBlockHeight = std::max<size_t>(1, (baseImage.height + 3) / 4);
        std::unique_ptr<BC4BlockMean[]> meanImage;
        std::unique_ptr<BC4BlockMean[]> meanScratch;

        // Mip 1 is direct; later levels require two ping-pong mean buffers.
        if (mipLevels > 2)
        {
            const size_t meanCount = baseBlockWidth * baseBlockHeight;
            const size_t scratchWidth = (baseBlockWidth + 1) / 2;
            const size_t scratchHeight = (baseBlockHeight + 1) / 2;
            meanImage.reset(new (std::nothrow) BC4BlockMean[meanCount]{});
            meanScratch.reset(new (std::nothrow) BC4BlockMean[scratchWidth * scratchHeight]{});
            if (!meanImage || !meanScratch)
            {
                return E_OUTOFMEMORY;
            }
        }

        BC4BlockMean* meanFront = meanImage.get();
        BC4BlockMean* meanBack = meanScratch.get();
        size_t meanWidth = baseBlockWidth;
        size_t meanHeight = baseBlockHeight;

        // BC4 stores linear scalar data, so no transfer curve is applied at any level.
        for (size_t mipLevel = 1; mipLevel < mipLevels; ++mipLevel)
        {
            Image* destination = const_cast<Image*>(mipChain.GetImage(mipLevel, 0, 0));
            if (!destination || !destination->pixels)
            {
                return E_FAIL;
            }

            const size_t destinationBlockHeight = std::max<size_t>(1, (destination->height + 3) / 4);

            // Before mip 3 and later, halve the mean pyramid to the required scale.
            if (mipLevel >= 3)
            {
                const size_t nextWidth = (meanWidth + 1) / 2;
                const size_t nextHeight = (meanHeight + 1) / 2;

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
                for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(nextHeight); ++row)
                {
                    DownsampleBC4MeanRow(meanFront, meanWidth, meanHeight, meanBack, nextWidth, static_cast<size_t>(row));
                }

                std::swap(meanFront, meanBack);
                meanWidth = nextWidth;
                meanHeight = nextHeight;
            }

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
            for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(destinationBlockHeight); ++row)
            {
                // Only mip 1 reads BC4 parents; later levels read the scalar mean pyramid.
                if (mipLevel == 1)
                {
                    ProcessCompressedRowBC4(baseImage, *destination, static_cast<size_t>(row), meanFront);
                }
                else
                {
                    ProcessLinearRowBC4(meanFront, meanWidth, meanHeight, *destination, static_cast<size_t>(row));
                }
            }
        }

        return S_OK;
    }

} // namespace for bc4

namespace // for bc7 
{
    constexpr uint8_t BC7_INVALID_MODE = 0xFF;

    // BC7 mode is encoded as zero bits followed by the first one bit.
    inline uint8_t GetBC7Mode(const uint8_t* block) noexcept
    {
        assert(block != nullptr);
        const uint8_t prefix = block[0];

        for (uint8_t mode = 0; mode < 8; ++mode)
        {
            if (prefix & (1u << mode))
            {
                return mode;
            }
        }
        return BC7_INVALID_MODE;
    }

    inline uint8_t ReadBC7Bits(const uint8_t* block, size_t &bitOffset, size_t bitCount) noexcept
    {
        assert(block != nullptr);
        assert(bitCount <= 8);
        assert(bitOffset + bitCount <= 128);

        if (bitCount == 0)
        {
            return 0;
        }

        const size_t byteIndex = bitOffset >> 3;
        const size_t bitIndex = bitOffset & 7;

        uint16_t bits = block[byteIndex];

        if (bitIndex + bitCount > 8)
        {
            bits |= static_cast<uint16_t>(block[byteIndex + 1]) << 8;
        }

        const uint16_t mask = static_cast<uint16_t>((1u << bitCount) - 1u);
        const uint8_t value = static_cast<uint8_t>((bits >> bitIndex) & mask);

        bitOffset += bitCount;
        return value;
    }

    // 2-Subset Partition Set (64 shapes x 16 texels)
    constexpr uint8_t g_bc7PartitionTable2Subsets[64][16] =
    {
        { 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1 }, // Shape 0
        { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 }, // Shape 1
        { 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1 }, // Shape 2
        { 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1 }, // Shape 3
        { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1 }, // Shape 4
        { 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 }, // Shape 5
        { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1 }, // Shape 6
        { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1 }, // Shape 7
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1 }, // Shape 8
        { 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // Shape 9
        { 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1 }, // Shape 10
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1 }, // Shape 11
        { 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // Shape 12
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1 }, // Shape 13
        { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }, // Shape 14
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 }, // Shape 15
        { 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1 }, // Shape 16
        { 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, // Shape 17
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0 }, // Shape 18
        { 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0 }, // Shape 19
        { 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, // Shape 20
        { 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0 }, // Shape 21
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0 }, // Shape 22
        { 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1 }, // Shape 23
        { 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 }, // Shape 24
        { 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0 }, // Shape 25
        { 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0 }, // Shape 26
        { 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0 }, // Shape 27
        { 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0 }, // Shape 28
        { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 }, // Shape 29
        { 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0 }, // Shape 30
        { 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0 }, // Shape 31
        { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1 }, // Shape 32
        { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1 }, // Shape 33
        { 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0 }, // Shape 34
        { 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0 }, // Shape 35
        { 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0 }, // Shape 36
        { 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0 }, // Shape 37
        { 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1 }, // Shape 38
        { 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1 }, // Shape 39
        { 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0 }, // Shape 40
        { 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0 }, // Shape 41
        { 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0 }, // Shape 42
        { 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0 }, // Shape 43
        { 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0 }, // Shape 44
        { 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1 }, // Shape 45
        { 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1 }, // Shape 46
        { 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0 }, // Shape 47
        { 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0 }, // Shape 48
        { 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0 }, // Shape 49
        { 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0 }, // Shape 50
        { 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0 }, // Shape 51
        { 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1 }, // Shape 52
        { 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1 }, // Shape 53
        { 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0 }, // Shape 54
        { 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0 }, // Shape 55
        { 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1 }, // Shape 56
        { 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1 }, // Shape 57
        { 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1 }, // Shape 58
        { 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1 }, // Shape 59
        { 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1 }, // Shape 60
        { 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 }, // Shape 61
        { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0 }, // Shape 62
        { 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1 }  // Shape 63
    };

    // BC7 partition set for three subsets. Mode 0 uses shapes 0..15;
    // Mode 2 uses the complete set of 64 shapes.
    constexpr uint8_t g_bc7PartitionTable3Subsets[64][16] =
    {
        { 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2 },
        { 0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1 },
        { 0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1 },
        { 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2 },
        { 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2 },
        { 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1 },
        { 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2 },
        { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2 },
        { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 },
        { 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2 },
        { 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2 },
        { 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2 },
        { 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2 },
        { 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0 },
        { 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2 },
        { 0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0 },
        { 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2 },
        { 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1 },
        { 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2 },
        { 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1 },
        { 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2 },
        { 0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0 },
        { 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0 },
        { 0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2 },
        { 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0 },
        { 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1 },
        { 0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2 },
        { 0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2 },
        { 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1 },
        { 0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1 },
        { 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2 },
        { 0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1 },
        { 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2 },
        { 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0 },
        { 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0 },
        { 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 },
        { 0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0 },
        { 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1 },
        { 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1 },
        { 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1 },
        { 0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2 },
        { 0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1 },
        { 0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1 },
        { 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1 },
        { 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1 },
        { 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2 },
        { 0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1 },
        { 0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2 },
        { 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2 },
        { 0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2 },
        { 0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2 },
        { 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2 },
        { 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2 },
        { 0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2 },
        { 0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2 },
        { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2 },
        { 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1 },
        { 0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2 },
        { 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 },
        { 0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0 }
    };

    // Fixup index for Subset 1 (Subset 0 fixup is always 0)
    constexpr uint8_t g_bc7FixUp2Subsets[64] =
    {
        15, 15, 15, 15,  15, 15, 15, 15,  15, 15, 15, 15,  15, 15, 15, 15,
        15,  2,  8,  2,   2,  8,  8, 15,   2,  8,  2,  2,   8,  8,  2,  2,
        15, 15,  6,  8,   2,  8, 15, 15,   2,  8,  2,  2,   2, 15, 15,  6,
         6,  2,  6,  8,  15, 15,  2,  2,  15, 15, 15, 15,  15,  2,  2, 15
    };

    // Anchor texels for subsets 1 and 2 of each three-subset partition.
    constexpr uint8_t g_bc7FixUp3Subsets[64][2] =
    {
        { 3,15 }, { 3, 8 }, {15, 8 }, {15, 3 }, { 8,15 }, { 3,15 }, {15, 3 }, {15, 8 },
        { 8,15 }, { 8,15 }, { 6,15 }, { 6,15 }, { 6,15 }, { 5,15 }, { 3,15 }, { 3, 8 },
        { 3,15 }, { 3, 8 }, { 8,15 }, {15, 3 }, { 3,15 }, { 3, 8 }, { 6,15 }, {10, 8 },
        { 5, 3 }, { 8,15 }, { 8, 6 }, { 6,10 }, { 8,15 }, { 5,15 }, {15,10 }, {15, 8 },
        { 8,15 }, {15, 3 }, { 3,15 }, { 5,10 }, { 6,10 }, {10, 8 }, { 8, 9 }, {15,10 },
        {15, 6 }, { 3,15 }, {15, 8 }, { 5,15 }, {15, 3 }, {15, 6 }, {15, 6 }, {15, 8 },
        { 3,15 }, {15, 3 }, { 5,15 }, { 5,15 }, { 5,15 }, { 8,15 }, { 5,15 }, {10,15 },
        { 5,15 }, {10,15 }, { 8,15 }, {13,15 }, {15, 3 }, {12,15 }, { 3,15 }, { 3, 8 }
    };

    // Common spatial representation shared by every BC7 mode.
    struct BC7PartitionLayout
    {
        const uint8_t* subsetByTexel;
        std::array<uint8_t, 3> anchorTexel;
        size_t subsetCount;
    };

    inline BC7PartitionLayout GetBC7PartitionLayout(size_t subsetCount, size_t partition) noexcept
    {
        assert(subsetCount >= 1 && subsetCount <= 3);
        assert(partition < 64);

        static constexpr uint8_t singleSubset[16]{};

        if (subsetCount == 1)
        {
            return { singleSubset, { 0, 0, 0 }, 1 };
        }

        if (subsetCount == 2)
        {
            return { g_bc7PartitionTable2Subsets[partition], { 0, g_bc7FixUp2Subsets[partition], 0 }, 2 };
        }

        return
        {
            g_bc7PartitionTable3Subsets[partition],
            { 0, g_bc7FixUp3Subsets[partition][0], g_bc7FixUp3Subsets[partition][1] },
            3
        };
    }

    // Bit writer writing up to 128 bits into an XMUINT4 block.
    inline void WriteBC7Bits(uint8_t* block, size_t& bitOffset, uint32_t value, size_t bitCount) noexcept
    {
        assert(block != nullptr);
        assert(bitOffset + bitCount <= 128);

        for (size_t i = 0; i < bitCount; ++i)
        {
            const size_t totalBit = bitOffset + i;
            const size_t byteIdx = totalBit >> 3;
            const size_t bitIdx = totalBit & 7;
            if ((value >> i) & 1u)
            {
                block[byteIdx] |= static_cast<uint8_t>(1u << bitIdx);
            }
            else
            {
                block[byteIdx] &= static_cast<uint8_t>(~(1u << bitIdx));
            }
        }
        bitOffset += bitCount;
    }


    // Each mode uses the same parser and stream fitter. P-bits belong to the
    // endpoint grid, so they are never added after the endpoint search.
    enum class Bc7PBits { kNone, kShared, kEndpoint };
    struct Bc7ModeSpec
    {
        uint8_t subsets, partition_bits, rgb_bits, alpha_bits;
        uint8_t primary_bits, secondary_bits;
        Bc7PBits pbits;
    };
    constexpr Bc7ModeSpec kBc7Modes[8] =
    {
        {3,4,4,0,3,0,Bc7PBits::kEndpoint},
        {2,6,6,0,3,0,Bc7PBits::kShared},
        {3,6,5,0,2,0,Bc7PBits::kNone},
        {2,6,7,0,2,0,Bc7PBits::kEndpoint},
        {1,0,5,6,2,3,Bc7PBits::kNone},
        {1,0,7,8,2,2,Bc7PBits::kNone},
        {1,0,7,7,4,0,Bc7PBits::kEndpoint},
        {2,6,5,5,2,0,Bc7PBits::kEndpoint}
    };
    constexpr uint8_t kBc7Weights[3][16] =
    {
        {0,21,43,64},
        {0,9,18,27,37,46,55,64},
        {0,4,9,13,17,21,26,30,34,38,43,47,51,55,60,64}
    };

    // A pixel has four adjacent float channels. A fitting block uses SoA:
    // one SIMD register can compare four texels of the same channel.
    struct Bc7LinearPixel { float channel[4]; };
    struct alignas(16) Bc7LinearBlock { float channel[4][16]; };
    struct Bc7Endpoint
    {
        uint8_t raw[4]{};
        uint8_t expanded[4]{};
        uint8_t pbit = 0;
    };
    struct Bc7Candidate
    {
        uint8_t mode = 0, partition = 0, rotation = 0, index_mode = 0;
        Bc7Endpoint endpoint[3][2]{};
        uint8_t primary[16]{}, secondary[16]{};
        float error = FLT_MAX;
    };
    struct Bc7SymbolicBlock
    {
        Bc7Candidate data;
        // Storage channel to logical RGBA channel. Rotation also moves alpha.
        uint8_t logical[4]{0,1,2,3};
    };
    struct Bc7PaletteCache
    {
        // The scalar stream uses only storage channel 3.
        float vector[3][16][4]{};
        float scalar[16]{};
    };
    enum class Bc7SearchPolicy { kFull, kParentGuided };
    struct Bc7SearchOptions
    {
        Bc7SearchPolicy policy = Bc7SearchPolicy::kParentGuided;
        size_t mode_candidates = 8;
        size_t partition_candidates = 64;
        unsigned refinement_iterations = 2;
        unsigned local_passes = 1;
        float weights[4]{1,1,1,1};
        bool use_parent_guidance = true;
        bool use_previous_encoded_guidance = true;
        // This limits only a continuous GN proposal, not the final search.
        float max_gn_step = 16.0f / 255.0f;
    };
    struct Bc7Telemetry
    {
        float proxy_error[8]{}, full_error[8]{};
        uint8_t parent_modes[4]{255,255,255,255};
        uint8_t selected_mode = 0, partition = 0, rotation = 0, index_mode = 0;
        size_t evaluated_modes = 0, evaluated_structures = 0;
    };

    inline float Bc7Clamp(float value) noexcept
    {
        return std::max(0.0f, std::min(1.0f, value));
    }
    inline float Bc7SrgbToLinear(float code) noexcept
    {
        return code <= 0.04045f ? code / 12.92f
            : std::pow((code + 0.055f) / 1.055f, 2.4f);
    }
    inline float Bc7SrgbDerivative(float code) noexcept
    {
        return code <= 0.04045f ? 1.0f / 12.92f
            : (2.4f / 1.055f) * std::pow((code + 0.055f) / 1.055f, 1.4f);
    }
    template<bool IsSrgb>
    inline float Bc7DecodeCode(uint8_t code, size_t logical) noexcept
    {
        // Alpha always stays linear, even inside a rotated vector stream.
        if (IsSrgb)
        {
            if (logical < 3) return GetSrgb8ToLinearTable()[code];
        }
        return static_cast<float>(code) / 255.0f;
    }
    template<bool IsSrgb>
    inline float Bc7EncodeCode(float linear, size_t logical) noexcept
    {
        linear = Bc7Clamp(linear);
        if (IsSrgb)
        {
            if (logical < 3)
                return linear <= 0.0031308f ? 12.92f * linear
                    : 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
        }
        return linear;
    }
    inline uint8_t Bc7Expand(uint8_t raw, unsigned bits, int pbit) noexcept
    {
        unsigned value = raw;
        if (pbit >= 0)
        {
            value = (value << 1) | static_cast<unsigned>(pbit);
            ++bits;
        }
        // BC7 repeats the high bits after shifting into the eight-bit range.
        return static_cast<uint8_t>((value << (8 - bits)) | (value >> (2 * bits - 8)));
    }
    inline void Bc7ChannelLayout(uint8_t rotation, uint8_t logical[4]) noexcept
    {
        for (uint8_t c = 0; c < 4; ++c) logical[c] = c;
        if (rotation) std::swap(logical[3], logical[rotation - 1]);
    }

    // Parse symbols, not a full decoded image. Each anchor omits its index MSB.
    inline bool ParseBc7SymbolicBlock(const uint8_t* bytes, Bc7SymbolicBlock& block) noexcept
    {
        block = {};
        auto& data = block.data;
        data.mode = GetBC7Mode(bytes);
        if (data.mode == BC7_INVALID_MODE) return false;
        const auto& spec = kBc7Modes[data.mode];
        size_t offset = data.mode + 1;
        data.partition = ReadBC7Bits(bytes, offset, spec.partition_bits);
        if (spec.secondary_bits)
        {
            data.rotation = ReadBC7Bits(bytes, offset, 2);
            if (data.mode == 4) data.index_mode = ReadBC7Bits(bytes, offset, 1);
        }
        Bc7ChannelLayout(data.rotation, block.logical);
        // Endpoints are stored by channel, then subset, then endpoint.
        for (size_t c = 0; c < 4; ++c)
            for (size_t s = 0; s < spec.subsets; ++s)
                for (size_t e = 0; e < 2; ++e)
                    data.endpoint[s][e].raw[c] = ReadBC7Bits(
                        bytes, offset, c == 3 ? spec.alpha_bits : spec.rgb_bits);
        for (size_t s = 0; s < spec.subsets; ++s)
        {
            if (spec.pbits == Bc7PBits::kShared)
            {
                const uint8_t p = ReadBC7Bits(bytes, offset, 1);
                data.endpoint[s][0].pbit = data.endpoint[s][1].pbit = p;
            }
            else if (spec.pbits == Bc7PBits::kEndpoint)
            {
                for (auto& ep : data.endpoint[s])
                    ep.pbit = ReadBC7Bits(bytes, offset, 1);
            }
            for (auto& ep : data.endpoint[s])
                for (size_t c = 0; c < 4; ++c)
                {
                    const unsigned bits = c == 3 ? spec.alpha_bits : spec.rgb_bits;
                    ep.expanded[c] = bits ? Bc7Expand(ep.raw[c], bits,
                        spec.pbits == Bc7PBits::kNone ? -1 : ep.pbit) : 255;
                }
        }
        const auto layout = GetBC7PartitionLayout(spec.subsets, data.partition);
        for (size_t t = 0; t < 16; ++t)
        {
            const size_t subset = layout.subsetByTexel[t];
            data.primary[t] = ReadBC7Bits(bytes, offset,
                spec.primary_bits - (t == layout.anchorTexel[subset] ? 1 : 0));
        }
        if (spec.secondary_bits)
            for (size_t t = 0; t < 16; ++t)
                data.secondary[t] = ReadBC7Bits(bytes, offset,
                    spec.secondary_bits - (t == 0 ? 1 : 0));
        assert(offset == 128);
        return offset == 128;
    }

    template<bool IsSrgb>
    inline void BuildBc7PaletteCache(const Bc7SymbolicBlock& block, Bc7PaletteCache& cache) noexcept
    {
        const auto& data = block.data;
        const auto& spec = kBc7Modes[data.mode];
        const unsigned vector_bits = data.index_mode ? spec.secondary_bits : spec.primary_bits;
        const unsigned scalar_bits = data.index_mode ? spec.primary_bits : spec.secondary_bits;
        for (size_t s = 0; s < spec.subsets; ++s)
        {
            for (unsigned j = 0; j < (1u << vector_bits); ++j)
            {
                const unsigned w = kBc7Weights[vector_bits - 2][j];
                for (size_t c = 0; c < (spec.secondary_bits ? 3u : 4u); ++c)
                {
                    const auto code = static_cast<uint8_t>(((64 - w) * data.endpoint[s][0].expanded[c]
                        + w * data.endpoint[s][1].expanded[c] + 32) >> 6);
                    cache.vector[s][j][c] = Bc7DecodeCode<IsSrgb>(code, block.logical[c]);
                }
            }
        }
        if (scalar_bits)
            for (unsigned j = 0; j < (1u << scalar_bits); ++j)
            {
                const unsigned w = kBc7Weights[scalar_bits - 2][j];
                const auto code = static_cast<uint8_t>(((64 - w) * data.endpoint[0][0].expanded[3]
                    + w * data.endpoint[0][1].expanded[3] + 32) >> 6);
                cache.scalar[j] = Bc7DecodeCode<IsSrgb>(code, block.logical[3]);
            }
    }
    inline Bc7LinearPixel EvaluateBc7SymbolicTexel(
        const Bc7SymbolicBlock& block, const Bc7PaletteCache& cache, size_t texel) noexcept
    {
        const auto& d = block.data;
        const auto& spec = kBc7Modes[d.mode];
        const auto layout = GetBC7PartitionLayout(spec.subsets, d.partition);
        const size_t subset = layout.subsetByTexel[texel];
        const unsigned vector_index = d.index_mode ? d.secondary[texel] : d.primary[texel];
        Bc7LinearPixel result{};
        for (size_t c = 0; c < (spec.secondary_bits ? 3u : 4u); ++c)
            result.channel[block.logical[c]] = cache.vector[subset][vector_index][c];
        if (spec.secondary_bits)
        {
            const unsigned scalar_index = d.index_mode ? d.primary[texel] : d.secondary[texel];
            result.channel[block.logical[3]] = cache.scalar[scalar_index];
        }
        return result;
    }

    // Decode a 2x2 quadrant through selector histograms. A palette entry is
    // multiplied once by its frequency instead of being looked up per texel.
    inline Bc7LinearPixel ComputeBc7QuadrantMean(
        const Bc7SymbolicBlock& block, const Bc7PaletteCache& cache, size_t quadrant) noexcept
    {
        const auto& data = block.data;
        const auto& spec = kBc7Modes[data.mode];
        const auto layout = GetBC7PartitionLayout(spec.subsets, data.partition);
        uint8_t vector_subset[4]{}, vector_index[4]{}, vector_frequency[4]{};
        uint8_t scalar_index[4]{}, scalar_frequency[4]{};
        size_t vector_entries = 0, scalar_entries = 0;
        const size_t x0 = (quadrant & 1) * 2;
        const size_t y0 = (quadrant >> 1) * 2;
        for (size_t y = 0; y < 2; ++y)
            for (size_t x = 0; x < 2; ++x)
            {
                const size_t texel = (y0 + y) * 4 + x0 + x;
                const size_t subset = layout.subsetByTexel[texel];
                const uint8_t index = data.index_mode
                    ? data.secondary[texel] : data.primary[texel];
                size_t entry = 0;
                while (entry < vector_entries
                    && (vector_subset[entry] != subset || vector_index[entry] != index)) ++entry;
                if (entry == vector_entries)
                {
                    vector_subset[entry] = static_cast<uint8_t>(subset);
                    vector_index[entry] = index;
                    ++vector_entries;
                }
                ++vector_frequency[entry];
                if (spec.secondary_bits)
                {
                    const uint8_t scalar = data.index_mode
                        ? data.primary[texel] : data.secondary[texel];
                    entry = 0;
                    while (entry < scalar_entries && scalar_index[entry] != scalar) ++entry;
                    if (entry == scalar_entries)
                    {
                        scalar_index[entry] = scalar;
                        ++scalar_entries;
                    }
                    ++scalar_frequency[entry];
                }
            }
        Bc7LinearPixel result{};
        for (size_t entry = 0; entry < vector_entries; ++entry)
        {
            const float weight = 0.25f * vector_frequency[entry];
            for (size_t c = 0; c < (spec.secondary_bits ? 3u : 4u); ++c)
                result.channel[block.logical[c]] += weight
                    * cache.vector[vector_subset[entry]][vector_index[entry]][c];
        }
        if (spec.secondary_bits)
            for (size_t entry = 0; entry < scalar_entries; ++entry)
                result.channel[block.logical[3]] += 0.25f * scalar_frequency[entry]
                    * cache.scalar[scalar_index[entry]];
        return result;
    }

    // Keep each parent subset/stream separate. Collapsing them into one line
    // loses the color families that distinguish one-, two-, and three-subset modes.
    struct Bc7GuidanceLine
    {
        float endpoint[2][4]{};
        float mean[4]{};
        float direction[4]{};
        float selector_mean = 0;
        float selector_variance = 0;
        float energy = 0;
        float coherence = 1;
        uint16_t selector_histogram[16]{};
        uint8_t logical_mask = 0;
        uint8_t parent = 0, subset = 0, index_bits = 0;
    };
    struct Bc7Guidance
    {
        Bc7GuidanceLine lines[24]{};
        float mean[4]{};
        uint8_t modes[4]{255,255,255,255};
        uint8_t subset_counts[4]{};
        uint8_t parent_count = 0, line_count = 0;
    };
    struct Bc7Stream
    {
        uint8_t logical[4]{0,1,2,3};
        uint8_t bits[4]{};
        uint8_t channels = 0, index_bits = 0;
        Bc7PBits pbits = Bc7PBits::kNone;
    };
    struct Bc7StreamFit
    {
        Bc7Endpoint endpoint[2]{};
        uint8_t indices[16]{};
        float error = FLT_MAX;
    };
    struct Bc7Seed { float code[2][4]{}; };
    struct Bc7SubsetStatistics
    {
        float mean[4]{}, low[4]{}, high[4]{}, covariance[4][4]{};
    };

    inline Bc7SubsetStatistics ComputeBc7Statistics(
        const Bc7LinearBlock& target, uint16_t mask, const Bc7Stream& stream) noexcept
    {
        Bc7SubsetStatistics stats{};
        size_t count = 0;
        for (size_t c = 0; c < stream.channels; ++c)
        {
            stats.low[c] = FLT_MAX;
            stats.high[c] = -FLT_MAX;
        }
        for (size_t t = 0; t < 16; ++t)
        {
            if (!(mask & (1u << t))) continue;
            ++count;
            for (size_t c = 0; c < stream.channels; ++c)
            {
                const float v = target.channel[stream.logical[c]][t];
                stats.mean[c] += v;
                stats.low[c] = std::min(stats.low[c], v);
                stats.high[c] = std::max(stats.high[c], v);
            }
        }
        assert(count != 0);
        for (size_t c = 0; c < stream.channels; ++c) stats.mean[c] /= static_cast<float>(count);
        // Center the samples before accumulating covariance to avoid cancellation.
        for (size_t t = 0; t < 16; ++t)
        {
            if (!(mask & (1u << t))) continue;
            for (size_t c = 0; c < stream.channels; ++c)
                for (size_t d = 0; d < stream.channels; ++d)
                    stats.covariance[c][d] +=
                        (target.channel[stream.logical[c]][t] - stats.mean[c]) *
                        (target.channel[stream.logical[d]][t] - stats.mean[d]);
        }
        return stats;
    }

    template<bool IsSrgb>
    inline size_t GenerateBc7Seeds(const Bc7LinearBlock& target, uint16_t mask,
        const Bc7Stream& stream, const Bc7Guidance* guidance, uint8_t expected_subsets,
        bool proxy, Bc7Seed seeds[3]) noexcept
    {
        const auto stats = ComputeBc7Statistics(target, mask, stream);
        for (size_t c = 0; c < stream.channels; ++c)
        {
            seeds[0].code[0][c] = Bc7EncodeCode<IsSrgb>(stats.low[c], stream.logical[c]);
            seeds[0].code[1][c] = Bc7EncodeCode<IsSrgb>(stats.high[c], stream.logical[c]);
        }
        size_t count = 1;
        // A proxy also evaluates a parent-derived endpoint proposal. This makes
        // parent endpoint and selector statistics part of the numerical mode score.
        if (guidance && guidance->line_count)
        {
            const uint8_t required_mask = [&]() noexcept
            {
                uint8_t result = 0;
                for (size_t c = 0; c < stream.channels; ++c)
                    result |= static_cast<uint8_t>(1u << stream.logical[c]);
                return result;
            }();
            const Bc7GuidanceLine* best_line = nullptr;
            float best_score = -FLT_MAX;
            for (size_t i = 0; i < guidance->line_count; ++i)
            {
                const auto& line = guidance->lines[i];
                if ((line.logical_mask & required_mask) != required_mask) continue;
                float distance = 0, child_energy = 0, span = 0;
                for (size_t c = 0; c < stream.channels; ++c)
                {
                    const size_t logical_c = stream.logical[c];
                    const float delta = stats.mean[c] - line.mean[logical_c];
                    distance += delta * delta;
                    const float extent = line.endpoint[1][logical_c] - line.endpoint[0][logical_c];
                    span += extent * extent;
                    for (size_t d = 0; d < stream.channels; ++d)
                        child_energy += line.direction[logical_c]
                            * stats.covariance[c][d]
                            * line.direction[stream.logical[d]];
                }
                unsigned samples = 0;
                for (uint16_t frequency : line.selector_histogram) samples += frequency;
                const float parent_energy = samples * line.selector_variance * span;
                const float subset_prior = guidance->subset_counts[line.parent] == expected_subsets
                    ? 1.0f : 0.5f;
                const float index_prior = line.index_bits == stream.index_bits ? 1.0f : 0.75f;
                // Selector variance favors a parent line that actually used its span.
                const float score = subset_prior * index_prior * (0.5f + 0.5f * line.coherence)
                    * (std::max(0.0f, child_energy) + parent_energy)
                    * (0.25f + line.selector_variance) / (1.0f + 8.0f * distance);
                if (score > best_score)
                {
                    best_score = score;
                    best_line = &line;
                }
            }
            if (best_line)
            {
                Bc7Seed parent_seed{};
                for (size_t c = 0; c < stream.channels; ++c)
                {
                    const size_t logical_c = stream.logical[c];
                    const float translation = stats.mean[c] - best_line->mean[logical_c];
                    for (size_t e = 0; e < 2; ++e)
                        parent_seed.code[e][c] = Bc7EncodeCode<IsSrgb>(
                            best_line->endpoint[e][logical_c] + translation, logical_c);
                }
                bool duplicate = true;
                for (size_t e = 0; e < 2; ++e)
                    for (size_t c = 0; c < stream.channels; ++c)
                        duplicate = duplicate && parent_seed.code[e][c] == seeds[0].code[e][c];
                if (!duplicate) seeds[count++] = parent_seed;
            }
        }
        if (proxy) return count;

        // Full fitting additionally tries the child's principal component.
        {
            float axis[4]{};
            // Start on the channel with greatest variance. An all-ones start
            // can be perpendicular to a line with negative color correlation.
            size_t start = 0;
            for (size_t c = 1; c < stream.channels; ++c)
                if (stats.covariance[c][c] > stats.covariance[start][start]) start = c;
            axis[start] = 1;
            for (unsigned step = 0; step < 4; ++step)
            {
                float next[4]{}, norm = 0;
                for (size_t c = 0; c < stream.channels; ++c)
                {
                    for (size_t d = 0; d < stream.channels; ++d)
                        next[c] += stats.covariance[c][d] * axis[d];
                    norm += next[c] * next[c];
                }
                if (norm <= 1e-20f) break;
                const float inverse = 1.0f / std::sqrt(norm);
                for (size_t c = 0; c < stream.channels; ++c) axis[c] = next[c] * inverse;
            }
            float norm = 0;
            for (size_t c = 0; c < stream.channels; ++c) norm += axis[c] * axis[c];
            if (norm <= 1e-20f) return count;
            for (size_t c = 0; c < stream.channels; ++c) axis[c] /= std::sqrt(norm);
            float low = FLT_MAX, high = -FLT_MAX;
            for (size_t t = 0; t < 16; ++t)
            {
                if (!(mask & (1u << t))) continue;
                float projection = 0;
                for (size_t c = 0; c < stream.channels; ++c)
                    projection += (target.channel[stream.logical[c]][t] - stats.mean[c]) * axis[c];
                low = std::min(low, projection);
                high = std::max(high, projection);
            }
            Bc7Seed proposal{};
            bool duplicate = false;
            for (size_t c = 0; c < stream.channels; ++c)
            {
                proposal.code[0][c] = Bc7EncodeCode<IsSrgb>(
                    stats.mean[c] + low * axis[c], stream.logical[c]);
                proposal.code[1][c] = Bc7EncodeCode<IsSrgb>(
                    stats.mean[c] + high * axis[c], stream.logical[c]);
            }
            for (size_t i = 0; i < count; ++i)
            {
                bool equal = true;
                for (size_t e = 0; e < 2; ++e)
                    for (size_t c = 0; c < stream.channels; ++c)
                        equal = equal && proposal.code[e][c] == seeds[i].code[e][c];
                duplicate = duplicate || equal;
            }
            if (!duplicate) seeds[count++] = proposal;
        }
        return count;
    }

    inline uint8_t QuantizeBc7Component(float code, unsigned bits, int pbit) noexcept
    {
        // Search the monotone expanded grid. Bit replication makes this grid
        // slightly different from uniformly spaced normalized endpoint values.
        const float target = Bc7Clamp(code) * 255.0f;
        unsigned low = 0, high = (1u << bits) - 1;
        while (low < high)
        {
            const unsigned middle = (low + high) / 2;
            if (Bc7Expand(static_cast<uint8_t>(middle), bits, pbit) < target) low = middle + 1;
            else high = middle;
        }
        if (low && std::abs(target - Bc7Expand(static_cast<uint8_t>(low - 1), bits, pbit))
            <= std::abs(target - Bc7Expand(static_cast<uint8_t>(low), bits, pbit))) --low;
        return static_cast<uint8_t>(low);
    }

    // Invariants B and D: score the real integer palette in linear light and
    // select a new nearest entry. Four texels share each SIMD distance operation.
    template<bool IsSrgb>
    inline void AssignBc7Selectors(const Bc7LinearBlock& target, uint16_t mask,
        const Bc7Stream& stream, const Bc7SearchOptions& options, Bc7StreamFit& fit) noexcept
    {
        float palette[16][4]{};
        const unsigned entries = 1u << stream.index_bits;
        for (unsigned j = 0; j < entries; ++j)
        {
            const unsigned weight = kBc7Weights[stream.index_bits - 2][j];
            for (size_t c = 0; c < stream.channels; ++c)
            {
                const auto code = static_cast<uint8_t>(((64 - weight) * fit.endpoint[0].expanded[c]
                    + weight * fit.endpoint[1].expanded[c] + 32) >> 6);
                palette[j][c] = Bc7DecodeCode<IsSrgb>(code, stream.logical[c]);
            }
        }
        fit.error = 0;
        for (size_t t = 0; t < 16; t += 4)
        {
            XMVECTOR best = XMVectorReplicate(FLT_MAX);
            XMVECTOR indices = XMVectorZero();
            for (unsigned j = 0; j < entries; ++j)
            {
                XMVECTOR error = XMVectorZero();
                for (size_t c = 0; c < stream.channels; ++c)
                {
                    const size_t logical = stream.logical[c];
                    const XMVECTOR delta = XMVectorSubtract(_mm_load_ps(target.channel[logical] + t),
                        XMVectorReplicate(palette[j][c]));
                    error = XMVectorAdd(error, XMVectorScale(XMVectorMultiply(delta, delta),
                        options.weights[logical]));
                }
                const XMVECTOR better = XMVectorLess(error, best);
                best = XMVectorSelect(best, error, better);
                indices = XMVectorSelect(indices, XMVectorReplicateInt(j), better);
            }
            alignas(16) float errors[4];
            alignas(16) uint32_t selected[4];
            _mm_store_ps(errors, best);
            _mm_store_si128(reinterpret_cast<__m128i*>(selected), _mm_castps_si128(indices));
            for (size_t lane = 0; lane < 4; ++lane)
                if (mask & (1u << (t + lane)))
                {
                    fit.indices[t + lane] = static_cast<uint8_t>(selected[lane]);
                    fit.error += errors[lane];
                }
        }
    }

    template<bool IsSrgb>
    inline Bc7StreamFit ProjectBc7Seed(const Bc7Seed& seed, const Bc7LinearBlock& target,
        uint16_t mask, const Bc7Stream& stream, const Bc7SearchOptions& options) noexcept
    {
        Bc7StreamFit best{};
        const unsigned states = stream.pbits == Bc7PBits::kNone ? 1
            : stream.pbits == Bc7PBits::kShared ? 2 : 4;
        // Invariant G: enumerate each legal P-bit lattice before scoring it.
        for (unsigned state = 0; state < states; ++state)
        {
            Bc7StreamFit trial{};
            for (unsigned e = 0; e < 2; ++e)
            {
                const int p = stream.pbits == Bc7PBits::kNone ? -1
                    : stream.pbits == Bc7PBits::kShared ? static_cast<int>(state)
                    : static_cast<int>((state >> e) & 1u);
                trial.endpoint[e].pbit = static_cast<uint8_t>(std::max(0, p));
                for (size_t c = 0; c < stream.channels; ++c)
                {
                    trial.endpoint[e].raw[c] = QuantizeBc7Component(seed.code[e][c], stream.bits[c], p);
                    trial.endpoint[e].expanded[c] = Bc7Expand(trial.endpoint[e].raw[c], stream.bits[c], p);
                }
            }
            AssignBc7Selectors<IsSrgb>(target, mask, stream, options, trial);
            if (trial.error < best.error) best = trial;
        }
        return best;
    }

    // Solve a selector-fixed LS or GN step in normalized endpoint code space.
    template<bool IsSrgb>
    inline Bc7Seed RefineBc7Endpoints(const Bc7LinearBlock& target, uint16_t mask,
        const Bc7Stream& stream, const Bc7StreamFit& fit, const Bc7SearchOptions& options) noexcept
    {
        Bc7Seed proposal{};
        for (size_t c = 0; c < stream.channels; ++c)
        {
            const float e0 = fit.endpoint[0].expanded[c] / 255.0f;
            const float e1 = fit.endpoint[1].expanded[c] / 255.0f;
            double aa = 0, ab = 0, bb = 0, ar = 0, br = 0;
            for (size_t t = 0; t < 16; ++t)
            {
                if (!(mask & (1u << t))) continue;
                const float lambda = kBc7Weights[stream.index_bits - 2][fit.indices[t]] / 64.0f;
                const float code = (1 - lambda) * e0 + lambda * e1;
                float decoded = code, derivative = 1;
                if (IsSrgb)
                {
                    if (stream.logical[c] < 3)
                    {
                        decoded = Bc7SrgbToLinear(code);
                        derivative = Bc7SrgbDerivative(code);
                    }
                }
                const double a = (1 - lambda) * derivative, b = lambda * derivative;
                const double residual = target.channel[stream.logical[c]][t] - decoded;
                aa += a * a; ab += a * b; bb += b * b;
                ar += a * residual; br += b * residual;
            }
            float delta0 = 0, delta1 = 0;
            const double determinant = aa * bb - ab * ab;
            // A constant selector gives a singular system. Keep that endpoint pair.
            if (determinant > 1e-12 * std::max(1.0, aa * bb))
            {
                delta0 = static_cast<float>((ar * bb - br * ab) / determinant);
                delta1 = static_cast<float>((br * aa - ar * ab) / determinant);
                if (IsSrgb)
                {
                    if (stream.logical[c] < 3)
                    {
                        delta0 = std::max(-options.max_gn_step, std::min(options.max_gn_step, delta0));
                        delta1 = std::max(-options.max_gn_step, std::min(options.max_gn_step, delta1));
                    }
                }
            }
            proposal.code[0][c] = Bc7Clamp(e0 + delta0);
            proposal.code[1][c] = Bc7Clamp(e1 + delta1);
        }
        return proposal;
    }

    template<bool IsSrgb>
    inline Bc7StreamFit FitBc7Stream(const Bc7LinearBlock& target, uint16_t mask,
        const Bc7Stream& stream, const Bc7Guidance* guidance,
        const Bc7SearchOptions& options, uint8_t expected_subsets, bool proxy) noexcept
    {
        Bc7Seed seeds[3]{};
        const size_t seed_count = GenerateBc7Seeds<IsSrgb>(
            target, mask, stream, guidance, expected_subsets, proxy, seeds);
        Bc7StreamFit best{};
        for (size_t i = 0; i < seed_count; ++i)
        {
            auto current = ProjectBc7Seed<IsSrgb>(seeds[i], target, mask, stream, options);
            if (!proxy)
            {
                // Invariants E and F: hold selectors for the solve, then project
                // endpoints and assign selectors again before accepting a step.
                for (unsigned iteration = 0; iteration < options.refinement_iterations; ++iteration)
                {
                    const auto seed = RefineBc7Endpoints<IsSrgb>(target, mask, stream, current, options);
                    auto next = ProjectBc7Seed<IsSrgb>(seed, target, mask, stream, options);
                    if (next.error >= current.error) break;
                    current = next;
                }
            }
            if (current.error < best.error) best = current;
        }
        if (!proxy)
            for (unsigned pass = 0; pass < options.local_passes; ++pass)
                for (size_t e = 0; e < 2; ++e)
                    for (size_t c = 0; c < stream.channels; ++c)
                    {
                        // Keep the P-bit fixed while trying neighboring raw codes.
                        const auto base = best;
                        // Test one grid point on each side without creating a container.
                        for (int step = -1; step <= 1; step += 2)
                        {
                            const int raw = base.endpoint[e].raw[c] + step;
                            if (raw < 0 || raw >= (1 << stream.bits[c])) continue;
                            auto trial = base;
                            trial.endpoint[e].raw[c] = static_cast<uint8_t>(raw);
                            trial.endpoint[e].expanded[c] = Bc7Expand(static_cast<uint8_t>(raw), stream.bits[c],
                                stream.pbits == Bc7PBits::kNone ? -1 : trial.endpoint[e].pbit);
                            AssignBc7Selectors<IsSrgb>(target, mask, stream, options, trial);
                            if (trial.error < best.error) best = trial;
                        }
                    }
        return best;
    }

    inline float Bc7OpaqueAlphaError(const Bc7LinearBlock& target, const Bc7SearchOptions& options) noexcept
    {
        float error = 0;
        for (float alpha : target.channel[3]) error += (alpha - 1) * (alpha - 1);
        return error * options.weights[3];
    }

    template<bool IsSrgb>
    inline Bc7Candidate FitBc7Structure(const Bc7LinearBlock& target,
        const Bc7Guidance* guidance, const Bc7SearchOptions& options,
        uint8_t mode, uint8_t partition, uint8_t rotation, uint8_t index_mode, bool proxy) noexcept
    {
        Bc7Candidate candidate{};
        candidate.mode = mode; candidate.partition = partition;
        candidate.rotation = rotation; candidate.index_mode = index_mode;
        candidate.error = mode < 4 ? Bc7OpaqueAlphaError(target, options) : 0;
        const auto& spec = kBc7Modes[mode];
        const auto layout = GetBC7PartitionLayout(spec.subsets, partition);
        uint8_t logical[4];
        Bc7ChannelLayout(rotation, logical);
        for (size_t subset = 0; subset < spec.subsets; ++subset)
        {
            uint16_t mask = 0;
            for (size_t t = 0; t < 16; ++t)
                if (layout.subsetByTexel[t] == subset) mask |= static_cast<uint16_t>(1u << t);
            Bc7Stream stream{};
            stream.channels = spec.secondary_bits || !spec.alpha_bits ? 3 : 4;
            stream.index_bits = index_mode ? spec.secondary_bits : spec.primary_bits;
            stream.pbits = spec.pbits;
            for (size_t c = 0; c < stream.channels; ++c)
            {
                stream.logical[c] = logical[c];
                stream.bits[c] = c == 3 ? spec.alpha_bits : spec.rgb_bits;
            }
            const auto fit = FitBc7Stream<IsSrgb>(
                target, mask, stream, guidance, options, spec.subsets, proxy);
            candidate.error += fit.error;
            for (size_t e = 0; e < 2; ++e)
            {
                candidate.endpoint[subset][e] = fit.endpoint[e];
                if (!spec.alpha_bits) candidate.endpoint[subset][e].expanded[3] = 255;
            }
            // Index selection chooses which physical stream carries vector colors.
            auto* vector_indices = index_mode ? candidate.secondary : candidate.primary;
            for (size_t t = 0; t < 16; ++t)
                if (mask & (1u << t)) vector_indices[t] = fit.indices[t];
        }
        if (spec.secondary_bits)
        {
            Bc7Stream scalar{};
            scalar.channels = 1;
            scalar.logical[0] = logical[3];
            scalar.bits[0] = spec.alpha_bits;
            scalar.index_bits = index_mode ? spec.primary_bits : spec.secondary_bits;
            const auto fit = FitBc7Stream<IsSrgb>(
                target, 0xFFFF, scalar, guidance, options, 1, proxy);
            candidate.error += fit.error;
            for (size_t e = 0; e < 2; ++e)
            {
                candidate.endpoint[0][e].raw[3] = fit.endpoint[e].raw[0];
                candidate.endpoint[0][e].expanded[3] = fit.endpoint[e].expanded[0];
            }
            auto* scalar_indices = index_mode ? candidate.primary : candidate.secondary;
            for (size_t t = 0; t < 16; ++t) scalar_indices[t] = fit.indices[t];
        }
        return candidate;
    }

    struct Bc7StructureScore
    {
        uint8_t partition = 0, rotation = 0, index_mode = 0;
        float error = FLT_MAX;
    };
    struct Bc7ModeProxy
    {
        uint8_t mode = 0;
        size_t count = 0;
        Bc7StructureScore structures[64]{};
        float error = FLT_MAX;
    };
    template<bool IsSrgb>
    inline Bc7ModeProxy EvaluateBc7ModeProxy(const Bc7LinearBlock& target,
        const Bc7Guidance* guidance, const Bc7SearchOptions& options, uint8_t mode) noexcept
    {
        Bc7ModeProxy proxy{};
        proxy.mode = mode;
        const auto& spec = kBc7Modes[mode];
        const unsigned partitions = 1u << spec.partition_bits;
        const unsigned rotations = spec.secondary_bits ? 4 : 1;
        const unsigned index_modes = mode == 4 ? 2 : 1;
        for (unsigned p = 0; p < partitions; ++p)
            for (unsigned r = 0; r < rotations; ++r)
                for (unsigned i = 0; i < index_modes; ++i)
                {
                    const auto candidate = FitBc7Structure<IsSrgb>(target, guidance, options, mode,
                        static_cast<uint8_t>(p), static_cast<uint8_t>(r), static_cast<uint8_t>(i), true);
                    proxy.structures[proxy.count++] =
                    {
                        static_cast<uint8_t>(p), static_cast<uint8_t>(r),
                        static_cast<uint8_t>(i), candidate.error
                    };
                    proxy.error = std::min(proxy.error, candidate.error);
                }
        // A proxy orders work; it is not a lower bound on a refined candidate.
        std::sort(proxy.structures, proxy.structures + proxy.count,
            [](const Bc7StructureScore& a, const Bc7StructureScore& b)
            {
                if (a.error != b.error) return a.error < b.error;
                if (a.partition != b.partition) return a.partition < b.partition;
                if (a.rotation != b.rotation) return a.rotation < b.rotation;
                return a.index_mode < b.index_mode;
            });
        return proxy;
    }

    // Correct anchor orientation without changing any reconstructed color.
    inline void CanonicalizeBc7Fixups(Bc7Candidate& candidate) noexcept
    {
        const auto& spec = kBc7Modes[candidate.mode];
        const auto layout = GetBC7PartitionLayout(spec.subsets, candidate.partition);
        if (spec.secondary_bits)
        {
            for (unsigned stream = 0; stream < 2; ++stream)
            {
                auto* indices = stream ? candidate.secondary : candidate.primary;
                const unsigned bits = stream ? spec.secondary_bits : spec.primary_bits;
                if (!(indices[0] & (1u << (bits - 1)))) continue;
                const bool vector = stream == candidate.index_mode;
                // A dual stream swaps only its own storage channel endpoints.
                for (size_t c = vector ? 0 : 3; c < (vector ? 3u : 4u); ++c)
                {
                    std::swap(candidate.endpoint[0][0].raw[c], candidate.endpoint[0][1].raw[c]);
                    std::swap(candidate.endpoint[0][0].expanded[c], candidate.endpoint[0][1].expanded[c]);
                }
                for (size_t t = 0; t < 16; ++t)
                    indices[t] = static_cast<uint8_t>((1u << bits) - 1 - indices[t]);
            }
        }
        else
            for (size_t s = 0; s < spec.subsets; ++s)
            {
                if (!(candidate.primary[layout.anchorTexel[s]] & (1u << (spec.primary_bits - 1)))) continue;
                // Per-endpoint P-bits move together with the endpoint.
                std::swap(candidate.endpoint[s][0], candidate.endpoint[s][1]);
                for (size_t t = 0; t < 16; ++t)
                    if (layout.subsetByTexel[t] == s)
                        candidate.primary[t] = static_cast<uint8_t>(
                            (1u << spec.primary_bits) - 1 - candidate.primary[t]);
            }
    }
    inline void EmitBc7Candidate(Bc7Candidate candidate, uint8_t* bytes) noexcept
    {
        CanonicalizeBc7Fixups(candidate);
        const auto& spec = kBc7Modes[candidate.mode];
        const auto layout = GetBC7PartitionLayout(spec.subsets, candidate.partition);
        size_t offset = 0;
        WriteBC7Bits(bytes, offset, 1u << candidate.mode, candidate.mode + 1);
        WriteBC7Bits(bytes, offset, candidate.partition, spec.partition_bits);
        if (spec.secondary_bits)
        {
            WriteBC7Bits(bytes, offset, candidate.rotation, 2);
            if (candidate.mode == 4) WriteBC7Bits(bytes, offset, candidate.index_mode, 1);
        }
        for (size_t c = 0; c < 4; ++c)
            for (size_t s = 0; s < spec.subsets; ++s)
                for (size_t e = 0; e < 2; ++e)
                    WriteBC7Bits(bytes, offset, candidate.endpoint[s][e].raw[c],
                        c == 3 ? spec.alpha_bits : spec.rgb_bits);
        for (size_t s = 0; s < spec.subsets; ++s)
        {
            if (spec.pbits == Bc7PBits::kShared)
                WriteBC7Bits(bytes, offset, candidate.endpoint[s][0].pbit, 1);
            else if (spec.pbits == Bc7PBits::kEndpoint)
                for (size_t e = 0; e < 2; ++e)
                    WriteBC7Bits(bytes, offset, candidate.endpoint[s][e].pbit, 1);
        }
        for (size_t t = 0; t < 16; ++t)
            WriteBC7Bits(bytes, offset, candidate.primary[t],
                spec.primary_bits - (t == layout.anchorTexel[layout.subsetByTexel[t]] ? 1 : 0));
        if (spec.secondary_bits)
            for (size_t t = 0; t < 16; ++t)
                WriteBC7Bits(bytes, offset, candidate.secondary[t], spec.secondary_bits - (t == 0 ? 1 : 0));
        assert(offset == 128);
    }

    template<bool IsSrgb>
    inline Bc7Candidate EncodeBc7LinearBlock(const Bc7LinearBlock& target,
        const Bc7Guidance* guidance, const Bc7SearchOptions& options, Bc7Telemetry* telemetry = nullptr) noexcept
    {
        Bc7ModeProxy proxies[8]{};
        uint8_t order[8]{0,1,2,3,4,5,6,7};
        for (uint8_t mode = 0; mode < 8; ++mode)
        {
            proxies[mode] = EvaluateBc7ModeProxy<IsSrgb>(target, guidance, options, mode);
            if (telemetry)
            {
                telemetry->proxy_error[mode] = proxies[mode].error;
                telemetry->full_error[mode] = FLT_MAX;
            }
        }
        std::sort(order, order + 8, [&](uint8_t a, uint8_t b)
            {
                if (proxies[a].error != proxies[b].error) return proxies[a].error < proxies[b].error;
                // Parent modes break ties only; they never force a child mode.
                unsigned count_a = 0, count_b = 0;
                if (guidance)
                    for (uint8_t mode : guidance->modes)
                    {
                        count_a += mode == a ? 1u : 0u;
                        count_b += mode == b ? 1u : 0u;
                    }
                return count_a != count_b ? count_a > count_b : a < b;
            });
        const size_t mode_count = options.policy == Bc7SearchPolicy::kFull ? 8
            : std::max<size_t>(1, std::min<size_t>(8, options.mode_candidates));
        Bc7Candidate best{};
        const float alpha_bound = Bc7OpaqueAlphaError(target, options);
        for (size_t i = 0; i < mode_count; ++i)
        {
            const uint8_t mode = order[i];
            // Only this exact alpha bound may skip a mode with default budgets.
            if (mode < 4 && alpha_bound >= best.error) continue;
            const auto& proxy = proxies[mode];
            const size_t structures = options.policy == Bc7SearchPolicy::kFull
                || kBc7Modes[mode].secondary_bits ? proxy.count
                : std::max<size_t>(1, std::min(proxy.count, options.partition_candidates));
            if (telemetry) ++telemetry->evaluated_modes;
            for (size_t j = 0; j < structures; ++j)
            {
                const auto& structure = proxy.structures[j];
                const auto candidate = FitBc7Structure<IsSrgb>(target, guidance, options, mode,
                    structure.partition, structure.rotation, structure.index_mode, false);
                if (telemetry)
                {
                    ++telemetry->evaluated_structures;
                    telemetry->full_error[mode] = std::min(telemetry->full_error[mode], candidate.error);
                }
                if (candidate.error < best.error) best = candidate;
            }
        }
        if (telemetry)
        {
            telemetry->selected_mode = best.mode;
            telemetry->partition = best.partition;
            telemetry->rotation = best.rotation;
            telemetry->index_mode = best.index_mode;
            if (guidance)
                for (size_t i = 0; i < 4; ++i) telemetry->parent_modes[i] = guidance->modes[i];
        }
        return best;
    }

    struct Bc7LinearImage
    {
        size_t width = 0, height = 0;
        Bc7LinearPixel* pixels = nullptr;
    };
    struct Bc7AxisFootprint
    {
        size_t index[4]{};
        float weight[4]{};
        size_t count = 0;
    };
    inline Bc7AxisFootprint ComputeBc7BoxFootprint(
        size_t source_size, size_t destination_size, size_t destination_index) noexcept
    {
        Bc7AxisFootprint footprint{};
        if (source_size == destination_size)
        {
            footprint.index[0] = destination_index;
            footprint.weight[0] = 1;
            footprint.count = 1;
            return footprint;
        }
        if (source_size == destination_size * 2)
        {
            // Exact even-size fast path: two texels, each with half the weight.
            footprint.index[0] = destination_index * 2;
            footprint.index[1] = destination_index * 2 + 1;
            footprint.weight[0] = footprint.weight[1] = 0.5f;
            footprint.count = 2;
            return footprint;
        }
        // An odd mip needs area weights. Repeating its last row would bias the mean.
        const double scale = static_cast<double>(source_size) / static_cast<double>(destination_size);
        const double start = static_cast<double>(destination_index) * scale;
        const double end = static_cast<double>(destination_index + 1) * scale;
        const size_t first = static_cast<size_t>(std::floor(start));
        const size_t last = std::min(source_size, static_cast<size_t>(std::ceil(end)));
        for (size_t source = first; source < last; ++source)
        {
            const double overlap = std::min(end, static_cast<double>(source + 1))
                - std::max(start, static_cast<double>(source));
            if (overlap <= 0) continue;
            assert(footprint.count < 4);
            footprint.index[footprint.count] = source;
            footprint.weight[footprint.count++] = static_cast<float>(overlap / scale);
        }
        return footprint;
    }

    // A small stack cache avoids repeatedly parsing a source block. No source
    // level is expanded into a full RGBA image.
    struct Bc7SourceCache
    {
        struct Entry
        {
            size_t x = SIZE_MAX, y = SIZE_MAX;
            Bc7SymbolicBlock block;
            Bc7PaletteCache palette;
        };
        Entry entries[4]{};
        size_t next = 0;
    };
    template<bool IsSrgb>
    inline const Bc7SourceCache::Entry& GetBc7CachedBlock(
        const Image& source, size_t x, size_t y, Bc7SourceCache& cache) noexcept
    {
        for (const auto& entry : cache.entries)
            if (entry.x == x && entry.y == y) return entry;
        auto& entry = cache.entries[cache.next];
        cache.next = (cache.next + 1) % 4;
        entry.x = x; entry.y = y;
        const bool valid = ParseBc7SymbolicBlock(source.pixels + y * source.rowPitch + x * 16, entry.block);
        assert(valid);
        (void)valid; // The public entry point validates every source block first.
        BuildBc7PaletteCache<IsSrgb>(entry.block, entry.palette);
        return entry;
    }

    template<bool IsSrgb>
    inline void GenerateBc7LinearMip1Row(
        const Image& source, Bc7LinearImage destination, size_t y) noexcept
    {
        Bc7SourceCache cache{};
        if (!(source.width & 1) && !(source.height & 1)
            && destination.width == source.width / 2 && destination.height == source.height / 2)
        {
            // An aligned 2x reduction maps every output texel to one BC7 quadrant.
            for (size_t x = 0; x < destination.width; ++x)
            {
                const auto& entry = GetBc7CachedBlock<IsSrgb>(source, x / 2, y / 2, cache);
                const auto pixel = ComputeBc7QuadrantMean(entry.block, entry.palette,
                    (y & 1) * 2 + (x & 1));
                _mm_storeu_ps(destination.pixels[y * destination.width + x].channel,
                    _mm_loadu_ps(pixel.channel));
            }
        }
        else
        {
            const auto rows = ComputeBc7BoxFootprint(source.height, destination.height, y);
            for (size_t x = 0; x < destination.width; ++x)
            {
                const auto columns = ComputeBc7BoxFootprint(source.width, destination.width, x);
                XMVECTOR sum = XMVectorZero();
                // On even images these four samples are one parent 2x2 quadrant.
                // The cached integer palette is converted before any averaging.
                for (size_t iy = 0; iy < rows.count; ++iy)
                    for (size_t ix = 0; ix < columns.count; ++ix)
                    {
                        const size_t sx = columns.index[ix], sy = rows.index[iy];
                        const auto& entry = GetBc7CachedBlock<IsSrgb>(source, sx / 4, sy / 4, cache);
                        const auto pixel = EvaluateBc7SymbolicTexel(
                            entry.block, entry.palette, (sy % 4) * 4 + sx % 4);
                        sum = XMVectorAdd(sum, XMVectorScale(_mm_loadu_ps(pixel.channel),
                            rows.weight[iy] * columns.weight[ix]));
                    }
                // Invariant A: the persistent target stays in float linear RGBA.
                _mm_storeu_ps(destination.pixels[y * destination.width + x].channel, sum);
            }
        }
    }
    inline void DownsampleBc7LinearRow(
        const Bc7LinearImage& source, Bc7LinearImage destination, size_t y) noexcept
    {
        const auto rows = ComputeBc7BoxFootprint(source.height, destination.height, y);
        for (size_t x = 0; x < destination.width; ++x)
        {
            const auto columns = ComputeBc7BoxFootprint(source.width, destination.width, x);
            XMVECTOR sum = XMVectorZero();
            for (size_t iy = 0; iy < rows.count; ++iy)
                for (size_t ix = 0; ix < columns.count; ++ix)
                {
                    const auto& pixel = source.pixels[rows.index[iy] * source.width + columns.index[ix]];
                    sum = XMVectorAdd(sum, XMVectorScale(_mm_loadu_ps(pixel.channel),
                        rows.weight[iy] * columns.weight[ix]));
                }
            _mm_storeu_ps(destination.pixels[y * destination.width + x].channel, sum);
        }
    }
    inline Bc7LinearBlock LoadBc7LinearBlock(const Bc7LinearImage& image, size_t bx, size_t by) noexcept
    {
        Bc7LinearBlock block{};
        for (size_t y = 0; y < 4; ++y)
            for (size_t x = 0; x < 4; ++x)
            {
                // Tiny levels still need sixteen fitting slots. Repeat the edge.
                const size_t sx = std::min(bx * 4 + x, image.width - 1);
                const size_t sy = std::min(by * 4 + y, image.height - 1);
                for (size_t c = 0; c < 4; ++c)
                    block.channel[c][y * 4 + x] = image.pixels[sy * image.width + sx].channel[c];
            }
        return block;
    }

    // Parent symbols only guide the search (Invariant C). Their colors never
    // replace a target in a later level of the linear pyramid (Invariant H).
    template<bool IsSrgb>
    inline Bc7Guidance BuildBc7ParentGuidance(
        const Image& source, const Bc7LinearImage& child, size_t bx, size_t by) noexcept
    {
        Bc7Guidance guidance{};
        Bc7SourceCache cache{};
        size_t used = 0;
        size_t previous_x[4]{}, previous_y[4]{};
        for (size_t corner = 0; corner < 4; ++corner)
        {
            // Choose up to four distinct parents across the child footprint.
            // Odd-size filtering can touch more parents; guidance is only a prior.
            const size_t cx = std::min(child.width - 1, bx * 4 + (corner & 1 ? 3 : 0));
            const size_t cy = std::min(child.height - 1, by * 4 + (corner & 2 ? 3 : 0));
            const auto xs = ComputeBc7BoxFootprint(source.width, child.width, cx);
            const auto ys = ComputeBc7BoxFootprint(source.height, child.height, cy);
            const size_t px = xs.index[corner & 1 ? xs.count - 1 : 0] / 4;
            const size_t py = ys.index[corner & 2 ? ys.count - 1 : 0] / 4;
            bool duplicate = false;
            for (size_t i = 0; i < used; ++i)
                duplicate = duplicate || (previous_x[i] == px && previous_y[i] == py);
            if (duplicate) continue;
            previous_x[used] = px; previous_y[used] = py;
            const auto& entry = GetBc7CachedBlock<IsSrgb>(source, px, py, cache);
            const auto& d = entry.block.data;
            const auto& spec = kBc7Modes[d.mode];
            const size_t parent = used++;
            guidance.modes[parent] = d.mode;
            guidance.subset_counts[parent] = spec.subsets;
            const auto layout = GetBC7PartitionLayout(spec.subsets, d.partition);
            float parent_mean[4]{};
            size_t valid_texels = 0;
            for (size_t t = 0; t < 16; ++t)
            {
                if (px * 4 + t % 4 >= source.width || py * 4 + t / 4 >= source.height) continue;
                const auto pixel = EvaluateBc7SymbolicTexel(entry.block, entry.palette, t);
                for (size_t c = 0; c < 4; ++c) parent_mean[c] += pixel.channel[c];
                ++valid_texels;
            }
            for (size_t c = 0; c < 4; ++c)
                guidance.mean[c] += parent_mean[c] / static_cast<float>(valid_texels);

            // Preserve every parent subset and selector stream as its own line.
            for (size_t s = 0; s < spec.subsets; ++s)
                for (unsigned stream = 0; stream < (spec.secondary_bits ? 2u : 1u); ++stream)
                {
                    assert(guidance.line_count < 24);
                    auto& line = guidance.lines[guidance.line_count];
                    line.parent = static_cast<uint8_t>(parent);
                    line.subset = static_cast<uint8_t>(s);
                    const unsigned bits = stream ? spec.secondary_bits : spec.primary_bits;
                    line.index_bits = static_cast<uint8_t>(bits);
                    float total = 0, square = 0, count = 0;
                    for (size_t t = 0; t < 16; ++t)
                    {
                        if (layout.subsetByTexel[t] != s ||
                            px * 4 + t % 4 >= source.width || py * 4 + t / 4 >= source.height) continue;
                        const uint8_t index = stream ? d.secondary[t] : d.primary[t];
                        const float lambda = kBc7Weights[bits - 2][index] / 64.0f;
                        total += lambda; square += lambda * lambda; ++count;
                        ++line.selector_histogram[index];
                        const auto pixel = EvaluateBc7SymbolicTexel(entry.block, entry.palette, t);
                        for (size_t c = 0; c < 4; ++c) line.mean[c] += pixel.channel[c];
                    }
                    if (!count) continue;
                    const bool vector = !spec.secondary_bits || stream == d.index_mode;
                    const size_t first_channel = vector ? 0 : 3;
                    const size_t last_channel = vector
                        ? (spec.secondary_bits ? 3 : (spec.alpha_bits ? 4 : 3)) : 4;
                    float norm = 0;
                    for (size_t c = 0; c < 4; ++c) line.mean[c] /= count;
                    for (size_t c = first_channel; c < last_channel; ++c)
                    {
                        const size_t logical = entry.block.logical[c];
                        line.logical_mask |= static_cast<uint8_t>(1u << logical);
                        for (size_t e = 0; e < 2; ++e)
                            line.endpoint[e][logical] = Bc7DecodeCode<IsSrgb>(
                                d.endpoint[s][e].expanded[c], logical);
                        line.direction[logical] = line.endpoint[1][logical] - line.endpoint[0][logical];
                        norm += line.direction[logical] * line.direction[logical];
                    }
                    line.selector_mean = total / count;
                    line.selector_variance = std::max(0.0f, square / count
                        - line.selector_mean * line.selector_mean);
                    line.energy = count * line.selector_variance * norm;
                    if (norm > 1e-20f)
                    {
                        const float inverse = 1.0f / std::sqrt(norm);
                        for (float& component : line.direction) component *= inverse;
                    }
                    ++guidance.line_count;
                }
        }
        guidance.parent_count = static_cast<uint8_t>(used);
        // Coherence measures whether different parents describe the same line.
        // Incoherent lines remain available but receive a weaker proxy score.
        for (size_t i = 0; i < guidance.line_count; ++i)
        {
            auto& line = guidance.lines[i];
            float total = 0, weight = 0;
            for (size_t j = 0; j < guidance.line_count; ++j)
            {
                const auto& other = guidance.lines[j];
                if (line.parent == other.parent || line.logical_mask != other.logical_mask) continue;
                float dot = 0;
                for (size_t c = 0; c < 4; ++c) dot += line.direction[c] * other.direction[c];
                const float importance = std::max(other.energy, 1e-8f);
                total += std::abs(dot) * importance;
                weight += importance;
            }
            if (weight > 0) line.coherence = total / weight;
        }
        if (used)
            for (float& channel : guidance.mean) channel /= static_cast<float>(used);
        return guidance;
    }

    inline bool IsSupportedBC7Image(const Image& image) noexcept
    {
        if (!image.pixels || !image.width || !image.height) return false;
        const size_t width = image.width / 4 + (image.width % 4 != 0);
        const size_t height = image.height / 4 + (image.height % 4 != 0);
        if (width > SIZE_MAX / 16 || image.rowPitch < width * 16) return false;
        if (height > SIZE_MAX / image.rowPitch || image.slicePitch < height * image.rowPitch) return false;
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                if (GetBC7Mode(image.pixels + y * image.rowPitch + x * 16) == BC7_INVALID_MODE) return false;
        return true;
    }

    // Mip 0 is already copied by the unchanged public API. These two buffers
    // hold only linear mip targets; encoded output is never a filtering source.
    template<bool IsSrgb>
    inline HRESULT GenerateCompressedMipMapsBC7(const Image& baseImage, ScratchImage& mipChain) noexcept
    {
        const size_t levels = mipChain.GetMetadata().mipLevels;
        if (levels <= 1) return S_OK;
        const size_t width = std::max<size_t>(1, baseImage.width / 2);
        const size_t height = std::max<size_t>(1, baseImage.height / 2);
        const size_t next_width = std::max<size_t>(1, width / 2);
        const size_t next_height = std::max<size_t>(1, height / 2);
        if (width > SIZE_MAX / height || width * height > SIZE_MAX / sizeof(Bc7LinearPixel))
            return E_OUTOFMEMORY;
        std::unique_ptr<Bc7LinearPixel[]> first(new (std::nothrow) Bc7LinearPixel[width * height]);
        std::unique_ptr<Bc7LinearPixel[]> second;
        if (levels > 2)
            second.reset(new (std::nothrow) Bc7LinearPixel[next_width * next_height]);
        if (!first || (levels > 2 && !second)) return E_OUTOFMEMORY;
        if (IsSrgb)
        {
            (void)GetSrgb8ToLinearTable();
        }
        Bc7LinearImage current{width, height, first.get()};
        Bc7LinearPixel* spare = second.get();
        const Bc7SearchOptions options{};
        for (size_t level = 1; level < levels; ++level)
        {
            if (level == 1)
            {
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
                for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(height); ++row)
                {
                    GenerateBc7LinearMip1Row<IsSrgb>(baseImage, current, static_cast<size_t>(row));
                }
            }
            else
            {
                Bc7LinearImage next{
                    std::max<size_t>(1, current.width / 2),
                    std::max<size_t>(1, current.height / 2), spare};
            #ifdef _OPENMP
            #pragma omp parallel for
                #endif
                for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(next.height); ++row)
                {
                    DownsampleBc7LinearRow(current, next, static_cast<size_t>(row));
                }
                // The old larger allocation can hold the following smaller level.
                spare = current.pixels;
                current = next;
            }
            const Image* destination = mipChain.GetImage(level, 0, 0);
            const Image* parent = level == 1 ? &baseImage : mipChain.GetImage(level - 1, 0, 0);
            if (!destination || !parent || !destination->pixels ||
                destination->width != current.width || destination->height != current.height) return E_FAIL;
            const size_t block_width = current.width / 4 + (current.width % 4 != 0);
            const size_t block_height = current.height / 4 + (current.height % 4 != 0);
        #ifdef _OPENMP
        #pragma omp parallel for
            #endif
            for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(block_height); ++row)
            {
                for (size_t x = 0; x < block_width; ++x)
                {
                    const size_t y = static_cast<size_t>(row);
                    const auto target = LoadBc7LinearBlock(current, x, y);
                    Bc7Guidance guidance{};
                    const bool use_guidance = options.use_parent_guidance
                        && (level == 1 || options.use_previous_encoded_guidance);
                    if (use_guidance) guidance = BuildBc7ParentGuidance<IsSrgb>(*parent, current, x, y);
                    const auto candidate = EncodeBc7LinearBlock<IsSrgb>(target,
                        use_guidance ? &guidance : nullptr, options);
                    EmitBc7Candidate(candidate, destination->pixels + y * destination->rowPitch + x * 16);
                }
            }
        }
        return S_OK;
    }

} // namespace for bc7

// Generate a compressed mip chain without materializing uncompressed images.
HRESULT DirectX::GenerateCompressedMipMaps(const Image& baseImage, size_t levels, ScratchImage& mipChain) noexcept
{
    // Validate the compressed source before allocating output memory.
    if (!baseImage.pixels)
    {
        return E_POINTER;
    }

    // Format validation and invariant checks for supported compressed formats.
    switch (baseImage.format)
    {
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
        // The current algorithm supports only opaque four-color BC1 blocks.
        if (!IsOpaqueBC1Image(baseImage))
        {
            return HRESULT_E_NOT_SUPPORTED;
        }
        break;

    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        // Verify that all blocks use supported modes (Mode 1, Mode 6, Mode 7).
        if (!IsSupportedBC7Image(baseImage))
        {
            return HRESULT_E_NOT_SUPPORTED;
        }
        break;

    case DXGI_FORMAT_BC4_UNORM:
        // Both BC4 interpolation modes are handled, so every block is supported.
        break;

    // TODO: Add support for BC2, BC3, BC5, BC6H here in future extensions.
    default:
        return HRESULT_E_NOT_SUPPORTED;
    }

    // Allocate the compressed output mip chain.
    HRESULT hr = mipChain.Initialize2D(baseImage.format, baseImage.width, baseImage.height, 1, levels);
    if (FAILED(hr))
    {
        return hr;
    }

    // Copy the original compressed level zero into the output chain.
    const Image* outputBase = mipChain.GetImage(0, 0, 0);
    if (!outputBase || !outputBase->pixels)
    {
        return E_POINTER;
    }

    // The source and destination layouts must match.
    if (baseImage.rowPitch != outputBase->rowPitch || baseImage.slicePitch != outputBase->slicePitch)
    {
        return E_FAIL;
    }

    // Preserve mip 0 bit-for-bit; only lower levels are generated here.
    memcpy_s(outputBase->pixels, outputBase->slicePitch, baseImage.pixels, baseImage.slicePitch);

    // Dispatch to the corresponding compressed mipmap generator.
    switch (baseImage.format)
    {
    case DXGI_FORMAT_BC1_UNORM:
        return GenerateCompressedMipMapsBC1<false>(baseImage, mipChain);

    case DXGI_FORMAT_BC1_UNORM_SRGB:
        return GenerateCompressedMipMapsBC1<true>(baseImage, mipChain);

    case DXGI_FORMAT_BC7_UNORM:
        return GenerateCompressedMipMapsBC7<false>(baseImage, mipChain);

    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return GenerateCompressedMipMapsBC7<true>(baseImage, mipChain);

    case DXGI_FORMAT_BC4_UNORM:
        return GenerateCompressedMipMapsBC4(baseImage, mipChain);

    // TODO: Add case dispatches for BC2, BC3, BC5 here.
    default:
        return HRESULT_E_NOT_SUPPORTED;
    }
}

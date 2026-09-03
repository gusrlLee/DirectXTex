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

    // word[wordIndex] is split into named members, and each SIMD lane stores one BC7 block.
    struct BC7BlockBatch
    {
        XMVECTOR word0;
        XMVECTOR word1;
        XMVECTOR word2;
        XMVECTOR word3;
    };

    // Spatial fields needed to map every BC7 mode onto a common subset layout.
    struct BC7ModeSpatialInfo
    {
        uint8_t subsetCount;
        uint8_t partitionBitOffset;
        uint8_t partitionBitCount;
    };

    constexpr BC7ModeSpatialInfo g_bc7ModeSpatialInfo[8] =
    {
        { 3, 1, 4 }, // Mode 0
        { 2, 2, 6 }, // Mode 1
        { 3, 3, 6 }, // Mode 2
        { 2, 4, 6 }, // Mode 3
        { 1, 0, 0 }, // Mode 4
        { 1, 0, 0 }, // Mode 5
        { 1, 0, 0 }, // Mode 6
        { 2, 8, 6 }  // Mode 7
    };

    // mode[modeIndex] contains 0xFFFFFFFF for matching lanes and zero otherwise.
    struct BC7ModeMaskBatch
    {
        XMVECTOR mode[8];
    };

    // value[endpointIndex][channelIndex], with one BC7 block in each SIMD lane.
    // Channels use the order R, G, B, A and contain unquantized 8-bit integers.
    struct BC7EndpointPairBatch
    {
        XMVECTOR value[2][4];
    };

    // value[endpointIndex][channelIndex], with one BC7 block in each SIMD lane.
    // Channels use the order R, G, B, A and contain floating-point UNORM [0, 1] values.
    struct BC7EndpointPairFloatBatch
    {
        XMVECTOR value[2][4];
    };

    // Endpoints for multi-subset modes (up to 3 subsets: Mode 0, Mode 1, Mode 2, Mode 3, Mode 7).
    // value[subsetIndex][endpointIndex][channelIndex] (RGBA 8-bit integers).
    struct BC7MultiSubsetEndpointBatch
    {
        XMVECTOR value[3][2][4];
    };
    using BC7TwoSubsetEndpointBatch = BC7MultiSubsetEndpointBatch;

    // Unpacked BC7 indices across four SIMD lanes.
    // indices[texelIndex] contains one mode-specific palette index in each lane.
    struct BC7IndexBatch
    {
        XMVECTOR indices[16];
    };

    // low stores texels 0-7 and high stores texels 8-15 as four-bit nibbles.
    struct BC7Mode6PackedIndexBatch
    {
        XMVECTOR low;
        XMVECTOR high;
    };

    // value[quadrantIndex] stores one palette-index count per BC7 block lane.
    struct BC7QuadrantCountBatch
    {
        XMVECTOR value[4];
    };



    // value[quadrantIndex][channelIndex] stores one mean per BC7 block lane.
    struct BC7QuadrantMeanBatch
    {
        XMVECTOR value[4][4];
    };

    // value[channelIndex] stores one block mean per SIMD lane (RGBA).
    struct BC7BlockMeanBatch
    {
        XMVECTOR value[4];
    };

    // Symmetric 4x4 covariance matrices for four SIMD lanes (RGBA).
    struct BC7CovarianceMatrixBatch
    {
        XMVECTOR rr;
        XMVECTOR gg;
        XMVECTOR bb;
        XMVECTOR aa;

        XMVECTOR rg;
        XMVECTOR rb;
        XMVECTOR ra;
        XMVECTOR gb;
        XMVECTOR ga;
        XMVECTOR ba;
    };

    // Mean and within-block covariance for four parent BC7 blocks.
    struct BC7ParentStatisticsBatch
    {
        BC7BlockMeanBatch mean;
        BC7CovarianceMatrixBatch withinCovariance;
    };

    // Linear means of the four source blocks used by one destination BC7 block.
    struct BC7SourceBlockMeansBatch
    {
        BC7BlockMeanBatch p00;
        BC7BlockMeanBatch p10;
        BC7BlockMeanBatch p01;
        BC7BlockMeanBatch p11;
    };

    // Canonical intermediate representation (IR) of the downsampled child block.
    // Contains the 16 child texels, block statistics, 4D principal axis, and opacity mask.
    struct BC7ChildCanvas
    {
        XMVECTOR texels[16][4];              // 16 child texels (RGBA [0, 1] float in SIMD lanes)
        BC7BlockMeanBatch mean;               // 4D block mean
        BC7CovarianceMatrixBatch covariance;  // 4D ANOVA covariance matrix
        BC7BlockMeanBatch axis;               // 4D principal axis from PCA power iteration
        XMVECTOR isOpaque;                    // 0xFFFFFFFF if all 16 texels have alpha >= 254/255
    };

    // Scalar RGBA block mean used by the higher mip mean pyramid.
    struct BC7BlockMean
    {
        float r;
        float g;
        float b;
        float a;
    };

    static_assert(sizeof(XMUINT4) == 16, "XMUINT4 must have the same size as one BC7 block");

    inline BC7BlockBatch LoadBC7BlockBatch(const XMUINT4* blocks) noexcept
    {
        assert(blocks != nullptr);

        // Load four compressed blocks without numeric conversion.
        const XMVECTOR block0 = XMLoadInt4(&blocks[0].x);
        const XMVECTOR block1 = XMLoadInt4(&blocks[1].x);
        const XMVECTOR block2 = XMLoadInt4(&blocks[2].x);
        const XMVECTOR block3 = XMLoadInt4(&blocks[3].x);

        // Transpose equal 32-bit words into the same SIMD vector.
        const XMMATRIX blockRows(block0, block1, block2, block3);
        const XMMATRIX transposed = XMMatrixTranspose(blockRows);

        BC7BlockBatch result;
        result.word0 = transposed.r[0];
        result.word1 = transposed.r[1];
        result.word2 = transposed.r[2];
        result.word3 = transposed.r[3];

        return result;
    }

    // Select one 32-bit word from every SIMD lane.
    template<size_t WordIndex>
    inline XMVECTOR GetBC7Word(const BC7BlockBatch& blocks) noexcept
    {
        static_assert(WordIndex < 4, "Invalid BC7 word index");

        if constexpr (WordIndex == 0) return blocks.word0;
        if constexpr (WordIndex == 1) return blocks.word1;
        if constexpr (WordIndex == 2) return blocks.word2;
        return blocks.word3;
    }

    // Extract the same BC7 bit field from four blocks.
    template<size_t BitOffset, size_t BitCount>
    inline XMVECTOR ExtractBC7Bits(const BC7BlockBatch& blocks) noexcept
    {
        static_assert(BitCount > 0 && BitCount <= 8, "Invalid BC7 field size");
        static_assert(BitOffset + BitCount <= 128, "BC7 field exceeds the block size");

        constexpr size_t wordIndex = BitOffset / 32;
        constexpr size_t bitIndex = BitOffset % 32;
        constexpr uint32_t valueMask = (1u << BitCount) - 1u;

        XMVECTOR value = ShiftRight32<bitIndex>(GetBC7Word<wordIndex>(blocks));

        if constexpr (bitIndex + BitCount > 32)
        {
            value = XMVectorOrInt(value, ShiftLeft32<32 - bitIndex>(GetBC7Word<wordIndex + 1>(blocks)));
        }

        return XMVectorAndInt(value, XMVectorReplicateInt(valueMask));
    }

    inline XMVECTOR AppendBC7PBit(FXMVECTOR value, FXMVECTOR pBit) noexcept
    {
        return XMVectorOrInt(ShiftLeft32<1>(value), pBit);
    }

    // Extract and expand mode 6 endpoints to RGBA8.
    inline BC7EndpointPairBatch ExtractBC7Mode6Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        const XMVECTOR pBit0 = ExtractBC7Bits<63, 1>(blocks);
        const XMVECTOR pBit1 = ExtractBC7Bits<64, 1>(blocks);

        BC7EndpointPairBatch result{};
        result.value[0][0] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<7, 7>(blocks), pBit0), activeMask);
        result.value[1][0] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<14, 7>(blocks), pBit1), activeMask);
        result.value[0][1] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<21, 7>(blocks), pBit0), activeMask);
        result.value[1][1] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<28, 7>(blocks), pBit1), activeMask);
        result.value[0][2] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<35, 7>(blocks), pBit0), activeMask);
        result.value[1][2] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<42, 7>(blocks), pBit1), activeMask);
        result.value[0][3] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<49, 7>(blocks), pBit0), activeMask);
        result.value[1][3] = XMVectorAndInt(AppendBC7PBit(ExtractBC7Bits<56, 7>(blocks), pBit1), activeMask);
        return result;
    }

    // Expand 6-bit channel with 1 P-bit to unquantized 8-bit integer (Mode 1).
    // Exact Direct3D bit replication: (val7 << 1) | (val7 >> 6)
    inline XMVECTOR UnquantizeBC7_6Bit(FXMVECTOR value6, FXMVECTOR pBit) noexcept
    {
        const XMVECTOR val7 = XMVectorOrInt(ShiftLeft32<1>(value6), pBit);
        return XMVectorOrInt(ShiftLeft32<1>(val7), ShiftRight32<6>(val7));
    }

    inline uint8_t UnquantizeBC7_6BitScalar(uint8_t value6, uint8_t pBit) noexcept
    {
        const uint8_t val7 = static_cast<uint8_t>((value6 << 1) | (pBit & 1u));
        return static_cast<uint8_t>((val7 << 1) | (val7 >> 6));
    }

    // Expand 5-bit channel with 1 P-bit to unquantized 8-bit integer (Mode 7).
    // Exact Direct3D bit replication: (val6 << 2) | (val6 >> 4)
    inline XMVECTOR UnquantizeBC7_5Bit(FXMVECTOR value5, FXMVECTOR pBit) noexcept
    {
        const XMVECTOR val6 = XMVectorOrInt(ShiftLeft32<1>(value5), pBit);
        return XMVectorOrInt(ShiftLeft32<2>(val6), ShiftRight32<4>(val6));
    }

    inline uint8_t UnquantizeBC7_5BitScalar(uint8_t value5, uint8_t pBit) noexcept
    {
        const uint8_t val6 = static_cast<uint8_t>((value5 << 1) | (pBit & 1u));
        return static_cast<uint8_t>((val6 << 2) | (val6 >> 4));
    }

    // Expand 4-bit channel with 1 P-bit to unquantized 8-bit integer (Mode 0).
    // Exact Direct3D bit replication: (val5 << 3) | (val5 >> 2)
    inline XMVECTOR UnquantizeBC7_4Bit(FXMVECTOR value4, FXMVECTOR pBit) noexcept
    {
        const XMVECTOR val5 = XMVectorOrInt(ShiftLeft32<1>(value4), pBit);
        return XMVectorOrInt(ShiftLeft32<3>(val5), ShiftRight32<2>(val5));
    }

    inline uint8_t UnquantizeBC7_4BitScalar(uint8_t value4, uint8_t pBit) noexcept
    {
        const uint8_t val5 = static_cast<uint8_t>((value4 << 1) | (pBit & 1u));
        return static_cast<uint8_t>((val5 << 3) | (val5 >> 2));
    }

    // Expand 5-bit channel without P-bit to unquantized 8-bit integer (Mode 2).
    // Exact Direct3D bit replication: (val5 << 3) | (val5 >> 2)
    inline XMVECTOR UnquantizeBC7_5Bit_NoPBit(FXMVECTOR value5) noexcept
    {
        return XMVectorOrInt(ShiftLeft32<3>(value5), ShiftRight32<2>(value5));
    }

    inline uint8_t UnquantizeBC7_5Bit_NoPBitScalar(uint8_t value5) noexcept
    {
        return static_cast<uint8_t>((value5 << 3) | (value5 >> 2));
    }

    // Expand 7-bit channel with 1 P-bit to unquantized 8-bit integer (Mode 3).
    // Exact Direct3D: (val7 << 1) | pBit
    inline XMVECTOR UnquantizeBC7_7Bit(FXMVECTOR value7, FXMVECTOR pBit) noexcept
    {
        return XMVectorOrInt(ShiftLeft32<1>(value7), pBit);
    }

    inline uint8_t UnquantizeBC7_7BitScalar(uint8_t value7, uint8_t pBit) noexcept
    {
        return static_cast<uint8_t>((value7 << 1) | (pBit & 1u));
    }

    // Expand 6-bit channel without P-bit to unquantized 8-bit integer (Mode 4 alpha).
    // Exact Direct3D bit replication: (val6 << 2) | (val6 >> 4)
    inline XMVECTOR UnquantizeBC7_6Bit_NoPBit(FXMVECTOR value6) noexcept
    {
        return XMVectorOrInt(ShiftLeft32<2>(value6), ShiftRight32<4>(value6));
    }

    inline uint8_t UnquantizeBC7_6Bit_NoPBitScalar(uint8_t value6) noexcept
    {
        return static_cast<uint8_t>((value6 << 2) | (value6 >> 4));
    }

    // Expand 7-bit channel without P-bit to unquantized 8-bit integer (Mode 5 RGB).
    // Exact Direct3D bit replication: (val7 << 1) | (val7 >> 6)
    inline XMVECTOR UnquantizeBC7_7Bit_NoPBit(FXMVECTOR value7) noexcept
    {
        return XMVectorOrInt(ShiftLeft32<1>(value7), ShiftRight32<6>(value7));
    }

    inline uint8_t UnquantizeBC7_7Bit_NoPBitScalar(uint8_t value7) noexcept
    {
        return static_cast<uint8_t>((value7 << 1) | (value7 >> 6));
    }

    // Extract the partition index for one BC7 mode from four block lanes.
    inline XMVECTOR ExtractBC7Partition(
        const BC7BlockBatch& blocks,
        uint8_t mode,
        FXMVECTOR activeMask) noexcept
    {
        assert(mode < 8);

        const BC7ModeSpatialInfo& info = g_bc7ModeSpatialInfo[mode];
        if (info.partitionBitCount == 0)
        {
            return XMVectorZero();
        }

        const __m128i shift = _mm_cvtsi32_si128(info.partitionBitOffset);
        const XMVECTOR shifted = _mm_castsi128_ps(_mm_srl_epi32(_mm_castps_si128(blocks.word0), shift));
        const uint32_t valueMask = (1u << info.partitionBitCount) - 1u;
        return XMVectorAndInt(XMVectorAndInt(shifted, XMVectorReplicateInt(valueMask)), activeMask);
    }

    // Extract and expand Mode 1 endpoints to RGBA8 (RGB from 6-bit + shared P-bit, Alpha = 255).
    inline BC7TwoSubsetEndpointBatch ExtractBC7Mode1Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        // Mode 1: 2 P-bits, shared per subset (pBit0 for subset 0, pBit1 for subset 1)
        const XMVECTOR pBit0 = ExtractBC7Bits<80, 1>(blocks);
        const XMVECTOR pBit1 = ExtractBC7Bits<81, 1>(blocks);
        const XMVECTOR alpha255 = XMVectorReplicateInt(255u);

        BC7TwoSubsetEndpointBatch result{};

        // Subset 0, Endpoint 0 (R: bit 8..13, G: bit 32..37, B: bit 56..61)
        result.value[0][0][0] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<8, 6>(blocks), pBit0), activeMask);
        result.value[0][0][1] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<32, 6>(blocks), pBit0), activeMask);
        result.value[0][0][2] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<56, 6>(blocks), pBit0), activeMask);
        result.value[0][0][3] = XMVectorAndInt(alpha255, activeMask);

        // Subset 0, Endpoint 1 (R: bit 14..19, G: bit 38..43, B: bit 62..67)
        result.value[0][1][0] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<14, 6>(blocks), pBit0), activeMask);
        result.value[0][1][1] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<38, 6>(blocks), pBit0), activeMask);
        result.value[0][1][2] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<62, 6>(blocks), pBit0), activeMask);
        result.value[0][1][3] = XMVectorAndInt(alpha255, activeMask);

        // Subset 1, Endpoint 0 (R: bit 20..25, G: bit 44..49, B: bit 68..73)
        result.value[1][0][0] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<20, 6>(blocks), pBit1), activeMask);
        result.value[1][0][1] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<44, 6>(blocks), pBit1), activeMask);
        result.value[1][0][2] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<68, 6>(blocks), pBit1), activeMask);
        result.value[1][0][3] = XMVectorAndInt(alpha255, activeMask);

        // Subset 1, Endpoint 1 (R: bit 26..31, G: bit 50..55, B: bit 74..79)
        result.value[1][1][0] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<26, 6>(blocks), pBit1), activeMask);
        result.value[1][1][1] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<50, 6>(blocks), pBit1), activeMask);
        result.value[1][1][2] = XMVectorAndInt(UnquantizeBC7_6Bit(ExtractBC7Bits<74, 6>(blocks), pBit1), activeMask);
        result.value[1][1][3] = XMVectorAndInt(alpha255, activeMask);

        return result;
    }

    // Extract and expand Mode 7 endpoints to RGBA8.
    inline BC7TwoSubsetEndpointBatch ExtractBC7Mode7Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        // Mode 7 stores one P-bit for each of its four endpoints.
        const XMVECTOR pBit00 = ExtractBC7Bits<94, 1>(blocks);
        const XMVECTOR pBit01 = ExtractBC7Bits<95, 1>(blocks);
        const XMVECTOR pBit10 = ExtractBC7Bits<96, 1>(blocks);
        const XMVECTOR pBit11 = ExtractBC7Bits<97, 1>(blocks);

        BC7TwoSubsetEndpointBatch result{};

        result.value[0][0][0] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<14, 5>(blocks), pBit00), activeMask);
        result.value[0][1][0] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<19, 5>(blocks), pBit01), activeMask);
        result.value[1][0][0] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<24, 5>(blocks), pBit10), activeMask);
        result.value[1][1][0] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<29, 5>(blocks), pBit11), activeMask);

        result.value[0][0][1] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<34, 5>(blocks), pBit00), activeMask);
        result.value[0][1][1] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<39, 5>(blocks), pBit01), activeMask);
        result.value[1][0][1] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<44, 5>(blocks), pBit10), activeMask);
        result.value[1][1][1] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<49, 5>(blocks), pBit11), activeMask);

        result.value[0][0][2] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<54, 5>(blocks), pBit00), activeMask);
        result.value[0][1][2] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<59, 5>(blocks), pBit01), activeMask);
        result.value[1][0][2] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<64, 5>(blocks), pBit10), activeMask);
        result.value[1][1][2] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<69, 5>(blocks), pBit11), activeMask);

        result.value[0][0][3] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<74, 5>(blocks), pBit00), activeMask);
        result.value[0][1][3] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<79, 5>(blocks), pBit01), activeMask);
        result.value[1][0][3] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<84, 5>(blocks), pBit10), activeMask);
        result.value[1][1][3] = XMVectorAndInt(UnquantizeBC7_5Bit(ExtractBC7Bits<89, 5>(blocks), pBit11), activeMask);

        return result;
    }

    // Extract and expand Mode 0 endpoints to RGBA8 (3 subsets, RGB 444 + 6 unique P-bits, Alpha = 255).
    inline BC7MultiSubsetEndpointBatch ExtractBC7Mode0Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        const XMVECTOR p0 = ExtractBC7Bits<77, 1>(blocks);
        const XMVECTOR p1 = ExtractBC7Bits<78, 1>(blocks);
        const XMVECTOR p2 = ExtractBC7Bits<79, 1>(blocks);
        const XMVECTOR p3 = ExtractBC7Bits<80, 1>(blocks);
        const XMVECTOR p4 = ExtractBC7Bits<81, 1>(blocks);
        const XMVECTOR p5 = ExtractBC7Bits<82, 1>(blocks);
        const XMVECTOR alpha255 = XMVectorReplicateInt(255u);

        BC7MultiSubsetEndpointBatch result{};

        // Red (bits 5..28)
        result.value[0][0][0] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<5, 4>(blocks), p0), activeMask);
        result.value[0][1][0] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<9, 4>(blocks), p1), activeMask);
        result.value[1][0][0] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<13, 4>(blocks), p2), activeMask);
        result.value[1][1][0] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<17, 4>(blocks), p3), activeMask);
        result.value[2][0][0] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<21, 4>(blocks), p4), activeMask);
        result.value[2][1][0] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<25, 4>(blocks), p5), activeMask);

        // Green (bits 29..52)
        result.value[0][0][1] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<29, 4>(blocks), p0), activeMask);
        result.value[0][1][1] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<33, 4>(blocks), p1), activeMask);
        result.value[1][0][1] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<37, 4>(blocks), p2), activeMask);
        result.value[1][1][1] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<41, 4>(blocks), p3), activeMask);
        result.value[2][0][1] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<45, 4>(blocks), p4), activeMask);
        result.value[2][1][1] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<49, 4>(blocks), p5), activeMask);

        // Blue (bits 53..76)
        result.value[0][0][2] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<53, 4>(blocks), p0), activeMask);
        result.value[0][1][2] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<57, 4>(blocks), p1), activeMask);
        result.value[1][0][2] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<61, 4>(blocks), p2), activeMask);
        result.value[1][1][2] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<65, 4>(blocks), p3), activeMask);
        result.value[2][0][2] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<69, 4>(blocks), p4), activeMask);
        result.value[2][1][2] = XMVectorAndInt(UnquantizeBC7_4Bit(ExtractBC7Bits<73, 4>(blocks), p5), activeMask);

        // Alpha = 255
        for (size_t s = 0; s < 3; ++s)
        {
            result.value[s][0][3] = XMVectorAndInt(alpha255, activeMask);
            result.value[s][1][3] = XMVectorAndInt(alpha255, activeMask);
        }

        return result;
    }

    // Extract and expand Mode 2 endpoints to RGBA8 (3 subsets, RGB 555 without P-bits, Alpha = 255).
    inline BC7MultiSubsetEndpointBatch ExtractBC7Mode2Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        const XMVECTOR alpha255 = XMVectorReplicateInt(255u);
        BC7MultiSubsetEndpointBatch result{};

        // Red (bits 9..38)
        result.value[0][0][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<9, 5>(blocks)), activeMask);
        result.value[0][1][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<14, 5>(blocks)), activeMask);
        result.value[1][0][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<19, 5>(blocks)), activeMask);
        result.value[1][1][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<24, 5>(blocks)), activeMask);
        result.value[2][0][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<29, 5>(blocks)), activeMask);
        result.value[2][1][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<34, 5>(blocks)), activeMask);

        // Green (bits 39..68)
        result.value[0][0][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<39, 5>(blocks)), activeMask);
        result.value[0][1][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<44, 5>(blocks)), activeMask);
        result.value[1][0][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<49, 5>(blocks)), activeMask);
        result.value[1][1][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<54, 5>(blocks)), activeMask);
        result.value[2][0][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<59, 5>(blocks)), activeMask);
        result.value[2][1][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<64, 5>(blocks)), activeMask);

        // Blue (bits 69..98)
        result.value[0][0][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<69, 5>(blocks)), activeMask);
        result.value[0][1][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<74, 5>(blocks)), activeMask);
        result.value[1][0][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<79, 5>(blocks)), activeMask);
        result.value[1][1][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<84, 5>(blocks)), activeMask);
        result.value[2][0][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<89, 5>(blocks)), activeMask);
        result.value[2][1][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<94, 5>(blocks)), activeMask);

        // Alpha = 255
        for (size_t s = 0; s < 3; ++s)
        {
            result.value[s][0][3] = XMVectorAndInt(alpha255, activeMask);
            result.value[s][1][3] = XMVectorAndInt(alpha255, activeMask);
        }

        return result;
    }

    // Extract and expand Mode 3 endpoints to RGBA8 (2 subsets, RGB 777 + 4 unique P-bits, Alpha = 255).
    inline BC7MultiSubsetEndpointBatch ExtractBC7Mode3Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        const XMVECTOR p0 = ExtractBC7Bits<94, 1>(blocks);
        const XMVECTOR p1 = ExtractBC7Bits<95, 1>(blocks);
        const XMVECTOR p2 = ExtractBC7Bits<96, 1>(blocks);
        const XMVECTOR p3 = ExtractBC7Bits<97, 1>(blocks);
        const XMVECTOR alpha255 = XMVectorReplicateInt(255u);

        BC7MultiSubsetEndpointBatch result{};

        // Red (bits 10..37)
        result.value[0][0][0] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<10, 7>(blocks), p0), activeMask);
        result.value[0][1][0] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<17, 7>(blocks), p1), activeMask);
        result.value[1][0][0] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<24, 7>(blocks), p2), activeMask);
        result.value[1][1][0] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<31, 7>(blocks), p3), activeMask);

        // Green (bits 38..65)
        result.value[0][0][1] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<38, 7>(blocks), p0), activeMask);
        result.value[0][1][1] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<45, 7>(blocks), p1), activeMask);
        result.value[1][0][1] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<52, 7>(blocks), p2), activeMask);
        result.value[1][1][1] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<59, 7>(blocks), p3), activeMask);

        // Blue (bits 66..93)
        result.value[0][0][2] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<66, 7>(blocks), p0), activeMask);
        result.value[0][1][2] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<73, 7>(blocks), p1), activeMask);
        result.value[1][0][2] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<80, 7>(blocks), p2), activeMask);
        result.value[1][1][2] = XMVectorAndInt(UnquantizeBC7_7Bit(ExtractBC7Bits<87, 7>(blocks), p3), activeMask);

        // Alpha = 255
        result.value[0][0][3] = XMVectorAndInt(alpha255, activeMask);
        result.value[0][1][3] = XMVectorAndInt(alpha255, activeMask);
        result.value[1][0][3] = XMVectorAndInt(alpha255, activeMask);
        result.value[1][1][3] = XMVectorAndInt(alpha255, activeMask);

        return result;
    }

    // Extract and expand Mode 4 endpoints to RGBA8 (1 subset, RGB 555, A 6, no P-bits).
    inline BC7EndpointPairBatch ExtractBC7Mode4Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        BC7EndpointPairBatch result{};

        // Red (bits 8..17)
        result.value[0][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<8, 5>(blocks)), activeMask);
        result.value[1][0] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<13, 5>(blocks)), activeMask);

        // Green (bits 18..27)
        result.value[0][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<18, 5>(blocks)), activeMask);
        result.value[1][1] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<23, 5>(blocks)), activeMask);

        // Blue (bits 28..37)
        result.value[0][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<28, 5>(blocks)), activeMask);
        result.value[1][2] = XMVectorAndInt(UnquantizeBC7_5Bit_NoPBit(ExtractBC7Bits<33, 5>(blocks)), activeMask);

        // Alpha (bits 38..49)
        result.value[0][3] = XMVectorAndInt(UnquantizeBC7_6Bit_NoPBit(ExtractBC7Bits<38, 6>(blocks)), activeMask);
        result.value[1][3] = XMVectorAndInt(UnquantizeBC7_6Bit_NoPBit(ExtractBC7Bits<44, 6>(blocks)), activeMask);

        return result;
    }

    // Extract and expand Mode 5 endpoints to RGBA8 (1 subset, RGB 777, A 8, no P-bits).
    inline BC7EndpointPairBatch ExtractBC7Mode5Endpoints(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        BC7EndpointPairBatch result{};

        // Red (bits 8..21)
        result.value[0][0] = XMVectorAndInt(UnquantizeBC7_7Bit_NoPBit(ExtractBC7Bits<8, 7>(blocks)), activeMask);
        result.value[1][0] = XMVectorAndInt(UnquantizeBC7_7Bit_NoPBit(ExtractBC7Bits<15, 7>(blocks)), activeMask);

        // Green (bits 22..35)
        result.value[0][1] = XMVectorAndInt(UnquantizeBC7_7Bit_NoPBit(ExtractBC7Bits<22, 7>(blocks)), activeMask);
        result.value[1][1] = XMVectorAndInt(UnquantizeBC7_7Bit_NoPBit(ExtractBC7Bits<29, 7>(blocks)), activeMask);

        // Blue (bits 36..49)
        result.value[0][2] = XMVectorAndInt(UnquantizeBC7_7Bit_NoPBit(ExtractBC7Bits<36, 7>(blocks)), activeMask);
        result.value[1][2] = XMVectorAndInt(UnquantizeBC7_7Bit_NoPBit(ExtractBC7Bits<43, 7>(blocks)), activeMask);

        // Alpha (bits 50..65)
        result.value[0][3] = XMVectorAndInt(ExtractBC7Bits<50, 8>(blocks), activeMask);
        result.value[1][3] = XMVectorAndInt(ExtractBC7Bits<58, 8>(blocks), activeMask);

        return result;
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

    // Resolve the spatial subset layout for any BC7 mode.
    inline BC7PartitionLayout GetBC7ModePartitionLayout(uint8_t mode, uint8_t partition) noexcept
    {
        assert(mode < 8);

        const BC7ModeSpatialInfo& info = g_bc7ModeSpatialInfo[mode];
        const size_t partitionCount = size_t(1) << info.partitionBitCount;
        assert(partition < partitionCount);
        (void)partitionCount;

        return GetBC7PartitionLayout(info.subsetCount, partition);
    }


    // Extract one partitioned index stream while restoring each subset's omitted anchor MSB to zero.
    inline BC7IndexBatch ExtractBC7PartitionedIndices(
        const BC7BlockBatch& blocks,
        FXMVECTOR partition,
        uint8_t mode,
        size_t firstIndexBit,
        size_t indexBitCount,
        FXMVECTOR activeMask) noexcept
    {
        assert(mode < 8);
        assert(indexBitCount >= 2 && indexBitCount <= 4);

        // Unswizzle the 4 block words so each lane can be read as a 16-byte block
        alignas(16) uint32_t w0[4], w1[4], w2[4], w3[4], shapes[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(w0), _mm_castps_si128(blocks.word0));
        _mm_store_si128(reinterpret_cast<__m128i*>(w1), _mm_castps_si128(blocks.word1));
        _mm_store_si128(reinterpret_cast<__m128i*>(w2), _mm_castps_si128(blocks.word2));
        _mm_store_si128(reinterpret_cast<__m128i*>(w3), _mm_castps_si128(blocks.word3));
        _mm_store_si128(reinterpret_cast<__m128i*>(shapes), _mm_castps_si128(partition));

        uint32_t unpackedIndices[16][4]{};

        for (size_t lane = 0; lane < 4; ++lane)
        {
            const uint32_t raw[4] = { w0[lane], w1[lane], w2[lane], w3[lane] };
            const auto* blockBytes = reinterpret_cast<const uint8_t*>(raw);
            const uint8_t shape = static_cast<uint8_t>(shapes[lane] & 0x3Fu);
            const BC7PartitionLayout layout = GetBC7ModePartitionLayout(mode, shape);

            size_t bitOffset = firstIndexBit;
            for (size_t t = 0; t < 16; ++t)
            {
                const uint8_t subset = layout.subsetByTexel[t];
                const size_t numBits = (t == layout.anchorTexel[subset]) ? indexBitCount - 1 : indexBitCount;
                unpackedIndices[t][lane] = ReadBC7Bits(blockBytes, bitOffset, numBits);
            }
        }

        BC7IndexBatch result{};
        for (size_t t = 0; t < 16; ++t)
        {
            const XMVECTOR idx = _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(unpackedIndices[t])));
            result.indices[t] = XMVectorAndInt(idx, activeMask);
        }

        return result;
    }

    inline BC7IndexBatch ExtractBC7Mode0Indices(
        const BC7BlockBatch& blocks,
        FXMVECTOR partition,
        FXMVECTOR activeMask) noexcept
    {
        return ExtractBC7PartitionedIndices(blocks, partition, 0, 83, 3, activeMask);
    }

    inline BC7IndexBatch ExtractBC7Mode1Indices(
        const BC7BlockBatch& blocks,
        FXMVECTOR partition,
        FXMVECTOR activeMask) noexcept
    {
        return ExtractBC7PartitionedIndices(blocks, partition, 1, 82, 3, activeMask);
    }

    inline BC7IndexBatch ExtractBC7Mode2Indices(
        const BC7BlockBatch& blocks,
        FXMVECTOR partition,
        FXMVECTOR activeMask) noexcept
    {
        return ExtractBC7PartitionedIndices(blocks, partition, 2, 99, 2, activeMask);
    }

    inline BC7IndexBatch ExtractBC7Mode3Indices(
        const BC7BlockBatch& blocks,
        FXMVECTOR partition,
        FXMVECTOR activeMask) noexcept
    {
        return ExtractBC7PartitionedIndices(blocks, partition, 3, 98, 2, activeMask);
    }

    inline BC7IndexBatch ExtractBC7Mode7Indices(
        const BC7BlockBatch& blocks,
        FXMVECTOR partition,
        FXMVECTOR activeMask) noexcept
    {
        return ExtractBC7PartitionedIndices(blocks, partition, 7, 98, 2, activeMask);
    }

    template<size_t IndexBits>
    inline XMVECTOR LookupBC7Weight(FXMVECTOR indices) noexcept
    {
        static_assert(IndexBits >= 2 && IndexBits <= 4, "Invalid BC7 index precision");

        __m128i weights;
        if constexpr (IndexBits == 2) weights = _mm_setr_epi8(0, 21, 43, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        else if constexpr (IndexBits == 3) weights = _mm_setr_epi8(0, 9, 18, 27, 37, 46, 55, 64, 0, 0, 0, 0, 0, 0, 0, 0);
        else weights = _mm_setr_epi8(0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);

        return _mm_castsi128_ps(_mm_shuffle_epi8(weights, _mm_castps_si128(indices)));
    }

    // Build rounded BC7 palette values with per-lane vector weights.
    inline XMVECTOR InterpolateBC7PaletteVector(FXMVECTOR endpoint0, FXMVECTOR endpoint1, FXMVECTOR weightFloat) noexcept
    {
        const XMVECTOR endpoint0Float = XMConvertVectorUIntToFloat(endpoint0, 0);
        const XMVECTOR endpoint1Float = XMConvertVectorUIntToFloat(endpoint1, 0);
        const XMVECTOR numerator = XMVectorMultiplyAdd(
            XMVectorSubtract(endpoint1Float, endpoint0Float),
            weightFloat,
            XMVectorScale(endpoint0Float, 64.0f));
        const XMVECTOR rounded = XMVectorTruncate(XMVectorScale(XMVectorAdd(numerator, XMVectorReplicate(32.0f)), 1.0f / 64.0f));
        return XMConvertVectorFloatToUInt(rounded, 0);
    }

    // Build one rounded BC7 palette value for four blocks at once.
    inline XMVECTOR InterpolateBC7PaletteValue(FXMVECTOR endpoint0, FXMVECTOR endpoint1, uint32_t weight) noexcept
    {
        assert(weight <= 64);
        return InterpolateBC7PaletteVector(endpoint0, endpoint1, XMVectorReplicate(static_cast<float>(weight)));
    }

    template<size_t Texel>
    inline XMVECTOR ExtractBC7Mode6Index(const BC7BlockBatch& blocks) noexcept
    {
        static_assert(Texel < 16, "Invalid BC7 texel index");
        constexpr size_t bitOffset = Texel == 0 ? 65 : 64 + Texel * 4;
        constexpr size_t bitCount = Texel == 0 ? 3 : 4;
        return ExtractBC7Bits<bitOffset, bitCount>(blocks);
    }

    // Restore the mode 6 anchor bit and keep all selectors packed for SWAR operations.
    inline BC7Mode6PackedIndexBatch ExtractBC7Mode6PackedIndices(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        const XMVECTOR lowThreeBits = XMVectorReplicateInt(0x7u);
        const XMVECTOR rawLow = XMVectorOrInt(ShiftRight32<1>(blocks.word2), ShiftLeft32<31>(blocks.word3));
        const XMVECTOR rawHigh = ShiftRight32<1>(blocks.word3);

        BC7Mode6PackedIndexBatch result{};
        result.low = XMVectorOrInt(XMVectorAndInt(rawLow, lowThreeBits), ShiftLeft32<1>(XMVectorAndCInt(rawLow, lowThreeBits)));
        result.high = XMVectorOrInt(ShiftLeft32<1>(rawHigh), ShiftRight32<31>(rawLow));
        result.low = XMVectorAndInt(result.low, activeMask);
        result.high = XMVectorAndInt(result.high, activeMask);
        return result;
    }

    // Mark matching four-bit selectors with one bit per nibble.
    inline XMVECTOR MatchBC7Nibbles(FXMVECTOR packedIndices, uint32_t paletteIndex) noexcept
    {
        assert(paletteIndex < 16);

        const XMVECTOR nibbleBits = XMVectorReplicateInt(0x11111111u);
        const XMVECTOR repeatedIndex = XMVectorReplicateInt(paletteIndex * 0x11111111u);
        const XMVECTOR difference = XMVectorXorInt(packedIndices, repeatedIndex);
        XMVECTOR differentBits = XMVectorOrInt(difference, ShiftRight32<1>(difference));
        differentBits = XMVectorOrInt(differentBits, ShiftRight32<2>(difference));
        differentBits = XMVectorOrInt(differentBits, ShiftRight32<3>(difference));
        return XMVectorAndCInt(nibbleBits, differentBits);
    }

    // Count matching selector bits in two 2x2 quadrants.
    inline XMVECTOR CountBC7NibbleQuadrants(FXMVECTOR matchingNibbles) noexcept
    {
        const XMVECTOR pairMask = XMVectorReplicateInt(0x01010101u);
        const XMVECTOR quadrantMask = XMVectorReplicateInt(0x00000303u);
        const XMVECTOR horizontal = AddInt32(XMVectorAndInt(matchingNibbles, pairMask), XMVectorAndInt(ShiftRight32<4>(matchingNibbles), pairMask));
        return AddInt32(XMVectorAndInt(horizontal, quadrantMask), XMVectorAndInt(ShiftRight32<16>(horizontal), quadrantMask));
    }

    // Count one mode 6 palette index in all four quadrants.
    inline BC7QuadrantCountBatch CountBC7Mode6Quadrants(const BC7Mode6PackedIndexBatch& indices, uint32_t paletteIndex, FXMVECTOR activeMask) noexcept
    {
        const XMVECTOR countMask = XMVectorReplicateInt(0xFFu);
        const XMVECTOR top = CountBC7NibbleQuadrants(MatchBC7Nibbles(indices.low, paletteIndex));
        const XMVECTOR bottom = CountBC7NibbleQuadrants(MatchBC7Nibbles(indices.high, paletteIndex));

        BC7QuadrantCountBatch result{};
        result.value[0] = XMVectorAndInt(XMVectorAndInt(top, countMask), activeMask);
        result.value[1] = XMVectorAndInt(XMVectorAndInt(ShiftRight32<8>(top), countMask), activeMask);
        result.value[2] = XMVectorAndInt(XMVectorAndInt(bottom, countMask), activeMask);
        result.value[3] = XMVectorAndInt(XMVectorAndInt(ShiftRight32<8>(bottom), countMask), activeMask);
        return result;
    }

    inline BC7ModeMaskBatch GetBC7ModeMasks(const BC7BlockBatch& blocks) noexcept
    {
        BC7ModeMaskBatch result{};

        for (uint32_t mode = 0; mode < 8; ++mode)
        {
            const uint32_t prefixMask = (1u << (mode + 1u)) - 1u;
            const uint32_t expectedPrefix = 1u << mode;
            const XMVECTOR prefix = XMVectorAndInt(blocks.word0, XMVectorReplicateInt(prefixMask));
            result.mode[mode] = XMVectorEqualInt(prefix, XMVectorReplicateInt(expectedPrefix));
        }

        return result;
    }


    // Compute the exact UNORM quadrant means using the BC7 hardware rounding rules.
    inline BC7QuadrantMeanBatch ComputeBC7Mode6QuadrantMeans(
        const BC7EndpointPairBatch& endpoints,
        const BC7Mode6PackedIndexBatch& indices,
        FXMVECTOR activeMask) noexcept
    {
        constexpr float colorScale = 1.0f / (255.0f * 4.0f);
        constexpr uint32_t weights[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

        BC7QuadrantMeanBatch result{};

        for (size_t q = 0; q < 4; ++q)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                result.value[q][c] = XMVectorZero();
            }
        }

        for (uint32_t i = 0; i < 16; ++i)
        {
            const BC7QuadrantCountBatch counts = CountBC7Mode6Quadrants(indices, i, activeMask);

            XMVECTOR colorFloat[4];
            for (size_t c = 0; c < 4; ++c)
            {
                const XMVECTOR rounded = InterpolateBC7PaletteValue(endpoints.value[0][c], endpoints.value[1][c], weights[i]);
                colorFloat[c] = XMConvertVectorUIntToFloat(rounded, 0);
            }

            for (size_t q = 0; q < 4; ++q)
            {
                const XMVECTOR countFloat = XMConvertVectorUIntToFloat(counts.value[q], 0);
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorMultiplyAdd(colorFloat[c], countFloat, result.value[q][c]);
                }
            }
        }

        for (size_t q = 0; q < 4; ++q)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                result.value[q][c] = XMVectorScale(result.value[q][c], colorScale);
            }
        }

        return result;
    }


    // Compute exact UNORM quadrant means for partitioned multi-subset BC7 modes (Mode 0, 1, 2, 3, 7).
    template<size_t IndexBits>
    inline BC7QuadrantMeanBatch ComputeBC7MultiSubsetQuadrantMeans(
        const BC7MultiSubsetEndpointBatch& endpoints,
        FXMVECTOR partition,
        const BC7IndexBatch& indices,
        uint8_t mode,
        FXMVECTOR activeMask) noexcept
    {
        static_assert(IndexBits == 2 || IndexBits == 3, "Invalid BC7 index precision");
        assert(mode == 0 || mode == 1 || mode == 2 || mode == 3 || mode == 7);

        constexpr float colorScale = 1.0f / (255.0f * 4.0f);

        alignas(16) uint32_t shapes[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(shapes), _mm_castps_si128(partition));

        const BC7PartitionLayout layouts[4] =
        {
            GetBC7ModePartitionLayout(mode, static_cast<uint8_t>(shapes[0] & 0x3Fu)),
            GetBC7ModePartitionLayout(mode, static_cast<uint8_t>(shapes[1] & 0x3Fu)),
            GetBC7ModePartitionLayout(mode, static_cast<uint8_t>(shapes[2] & 0x3Fu)),
            GetBC7ModePartitionLayout(mode, static_cast<uint8_t>(shapes[3] & 0x3Fu))
        };

        // Evaluate all 16 texels in float across 4 channels
        XMVECTOR texelColorsFloat[16][4];

        for (size_t t = 0; t < 16; ++t)
        {
            // Build per-lane subset masks for texel t (supports subset 0, 1, 2)
            alignas(16) uint32_t maskS1[4], maskS2[4];
            for (size_t l = 0; l < 4; ++l)
            {
                const uint8_t s = layouts[l].subsetByTexel[t];
                maskS1[l] = (s == 1) ? 0xFFFFFFFFu : 0u;
                maskS2[l] = (s == 2) ? 0xFFFFFFFFu : 0u;
            }
            const XMVECTOR isS1 = _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(maskS1)));
            const XMVECTOR isS2 = _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(maskS2)));

            // Convert the mode's hardware interpolation weights to floating point.
            const XMVECTOR weightFloat = XMConvertVectorUIntToFloat(LookupBC7Weight<IndexBits>(indices.indices[t]), 0);

            // Interpolate each channel using the selected subset endpoints
            for (size_t c = 0; c < 4; ++c)
            {
                XMVECTOR ep0 = XMVectorSelect(endpoints.value[0][0][c], endpoints.value[1][0][c], isS1);
                ep0 = XMVectorSelect(ep0, endpoints.value[2][0][c], isS2);

                XMVECTOR ep1 = XMVectorSelect(endpoints.value[0][1][c], endpoints.value[1][1][c], isS1);
                ep1 = XMVectorSelect(ep1, endpoints.value[2][1][c], isS2);

                const XMVECTOR colorUInt = InterpolateBC7PaletteVector(ep0, ep1, weightFloat);
                texelColorsFloat[t][c] = XMConvertVectorUIntToFloat(colorUInt, 0);
            }
        }

        // Sum texels for each of the four 2x2 quadrants
        constexpr size_t quadrantTexels[4][4] =
        {
            { 0, 1, 4, 5 },     // Top-left
            { 2, 3, 6, 7 },     // Top-right
            { 8, 9, 12, 13 },   // Bottom-left
            { 10, 11, 14, 15 }  // Bottom-right
        };

        BC7QuadrantMeanBatch result{};

        for (size_t q = 0; q < 4; ++q)
        {
            const size_t t0 = quadrantTexels[q][0];
            const size_t t1 = quadrantTexels[q][1];
            const size_t t2 = quadrantTexels[q][2];
            const size_t t3 = quadrantTexels[q][3];

            for (size_t c = 0; c < 4; ++c)
            {
                XMVECTOR sum = XMVectorAdd(texelColorsFloat[t0][c], texelColorsFloat[t1][c]);
                sum = XMVectorAdd(sum, texelColorsFloat[t2][c]);
                sum = XMVectorAdd(sum, texelColorsFloat[t3][c]);
                result.value[q][c] = XMVectorAndInt(XMVectorScale(sum, colorScale), activeMask);
            }
        }

        return result;
    }

    // Backwards-compatible alias for two-subset callers
    template<size_t IndexBits>
    inline BC7QuadrantMeanBatch ComputeBC7TwoSubsetQuadrantMeans(
        const BC7MultiSubsetEndpointBatch& endpoints,
        FXMVECTOR partition,
        const BC7IndexBatch& indices,
        uint8_t mode,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7MultiSubsetQuadrantMeans<IndexBits>(endpoints, partition, indices, mode, activeMask);
    }

    inline BC7QuadrantMeanBatch ComputeBC7Mode0QuadrantMeans(
        const BC7MultiSubsetEndpointBatch& endpoints,
        FXMVECTOR partition,
        const BC7IndexBatch& indices,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7MultiSubsetQuadrantMeans<3>(endpoints, partition, indices, 0, activeMask);
    }

    inline BC7QuadrantMeanBatch ComputeBC7Mode1QuadrantMeans(
        const BC7MultiSubsetEndpointBatch& endpoints,
        FXMVECTOR partition,
        const BC7IndexBatch& indices,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7MultiSubsetQuadrantMeans<3>(endpoints, partition, indices, 1, activeMask);
    }

    inline BC7QuadrantMeanBatch ComputeBC7Mode2QuadrantMeans(
        const BC7MultiSubsetEndpointBatch& endpoints,
        FXMVECTOR partition,
        const BC7IndexBatch& indices,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7MultiSubsetQuadrantMeans<2>(endpoints, partition, indices, 2, activeMask);
    }

    inline BC7QuadrantMeanBatch ComputeBC7Mode3QuadrantMeans(
        const BC7MultiSubsetEndpointBatch& endpoints,
        FXMVECTOR partition,
        const BC7IndexBatch& indices,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7MultiSubsetQuadrantMeans<2>(endpoints, partition, indices, 3, activeMask);
    }

    inline BC7QuadrantMeanBatch ComputeBC7Mode7QuadrantMeans(
        const BC7MultiSubsetEndpointBatch& endpoints,
        FXMVECTOR partition,
        const BC7IndexBatch& indices,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7MultiSubsetQuadrantMeans<2>(endpoints, partition, indices, 7, activeMask);
    }

    // Dispatch and compute exact quadrant means across mixed parent block modes (Mode 0, 1, 2, 3, 6, 7).
    // Compute exact UNORM quadrant means for 1-subset dual-index BC7 modes with rotation (Mode 4 and Mode 5).
    inline BC7QuadrantMeanBatch ComputeBC7DualIndexQuadrantMeans(
        const BC7EndpointPairBatch& endpoints,
        const BC7IndexBatch& idx1,
        const BC7IndexBatch& idx2,
        FXMVECTOR rotation,
        FXMVECTOR indexMode,
        bool isMode4,
        FXMVECTOR activeMask) noexcept
    {
        constexpr float colorScale = 1.0f / (255.0f * 4.0f);

        const XMVECTOR isRot1 = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(rotation), _mm_set1_epi32(1)));
        const XMVECTOR isRot2 = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(rotation), _mm_set1_epi32(2)));
        const XMVECTOR isRot3 = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(rotation), _mm_set1_epi32(3)));
        const XMVECTOR isIdxMode1 = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_castps_si128(indexMode), _mm_set1_epi32(1)));

        XMVECTOR texelColorsFloat[16][4];

        for (size_t t = 0; t < 16; ++t)
        {
            XMVECTOR wcFloat, waFloat;
            if (isMode4)
            {
                // Mode 4: idx1 is 2-bit, idx2 is 3-bit
                const XMVECTOR w2 = XMConvertVectorUIntToFloat(LookupBC7Weight<2>(idx1.indices[t]), 0);
                const XMVECTOR w3 = XMConvertVectorUIntToFloat(LookupBC7Weight<3>(idx2.indices[t]), 0);
                wcFloat = XMVectorSelect(w2, w3, isIdxMode1);
                waFloat = XMVectorSelect(w3, w2, isIdxMode1);
            }
            else
            {
                // Mode 5: both idx1 (color) and idx2 (alpha) are 2-bit
                wcFloat = XMConvertVectorUIntToFloat(LookupBC7Weight<2>(idx1.indices[t]), 0);
                waFloat = XMConvertVectorUIntToFloat(LookupBC7Weight<2>(idx2.indices[t]), 0);
            }

            // Interpolate RGB using wcFloat, A using waFloat
            const XMVECTOR rInt = InterpolateBC7PaletteVector(endpoints.value[0][0], endpoints.value[1][0], wcFloat);
            const XMVECTOR gInt = InterpolateBC7PaletteVector(endpoints.value[0][1], endpoints.value[1][1], wcFloat);
            const XMVECTOR bInt = InterpolateBC7PaletteVector(endpoints.value[0][2], endpoints.value[1][2], wcFloat);
            const XMVECTOR aInt = InterpolateBC7PaletteVector(endpoints.value[0][3], endpoints.value[1][3], waFloat);

            const XMVECTOR r = XMConvertVectorUIntToFloat(rInt, 0);
            const XMVECTOR g = XMConvertVectorUIntToFloat(gInt, 0);
            const XMVECTOR b = XMConvertVectorUIntToFloat(bInt, 0);
            const XMVECTOR a = XMConvertVectorUIntToFloat(aInt, 0);

            // Channel rotation (swap with Alpha):
            // rot=1: swap(R, A)
            // rot=2: swap(G, A)
            // rot=3: swap(B, A)
            const XMVECTOR outR = XMVectorSelect(r, a, isRot1);
            const XMVECTOR outG = XMVectorSelect(g, a, isRot2);
            const XMVECTOR outB = XMVectorSelect(b, a, isRot3);
            XMVECTOR outA = XMVectorSelect(a, r, isRot1);
            outA = XMVectorSelect(outA, g, isRot2);
            outA = XMVectorSelect(outA, b, isRot3);

            texelColorsFloat[t][0] = outR;
            texelColorsFloat[t][1] = outG;
            texelColorsFloat[t][2] = outB;
            texelColorsFloat[t][3] = outA;
        }

        constexpr size_t quadrantTexels[4][4] =
        {
            { 0, 1, 4, 5 },     // Top-left
            { 2, 3, 6, 7 },     // Top-right
            { 8, 9, 12, 13 },   // Bottom-left
            { 10, 11, 14, 15 }  // Bottom-right
        };

        BC7QuadrantMeanBatch result{};
        for (size_t q = 0; q < 4; ++q)
        {
            const size_t t0 = quadrantTexels[q][0];
            const size_t t1 = quadrantTexels[q][1];
            const size_t t2 = quadrantTexels[q][2];
            const size_t t3 = quadrantTexels[q][3];

            for (size_t c = 0; c < 4; ++c)
            {
                XMVECTOR sum = XMVectorAdd(texelColorsFloat[t0][c], texelColorsFloat[t1][c]);
                sum = XMVectorAdd(sum, texelColorsFloat[t2][c]);
                sum = XMVectorAdd(sum, texelColorsFloat[t3][c]);
                result.value[q][c] = XMVectorAndInt(XMVectorScale(sum, colorScale), activeMask);
            }
        }

        return result;
    }

    inline BC7QuadrantMeanBatch ComputeBC7Mode4QuadrantMeans(
        const BC7EndpointPairBatch& endpoints,
        const BC7IndexBatch& idx1,
        const BC7IndexBatch& idx2,
        FXMVECTOR rotation,
        FXMVECTOR indexMode,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7DualIndexQuadrantMeans(endpoints, idx1, idx2, rotation, indexMode, true, activeMask);
    }

    inline BC7QuadrantMeanBatch ComputeBC7Mode5QuadrantMeans(
        const BC7EndpointPairBatch& endpoints,
        const BC7IndexBatch& idx1,
        const BC7IndexBatch& idx2,
        FXMVECTOR rotation,
        FXMVECTOR activeMask) noexcept
    {
        return ComputeBC7DualIndexQuadrantMeans(endpoints, idx1, idx2, rotation, XMVectorZero(), false, activeMask);
    }

    // Dispatch and compute exact quadrant means across mixed parent block modes (Mode 0 through 7).
    inline BC7QuadrantMeanBatch ComputeBC7ParentQuadrantMeans(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        const BC7ModeMaskBatch modeMasks = GetBC7ModeMasks(blocks);

        BC7QuadrantMeanBatch result{};
        for (size_t q = 0; q < 4; ++q)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                result.value[q][c] = XMVectorZero();
            }
        }

        // Mode 0 parent blocks (3 subsets, RGB 444 + 6 P-bits, 3-bit indices)
        const XMVECTOR mask0 = XMVectorAndInt(modeMasks.mode[0], activeMask);
        if (_mm_movemask_ps(mask0) != 0)
        {
            const XMVECTOR part0 = ExtractBC7Partition(blocks, 0, mask0);
            const BC7MultiSubsetEndpointBatch ep0 = ExtractBC7Mode0Endpoints(blocks, mask0);
            const BC7IndexBatch idx0 = ExtractBC7Mode0Indices(blocks, part0, mask0);
            const BC7QuadrantMeanBatch q0 = ComputeBC7Mode0QuadrantMeans(ep0, part0, idx0, mask0);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q0.value[q][c]);
                }
            }
        }

        // Mode 1 parent blocks (2 subsets, RGB 666 + 2 P-bits, 3-bit indices)
        const XMVECTOR mask1 = XMVectorAndInt(modeMasks.mode[1], activeMask);
        if (_mm_movemask_ps(mask1) != 0)
        {
            const XMVECTOR part1 = ExtractBC7Partition(blocks, 1, mask1);
            const BC7MultiSubsetEndpointBatch ep1 = ExtractBC7Mode1Endpoints(blocks, mask1);
            const BC7IndexBatch idx1 = ExtractBC7Mode1Indices(blocks, part1, mask1);
            const BC7QuadrantMeanBatch q1 = ComputeBC7Mode1QuadrantMeans(ep1, part1, idx1, mask1);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q1.value[q][c]);
                }
            }
        }

        // Mode 2 parent blocks (3 subsets, RGB 555, 2-bit indices)
        const XMVECTOR mask2 = XMVectorAndInt(modeMasks.mode[2], activeMask);
        if (_mm_movemask_ps(mask2) != 0)
        {
            const XMVECTOR part2 = ExtractBC7Partition(blocks, 2, mask2);
            const BC7MultiSubsetEndpointBatch ep2 = ExtractBC7Mode2Endpoints(blocks, mask2);
            const BC7IndexBatch idx2 = ExtractBC7Mode2Indices(blocks, part2, mask2);
            const BC7QuadrantMeanBatch q2 = ComputeBC7Mode2QuadrantMeans(ep2, part2, idx2, mask2);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q2.value[q][c]);
                }
            }
        }

        // Mode 3 parent blocks (2 subsets, RGB 777 + 4 P-bits, 2-bit indices)
        const XMVECTOR mask3 = XMVectorAndInt(modeMasks.mode[3], activeMask);
        if (_mm_movemask_ps(mask3) != 0)
        {
            const XMVECTOR part3 = ExtractBC7Partition(blocks, 3, mask3);
            const BC7MultiSubsetEndpointBatch ep3 = ExtractBC7Mode3Endpoints(blocks, mask3);
            const BC7IndexBatch idx3 = ExtractBC7Mode3Indices(blocks, part3, mask3);
            const BC7QuadrantMeanBatch q3 = ComputeBC7Mode3QuadrantMeans(ep3, part3, idx3, mask3);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q3.value[q][c]);
                }
            }
        }

        // Mode 4 parent blocks (1 subset, RGB 555 + A 6, 2-bit/3-bit dual indices, rotation)
        const XMVECTOR mask4 = XMVectorAndInt(modeMasks.mode[4], activeMask);
        if (_mm_movemask_ps(mask4) != 0)
        {
            const XMVECTOR rot4 = ExtractBC7Bits<5, 2>(blocks);
            const XMVECTOR idxMode4 = ExtractBC7Bits<7, 1>(blocks);
            const BC7EndpointPairBatch ep4 = ExtractBC7Mode4Endpoints(blocks, mask4);
            const BC7IndexBatch idx1 = ExtractBC7PartitionedIndices(blocks, XMVectorZero(), 4, 50, 2, mask4);
            const BC7IndexBatch idx2 = ExtractBC7PartitionedIndices(blocks, XMVectorZero(), 4, 81, 3, mask4);
            const BC7QuadrantMeanBatch q4 = ComputeBC7Mode4QuadrantMeans(ep4, idx1, idx2, rot4, idxMode4, mask4);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q4.value[q][c]);
                }
            }
        }

        // Mode 5 parent blocks (1 subset, RGB 777 + A 8, 2-bit/2-bit dual indices, rotation)
        const XMVECTOR mask5 = XMVectorAndInt(modeMasks.mode[5], activeMask);
        if (_mm_movemask_ps(mask5) != 0)
        {
            const XMVECTOR rot5 = ExtractBC7Bits<6, 2>(blocks);
            const BC7EndpointPairBatch ep5 = ExtractBC7Mode5Endpoints(blocks, mask5);
            const BC7IndexBatch idx1 = ExtractBC7PartitionedIndices(blocks, XMVectorZero(), 5, 66, 2, mask5);
            const BC7IndexBatch idx2 = ExtractBC7PartitionedIndices(blocks, XMVectorZero(), 5, 97, 2, mask5);
            const BC7QuadrantMeanBatch q5 = ComputeBC7Mode5QuadrantMeans(ep5, idx1, idx2, rot5, mask5);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q5.value[q][c]);
                }
            }
        }

        // Mode 6 parent blocks (1 subset, RGBA 7777 + 2 P-bits, 4-bit indices)
        const XMVECTOR mask6 = XMVectorAndInt(modeMasks.mode[6], activeMask);
        if (_mm_movemask_ps(mask6) != 0)
        {
            const BC7EndpointPairBatch ep6 = ExtractBC7Mode6Endpoints(blocks, mask6);
            const BC7Mode6PackedIndexBatch idx6 = ExtractBC7Mode6PackedIndices(blocks, mask6);
            const BC7QuadrantMeanBatch q6 = ComputeBC7Mode6QuadrantMeans(ep6, idx6, mask6);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q6.value[q][c]);
                }
            }
        }

        // Mode 7 parent blocks (2 subsets, RGBA 5555 + 4 P-bits, 2-bit indices)
        const XMVECTOR mask7 = XMVectorAndInt(modeMasks.mode[7], activeMask);
        if (_mm_movemask_ps(mask7) != 0)
        {
            const XMVECTOR part7 = ExtractBC7Partition(blocks, 7, mask7);
            const BC7MultiSubsetEndpointBatch ep7 = ExtractBC7Mode7Endpoints(blocks, mask7);
            const BC7IndexBatch idx7 = ExtractBC7Mode7Indices(blocks, part7, mask7);
            const BC7QuadrantMeanBatch q7 = ComputeBC7Mode7QuadrantMeans(ep7, part7, idx7, mask7);
            for (size_t q = 0; q < 4; ++q)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    result.value[q][c] = XMVectorOrInt(result.value[q][c], q7.value[q][c]);
                }
            }
        }

        return result;
    }

    // Compute the 4-channel block mean (mu_b) from four quadrant means.
    inline BC7BlockMeanBatch ComputeBC7BlockMeans(const BC7QuadrantMeanBatch& quadrants) noexcept
    {
        BC7BlockMeanBatch result{};

        for (size_t channel = 0; channel < 4; ++channel)
        {
            XMVECTOR sum = XMVectorAdd(quadrants.value[0][channel], quadrants.value[1][channel]);
            sum = XMVectorAdd(sum, quadrants.value[2][channel]);
            sum = XMVectorAdd(sum, quadrants.value[3][channel]);
            result.value[channel] = XMVectorScale(sum, 0.25f);
        }

        return result;
    }

    // Compute mean and within-block covariance for four parent BC7 blocks.
    inline BC7ParentStatisticsBatch ComputeBC7ParentStatisticsBatch(const BC7QuadrantMeanBatch& parent) noexcept
    {
        BC7ParentStatisticsBatch result{};
        result.mean = ComputeBC7BlockMeans(parent);

        // Measure each quadrant's displacement from the parent block mean.
        XMVECTOR d[4][4];
        for (size_t q = 0; q < 4; ++q)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                d[q][c] = XMVectorSubtract(parent.value[q][c], result.mean.value[c]);
            }
        }

        // Accumulate within-block covariance: 1/4 * sum_q(d_q[c0] * d_q[c1])
        auto AccumulateCovariance = [&](size_t c0, size_t c1) noexcept -> XMVECTOR
            {
                XMVECTOR cov = XMVectorMultiply(d[0][c0], d[0][c1]);
                cov = XMVectorMultiplyAdd(d[1][c0], d[1][c1], cov);
                cov = XMVectorMultiplyAdd(d[2][c0], d[2][c1], cov);
                cov = XMVectorMultiplyAdd(d[3][c0], d[3][c1], cov);
                return XMVectorScale(cov, 0.25f);
            };

        result.withinCovariance.rr = AccumulateCovariance(0, 0);
        result.withinCovariance.gg = AccumulateCovariance(1, 1);
        result.withinCovariance.bb = AccumulateCovariance(2, 2);
        result.withinCovariance.aa = AccumulateCovariance(3, 3);

        result.withinCovariance.rg = AccumulateCovariance(0, 1);
        result.withinCovariance.rb = AccumulateCovariance(0, 2);
        result.withinCovariance.ra = AccumulateCovariance(0, 3);
        result.withinCovariance.gb = AccumulateCovariance(1, 2);
        result.withinCovariance.ga = AccumulateCovariance(1, 3);
        result.withinCovariance.ba = AccumulateCovariance(2, 3);

        return result;
    }

    // Average four floating-point SIMD vectors.
    inline XMVECTOR AverageFourVectors(
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

    // Add the between-parent covariance contribution for BC7 (RGBA 4D).
    inline void AccumulateBetweenParentCovarianceBC7(
        const BC7ParentStatisticsBatch& parent,
        const BC7BlockMeanBatch& mean,
        BC7CovarianceMatrixBatch& covariance) noexcept
    {
        // This is the between-group term of the law of total covariance in 4D.
        const XMVECTOR deltaR = XMVectorSubtract(parent.mean.value[0], mean.value[0]);
        const XMVECTOR deltaG = XMVectorSubtract(parent.mean.value[1], mean.value[1]);
        const XMVECTOR deltaB = XMVectorSubtract(parent.mean.value[2], mean.value[2]);
        const XMVECTOR deltaA = XMVectorSubtract(parent.mean.value[3], mean.value[3]);

        const XMVECTOR scaleR = XMVectorScale(deltaR, 0.25f);
        const XMVECTOR scaleG = XMVectorScale(deltaG, 0.25f);
        const XMVECTOR scaleB = XMVectorScale(deltaB, 0.25f);
        const XMVECTOR scaleA = XMVectorScale(deltaA, 0.25f);

        covariance.rr = XMVectorMultiplyAdd(deltaR, scaleR, covariance.rr);
        covariance.gg = XMVectorMultiplyAdd(deltaG, scaleG, covariance.gg);
        covariance.bb = XMVectorMultiplyAdd(deltaB, scaleB, covariance.bb);
        covariance.aa = XMVectorMultiplyAdd(deltaA, scaleA, covariance.aa);

        covariance.rg = XMVectorMultiplyAdd(deltaR, scaleG, covariance.rg);
        covariance.rb = XMVectorMultiplyAdd(deltaR, scaleB, covariance.rb);
        covariance.ra = XMVectorMultiplyAdd(deltaR, scaleA, covariance.ra);
        covariance.gb = XMVectorMultiplyAdd(deltaG, scaleB, covariance.gb);
        covariance.ga = XMVectorMultiplyAdd(deltaG, scaleA, covariance.ga);
        covariance.ba = XMVectorMultiplyAdd(deltaB, scaleA, covariance.ba);
    }

    // Combine four parent BC7 blocks with the ANOVA covariance identity.
    inline void ComputeBC7ChildBlockMoments(
        const BC7QuadrantMeanBatch& p00,
        const BC7QuadrantMeanBatch& p10,
        const BC7QuadrantMeanBatch& p01,
        const BC7QuadrantMeanBatch& p11,
        BC7SourceBlockMeansBatch& sourceMeans,
        BC7BlockMeanBatch& mean,
        BC7CovarianceMatrixBatch& covariance) noexcept
    {
        // Recover the mean and within-block covariance of each parent block.
        const BC7ParentStatisticsBatch stats00 = ComputeBC7ParentStatisticsBatch(p00);
        const BC7ParentStatisticsBatch stats10 = ComputeBC7ParentStatisticsBatch(p10);
        const BC7ParentStatisticsBatch stats01 = ComputeBC7ParentStatisticsBatch(p01);
        const BC7ParentStatisticsBatch stats11 = ComputeBC7ParentStatisticsBatch(p11);

        // Preserve parent block means for higher mip pyramid levels.
        sourceMeans.p00 = stats00.mean;
        sourceMeans.p10 = stats10.mean;
        sourceMeans.p01 = stats01.mean;
        sourceMeans.p11 = stats11.mean;

        // Child block mean is the mean of the four parent block means.
        for (size_t c = 0; c < 4; ++c)
        {
            mean.value[c] = AverageFourVectors(
                stats00.mean.value[c],
                stats10.mean.value[c],
                stats01.mean.value[c],
                stats11.mean.value[c]);
        }

        // Average within-block variation from the four parent blocks.
        BC7CovarianceMatrixBatch within{};
        within.rr = AverageFourVectors(stats00.withinCovariance.rr, stats10.withinCovariance.rr, stats01.withinCovariance.rr, stats11.withinCovariance.rr);
        within.gg = AverageFourVectors(stats00.withinCovariance.gg, stats10.withinCovariance.gg, stats01.withinCovariance.gg, stats11.withinCovariance.gg);
        within.bb = AverageFourVectors(stats00.withinCovariance.bb, stats10.withinCovariance.bb, stats01.withinCovariance.bb, stats11.withinCovariance.bb);
        within.aa = AverageFourVectors(stats00.withinCovariance.aa, stats10.withinCovariance.aa, stats01.withinCovariance.aa, stats11.withinCovariance.aa);

        within.rg = AverageFourVectors(stats00.withinCovariance.rg, stats10.withinCovariance.rg, stats01.withinCovariance.rg, stats11.withinCovariance.rg);
        within.rb = AverageFourVectors(stats00.withinCovariance.rb, stats10.withinCovariance.rb, stats01.withinCovariance.rb, stats11.withinCovariance.rb);
        within.ra = AverageFourVectors(stats00.withinCovariance.ra, stats10.withinCovariance.ra, stats01.withinCovariance.ra, stats11.withinCovariance.ra);
        within.gb = AverageFourVectors(stats00.withinCovariance.gb, stats10.withinCovariance.gb, stats01.withinCovariance.gb, stats11.withinCovariance.gb);
        within.ga = AverageFourVectors(stats00.withinCovariance.ga, stats10.withinCovariance.ga, stats01.withinCovariance.ga, stats11.withinCovariance.ga);
        within.ba = AverageFourVectors(stats00.withinCovariance.ba, stats10.withinCovariance.ba, stats01.withinCovariance.ba, stats11.withinCovariance.ba);

        // Add between-parent variation from parent mean differences.
        BC7CovarianceMatrixBatch between{};
        const XMVECTOR zero = XMVectorZero();
        between.rr = zero;
        between.gg = zero;
        between.bb = zero;
        between.aa = zero;
        between.rg = zero;
        between.rb = zero;
        between.ra = zero;
        between.gb = zero;
        between.ga = zero;
        between.ba = zero;

        AccumulateBetweenParentCovarianceBC7(stats00, mean, between);
        AccumulateBetweenParentCovarianceBC7(stats10, mean, between);
        AccumulateBetweenParentCovarianceBC7(stats01, mean, between);
        AccumulateBetweenParentCovarianceBC7(stats11, mean, between);

        // ANOVA covariance identity: total = within + between.
        covariance.rr = XMVectorAdd(between.rr, within.rr);
        covariance.gg = XMVectorAdd(between.gg, within.gg);
        covariance.bb = XMVectorAdd(between.bb, within.bb);
        covariance.aa = XMVectorAdd(between.aa, within.aa);

        covariance.rg = XMVectorAdd(between.rg, within.rg);
        covariance.rb = XMVectorAdd(between.rb, within.rb);
        covariance.ra = XMVectorAdd(between.ra, within.ra);
        covariance.gb = XMVectorAdd(between.gb, within.gb);
        covariance.ga = XMVectorAdd(between.ga, within.ga);
        covariance.ba = XMVectorAdd(between.ba, within.ba);
    }

    // Build the canonical intermediate representation (IR) of the downsampled child block.
    inline BC7ChildCanvas BuildBC7ChildCanvas(
        const BC7QuadrantMeanBatch& p00,
        const BC7QuadrantMeanBatch& p10,
        const BC7QuadrantMeanBatch& p01,
        const BC7QuadrantMeanBatch& p11,
        BC7SourceBlockMeansBatch& sourceMeans) noexcept
    {
        BC7ChildCanvas canvas{};

        // 1. Map parent quadrants to the 16 child texels (RGBA [0, 1] float)
        for (size_t c = 0; c < 4; ++c)
        {
            canvas.texels[0][c]  = p00.value[0][c];
            canvas.texels[1][c]  = p00.value[1][c];
            canvas.texels[2][c]  = p10.value[0][c];
            canvas.texels[3][c]  = p10.value[1][c];

            canvas.texels[4][c]  = p00.value[2][c];
            canvas.texels[5][c]  = p00.value[3][c];
            canvas.texels[6][c]  = p10.value[2][c];
            canvas.texels[7][c]  = p10.value[3][c];

            canvas.texels[8][c]  = p01.value[0][c];
            canvas.texels[9][c]  = p01.value[1][c];
            canvas.texels[10][c] = p11.value[0][c];
            canvas.texels[11][c] = p11.value[1][c];

            canvas.texels[12][c] = p01.value[2][c];
            canvas.texels[13][c] = p01.value[3][c];
            canvas.texels[14][c] = p11.value[2][c];
            canvas.texels[15][c] = p11.value[3][c];
        }

        // 2. Compute child block mean and 4D ANOVA covariance matrix
        ComputeBC7ChildBlockMoments(p00, p10, p01, p11, sourceMeans, canvas.mean, canvas.covariance);

        // 3. Measure opacity across all 16 child texels (Alpha >= 254.0/255.0)
        const XMVECTOR alphaThreshold = XMVectorReplicate(254.0f / 255.0f);
        XMVECTOR isOpaque = XMVectorTrueInt();
        for (size_t t = 0; t < 16; ++t)
        {
            const XMVECTOR opaqueTexel = XMVectorGreaterOrEqual(canvas.texels[t][3], alphaThreshold);
            isOpaque = XMVectorAndInt(isOpaque, opaqueTexel);
        }
        canvas.isOpaque = isOpaque;

        // 4. Extract 4D principal axis using power iteration PCA
        const XMVECTOR diag = XMVectorReplicate(0.5f);
        BC7BlockMeanBatch next{};
        next.value[0] = XMVectorMultiply(canvas.covariance.rr, diag);
        next.value[0] = XMVectorMultiplyAdd(canvas.covariance.rg, diag, next.value[0]);
        next.value[0] = XMVectorMultiplyAdd(canvas.covariance.rb, diag, next.value[0]);
        next.value[0] = XMVectorMultiplyAdd(canvas.covariance.ra, diag, next.value[0]);

        next.value[1] = XMVectorMultiply(canvas.covariance.rg, diag);
        next.value[1] = XMVectorMultiplyAdd(canvas.covariance.gg, diag, next.value[1]);
        next.value[1] = XMVectorMultiplyAdd(canvas.covariance.gb, diag, next.value[1]);
        next.value[1] = XMVectorMultiplyAdd(canvas.covariance.ga, diag, next.value[1]);

        next.value[2] = XMVectorMultiply(canvas.covariance.rb, diag);
        next.value[2] = XMVectorMultiplyAdd(canvas.covariance.gb, diag, next.value[2]);
        next.value[2] = XMVectorMultiplyAdd(canvas.covariance.bb, diag, next.value[2]);
        next.value[2] = XMVectorMultiplyAdd(canvas.covariance.ba, diag, next.value[2]);

        next.value[3] = XMVectorMultiply(canvas.covariance.ra, diag);
        next.value[3] = XMVectorMultiplyAdd(canvas.covariance.ga, diag, next.value[3]);
        next.value[3] = XMVectorMultiplyAdd(canvas.covariance.ba, diag, next.value[3]);
        next.value[3] = XMVectorMultiplyAdd(canvas.covariance.aa, diag, next.value[3]);

        XMVECTOR lengthSquared = XMVectorMultiply(next.value[0], next.value[0]);
        lengthSquared = XMVectorMultiplyAdd(next.value[1], next.value[1], lengthSquared);
        lengthSquared = XMVectorMultiplyAdd(next.value[2], next.value[2], lengthSquared);
        lengthSquared = XMVectorMultiplyAdd(next.value[3], next.value[3], lengthSquared);
        lengthSquared = XMVectorAdd(lengthSquared, XMVectorReplicate(1e-20f));

        const XMVECTOR inverseLength = XMVectorReciprocalSqrt(lengthSquared);
        canvas.axis.value[0] = XMVectorMultiply(next.value[0], inverseLength);
        canvas.axis.value[1] = XMVectorMultiply(next.value[1], inverseLength);
        canvas.axis.value[2] = XMVectorMultiply(next.value[2], inverseLength);
        canvas.axis.value[3] = XMVectorMultiply(next.value[3], inverseLength);

        return canvas;
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

    // Precomputed 16-bit partition masks for 64 2-subset shapes.
    struct BC7PartitionMaskTable2Subsets
    {
        uint16_t mask[64];
        constexpr BC7PartitionMaskTable2Subsets() : mask{}
        {
            for (uint32_t s = 0; s < 64; ++s)
            {
                uint16_t m = 0;
                for (uint32_t i = 0; i < 16; ++i)
                {
                    m |= static_cast<uint16_t>(g_bc7PartitionTable2Subsets[s][i] << i);
                }
                mask[s] = m;
            }
        }
    };
    constexpr BC7PartitionMaskTable2Subsets g_bc7PartitionMasks2Subsets{};

    // FasTC-style Hamming distance partition estimation for 2-subset modes.
    // Zero arbitrary thresholds: finds the exact argmin Hamming distance shape!
    inline XMVECTOR EstimateBC7Partition2Subsets(const BC7ChildCanvas& canvas) noexcept
    {
        // 1. Project 16 child texels onto the canvas principal axis: p_t = (C_t - mean) . axis
        alignas(16) float projLanes[16][4];
        for (size_t t = 0; t < 16; ++t)
        {
            XMVECTOR proj = XMVectorMultiply(canvas.axis.value[0], XMVectorSubtract(canvas.texels[t][0], canvas.mean.value[0]));
            proj = XMVectorMultiplyAdd(canvas.axis.value[1], XMVectorSubtract(canvas.texels[t][1], canvas.mean.value[1]), proj);
            proj = XMVectorMultiplyAdd(canvas.axis.value[2], XMVectorSubtract(canvas.texels[t][2], canvas.mean.value[2]), proj);
            proj = XMVectorMultiplyAdd(canvas.axis.value[3], XMVectorSubtract(canvas.texels[t][3], canvas.mean.value[3]), proj);
            _mm_store_ps(projLanes[t], proj);
        }

        // 2. For each SIMD lane, construct 16-bit projection mask and find argmin Hamming distance
        alignas(16) uint32_t bestShapes[4]{};
        for (size_t lane = 0; lane < 4; ++lane)
        {
            uint16_t projMask = 0;
            for (size_t t = 0; t < 16; ++t)
            {
                if (projLanes[t][lane] >= 0.0f)
                {
                    projMask |= static_cast<uint16_t>(1u << t);
                }
            }

            uint32_t minDistance = 999;
            uint32_t bestS = 0;
            const uint16_t invProjMask = static_cast<uint16_t>(~projMask);

            for (uint32_t s = 0; s < 64; ++s)
            {
                const uint16_t shapeMask = g_bc7PartitionMasks2Subsets.mask[s];
                // Subset polarity invariant: min(popcount(M ^ S), popcount((~M) ^ S))
                const uint32_t d0 = static_cast<uint32_t>(_mm_popcnt_u32(projMask ^ shapeMask));
                const uint32_t d1 = static_cast<uint32_t>(_mm_popcnt_u32(invProjMask ^ shapeMask));
                const uint32_t d = (d0 < d1) ? d0 : d1;

                if (d < minDistance)
                {
                    minDistance = d;
                    bestS = s;
                }
            }
            bestShapes[lane] = bestS;
        }

        return _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(bestShapes)));
    }

    struct BC7Partition3SubsetsResult
    {
        XMVECTOR mode0Shape; // Shape 0..15 per lane
        XMVECTOR mode2Shape; // Shape 0..63 per lane
    };

    // FasTC-style 3-subset partition estimation for Mode 0 (shapes 0..15) and Mode 2 (shapes 0..63).
    // Zero arbitrary thresholds: projects onto principal axis, splits into 3 tertiles, and finds argmax match.
    inline BC7Partition3SubsetsResult EstimateBC7Partition3Subsets(const BC7ChildCanvas& canvas) noexcept
    {
        alignas(16) float projLanes[16][4];
        for (size_t t = 0; t < 16; ++t)
        {
            XMVECTOR proj = XMVectorMultiply(canvas.axis.value[0], XMVectorSubtract(canvas.texels[t][0], canvas.mean.value[0]));
            proj = XMVectorMultiplyAdd(canvas.axis.value[1], XMVectorSubtract(canvas.texels[t][1], canvas.mean.value[1]), proj);
            proj = XMVectorMultiplyAdd(canvas.axis.value[2], XMVectorSubtract(canvas.texels[t][2], canvas.mean.value[2]), proj);
            proj = XMVectorMultiplyAdd(canvas.axis.value[3], XMVectorSubtract(canvas.texels[t][3], canvas.mean.value[3]), proj);
            _mm_store_ps(projLanes[t], proj);
        }

        alignas(16) uint32_t bestMode0Shapes[4]{};
        alignas(16) uint32_t bestMode2Shapes[4]{};

        for (size_t lane = 0; lane < 4; ++lane)
        {
            float pMin = projLanes[0][lane];
            float pMax = projLanes[0][lane];
            for (size_t t = 1; t < 16; ++t)
            {
                if (projLanes[t][lane] < pMin) pMin = projLanes[t][lane];
                if (projLanes[t][lane] > pMax) pMax = projLanes[t][lane];
            }

            const float delta = pMax - pMin;
            const float b1 = pMin + delta * (1.0f / 3.0f);
            const float b2 = pMin + delta * (2.0f / 3.0f);

            uint8_t L[16]{};
            for (size_t t = 0; t < 16; ++t)
            {
                const float p = projLanes[t][lane];
                L[t] = (p < b1) ? 0 : ((p < b2) ? 1 : 2);
            }

            // Invariant: Texel 0 in BC7 3-subset partitions is always in subset 0.
            // Renumber labels so L[0] == 0.
            const uint8_t l0 = L[0];
            if (l0 != 0)
            {
                for (size_t t = 0; t < 16; ++t)
                {
                    if (L[t] == 0) L[t] = l0;
                    else if (L[t] == l0) L[t] = 0;
                }
            }

            // Permuted label where subsets 1 and 2 are swapped
            uint8_t L_swap[16]{};
            for (size_t t = 0; t < 16; ++t)
            {
                L_swap[t] = (L[t] == 1) ? 2 : ((L[t] == 2) ? 1 : 0);
            }

            // Find best shape for Mode 0 (shapes 0..15) and Mode 2 (shapes 0..63)
            uint32_t bestMatchM0 = 0;
            uint32_t bestShapeM0 = 0;

            uint32_t bestMatchM2 = 0;
            uint32_t bestShapeM2 = 0;

            for (uint32_t s = 0; s < 64; ++s)
            {
                uint32_t matchA = 0;
                uint32_t matchB = 0;
                for (size_t t = 0; t < 16; ++t)
                {
                    const uint8_t tableVal = g_bc7PartitionTable3Subsets[s][t];
                    if (tableVal == L[t]) ++matchA;
                    if (tableVal == L_swap[t]) ++matchB;
                }
                const uint32_t match = (matchA > matchB) ? matchA : matchB;

                if (s < 16 && match > bestMatchM0)
                {
                    bestMatchM0 = match;
                    bestShapeM0 = s;
                }
                if (match > bestMatchM2)
                {
                    bestMatchM2 = match;
                    bestShapeM2 = s;
                }
            }

            bestMode0Shapes[lane] = bestShapeM0;
            bestMode2Shapes[lane] = bestShapeM2;
        }

        BC7Partition3SubsetsResult result{};
        result.mode0Shape = _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(bestMode0Shapes)));
        result.mode2Shape = _mm_castsi128_ps(_mm_load_si128(reinterpret_cast<const __m128i*>(bestMode2Shapes)));
        return result;
    }

    // Pack Mode 1 fields into one 128-bit BC7 block.
    inline XMUINT4 EmitBC7Mode1BlockScalar(
        uint32_t partition,
        const uint8_t ep[2][2][3], // [subset 0..1][ep 0..1][rgb 0..2] (6-bit values)
        const uint8_t pBit[2],     // [subset 0..1]
        const uint8_t indices[16]) noexcept
    {
        uint8_t bytes[16]{};
        size_t bitOffset = 0;

        // Mode 1 prefix: 0b10 (bit 0 = 0, bit 1 = 1) -> 2 bits
        WriteBC7Bits(bytes, bitOffset, 0b10, 2);

        // Partition: 6 bits
        WriteBC7Bits(bytes, bitOffset, partition, 6);

        // Endpoints (R, G, B order, subset 0 ep 0, ep 1, subset 1 ep 0, ep 1)
        for (size_t c = 0; c < 3; ++c)
        {
            WriteBC7Bits(bytes, bitOffset, ep[0][0][c], 6);
            WriteBC7Bits(bytes, bitOffset, ep[0][1][c], 6);
            WriteBC7Bits(bytes, bitOffset, ep[1][0][c], 6);
            WriteBC7Bits(bytes, bitOffset, ep[1][1][c], 6);
        }

        // P-bits (subset 0, subset 1)
        WriteBC7Bits(bytes, bitOffset, pBit[0], 1);
        WriteBC7Bits(bytes, bitOffset, pBit[1], 1);

        // Indices (3 bits each, except anchor texels which have 2 bits)
        const uint8_t anchor1 = g_bc7FixUp2Subsets[partition];

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0 || t == anchor1) ? 2 : 3;
            WriteBC7Bits(bytes, bitOffset, indices[t], bitCount);
        }

        assert(bitOffset == 128);

        XMUINT4 result;
        memcpy(&result, bytes, sizeof(result));
        return result;
    }

    struct BC7Mode1FitResult
    {
        XMUINT4 block;
        float totalError; // Sum of squared RGB errors across all 16 texels
    };

    // Fit Mode 1 endpoints and 3-bit indices for a single lane, returning 128-bit block and exact reconstruction error.
    inline BC7Mode1FitResult FitBC7Mode1SingleLane(
        const BC7ChildCanvas& canvas,
        uint32_t partition,
        size_t lane) noexcept
    {
        alignas(16) float texelsRGB[16][3];
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                texelsRGB[t][c] = XMVectorGetByIndex(canvas.texels[t][c], lane);
            }
        }

        uint8_t ep6[2][2][3]{};
        uint8_t pBits[2]{};
        uint8_t unquantEP[2][2][3]{};

        // 1. Quantize endpoints for each subset
        for (size_t s = 0; s < 2; ++s)
        {
            float minC[3] = { 1.0f, 1.0f, 1.0f };
            float maxC[3] = { 0.0f, 0.0f, 0.0f };
            bool hasTexels = false;

            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == s)
                {
                    hasTexels = true;
                    for (size_t c = 0; c < 3; ++c)
                    {
                        if (texelsRGB[t][c] < minC[c]) minC[c] = texelsRGB[t][c];
                        if (texelsRGB[t][c] > maxC[c]) maxC[c] = texelsRGB[t][c];
                    }
                }
            }

            if (!hasTexels)
            {
                minC[0] = minC[1] = minC[2] = 0.0f;
                maxC[0] = maxC[1] = maxC[2] = 0.0f;
            }

            uint32_t lsbSum = 0;
            int q0[3]{}, q1[3]{};
            for (size_t c = 0; c < 3; ++c)
            {
                q0[c] = std::min(127, std::max(0, static_cast<int>(std::round(minC[c] * 127.0f))));
                q1[c] = std::min(127, std::max(0, static_cast<int>(std::round(maxC[c] * 127.0f))));
                lsbSum += (q0[c] & 1) + (q1[c] & 1);
            }

            // Majority vote for shared P-bit
            const uint8_t p = (lsbSum >= 3) ? 1 : 0;
            pBits[s] = p;

            for (size_t c = 0; c < 3; ++c)
            {
                ep6[s][0][c] = static_cast<uint8_t>(std::min(63, std::max(0, (q0[c] - p + 1) / 2)));
                ep6[s][1][c] = static_cast<uint8_t>(std::min(63, std::max(0, (q1[c] - p + 1) / 2)));

                // D3D Mode 1 unquantization: v7 = (q << 1) | p, v8 = (v7 << 1) | (v7 >> 6)
                unquantEP[s][0][c] = UnquantizeBC7_6BitScalar(ep6[s][0][c], p);
                unquantEP[s][1][c] = UnquantizeBC7_6BitScalar(ep6[s][1][c], p);
            }
        }

        // 2. Build 8-color palette for each subset
        constexpr uint32_t weights[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
        float palette[2][8][3];
        for (size_t s = 0; s < 2; ++s)
        {
            for (uint32_t k = 0; k < 8; ++k)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    const uint32_t val = ((64u - weights[k]) * unquantEP[s][0][c] + weights[k] * unquantEP[s][1][c] + 32u) >> 6;
                    palette[s][k][c] = static_cast<float>(val) / 255.0f;
                }
            }
        }

        // 3. Assign nearest index and accumulate reconstruction error
        uint8_t indices[16]{};
        float totalError = 0.0f;

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t s = g_bc7PartitionTable2Subsets[partition][t];
            float bestDist = 1e10f;
            uint8_t bestK = 0;

            for (uint8_t k = 0; k < 8; ++k)
            {
                float d = 0.0f;
                for (size_t c = 0; c < 3; ++c)
                {
                    const float diff = texelsRGB[t][c] - palette[s][k][c];
                    d += diff * diff;
                }
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }

            indices[t] = bestK;
            totalError += bestDist;
        }

        // 4. Anchor constraints:
        // Subset 0 anchor: texel 0. MSB must be 0 (index <= 3).
        if (indices[0] >= 4)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                std::swap(ep6[0][0][c], ep6[0][1][c]);
            }
            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == 0)
                {
                    indices[t] = static_cast<uint8_t>(7u - indices[t]);
                }
            }
        }

        // Subset 1 anchor: texel anchor1. MSB must be 0 (index <= 3).
        const uint8_t anchor1 = g_bc7FixUp2Subsets[partition];
        if (indices[anchor1] >= 4)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                std::swap(ep6[1][0][c], ep6[1][1][c]);
            }
            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == 1)
                {
                    indices[t] = static_cast<uint8_t>(7u - indices[t]);
                }
            }
        }

        // 5. Emit 128-bit block
        BC7Mode1FitResult result{};
        result.block = EmitBC7Mode1BlockScalar(partition, ep6, pBits, indices);
        result.totalError = totalError;
        return result;
    }

    // Pack Mode 3 fields into one 128-bit BC7 block.
    inline XMUINT4 EmitBC7Mode3BlockScalar(
        uint32_t partition,
        const uint8_t ep[2][2][3], // [subset 0..1][ep 0..1][rgb 0..2] (7-bit values)
        const uint8_t pBit[2][2],  // [subset 0..1][ep 0..1]
        const uint8_t indices[16]) noexcept
    {
        uint8_t bytes[16]{};
        size_t bitOffset = 0;

        // Mode 3 prefix: 0b1000 (bit 0..2 = 0, bit 3 = 1) -> 4 bits
        WriteBC7Bits(bytes, bitOffset, 0b1000, 4);

        // Partition: 6 bits
        WriteBC7Bits(bytes, bitOffset, partition, 6);

        // Endpoints (R, G, B order, subset 0 ep 0, ep 1, subset 1 ep 0, ep 1)
        for (size_t c = 0; c < 3; ++c)
        {
            WriteBC7Bits(bytes, bitOffset, ep[0][0][c], 7);
            WriteBC7Bits(bytes, bitOffset, ep[0][1][c], 7);
            WriteBC7Bits(bytes, bitOffset, ep[1][0][c], 7);
            WriteBC7Bits(bytes, bitOffset, ep[1][1][c], 7);
        }

        // P-bits: 4 bits (subset 0 ep 0, ep 1, subset 1 ep 0, ep 1)
        WriteBC7Bits(bytes, bitOffset, pBit[0][0], 1);
        WriteBC7Bits(bytes, bitOffset, pBit[0][1], 1);
        WriteBC7Bits(bytes, bitOffset, pBit[1][0], 1);
        WriteBC7Bits(bytes, bitOffset, pBit[1][1], 1);

        // Indices (2 bits each, except anchor texels which have 1 bit)
        const uint8_t anchor1 = g_bc7FixUp2Subsets[partition];

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0 || t == anchor1) ? 1 : 2;
            WriteBC7Bits(bytes, bitOffset, indices[t], bitCount);
        }

        assert(bitOffset == 128);

        XMUINT4 result;
        memcpy(&result, bytes, sizeof(result));
        return result;
    }

    struct BC7Mode3FitResult
    {
        XMUINT4 block;
        float totalError;
    };

    // Fit Mode 3 endpoints and 2-bit indices for a single lane, returning 128-bit block and exact reconstruction error.
    inline BC7Mode3FitResult FitBC7Mode3SingleLane(
        const BC7ChildCanvas& canvas,
        uint32_t partition,
        size_t lane) noexcept
    {
        alignas(16) float texelsRGB[16][3];
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                texelsRGB[t][c] = XMVectorGetByIndex(canvas.texels[t][c], lane);
            }
        }

        uint8_t ep7[2][2][3]{};
        uint8_t pBits[2][2]{};
        uint8_t unquantEP[2][2][3]{};

        // 1. Quantize endpoints for each subset (7-bit + unique P-bit per endpoint)
        for (size_t s = 0; s < 2; ++s)
        {
            float minC[3] = { 1.0f, 1.0f, 1.0f };
            float maxC[3] = { 0.0f, 0.0f, 0.0f };
            bool hasTexels = false;

            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == s)
                {
                    hasTexels = true;
                    for (size_t c = 0; c < 3; ++c)
                    {
                        if (texelsRGB[t][c] < minC[c]) minC[c] = texelsRGB[t][c];
                        if (texelsRGB[t][c] > maxC[c]) maxC[c] = texelsRGB[t][c];
                    }
                }
            }

            if (!hasTexels)
            {
                minC[0] = minC[1] = minC[2] = 0.0f;
                maxC[0] = maxC[1] = maxC[2] = 0.0f;
            }

            // In Mode 3, 7-bit + P-bit gives an 8-bit value directly: (val7 << 1) | pBit
            for (size_t c = 0; c < 3; ++c)
            {
                const int q0 = std::min(255, std::max(0, static_cast<int>(std::round(minC[c] * 255.0f))));
                const int q1 = std::min(255, std::max(0, static_cast<int>(std::round(maxC[c] * 255.0f))));
                ep7[s][0][c] = static_cast<uint8_t>(q0 >> 1);
                ep7[s][1][c] = static_cast<uint8_t>(q1 >> 1);
                pBits[s][0] = static_cast<uint8_t>(q0 & 1);
                pBits[s][1] = static_cast<uint8_t>(q1 & 1);

                unquantEP[s][0][c] = UnquantizeBC7_7BitScalar(ep7[s][0][c], pBits[s][0]);
                unquantEP[s][1][c] = UnquantizeBC7_7BitScalar(ep7[s][1][c], pBits[s][1]);
            }
        }

        // 2. Build 4-color palette for each subset (2-bit weights: { 0, 21, 43, 64 })
        constexpr uint32_t weights[4] = { 0, 21, 43, 64 };
        float palette[2][4][3];
        for (size_t s = 0; s < 2; ++s)
        {
            for (uint32_t k = 0; k < 4; ++k)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    const uint32_t val = ((64u - weights[k]) * unquantEP[s][0][c] + weights[k] * unquantEP[s][1][c] + 32u) >> 6;
                    palette[s][k][c] = static_cast<float>(val) / 255.0f;
                }
            }
        }

        // 3. Assign nearest index and accumulate reconstruction error
        uint8_t indices[16]{};
        float totalError = 0.0f;

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t s = g_bc7PartitionTable2Subsets[partition][t];
            float bestDist = 1e10f;
            uint8_t bestK = 0;

            for (uint8_t k = 0; k < 4; ++k)
            {
                float d = 0.0f;
                for (size_t c = 0; c < 3; ++c)
                {
                    const float diff = texelsRGB[t][c] - palette[s][k][c];
                    d += diff * diff;
                }
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }

            indices[t] = bestK;
            totalError += bestDist;
        }

        // 4. Anchor constraints:
        // Subset 0 anchor: texel 0. 1 bit (index <= 1).
        if (indices[0] >= 2)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                std::swap(ep7[0][0][c], ep7[0][1][c]);
            }
            std::swap(pBits[0][0], pBits[0][1]);
            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == 0)
                {
                    indices[t] = static_cast<uint8_t>(3u - indices[t]);
                }
            }
        }

        // Subset 1 anchor: texel anchor1. 1 bit (index <= 1).
        const uint8_t anchor1 = g_bc7FixUp2Subsets[partition];
        if (indices[anchor1] >= 2)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                std::swap(ep7[1][0][c], ep7[1][1][c]);
            }
            std::swap(pBits[1][0], pBits[1][1]);
            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == 1)
                {
                    indices[t] = static_cast<uint8_t>(3u - indices[t]);
                }
            }
        }

        // 5. Emit 128-bit block
        BC7Mode3FitResult result{};
        result.block = EmitBC7Mode3BlockScalar(partition, ep7, pBits, indices);
        result.totalError = totalError;
        return result;
    }

    struct BC7Mode0FitResult
    {
        XMUINT4 block;
        float totalError;
    };

    // Pack Mode 0 fields into one 128-bit BC7 block.
    inline XMUINT4 EmitBC7Mode0BlockScalar(
        uint32_t partition,
        const uint8_t ep[3][2][3], // [subset 0..2][ep 0..1][rgb 0..2] (4-bit values)
        const uint8_t pBit[3][2],  // [subset 0..2][ep 0..1]
        const uint8_t indices[16]) noexcept
    {
        uint8_t bytes[16]{};
        size_t bitOffset = 0;

        // Mode 0 prefix: 0b1 (bit 0 = 1) -> 1 bit
        WriteBC7Bits(bytes, bitOffset, 1, 1);

        // Partition: 4 bits
        WriteBC7Bits(bytes, bitOffset, partition, 4);

        // Endpoints (R, G, B order, subset 0 ep 0..1, subset 1 ep 0..1, subset 2 ep 0..1)
        for (size_t c = 0; c < 3; ++c)
        {
            for (size_t s = 0; s < 3; ++s)
            {
                WriteBC7Bits(bytes, bitOffset, ep[s][0][c], 4);
                WriteBC7Bits(bytes, bitOffset, ep[s][1][c], 4);
            }
        }

        // P-bits: 6 bits (subset 0 ep 0..1, subset 1 ep 0..1, subset 2 ep 0..1)
        for (size_t s = 0; s < 3; ++s)
        {
            WriteBC7Bits(bytes, bitOffset, pBit[s][0], 1);
            WriteBC7Bits(bytes, bitOffset, pBit[s][1], 1);
        }

        // Indices (3 bits each, except 3 anchor texels which have 2 bits)
        const uint8_t anchor1 = g_bc7FixUp3Subsets[partition][0];
        const uint8_t anchor2 = g_bc7FixUp3Subsets[partition][1];

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0 || t == anchor1 || t == anchor2) ? 2 : 3;
            WriteBC7Bits(bytes, bitOffset, indices[t], bitCount);
        }

        assert(bitOffset == 128);

        XMUINT4 result;
        memcpy(&result, bytes, sizeof(result));
        return result;
    }

    // Fit Mode 0 endpoints and 3-bit indices for a single lane, returning 128-bit block and exact reconstruction error.
    inline BC7Mode0FitResult FitBC7Mode0SingleLane(
        const BC7ChildCanvas& canvas,
        uint32_t partition,
        size_t lane) noexcept
    {
        alignas(16) float texelsRGB[16][3];
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                texelsRGB[t][c] = XMVectorGetByIndex(canvas.texels[t][c], lane);
            }
        }

        uint8_t ep4[3][2][3]{};
        uint8_t pBits[3][2]{};
        uint8_t unquantEP[3][2][3]{};

        // 1. Quantize endpoints for each of the 3 subsets (4-bit + 1 P-bit per endpoint = 5 bits: [0, 31])
        for (size_t s = 0; s < 3; ++s)
        {
            float minC[3] = { 1.0f, 1.0f, 1.0f };
            float maxC[3] = { 0.0f, 0.0f, 0.0f };
            bool hasTexels = false;

            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable3Subsets[partition][t] == s)
                {
                    hasTexels = true;
                    for (size_t c = 0; c < 3; ++c)
                    {
                        if (texelsRGB[t][c] < minC[c]) minC[c] = texelsRGB[t][c];
                        if (texelsRGB[t][c] > maxC[c]) maxC[c] = texelsRGB[t][c];
                    }
                }
            }

            if (!hasTexels)
            {
                minC[0] = minC[1] = minC[2] = 0.0f;
                maxC[0] = maxC[1] = maxC[2] = 0.0f;
            }

            for (size_t c = 0; c < 3; ++c)
            {
                const int q0 = std::min(31, std::max(0, static_cast<int>(std::round(minC[c] * 31.0f))));
                const int q1 = std::min(31, std::max(0, static_cast<int>(std::round(maxC[c] * 31.0f))));
                ep4[s][0][c] = static_cast<uint8_t>(q0 >> 1);
                ep4[s][1][c] = static_cast<uint8_t>(q1 >> 1);
                pBits[s][0] = static_cast<uint8_t>(q0 & 1);
                pBits[s][1] = static_cast<uint8_t>(q1 & 1);

                unquantEP[s][0][c] = UnquantizeBC7_4BitScalar(ep4[s][0][c], pBits[s][0]);
                unquantEP[s][1][c] = UnquantizeBC7_4BitScalar(ep4[s][1][c], pBits[s][1]);
            }
        }

        // 2. Build 8-color palette for each subset (3-bit weights)
        constexpr uint32_t weights[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
        float palette[3][8][3];
        for (size_t s = 0; s < 3; ++s)
        {
            for (uint32_t k = 0; k < 8; ++k)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    const uint32_t val = ((64u - weights[k]) * unquantEP[s][0][c] + weights[k] * unquantEP[s][1][c] + 32u) >> 6;
                    palette[s][k][c] = static_cast<float>(val) / 255.0f;
                }
            }
        }

        // 3. Assign nearest index and accumulate error
        uint8_t indices[16]{};
        float totalError = 0.0f;

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t s = g_bc7PartitionTable3Subsets[partition][t];
            float bestDist = 1e10f;
            uint8_t bestK = 0;

            for (uint8_t k = 0; k < 8; ++k)
            {
                float d = 0.0f;
                for (size_t c = 0; c < 3; ++c)
                {
                    const float diff = texelsRGB[t][c] - palette[s][k][c];
                    d += diff * diff;
                }
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }

            indices[t] = bestK;
            totalError += bestDist;
        }

        // 4. Anchor constraints: 3 anchors (texel 0, anchor1, anchor2). MSB must be 0 (index <= 3).
        const uint8_t anchors[3] = { 0, g_bc7FixUp3Subsets[partition][0], g_bc7FixUp3Subsets[partition][1] };
        for (size_t s = 0; s < 3; ++s)
        {
            const uint8_t a = anchors[s];
            if (indices[a] >= 4)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    std::swap(ep4[s][0][c], ep4[s][1][c]);
                }
                std::swap(pBits[s][0], pBits[s][1]);
                for (size_t t = 0; t < 16; ++t)
                {
                    if (g_bc7PartitionTable3Subsets[partition][t] == s)
                    {
                        indices[t] = static_cast<uint8_t>(7u - indices[t]);
                    }
                }
            }
        }

        BC7Mode0FitResult result{};
        result.block = EmitBC7Mode0BlockScalar(partition, ep4, pBits, indices);
        result.totalError = totalError;
        return result;
    }

    struct BC7Mode2FitResult
    {
        XMUINT4 block;
        float totalError;
    };

    // Pack Mode 2 fields into one 128-bit BC7 block.
    inline XMUINT4 EmitBC7Mode2BlockScalar(
        uint32_t partition,
        const uint8_t ep[3][2][3], // [subset 0..2][ep 0..1][rgb 0..2] (5-bit values)
        const uint8_t indices[16]) noexcept
    {
        uint8_t bytes[16]{};
        size_t bitOffset = 0;

        // Mode 2 prefix: 0b100 (bit 0..1 = 0, bit 2 = 1) -> 3 bits
        WriteBC7Bits(bytes, bitOffset, 0b100, 3);

        // Partition: 6 bits
        WriteBC7Bits(bytes, bitOffset, partition, 6);

        // Endpoints (R, G, B order, subset 0 ep 0..1, subset 1 ep 0..1, subset 2 ep 0..1)
        for (size_t c = 0; c < 3; ++c)
        {
            for (size_t s = 0; s < 3; ++s)
            {
                WriteBC7Bits(bytes, bitOffset, ep[s][0][c], 5);
                WriteBC7Bits(bytes, bitOffset, ep[s][1][c], 5);
            }
        }

        // Indices (2 bits each, except 3 anchor texels which have 1 bit)
        const uint8_t anchor1 = g_bc7FixUp3Subsets[partition][0];
        const uint8_t anchor2 = g_bc7FixUp3Subsets[partition][1];

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0 || t == anchor1 || t == anchor2) ? 1 : 2;
            WriteBC7Bits(bytes, bitOffset, indices[t], bitCount);
        }

        assert(bitOffset == 128);

        XMUINT4 result;
        memcpy(&result, bytes, sizeof(result));
        return result;
    }

    // Fit Mode 2 endpoints and 2-bit indices for a single lane, returning 128-bit block and exact reconstruction error.
    inline BC7Mode2FitResult FitBC7Mode2SingleLane(
        const BC7ChildCanvas& canvas,
        uint32_t partition,
        size_t lane) noexcept
    {
        alignas(16) float texelsRGB[16][3];
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                texelsRGB[t][c] = XMVectorGetByIndex(canvas.texels[t][c], lane);
            }
        }

        uint8_t ep5[3][2][3]{};
        uint8_t unquantEP[3][2][3]{};

        // 1. Quantize endpoints for each of the 3 subsets (5-bit without P-bit: [0, 31])
        for (size_t s = 0; s < 3; ++s)
        {
            float minC[3] = { 1.0f, 1.0f, 1.0f };
            float maxC[3] = { 0.0f, 0.0f, 0.0f };
            bool hasTexels = false;

            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable3Subsets[partition][t] == s)
                {
                    hasTexels = true;
                    for (size_t c = 0; c < 3; ++c)
                    {
                        if (texelsRGB[t][c] < minC[c]) minC[c] = texelsRGB[t][c];
                        if (texelsRGB[t][c] > maxC[c]) maxC[c] = texelsRGB[t][c];
                    }
                }
            }

            if (!hasTexels)
            {
                minC[0] = minC[1] = minC[2] = 0.0f;
                maxC[0] = maxC[1] = maxC[2] = 0.0f;
            }

            for (size_t c = 0; c < 3; ++c)
            {
                ep5[s][0][c] = static_cast<uint8_t>(std::min(31, std::max(0, static_cast<int>(std::round(minC[c] * 31.0f)))));
                ep5[s][1][c] = static_cast<uint8_t>(std::min(31, std::max(0, static_cast<int>(std::round(maxC[c] * 31.0f)))));

                unquantEP[s][0][c] = UnquantizeBC7_5Bit_NoPBitScalar(ep5[s][0][c]);
                unquantEP[s][1][c] = UnquantizeBC7_5Bit_NoPBitScalar(ep5[s][1][c]);
            }
        }

        // 2. Build 4-color palette for each subset (2-bit weights: { 0, 21, 43, 64 })
        constexpr uint32_t weights[4] = { 0, 21, 43, 64 };
        float palette[3][4][3];
        for (size_t s = 0; s < 3; ++s)
        {
            for (uint32_t k = 0; k < 4; ++k)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    const uint32_t val = ((64u - weights[k]) * unquantEP[s][0][c] + weights[k] * unquantEP[s][1][c] + 32u) >> 6;
                    palette[s][k][c] = static_cast<float>(val) / 255.0f;
                }
            }
        }

        // 3. Assign nearest index and accumulate error
        uint8_t indices[16]{};
        float totalError = 0.0f;

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t s = g_bc7PartitionTable3Subsets[partition][t];
            float bestDist = 1e10f;
            uint8_t bestK = 0;

            for (uint8_t k = 0; k < 4; ++k)
            {
                float d = 0.0f;
                for (size_t c = 0; c < 3; ++c)
                {
                    const float diff = texelsRGB[t][c] - palette[s][k][c];
                    d += diff * diff;
                }
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }

            indices[t] = bestK;
            totalError += bestDist;
        }

        // 4. Anchor constraints: 3 anchors (texel 0, anchor1, anchor2). MSB must be 0 (index <= 1).
        const uint8_t anchors[3] = { 0, g_bc7FixUp3Subsets[partition][0], g_bc7FixUp3Subsets[partition][1] };
        for (size_t s = 0; s < 3; ++s)
        {
            const uint8_t a = anchors[s];
            if (indices[a] >= 2)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    std::swap(ep5[s][0][c], ep5[s][1][c]);
                }
                for (size_t t = 0; t < 16; ++t)
                {
                    if (g_bc7PartitionTable3Subsets[partition][t] == s)
                    {
                        indices[t] = static_cast<uint8_t>(3u - indices[t]);
                    }
                }
            }
        }

        BC7Mode2FitResult result{};
        result.block = EmitBC7Mode2BlockScalar(partition, ep5, indices);
        result.totalError = totalError;
        return result;
    }

    struct BC7Mode7FitResult
    {
        XMUINT4 block;
        float totalError;
    };

    // Pack Mode 7 fields into one 128-bit BC7 block.
    inline XMUINT4 EmitBC7Mode7BlockScalar(
        uint32_t partition,
        const uint8_t ep[2][2][4], // [subset 0..1][ep 0..1][rgba 0..3] (5-bit values)
        const uint8_t pBit[2][2],  // [subset 0..1][ep 0..1]
        const uint8_t indices[16]) noexcept
    {
        uint8_t bytes[16]{};
        size_t bitOffset = 0;

        // Mode 7 prefix: 0b10000000 (bit 0..6 = 0, bit 7 = 1) -> 8 bits
        WriteBC7Bits(bytes, bitOffset, 0b10000000, 8);

        // Partition: 6 bits
        WriteBC7Bits(bytes, bitOffset, partition, 6);

        // Endpoints (R, G, B, A order, subset 0 ep 0..1, subset 1 ep 0..1)
        for (size_t c = 0; c < 4; ++c)
        {
            WriteBC7Bits(bytes, bitOffset, ep[0][0][c], 5);
            WriteBC7Bits(bytes, bitOffset, ep[0][1][c], 5);
            WriteBC7Bits(bytes, bitOffset, ep[1][0][c], 5);
            WriteBC7Bits(bytes, bitOffset, ep[1][1][c], 5);
        }

        // P-bits: 4 bits (subset 0 ep 0..1, subset 1 ep 0..1)
        WriteBC7Bits(bytes, bitOffset, pBit[0][0], 1);
        WriteBC7Bits(bytes, bitOffset, pBit[0][1], 1);
        WriteBC7Bits(bytes, bitOffset, pBit[1][0], 1);
        WriteBC7Bits(bytes, bitOffset, pBit[1][1], 1);

        // Indices (2 bits each, except anchor texels which have 1 bit)
        const uint8_t anchor1 = g_bc7FixUp2Subsets[partition];

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0 || t == anchor1) ? 1 : 2;
            WriteBC7Bits(bytes, bitOffset, indices[t], bitCount);
        }

        assert(bitOffset == 128);

        XMUINT4 result;
        memcpy(&result, bytes, sizeof(result));
        return result;
    }

    // Fit Mode 7 endpoints and 2-bit indices for a single lane, returning 128-bit block and exact reconstruction error.
    inline BC7Mode7FitResult FitBC7Mode7SingleLane(
        const BC7ChildCanvas& canvas,
        uint32_t partition,
        size_t lane) noexcept
    {
        alignas(16) float texelsRGBA[16][4];
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                texelsRGBA[t][c] = XMVectorGetByIndex(canvas.texels[t][c], lane);
            }
        }

        uint8_t ep5[2][2][4]{};
        uint8_t pBits[2][2]{};
        uint8_t unquantEP[2][2][4]{};

        // 1. Quantize endpoints for each of the 2 subsets (5-bit + 1 P-bit per endpoint = 6 bits: [0, 63])
        for (size_t s = 0; s < 2; ++s)
        {
            float minC[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
            float maxC[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            bool hasTexels = false;

            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == s)
                {
                    hasTexels = true;
                    for (size_t c = 0; c < 4; ++c)
                    {
                        if (texelsRGBA[t][c] < minC[c]) minC[c] = texelsRGBA[t][c];
                        if (texelsRGBA[t][c] > maxC[c]) maxC[c] = texelsRGBA[t][c];
                    }
                }
            }

            if (!hasTexels)
            {
                minC[0] = minC[1] = minC[2] = minC[3] = 0.0f;
                maxC[0] = maxC[1] = maxC[2] = maxC[3] = 0.0f;
            }

            for (size_t c = 0; c < 4; ++c)
            {
                const int q0 = std::min(63, std::max(0, static_cast<int>(std::round(minC[c] * 63.0f))));
                const int q1 = std::min(63, std::max(0, static_cast<int>(std::round(maxC[c] * 63.0f))));
                ep5[s][0][c] = static_cast<uint8_t>(q0 >> 1);
                ep5[s][1][c] = static_cast<uint8_t>(q1 >> 1);
                pBits[s][0] = static_cast<uint8_t>(q0 & 1);
                pBits[s][1] = static_cast<uint8_t>(q1 & 1);

                unquantEP[s][0][c] = UnquantizeBC7_5BitScalar(ep5[s][0][c], pBits[s][0]);
                unquantEP[s][1][c] = UnquantizeBC7_5BitScalar(ep5[s][1][c], pBits[s][1]);
            }
        }

        // 2. Build 4-color palette for each subset (2-bit weights: { 0, 21, 43, 64 })
        constexpr uint32_t weights[4] = { 0, 21, 43, 64 };
        float palette[2][4][4];
        for (size_t s = 0; s < 2; ++s)
        {
            for (uint32_t k = 0; k < 4; ++k)
            {
                for (size_t c = 0; c < 4; ++c)
                {
                    const uint32_t val = ((64u - weights[k]) * unquantEP[s][0][c] + weights[k] * unquantEP[s][1][c] + 32u) >> 6;
                    palette[s][k][c] = static_cast<float>(val) / 255.0f;
                }
            }
        }

        // 3. Assign nearest index and accumulate RGBA error
        uint8_t indices[16]{};
        float totalError = 0.0f;

        for (size_t t = 0; t < 16; ++t)
        {
            const size_t s = g_bc7PartitionTable2Subsets[partition][t];
            float bestDist = 1e10f;
            uint8_t bestK = 0;

            for (uint8_t k = 0; k < 4; ++k)
            {
                float d = 0.0f;
                for (size_t c = 0; c < 4; ++c)
                {
                    const float diff = texelsRGBA[t][c] - palette[s][k][c];
                    d += diff * diff;
                }
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }

            indices[t] = bestK;
            totalError += bestDist;
        }

        // 4. Anchor constraints:
        // Subset 0 anchor: texel 0. 1 bit (index <= 1).
        if (indices[0] >= 2)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                std::swap(ep5[0][0][c], ep5[0][1][c]);
            }
            std::swap(pBits[0][0], pBits[0][1]);
            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == 0)
                {
                    indices[t] = static_cast<uint8_t>(3u - indices[t]);
                }
            }
        }

        // Subset 1 anchor: texel anchor1. 1 bit (index <= 1).
        const uint8_t anchor1 = g_bc7FixUp2Subsets[partition];
        if (indices[anchor1] >= 2)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                std::swap(ep5[1][0][c], ep5[1][1][c]);
            }
            std::swap(pBits[1][0], pBits[1][1]);
            for (size_t t = 0; t < 16; ++t)
            {
                if (g_bc7PartitionTable2Subsets[partition][t] == 1)
                {
                    indices[t] = static_cast<uint8_t>(3u - indices[t]);
                }
            }
        }

        BC7Mode7FitResult result{};
        result.block = EmitBC7Mode7BlockScalar(partition, ep5, pBits, indices);
        result.totalError = totalError;
        return result;
    }

    struct BC7Mode4FitResult
    {
        XMUINT4 block;
        float totalError;
    };

    // Pack Mode 4 fields into one 128-bit BC7 block.
    inline XMUINT4 EmitBC7Mode4BlockScalar(
        const uint8_t epRGB[2][3], // [ep 0..1][rgb 0..2] (5-bit values)
        const uint8_t epA[2],      // [ep 0..1] (6-bit values)
        const uint8_t colorIndices[16], // 2-bit indices
        const uint8_t alphaIndices[16]) // 3-bit indices
    {
        uint8_t bytes[16]{};
        size_t bitOffset = 0;

        // Mode 4 prefix: 0b10000 (bit 0..3 = 0, bit 4 = 1) -> 5 bits
        WriteBC7Bits(bytes, bitOffset, 0b10000, 5);

        // Rotation: 2 bits (0)
        WriteBC7Bits(bytes, bitOffset, 0, 2);

        // Index Mode: 1 bit (0 = 2-bit RGB, 3-bit Alpha)
        WriteBC7Bits(bytes, bitOffset, 0, 1);

        // Endpoints RGB (5 bits each: r0, r1, g0, g1, b0, b1)
        for (size_t c = 0; c < 3; ++c)
        {
            WriteBC7Bits(bytes, bitOffset, epRGB[0][c], 5);
            WriteBC7Bits(bytes, bitOffset, epRGB[1][c], 5);
        }

        // Endpoints Alpha (6 bits each: a0, a1)
        WriteBC7Bits(bytes, bitOffset, epA[0], 6);
        WriteBC7Bits(bytes, bitOffset, epA[1], 6);

        // Stream 0: Color indices (texel 0 is 1 bit, others 2 bits)
        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0) ? 1 : 2;
            WriteBC7Bits(bytes, bitOffset, colorIndices[t], bitCount);
        }

        // Stream 1: Alpha indices (texel 0 is 2 bits, others 3 bits)
        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0) ? 2 : 3;
            WriteBC7Bits(bytes, bitOffset, alphaIndices[t], bitCount);
        }

        assert(bitOffset == 128);

        XMUINT4 result;
        memcpy(&result, bytes, sizeof(result));
        return result;
    }

    // Fit Mode 4 endpoints and dual indices for a single lane, returning 128-bit block and exact reconstruction error.
    inline BC7Mode4FitResult FitBC7Mode4SingleLane(
        const BC7ChildCanvas& canvas,
        size_t lane) noexcept
    {
        alignas(16) float texelsRGBA[16][4];
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                texelsRGBA[t][c] = XMVectorGetByIndex(canvas.texels[t][c], lane);
            }
        }

        // 1. Color: 5-bit endpoints without P-bit, 2-bit palette (weights: {0, 21, 43, 64})
        float minRGB[3] = { 1.0f, 1.0f, 1.0f };
        float maxRGB[3] = { 0.0f, 0.0f, 0.0f };
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                if (texelsRGBA[t][c] < minRGB[c]) minRGB[c] = texelsRGBA[t][c];
                if (texelsRGBA[t][c] > maxRGB[c]) maxRGB[c] = texelsRGBA[t][c];
            }
        }

        uint8_t epRGB[2][3]{};
        uint8_t unquantRGB[2][3]{};
        for (size_t c = 0; c < 3; ++c)
        {
            epRGB[0][c] = static_cast<uint8_t>(std::min(31, std::max(0, static_cast<int>(std::round(minRGB[c] * 31.0f)))));
            epRGB[1][c] = static_cast<uint8_t>(std::min(31, std::max(0, static_cast<int>(std::round(maxRGB[c] * 31.0f)))));
            unquantRGB[0][c] = UnquantizeBC7_5Bit_NoPBitScalar(epRGB[0][c]);
            unquantRGB[1][c] = UnquantizeBC7_5Bit_NoPBitScalar(epRGB[1][c]);
        }

        constexpr uint32_t weights2Bit[4] = { 0, 21, 43, 64 };
        float paletteRGB[4][3];
        for (uint32_t k = 0; k < 4; ++k)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                const uint32_t val = ((64u - weights2Bit[k]) * unquantRGB[0][c] + weights2Bit[k] * unquantRGB[1][c] + 32u) >> 6;
                paletteRGB[k][c] = static_cast<float>(val) / 255.0f;
            }
        }

        uint8_t colorIndices[16]{};
        float colorError = 0.0f;
        for (size_t t = 0; t < 16; ++t)
        {
            float bestDist = 1e10f;
            uint8_t bestK = 0;
            for (uint8_t k = 0; k < 4; ++k)
            {
                float d = 0.0f;
                for (size_t c = 0; c < 3; ++c)
                {
                    const float diff = texelsRGBA[t][c] - paletteRGB[k][c];
                    d += diff * diff;
                }
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }
            colorIndices[t] = bestK;
            colorError += bestDist;
        }

        if (colorIndices[0] >= 2)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                std::swap(epRGB[0][c], epRGB[1][c]);
            }
            for (size_t t = 0; t < 16; ++t)
            {
                colorIndices[t] = static_cast<uint8_t>(3u - colorIndices[t]);
            }
        }

        // 2. Alpha: 6-bit endpoints without P-bit, 3-bit palette (weights: {0, 9, 18, 27, 37, 46, 55, 64})
        float minA = 1.0f;
        float maxA = 0.0f;
        for (size_t t = 0; t < 16; ++t)
        {
            if (texelsRGBA[t][3] < minA) minA = texelsRGBA[t][3];
            if (texelsRGBA[t][3] > maxA) maxA = texelsRGBA[t][3];
        }

        uint8_t epA[2]{};
        epA[0] = static_cast<uint8_t>(std::min(63, std::max(0, static_cast<int>(std::round(minA * 63.0f)))));
        epA[1] = static_cast<uint8_t>(std::min(63, std::max(0, static_cast<int>(std::round(maxA * 63.0f)))));
        const uint8_t unquantA0 = UnquantizeBC7_6Bit_NoPBitScalar(epA[0]);
        const uint8_t unquantA1 = UnquantizeBC7_6Bit_NoPBitScalar(epA[1]);

        constexpr uint32_t weights3Bit[8] = { 0, 9, 18, 27, 37, 46, 55, 64 };
        float paletteA[8];
        for (uint32_t k = 0; k < 8; ++k)
        {
            const uint32_t val = ((64u - weights3Bit[k]) * unquantA0 + weights3Bit[k] * unquantA1 + 32u) >> 6;
            paletteA[k] = static_cast<float>(val) / 255.0f;
        }

        uint8_t alphaIndices[16]{};
        float alphaError = 0.0f;
        for (size_t t = 0; t < 16; ++t)
        {
            float bestDist = 1e10f;
            uint8_t bestK = 0;
            for (uint8_t k = 0; k < 8; ++k)
            {
                const float diff = texelsRGBA[t][3] - paletteA[k];
                const float d = diff * diff;
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }
            alphaIndices[t] = bestK;
            alphaError += bestDist;
        }

        if (alphaIndices[0] >= 4)
        {
            std::swap(epA[0], epA[1]);
            for (size_t t = 0; t < 16; ++t)
            {
                alphaIndices[t] = static_cast<uint8_t>(7u - alphaIndices[t]);
            }
        }

        BC7Mode4FitResult result{};
        result.block = EmitBC7Mode4BlockScalar(epRGB, epA, colorIndices, alphaIndices);
        result.totalError = colorError + alphaError;
        return result;
    }

    struct BC7Mode5FitResult
    {
        XMUINT4 block;
        float totalError;
    };

    // Pack Mode 5 fields into one 128-bit BC7 block.
    inline XMUINT4 EmitBC7Mode5BlockScalar(
        const uint8_t epRGB[2][3], // [ep 0..1][rgb 0..2] (7-bit values)
        const uint8_t epA[2],      // [ep 0..1] (8-bit values)
        const uint8_t colorIndices[16], // 2-bit indices
        const uint8_t alphaIndices[16]) // 2-bit indices
    {
        uint8_t bytes[16]{};
        size_t bitOffset = 0;

        // Mode 5 prefix: 0b100000 (bit 0..4 = 0, bit 5 = 1) -> 6 bits
        WriteBC7Bits(bytes, bitOffset, 0b100000, 6);

        // Rotation: 2 bits (0)
        WriteBC7Bits(bytes, bitOffset, 0, 2);

        // Endpoints RGB (7 bits each: r0, r1, g0, g1, b0, b1)
        for (size_t c = 0; c < 3; ++c)
        {
            WriteBC7Bits(bytes, bitOffset, epRGB[0][c], 7);
            WriteBC7Bits(bytes, bitOffset, epRGB[1][c], 7);
        }

        // Endpoints Alpha (8 bits each: a0, a1)
        WriteBC7Bits(bytes, bitOffset, epA[0], 8);
        WriteBC7Bits(bytes, bitOffset, epA[1], 8);

        // Stream 0: Color indices (texel 0 is 1 bit, others 2 bits)
        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0) ? 1 : 2;
            WriteBC7Bits(bytes, bitOffset, colorIndices[t], bitCount);
        }

        // Stream 1: Alpha indices (texel 0 is 1 bit, others 2 bits)
        for (size_t t = 0; t < 16; ++t)
        {
            const size_t bitCount = (t == 0) ? 1 : 2;
            WriteBC7Bits(bytes, bitOffset, alphaIndices[t], bitCount);
        }

        assert(bitOffset == 128);

        XMUINT4 result;
        memcpy(&result, bytes, sizeof(result));
        return result;
    }

    // Fit Mode 5 endpoints and dual 2-bit indices for a single lane, returning 128-bit block and exact reconstruction error.
    inline BC7Mode5FitResult FitBC7Mode5SingleLane(
        const BC7ChildCanvas& canvas,
        size_t lane) noexcept
    {
        alignas(16) float texelsRGBA[16][4];
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                texelsRGBA[t][c] = XMVectorGetByIndex(canvas.texels[t][c], lane);
            }
        }

        // 1. Color: 7-bit endpoints without P-bit, 2-bit palette (weights: {0, 21, 43, 64})
        float minRGB[3] = { 1.0f, 1.0f, 1.0f };
        float maxRGB[3] = { 0.0f, 0.0f, 0.0f };
        for (size_t t = 0; t < 16; ++t)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                if (texelsRGBA[t][c] < minRGB[c]) minRGB[c] = texelsRGBA[t][c];
                if (texelsRGBA[t][c] > maxRGB[c]) maxRGB[c] = texelsRGBA[t][c];
            }
        }

        uint8_t epRGB[2][3]{};
        uint8_t unquantRGB[2][3]{};
        for (size_t c = 0; c < 3; ++c)
        {
            epRGB[0][c] = static_cast<uint8_t>(std::min(127, std::max(0, static_cast<int>(std::round(minRGB[c] * 127.0f)))));
            epRGB[1][c] = static_cast<uint8_t>(std::min(127, std::max(0, static_cast<int>(std::round(maxRGB[c] * 127.0f)))));
            unquantRGB[0][c] = UnquantizeBC7_7Bit_NoPBitScalar(epRGB[0][c]);
            unquantRGB[1][c] = UnquantizeBC7_7Bit_NoPBitScalar(epRGB[1][c]);
        }

        constexpr uint32_t weights2Bit[4] = { 0, 21, 43, 64 };
        float paletteRGB[4][3];
        for (uint32_t k = 0; k < 4; ++k)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                const uint32_t val = ((64u - weights2Bit[k]) * unquantRGB[0][c] + weights2Bit[k] * unquantRGB[1][c] + 32u) >> 6;
                paletteRGB[k][c] = static_cast<float>(val) / 255.0f;
            }
        }

        uint8_t colorIndices[16]{};
        float colorError = 0.0f;
        for (size_t t = 0; t < 16; ++t)
        {
            float bestDist = 1e10f;
            uint8_t bestK = 0;
            for (uint8_t k = 0; k < 4; ++k)
            {
                float d = 0.0f;
                for (size_t c = 0; c < 3; ++c)
                {
                    const float diff = texelsRGBA[t][c] - paletteRGB[k][c];
                    d += diff * diff;
                }
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }
            colorIndices[t] = bestK;
            colorError += bestDist;
        }

        if (colorIndices[0] >= 2)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                std::swap(epRGB[0][c], epRGB[1][c]);
            }
            for (size_t t = 0; t < 16; ++t)
            {
                colorIndices[t] = static_cast<uint8_t>(3u - colorIndices[t]);
            }
        }

        // 2. Alpha: 8-bit endpoints directly, 2-bit palette (weights: {0, 21, 43, 64})
        float minA = 1.0f;
        float maxA = 0.0f;
        for (size_t t = 0; t < 16; ++t)
        {
            if (texelsRGBA[t][3] < minA) minA = texelsRGBA[t][3];
            if (texelsRGBA[t][3] > maxA) maxA = texelsRGBA[t][3];
        }

        uint8_t epA[2]{};
        epA[0] = static_cast<uint8_t>(std::min(255, std::max(0, static_cast<int>(std::round(minA * 255.0f)))));
        epA[1] = static_cast<uint8_t>(std::min(255, std::max(0, static_cast<int>(std::round(maxA * 255.0f)))));

        float paletteA[4];
        for (uint32_t k = 0; k < 4; ++k)
        {
            const uint32_t val = ((64u - weights2Bit[k]) * epA[0] + weights2Bit[k] * epA[1] + 32u) >> 6;
            paletteA[k] = static_cast<float>(val) / 255.0f;
        }

        uint8_t alphaIndices[16]{};
        float alphaError = 0.0f;
        for (size_t t = 0; t < 16; ++t)
        {
            float bestDist = 1e10f;
            uint8_t bestK = 0;
            for (uint8_t k = 0; k < 4; ++k)
            {
                const float diff = texelsRGBA[t][3] - paletteA[k];
                const float d = diff * diff;
                if (d < bestDist)
                {
                    bestDist = d;
                    bestK = k;
                }
            }
            alphaIndices[t] = bestK;
            alphaError += bestDist;
        }

        if (alphaIndices[0] >= 2)
        {
            std::swap(epA[0], epA[1]);
            for (size_t t = 0; t < 16; ++t)
            {
                alphaIndices[t] = static_cast<uint8_t>(3u - alphaIndices[t]);
            }
        }

        BC7Mode5FitResult result{};
        result.block = EmitBC7Mode5BlockScalar(epRGB, epA, colorIndices, alphaIndices);
        result.totalError = colorError + alphaError;
        return result;
    }

    // Estimate initial RGBA endpoints for Mode 6 using the canvas principal axis.
    inline BC7EndpointPairFloatBatch ComputeInitialEndpointsBC7Mode6PCA(const BC7ChildCanvas& canvas) noexcept
    {
        XMVECTOR minimum = XMVectorReplicate(10000.0f);
        XMVECTOR maximum = XMVectorReplicate(-10000.0f);

        for (size_t t = 0; t < 16; ++t)
        {
            XMVECTOR proj = XMVectorMultiply(canvas.axis.value[0], XMVectorSubtract(canvas.texels[t][0], canvas.mean.value[0]));
            proj = XMVectorMultiplyAdd(canvas.axis.value[1], XMVectorSubtract(canvas.texels[t][1], canvas.mean.value[1]), proj);
            proj = XMVectorMultiplyAdd(canvas.axis.value[2], XMVectorSubtract(canvas.texels[t][2], canvas.mean.value[2]), proj);
            proj = XMVectorMultiplyAdd(canvas.axis.value[3], XMVectorSubtract(canvas.texels[t][3], canvas.mean.value[3]), proj);
            minimum = XMVectorMin(minimum, proj);
            maximum = XMVectorMax(maximum, proj);
        }

        BC7EndpointPairFloatBatch result{};
        for (size_t c = 0; c < 4; ++c)
        {
            result.value[0][c] = XMVectorMultiplyAdd(canvas.axis.value[c], minimum, canvas.mean.value[c]);
            result.value[1][c] = XMVectorMultiplyAdd(canvas.axis.value[c], maximum, canvas.mean.value[c]);
        }

        return result;
    }

    // Quantize floating-point [0, 1] RGBA endpoints to 8-bit Mode 6 endpoints with shared P-bit.
    inline BC7EndpointPairBatch QuantizeBC7Mode6Endpoints(const BC7EndpointPairFloatBatch& floatEndpoints) noexcept
    {
        const XMVECTOR zero = XMVectorZero();
        const XMVECTOR one = XMVectorReplicate(1.0f);
        const XMVECTOR scale255 = XMVectorReplicate(255.0f);
        const XMVECTOR two = XMVectorReplicateInt(2);
        const XMVECTOR oneInt = XMVectorReplicateInt(1);
        const XMVECTOR clearLsb = XMVectorReplicateInt(0xFEu);

        BC7EndpointPairBatch result{};

        for (size_t ep = 0; ep < 2; ++ep)
        {
            XMVECTOR intChannels[4];
            XMVECTOR lsbs[4];

            for (size_t c = 0; c < 4; ++c)
            {
                const XMVECTOR clamped = XMVectorClamp(floatEndpoints.value[ep][c], zero, one);
                const XMVECTOR scaled = XMVectorRound(XMVectorMultiply(clamped, scale255));
                intChannels[c] = XMConvertVectorFloatToUInt(scaled, 0);
                lsbs[c] = XMVectorAndInt(intChannels[c], oneInt);
            }

            // Majority vote: if sum of LSBs > 2, pBit = 1, otherwise 0.
            const XMVECTOR sumLsb = AddInt32(AddInt32(lsbs[0], lsbs[1]), AddInt32(lsbs[2], lsbs[3]));
            const XMVECTOR pBit = XMVectorAndInt(GreaterInt32(sumLsb, two), oneInt);

            for (size_t c = 0; c < 4; ++c)
            {
                result.value[ep][c] = XMVectorOrInt(XMVectorAndInt(intChannels[c], clearLsb), pBit);
            }
        }

        return result;
    }

    // Assign 4-bit palette indices to sixteen child texels with anchor bit enforcement.
    inline BC7Mode6PackedIndexBatch AssignBC7Mode6Indices(
        const BC7ChildCanvas& canvas,
        BC7EndpointPairBatch& endpoints,
        FXMVECTOR activeMask,
        XMVECTOR* outTotalError = nullptr) noexcept
    {
        constexpr float colorScale = 1.0f / 255.0f;
        constexpr uint32_t weights[16] = { 0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64 };

        // 1. Generate the 16 hardware-rounded palette colors in [0, 1] float.
        XMVECTOR paletteColors[16][4];
        for (uint32_t k = 0; k < 16; ++k)
        {
            for (size_t c = 0; c < 4; ++c)
            {
                const XMVECTOR rounded = InterpolateBC7PaletteValue(endpoints.value[0][c], endpoints.value[1][c], weights[k]);
                paletteColors[k][c] = XMVectorMultiply(XMConvertVectorUIntToFloat(rounded, 0), XMVectorReplicate(colorScale));
            }
        }

        // 2. Find nearest palette color for each of the 16 child texels.
        XMVECTOR texelIndices[16];
        XMVECTOR totalError = XMVectorZero();

        for (size_t t = 0; t < 16; ++t)
        {
            // Distance to palette color 0
            XMVECTOR bestDist = XMVectorZero();
            for (size_t c = 0; c < 4; ++c)
            {
                const XMVECTOR diff = XMVectorSubtract(canvas.texels[t][c], paletteColors[0][c]);
                bestDist = XMVectorMultiplyAdd(diff, diff, bestDist);
            }
            XMVECTOR bestIndex = XMVectorZero();

            // Compare against palette colors 1..15
            for (uint32_t k = 1; k < 16; ++k)
            {
                XMVECTOR distK = XMVectorZero();
                for (size_t c = 0; c < 4; ++c)
                {
                    const XMVECTOR diff = XMVectorSubtract(canvas.texels[t][c], paletteColors[k][c]);
                    distK = XMVectorMultiplyAdd(diff, diff, distK);
                }

                const XMVECTOR closer = XMVectorLess(distK, bestDist);
                bestDist = XMVectorMin(bestDist, distK);
                bestIndex = XMVectorSelect(bestIndex, XMVectorReplicateInt(k), closer);
            }

            texelIndices[t] = bestIndex;
            totalError = XMVectorAdd(totalError, bestDist);
        }

        if (outTotalError)
        {
            *outTotalError = XMVectorAndInt(totalError, activeMask);
        }

        // 3. Anchor constraint: texel 0 MSB must be 0 (index <= 7).
        // If texel 0 index >= 8, swap endpoints and invert all indices (15 - index).
        const XMVECTOR anchorHasMsb = GreaterInt32(texelIndices[0], XMVectorReplicateInt(7));

        for (size_t c = 0; c < 4; ++c)
        {
            const XMVECTOR p0 = endpoints.value[0][c];
            const XMVECTOR p1 = endpoints.value[1][c];
            endpoints.value[0][c] = XMVectorSelect(p0, p1, anchorHasMsb);
            endpoints.value[1][c] = XMVectorSelect(p1, p0, anchorHasMsb);
        }

        const XMVECTOR fifteen = XMVectorReplicateInt(15);
        for (size_t t = 0; t < 16; ++t)
        {
            texelIndices[t] = XMVectorSelect(texelIndices[t], SubtractInt32(fifteen, texelIndices[t]), anchorHasMsb);
        }

        // 4. Pack 16 4-bit nibbles into low (0..7) and high (8..15).
        BC7Mode6PackedIndexBatch result{};

        result.low = texelIndices[0];
        result.low = XMVectorOrInt(result.low, ShiftLeft32<4>(texelIndices[1]));
        result.low = XMVectorOrInt(result.low, ShiftLeft32<8>(texelIndices[2]));
        result.low = XMVectorOrInt(result.low, ShiftLeft32<12>(texelIndices[3]));
        result.low = XMVectorOrInt(result.low, ShiftLeft32<16>(texelIndices[4]));
        result.low = XMVectorOrInt(result.low, ShiftLeft32<20>(texelIndices[5]));
        result.low = XMVectorOrInt(result.low, ShiftLeft32<24>(texelIndices[6]));
        result.low = XMVectorOrInt(result.low, ShiftLeft32<28>(texelIndices[7]));

        result.high = texelIndices[8];
        result.high = XMVectorOrInt(result.high, ShiftLeft32<4>(texelIndices[9]));
        result.high = XMVectorOrInt(result.high, ShiftLeft32<8>(texelIndices[10]));
        result.high = XMVectorOrInt(result.high, ShiftLeft32<12>(texelIndices[11]));
        result.high = XMVectorOrInt(result.high, ShiftLeft32<16>(texelIndices[12]));
        result.high = XMVectorOrInt(result.high, ShiftLeft32<20>(texelIndices[13]));
        result.high = XMVectorOrInt(result.high, ShiftLeft32<24>(texelIndices[14]));
        result.high = XMVectorOrInt(result.high, ShiftLeft32<28>(texelIndices[15]));

        result.low = XMVectorAndInt(result.low, activeMask);
        result.high = XMVectorAndInt(result.high, activeMask);

        return result;
    }

    // Pack Mode 6 endpoints and indices into four 128-bit BC7 blocks.
    inline BC7BlockBatch EmitBC7Mode6BlockBatch(
        const BC7EndpointPairBatch& endpoints,
        const BC7Mode6PackedIndexBatch& indices) noexcept
    {
        // Extract 7-bit endpoints
        const XMVECTOR r0 = ShiftRight32<1>(endpoints.value[0][0]);
        const XMVECTOR r1 = ShiftRight32<1>(endpoints.value[1][0]);
        const XMVECTOR g0 = ShiftRight32<1>(endpoints.value[0][1]);
        const XMVECTOR g1 = ShiftRight32<1>(endpoints.value[1][1]);
        const XMVECTOR b0 = ShiftRight32<1>(endpoints.value[0][2]);
        const XMVECTOR b1 = ShiftRight32<1>(endpoints.value[1][2]);
        const XMVECTOR a0 = ShiftRight32<1>(endpoints.value[0][3]);
        const XMVECTOR a1 = ShiftRight32<1>(endpoints.value[1][3]);

        // Extract P-bits
        const XMVECTOR oneInt = XMVectorReplicateInt(1);
        const XMVECTOR pBit0 = XMVectorAndInt(endpoints.value[0][0], oneInt);
        const XMVECTOR pBit1 = XMVectorAndInt(endpoints.value[1][0], oneInt);

        // Word 0: Mode 6 prefix (0x40) + R0 + R1 + G0 + low 4 bits of G1
        const XMVECTOR mode6Prefix = XMVectorReplicateInt(0x40u);
        const XMVECTOR mask4 = XMVectorReplicateInt(0x0Fu);
        XMVECTOR word0 = XMVectorOrInt(mode6Prefix, ShiftLeft32<7>(r0));
        word0 = XMVectorOrInt(word0, ShiftLeft32<14>(r1));
        word0 = XMVectorOrInt(word0, ShiftLeft32<21>(g0));
        word0 = XMVectorOrInt(word0, ShiftLeft32<28>(XMVectorAndInt(g1, mask4)));

        // Word 1: high 3 bits of G1 + B0 + B1 + A0 + A1 + pBit0
        XMVECTOR word1 = ShiftRight32<4>(g1);
        word1 = XMVectorOrInt(word1, ShiftLeft32<3>(b0));
        word1 = XMVectorOrInt(word1, ShiftLeft32<10>(b1));
        word1 = XMVectorOrInt(word1, ShiftLeft32<17>(a0));
        word1 = XMVectorOrInt(word1, ShiftLeft32<24>(a1));
        word1 = XMVectorOrInt(word1, ShiftLeft32<31>(pBit0));

        // Words 2 & 3: pBit1 + 63 index bits (with texel 0's MSB omitted)
        const XMVECTOR lowThreeBits = XMVectorReplicateInt(0x7u);
        const XMVECTOR rawLow0_2 = XMVectorAndInt(indices.low, lowThreeBits);
        const XMVECTOR rawLow3_30 = XMVectorAndCInt(ShiftRight32<1>(indices.low), lowThreeBits);
        const XMVECTOR rawLow31 = ShiftLeft32<31>(indices.high);
        const XMVECTOR rawLow = XMVectorOrInt(XMVectorOrInt(rawLow0_2, rawLow3_30), rawLow31);
        const XMVECTOR rawHigh = ShiftRight32<1>(indices.high);

        const XMVECTOR word2 = XMVectorOrInt(pBit1, ShiftLeft32<1>(rawLow));
        const XMVECTOR word3 = XMVectorOrInt(ShiftRight32<31>(rawLow), ShiftLeft32<1>(rawHigh));

        BC7BlockBatch result{};
        result.word0 = word0;
        result.word1 = word1;
        result.word2 = word2;
        result.word3 = word3;
        return result;
    }

    // Store four 128-bit BC7 blocks from SIMD vector lanes to destination memory.
    inline void StoreBC7BlockBatch(XMUINT4* blocks, const BC7BlockBatch& batch, size_t validLanes = 4) noexcept
    {
        assert(blocks != nullptr);
        assert(validLanes <= 4);
        const XMMATRIX transposed(batch.word0, batch.word1, batch.word2, batch.word3);
        const XMMATRIX blockRows = XMMatrixTranspose(transposed);

        if (validLanes > 0) XMStoreInt4(&blocks[0].x, blockRows.r[0]);
        if (validLanes > 1) XMStoreInt4(&blocks[1].x, blockRows.r[1]);
        if (validLanes > 2) XMStoreInt4(&blocks[2].x, blockRows.r[2]);
        if (validLanes > 3) XMStoreInt4(&blocks[3].x, blockRows.r[3]);
    }

    // Dedicated function to evaluate all candidate BC7 modes via Rate-Distortion MSE,
    // pick the winning mode, and store the resulting 128-bit block into destination memory.
    inline uint8_t FitAndStoreBC7ChildBlock(
        XMUINT4* destinationBlock,
        const BC7ChildCanvas& canvas,
        uint32_t part2S,
        uint32_t part3SM0,
        uint32_t part3SM2,
        const XMUINT4& mode6Block,
        float errorMode6,
        bool isOpaque,
        size_t lane) noexcept
    {
        assert(destinationBlock != nullptr);

        uint8_t selectedMode = 6;
        float minError = errorMode6;
        XMUINT4 winningBlock = mode6Block;

        if (isOpaque)
        {
            // 5-way Rate-Distortion showdown for opaque blocks: Mode 0, Mode 1, Mode 2, Mode 3, Mode 6
            const BC7Mode0FitResult m0 = FitBC7Mode0SingleLane(canvas, part3SM0, lane);
            const BC7Mode1FitResult m1 = FitBC7Mode1SingleLane(canvas, part2S, lane);
            const BC7Mode2FitResult m2 = FitBC7Mode2SingleLane(canvas, part3SM2, lane);
            const BC7Mode3FitResult m3 = FitBC7Mode3SingleLane(canvas, part2S, lane);

            if (m0.totalError < minError)
            {
                minError = m0.totalError;
                selectedMode = 0;
                winningBlock = m0.block;
            }
            if (m1.totalError < minError)
            {
                minError = m1.totalError;
                selectedMode = 1;
                winningBlock = m1.block;
            }
            if (m2.totalError < minError)
            {
                minError = m2.totalError;
                selectedMode = 2;
                winningBlock = m2.block;
            }
            if (m3.totalError < minError)
            {
                minError = m3.totalError;
                selectedMode = 3;
                winningBlock = m3.block;
            }
        }
        else
        {
            // 4-way Rate-Distortion showdown for translucent blocks: Mode 4, Mode 5, Mode 6, Mode 7
            const BC7Mode7FitResult m7 = FitBC7Mode7SingleLane(canvas, part2S, lane);
            const BC7Mode4FitResult m4 = FitBC7Mode4SingleLane(canvas, lane);
            const BC7Mode5FitResult m5 = FitBC7Mode5SingleLane(canvas, lane);

            if (m7.totalError < minError)
            {
                minError = m7.totalError;
                selectedMode = 7;
                winningBlock = m7.block;
            }
            if (m4.totalError < minError)
            {
                minError = m4.totalError;
                selectedMode = 4;
                winningBlock = m4.block;
            }
            if (m5.totalError < minError)
            {
                minError = m5.totalError;
                selectedMode = 5;
                winningBlock = m5.block;
            }
        }

        // Store the winning mode's 128-bit block directly into destination memory
        *destinationBlock = winningBlock;

        return selectedMode;
    }

    // Process one mip-1 block row directly from compressed BC7 Mode 6 parent blocks.
    inline void ProcessCompressedRowBC7(
        const Image& source,
        Image& destination,
        size_t destinationRow,
        BC7BlockMean* sourceBlockMeans = nullptr) noexcept
    {
        constexpr size_t laneCount = 4;
        const size_t sourceBlockWidth = std::max<size_t>(1, (source.width + 3) / 4);
        const size_t sourceBlockHeight = std::max<size_t>(1, (source.height + 3) / 4);
        const size_t destinationBlockWidth = std::max<size_t>(1, (destination.width + 3) / 4);

        // One destination block covers a 2x2 group of compressed source blocks.
        const size_t sourceY0 = std::min(destinationRow * 2, sourceBlockHeight - 1);
        const size_t sourceY1 = std::min(sourceY0 + 1, sourceBlockHeight - 1);
        const auto* sourceRow0 = reinterpret_cast<const XMUINT4*>(source.pixels + sourceY0 * source.rowPitch);
        const auto* sourceRow1 = reinterpret_cast<const XMUINT4*>(source.pixels + sourceY1 * source.rowPitch);
        auto* destinationBlocks = reinterpret_cast<XMUINT4*>(destination.pixels + destinationRow * destination.rowPitch);

        for (size_t destinationX = 0; destinationX < destinationBlockWidth; destinationX += laneCount)
        {
            XMUINT4 p00Blocks[laneCount]{};
            XMUINT4 p10Blocks[laneCount]{};
            XMUINT4 p01Blocks[laneCount]{};
            XMUINT4 p11Blocks[laneCount]{};
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

            const XMVECTOR activeMask = XMVectorSetInt(
                0 < validLanes ? 0xFFFFFFFFu : 0,
                1 < validLanes ? 0xFFFFFFFFu : 0,
                2 < validLanes ? 0xFFFFFFFFu : 0,
                3 < validLanes ? 0xFFFFFFFFu : 0
            );

            // 1. Load 4 parent blocks per lane
            const BC7BlockBatch b00 = LoadBC7BlockBatch(p00Blocks);
            const BC7BlockBatch b10 = LoadBC7BlockBatch(p10Blocks);
            const BC7BlockBatch b01 = LoadBC7BlockBatch(p01Blocks);
            const BC7BlockBatch b11 = LoadBC7BlockBatch(p11Blocks);

            // 2. Compute exact UNORM quadrant means across parent modes (Mode 1, Mode 6)
            const BC7QuadrantMeanBatch q00 = ComputeBC7ParentQuadrantMeans(b00, activeMask);
            const BC7QuadrantMeanBatch q10 = ComputeBC7ParentQuadrantMeans(b10, activeMask);
            const BC7QuadrantMeanBatch q01 = ComputeBC7ParentQuadrantMeans(b01, activeMask);
            const BC7QuadrantMeanBatch q11 = ComputeBC7ParentQuadrantMeans(b11, activeMask);

            // 3. Build child canonical canvas (16 texels, moments, 4D axis, opacity)
            BC7SourceBlockMeansBatch sourceMeans{};
            const BC7ChildCanvas canvas = BuildBC7ChildCanvas(q00, q10, q01, q11, sourceMeans);

            // 4. Fit Mode 6 endpoints and 4-bit indices
            const BC7EndpointPairFloatBatch floatEndpoints = ComputeInitialEndpointsBC7Mode6PCA(canvas);
            BC7EndpointPairBatch quantizedEndpoints = QuantizeBC7Mode6Endpoints(floatEndpoints);
            XMVECTOR errorMode6 = XMVectorZero();
            const BC7Mode6PackedIndexBatch childIndices = AssignBC7Mode6Indices(canvas, quantizedEndpoints, activeMask, &errorMode6);
            const BC7BlockBatch mode6Encoded = EmitBC7Mode6BlockBatch(quantizedEndpoints, childIndices);

            alignas(16) XMUINT4 mode6Blocks[4]{};
            StoreBC7BlockBatch(mode6Blocks, mode6Encoded, validLanes);

            // 5. FasTC-style 2-subset and 3-subset partition estimation (arg-min Hamming distance, 0 thresholds)
            const XMVECTOR part2S = EstimateBC7Partition2Subsets(canvas);
            alignas(16) uint32_t part2SLanes[4]{};
            _mm_store_si128(reinterpret_cast<__m128i*>(part2SLanes), _mm_castps_si128(part2S));

            const BC7Partition3SubsetsResult part3S = EstimateBC7Partition3Subsets(canvas);
            alignas(16) uint32_t part3SMode0Lanes[4]{};
            alignas(16) uint32_t part3SMode2Lanes[4]{};
            _mm_store_si128(reinterpret_cast<__m128i*>(part3SMode0Lanes), _mm_castps_si128(part3S.mode0Shape));
            _mm_store_si128(reinterpret_cast<__m128i*>(part3SMode2Lanes), _mm_castps_si128(part3S.mode2Shape));

            alignas(16) uint32_t opaqueLanes[4]{};
            _mm_store_si128(reinterpret_cast<__m128i*>(opaqueLanes), _mm_castps_si128(canvas.isOpaque));

            // 6. Pure Rate-Distortion MSE comparison: min(E_0, E_1, E_2, E_3, E_6) without arbitrary thresholds
            // 6. Rate-Distortion mode evaluation and child block storage for all lanes
            for (size_t lane = 0; lane < validLanes; ++lane)
            {
                const bool isOpaque = (opaqueLanes[lane] != 0);
                const float e6 = XMVectorGetByIndex(errorMode6, lane);

                if (isOpaque)
                {
                    const BC7Mode0FitResult m0 = FitBC7Mode0SingleLane(canvas, part3SMode0Lanes[lane], lane);
                    const BC7Mode1FitResult m1 = FitBC7Mode1SingleLane(canvas, part2SLanes[lane], lane);
                    const BC7Mode2FitResult m2 = FitBC7Mode2SingleLane(canvas, part3SMode2Lanes[lane], lane);
                    const BC7Mode3FitResult m3 = FitBC7Mode3SingleLane(canvas, part2SLanes[lane], lane);

                    float minError = e6;
                    XMUINT4 bestBlock = mode6Blocks[lane];

                    if (m0.totalError < minError)
                    {
                        minError = m0.totalError;
                        bestBlock = m0.block;
                    }
                    if (m1.totalError < minError)
                    {
                        minError = m1.totalError;
                        bestBlock = m1.block;
                    }
                    if (m2.totalError < minError)
                    {
                        minError = m2.totalError;
                        bestBlock = m2.block;
                    }
                    if (m3.totalError < minError)
                    {
                        minError = m3.totalError;
                        bestBlock = m3.block;
                    }

                    destinationBlocks[destinationX + lane] = bestBlock;
                    continue;
                }

                // Block has varying alpha: 4-way Rate-Distortion MSE comparison min(E_4, E_5, E_6, E_7)
                const BC7Mode7FitResult m7 = FitBC7Mode7SingleLane(canvas, part2SLanes[lane], lane);
                const BC7Mode4FitResult m4 = FitBC7Mode4SingleLane(canvas, lane);
                const BC7Mode5FitResult m5 = FitBC7Mode5SingleLane(canvas, lane);

                float minAlphaError = e6;
                XMUINT4 bestAlphaBlock = mode6Blocks[lane];

                if (m7.totalError < minAlphaError)
                {
                    minAlphaError = m7.totalError;
                    bestAlphaBlock = m7.block;
                }
                if (m4.totalError < minAlphaError)
                {
                    minAlphaError = m4.totalError;
                    bestAlphaBlock = m4.block;
                }
                if (m5.totalError < minAlphaError)
                {
                    minAlphaError = m5.totalError;
                    bestAlphaBlock = m5.block;
                }

                destinationBlocks[destinationX + lane] = bestAlphaBlock;
                FitAndStoreBC7ChildBlock(
                    &destinationBlocks[destinationX + lane],
                    canvas,
                    part2SLanes[lane],
                    part3SMode0Lanes[lane],
                    part3SMode2Lanes[lane],
                    mode6Blocks[lane],
                    e6,
                    isOpaque,
                    lane
                );
            }
        }
    }

    // Verify that all blocks in the image use supported BC7 modes (Mode 0 through 7).
    inline bool IsSupportedBC7Image(const Image& image) noexcept
    {
        const size_t blockWidth = std::max<size_t>(1, (image.width + 3) / 4);
        const size_t blockHeight = std::max<size_t>(1, (image.height + 3) / 4);

        for (size_t y = 0; y < blockHeight; ++y)
        {
            const auto* row = reinterpret_cast<const uint8_t*>(image.pixels + y * image.rowPitch);
            for (size_t x = 0; x < blockWidth; ++x)
            {
                const uint8_t mode = GetBC7Mode(row + x * 16);
                if (mode >= 8)
                {
                    return false;
                }
            }
        }

        return true;
    }

    // Generate all compressed BC7 mip levels after level zero.
    inline HRESULT GenerateCompressedMipMapsBC7(const Image& baseImage, ScratchImage& mipChain) noexcept
    {
        const size_t mipLevels = mipChain.GetMetadata().mipLevels;
        if (mipLevels <= 1)
        {
            return S_OK;
        }

        for (size_t mipLevel = 1; mipLevel < mipLevels; ++mipLevel)
        {
            const Image* source = (mipLevel == 1) ? &baseImage : mipChain.GetImage(mipLevel - 1, 0, 0);
            Image* destination = const_cast<Image*>(mipChain.GetImage(mipLevel, 0, 0));
            if (!source || !source->pixels || !destination || !destination->pixels)
            {
                return E_FAIL;
            }

            const size_t destinationBlockHeight = std::max<size_t>(1, (destination->height + 3) / 4);

        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
            for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(destinationBlockHeight); ++row)
            {
                ProcessCompressedRowBC7(*source, *destination, static_cast<size_t>(row));
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

    // TODO: Add support for BC2, BC3, BC4, BC5, BC6H here in future extensions.
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
    case DXGI_FORMAT_BC7_UNORM_SRGB:
        return GenerateCompressedMipMapsBC7(baseImage, mipChain);

    // TODO: Add case dispatches for BC2, BC3, BC4, BC5 here.
    default:
        return HRESULT_E_NOT_SUPPORTED;
    }
}

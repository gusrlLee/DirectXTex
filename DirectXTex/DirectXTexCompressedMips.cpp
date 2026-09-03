#include "DirectXTexP.h"
#include "BC.h"

// Standard containers and allocation helpers used by the mean-image pyramid.
#include <algorithm>
#include <array>
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

    // Endpoints for two-subset modes (Mode 1, Mode 7).
    // value[subsetIndex][endpointIndex][channelIndex] (RGBA 8-bit integers).
    struct BC7TwoSubsetEndpointBatch
    {
        XMVECTOR value[2][2][4];
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

    // Mode 6 data reduced to an endpoint basis and four quadrant symbols.
    struct BC7Mode6SymbolBatch
    {
        XMVECTOR activeMask;
        BC7EndpointPairBatch endpoints;
        BC7Mode6PackedIndexBatch indices;
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

    // Extract 6-bit partition shape index (0..63) for 2-subset modes (Mode 1, Mode 7).
    inline XMVECTOR ExtractBC7Partition(const BC7BlockBatch& blocks, FXMVECTOR activeMask) noexcept
    {
        return XMVectorAndInt(ExtractBC7Bits<2, 6>(blocks), activeMask);
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

    // Build one rounded BC7 palette value for four blocks at once.
    inline XMVECTOR InterpolateBC7PaletteValue(FXMVECTOR endpoint0, FXMVECTOR endpoint1, uint32_t weight) noexcept
    {
        assert(weight <= 64);

        const XMVECTOR endpoint0Float = XMConvertVectorUIntToFloat(endpoint0, 0);
        const XMVECTOR endpoint1Float = XMConvertVectorUIntToFloat(endpoint1, 0);
        const XMVECTOR numerator = XMVectorMultiplyAdd(XMVectorSubtract(endpoint1Float, endpoint0Float), XMVectorReplicate(static_cast<float>(weight)), XMVectorScale(endpoint0Float, 64.0f));
        const XMVECTOR rounded = XMVectorTruncate(XMVectorScale(XMVectorAdd(numerator, XMVectorReplicate(32.0f)), 1.0f / 64.0f));
        return XMConvertVectorFloatToUInt(rounded, 0);
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

    inline BC7Mode6SymbolBatch ExtractBC7Mode6Symbols(const BC7BlockBatch& blocks) noexcept
    {
        const BC7ModeMaskBatch modeMasks = GetBC7ModeMasks(blocks);

        BC7Mode6SymbolBatch result{};
        result.activeMask = modeMasks.mode[6];
        result.endpoints = ExtractBC7Mode6Endpoints(blocks, result.activeMask);
        result.indices = ExtractBC7Mode6PackedIndices(blocks, result.activeMask);
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

    // Expand the 4D projection range with one RGBA quadrant sample.
    inline void ExpandBC7ProjectionRange(
        const XMVECTOR sample[4],
        const BC7BlockMeanBatch& axis,
        const BC7BlockMeanBatch& mean,
        XMVECTOR& minimum,
        XMVECTOR& maximum) noexcept
    {
        XMVECTOR proj = XMVectorMultiply(axis.value[0], XMVectorSubtract(sample[0], mean.value[0]));
        proj = XMVectorMultiplyAdd(axis.value[1], XMVectorSubtract(sample[1], mean.value[1]), proj);
        proj = XMVectorMultiplyAdd(axis.value[2], XMVectorSubtract(sample[2], mean.value[2]), proj);
        proj = XMVectorMultiplyAdd(axis.value[3], XMVectorSubtract(sample[3], mean.value[3]), proj);
        minimum = XMVectorMin(minimum, proj);
        maximum = XMVectorMax(maximum, proj);
    }

    // Expand the 4D projection range with all four quadrants of one parent block.
    inline void ExpandBC7ParentProjectionRange(
        const BC7QuadrantMeanBatch& parent,
        const BC7BlockMeanBatch& axis,
        const BC7BlockMeanBatch& mean,
        XMVECTOR& minimum,
        XMVECTOR& maximum) noexcept
    {
        for (size_t q = 0; q < 4; ++q)
        {
            ExpandBC7ProjectionRange(parent.value[q], axis, mean, minimum, maximum);
        }
    }

    // Estimate the 4D principal axis and initial RGBA endpoints using Power Iteration PCA.
    inline BC7EndpointPairFloatBatch ComputeInitialEndpointsBC7Mode6PCA(
        const BC7CovarianceMatrixBatch& covariance,
        const BC7BlockMeanBatch& mean,
        const BC7QuadrantMeanBatch& p00,
        const BC7QuadrantMeanBatch& p10,
        const BC7QuadrantMeanBatch& p01,
        const BC7QuadrantMeanBatch& p11) noexcept
    {
        // Begin power iteration with a normalized diagonal direction in 4D (RGBA) space.
        // In 4D, (0.5, 0.5, 0.5, 0.5) has length 1.0 (0.5^2 * 4 = 1.0).
        BC7BlockMeanBatch axis{};
        const XMVECTOR diag = XMVectorReplicate(0.5f);
        axis.value[0] = diag;
        axis.value[1] = diag;
        axis.value[2] = diag;
        axis.value[3] = diag;

        // Multiply the initial direction by the 4x4 covariance matrix once to approach principal axis.
        BC7BlockMeanBatch next{};
        next.value[0] = XMVectorMultiply(covariance.rr, axis.value[0]);
        next.value[0] = XMVectorMultiplyAdd(covariance.rg, axis.value[1], next.value[0]);
        next.value[0] = XMVectorMultiplyAdd(covariance.rb, axis.value[2], next.value[0]);
        next.value[0] = XMVectorMultiplyAdd(covariance.ra, axis.value[3], next.value[0]);

        next.value[1] = XMVectorMultiply(covariance.rg, axis.value[0]);
        next.value[1] = XMVectorMultiplyAdd(covariance.gg, axis.value[1], next.value[1]);
        next.value[1] = XMVectorMultiplyAdd(covariance.gb, axis.value[2], next.value[1]);
        next.value[1] = XMVectorMultiplyAdd(covariance.ga, axis.value[3], next.value[1]);

        next.value[2] = XMVectorMultiply(covariance.rb, axis.value[0]);
        next.value[2] = XMVectorMultiplyAdd(covariance.gb, axis.value[1], next.value[2]);
        next.value[2] = XMVectorMultiplyAdd(covariance.bb, axis.value[2], next.value[2]);
        next.value[2] = XMVectorMultiplyAdd(covariance.ba, axis.value[3], next.value[2]);

        next.value[3] = XMVectorMultiply(covariance.ra, axis.value[0]);
        next.value[3] = XMVectorMultiplyAdd(covariance.ga, axis.value[1], next.value[3]);
        next.value[3] = XMVectorMultiplyAdd(covariance.ba, axis.value[2], next.value[3]);
        next.value[3] = XMVectorMultiplyAdd(covariance.aa, axis.value[3], next.value[3]);

        // Normalize the estimated 4D axis; epsilon keeps flat-color blocks finite.
        XMVECTOR lengthSquared = XMVectorMultiply(next.value[0], next.value[0]);
        lengthSquared = XMVectorMultiplyAdd(next.value[1], next.value[1], lengthSquared);
        lengthSquared = XMVectorMultiplyAdd(next.value[2], next.value[2], lengthSquared);
        lengthSquared = XMVectorMultiplyAdd(next.value[3], next.value[3], lengthSquared);
        lengthSquared = XMVectorAdd(lengthSquared, XMVectorReplicate(1e-20f));

        const XMVECTOR inverseLength = XMVectorReciprocalSqrt(lengthSquared);
        axis.value[0] = XMVectorMultiply(next.value[0], inverseLength);
        axis.value[1] = XMVectorMultiply(next.value[1], inverseLength);
        axis.value[2] = XMVectorMultiply(next.value[2], inverseLength);
        axis.value[3] = XMVectorMultiply(next.value[3], inverseLength);

        // Project all 16 quadrant means and retain the minimum and maximum positions.
        XMVECTOR minimum = XMVectorReplicate(10000.0f);
        XMVECTOR maximum = XMVectorReplicate(-10000.0f);

        ExpandBC7ParentProjectionRange(p00, axis, mean, minimum, maximum);
        ExpandBC7ParentProjectionRange(p10, axis, mean, minimum, maximum);
        ExpandBC7ParentProjectionRange(p01, axis, mean, minimum, maximum);
        ExpandBC7ParentProjectionRange(p11, axis, mean, minimum, maximum);

        // Convert the two extreme scalar projections back into 4D endpoint candidates.
        BC7EndpointPairFloatBatch result{};
        for (size_t c = 0; c < 4; ++c)
        {
            result.value[0][c] = XMVectorMultiplyAdd(axis.value[c], minimum, mean.value[c]);
            result.value[1][c] = XMVectorMultiplyAdd(axis.value[c], maximum, mean.value[c]);
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

    // Map child texel index (0..15) to parent block quadrant.
    inline void GetBC7ChildTexel(
        const BC7QuadrantMeanBatch& p00,
        const BC7QuadrantMeanBatch& p10,
        const BC7QuadrantMeanBatch& p01,
        const BC7QuadrantMeanBatch& p11,
        size_t texelIndex,
        XMVECTOR texelOut[4]) noexcept
    {
        const BC7QuadrantMeanBatch* parent;
        size_t quadrant;

        switch (texelIndex)
        {
        case 0:  parent = &p00; quadrant = 0; break;
        case 1:  parent = &p00; quadrant = 1; break;
        case 2:  parent = &p10; quadrant = 0; break;
        case 3:  parent = &p10; quadrant = 1; break;
        case 4:  parent = &p00; quadrant = 2; break;
        case 5:  parent = &p00; quadrant = 3; break;
        case 6:  parent = &p10; quadrant = 2; break;
        case 7:  parent = &p10; quadrant = 3; break;
        case 8:  parent = &p01; quadrant = 0; break;
        case 9:  parent = &p01; quadrant = 1; break;
        case 10: parent = &p11; quadrant = 0; break;
        case 11: parent = &p11; quadrant = 1; break;
        case 12: parent = &p01; quadrant = 2; break;
        case 13: parent = &p01; quadrant = 3; break;
        case 14: parent = &p11; quadrant = 2; break;
        case 15: default: parent = &p11; quadrant = 3; break;
        }

        texelOut[0] = parent->value[quadrant][0];
        texelOut[1] = parent->value[quadrant][1];
        texelOut[2] = parent->value[quadrant][2];
        texelOut[3] = parent->value[quadrant][3];
    }

    // Assign 4-bit palette indices to sixteen child texels with anchor bit enforcement.
    inline BC7Mode6PackedIndexBatch AssignBC7Mode6Indices(
        const BC7QuadrantMeanBatch& p00,
        const BC7QuadrantMeanBatch& p10,
        const BC7QuadrantMeanBatch& p01,
        const BC7QuadrantMeanBatch& p11,
        BC7EndpointPairBatch& endpoints,
        FXMVECTOR activeMask) noexcept
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

        for (size_t t = 0; t < 16; ++t)
        {
            XMVECTOR texel[4];
            GetBC7ChildTexel(p00, p10, p01, p11, t, texel);

            // Distance to palette color 0
            XMVECTOR bestDist = XMVectorZero();
            for (size_t c = 0; c < 4; ++c)
            {
                const XMVECTOR diff = XMVectorSubtract(texel[c], paletteColors[0][c]);
                bestDist = XMVectorMultiplyAdd(diff, diff, bestDist);
            }
            XMVECTOR bestIndex = XMVectorZero();

            // Compare against palette colors 1..15
            for (uint32_t k = 1; k < 16; ++k)
            {
                XMVECTOR distK = XMVectorZero();
                for (size_t c = 0; c < 4; ++c)
                {
                    const XMVECTOR diff = XMVectorSubtract(texel[c], paletteColors[k][c]);
                    distK = XMVectorMultiplyAdd(diff, diff, distK);
                }

                const XMVECTOR closer = XMVectorLess(distK, bestDist);
                bestDist = XMVectorMin(bestDist, distK);
                bestIndex = XMVectorSelect(bestIndex, XMVectorReplicateInt(k), closer);
            }

            texelIndices[t] = bestIndex;
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

            // 2. Extract Mode 6 symbols & compute exact UNORM quadrant means
            const BC7Mode6SymbolBatch s00 = ExtractBC7Mode6Symbols(b00);
            const BC7Mode6SymbolBatch s10 = ExtractBC7Mode6Symbols(b10);
            const BC7Mode6SymbolBatch s01 = ExtractBC7Mode6Symbols(b01);
            const BC7Mode6SymbolBatch s11 = ExtractBC7Mode6Symbols(b11);

            const BC7QuadrantMeanBatch q00 = ComputeBC7Mode6QuadrantMeans(s00.endpoints, s00.indices, activeMask);
            const BC7QuadrantMeanBatch q10 = ComputeBC7Mode6QuadrantMeans(s10.endpoints, s10.indices, activeMask);
            const BC7QuadrantMeanBatch q01 = ComputeBC7Mode6QuadrantMeans(s01.endpoints, s01.indices, activeMask);
            const BC7QuadrantMeanBatch q11 = ComputeBC7Mode6QuadrantMeans(s11.endpoints, s11.indices, activeMask);

            // 3. Child block moments (ANOVA identity)
            BC7SourceBlockMeansBatch sourceMeans{};
            BC7BlockMeanBatch childMean{};
            BC7CovarianceMatrixBatch childCovariance{};
            ComputeBC7ChildBlockMoments(q00, q10, q01, q11, sourceMeans, childMean, childCovariance);

            // 4. Estimate PCA initial endpoints in 4D
            const BC7EndpointPairFloatBatch floatEndpoints = ComputeInitialEndpointsBC7Mode6PCA(childCovariance, childMean, q00, q10, q01, q11);

            // 5. Quantize endpoints to 7-bit with majority-voted P-bits
            BC7EndpointPairBatch quantizedEndpoints = QuantizeBC7Mode6Endpoints(floatEndpoints);

            // 6. Assign 4-bit indices to the 16 child texels with anchor bit enforcement
            const BC7Mode6PackedIndexBatch childIndices = AssignBC7Mode6Indices(q00, q10, q01, q11, quantizedEndpoints, activeMask);

            // 7. Emit 128-bit Mode 6 blocks and store
            const BC7BlockBatch encoded = EmitBC7Mode6BlockBatch(quantizedEndpoints, childIndices);
            StoreBC7BlockBatch(destinationBlocks + destinationX, encoded, validLanes);
        }
    }

    // Verify that all blocks in the image use BC7 Mode 6.
    inline bool IsMode6BC7Image(const Image& image) noexcept
    {
        const size_t blockWidth = std::max<size_t>(1, (image.width + 3) / 4);
        const size_t blockHeight = std::max<size_t>(1, (image.height + 3) / 4);

        for (size_t y = 0; y < blockHeight; ++y)
        {
            const auto* row = reinterpret_cast<const uint8_t*>(image.pixels + y * image.rowPitch);
            for (size_t x = 0; x < blockWidth; ++x)
            {
                if (GetBC7Mode(row + x * 16) != 6)
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
        // For BC7, verify that all blocks use Mode 6.
        if (!IsMode6BC7Image(baseImage))
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

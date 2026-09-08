#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Shared parameter structs — layouts MUST stay in sync with the matching
// Swift structs in MetalMoshRenderer.swift (field order and types).
// ============================================================================

struct MotionSearchParams {
    uint2 lowResSize;     // size of the downsampled luma textures
    int   searchRadius;   // +/- pixels searched, in low-res space
    int   blockSize;      // block edge length, in low-res space
};

struct CompositeParams {
    uint2 fullResSize;
    uint2 lowResSize;
    int   blockSize;        // low-res block size used during search
    int   fullResPerBlock;  // full-res pixels per low-res block edge
    float residualGain;     // 0...2, "how much of the new footage bleeds through"
};

inline float luma(float4 c) {
    return dot(c.rgb, float3(0.299, 0.587, 0.114));
}

// ============================================================================
// Pass 1 — downsample + convert to luma.
// Motion estimation doesn't need full color or full resolution; running it
// on a quarter-res luma image cuts the search cost by roughly 16x with
// negligible impact on the visual result.
// ============================================================================

kernel void downsampleLuma(texture2d<float, access::sample> src [[texture(0)]],
                           texture2d<float, access::write> dst [[texture(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;

    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float4 c = src.sample(s, uv);
    dst.write(float4(luma(c), 0, 0, 1), gid);
}

// ============================================================================
// Pass 2 — block motion search.
// A straightforward full-search block matcher, minimizing sum-of-absolute-
// differences. This stands in for the motion-estimation stage of a real
// H.264 encoder: for each block of the CURRENT live frame, it finds the
// offset into the PREVIOUS live frame that best explains it. Those vectors
// are exactly the kind of data a real encoder would package into a P-frame.
//
// Note: this is intentionally the simplest possible correct implementation
// (full search, no early-out). It is the first thing to optimize — a
// diamond or three-step search, or a smaller radius/block grid, buys back
// meaningful GPU time on older devices. See README "Performance tuning".
// ============================================================================

kernel void blockMotionSearch(texture2d<float, access::sample> curLuma [[texture(0)]],
                              texture2d<float, access::sample> prevLuma [[texture(1)]],
                              texture2d<float, access::write> mvOut [[texture(2)]],
                              constant MotionSearchParams &p [[buffer(0)]],
                              uint2 blockId [[thread_position_in_grid]]) {
    uint blocksX = (p.lowResSize.x + uint(p.blockSize) - 1) / uint(p.blockSize);
    uint blocksY = (p.lowResSize.y + uint(p.blockSize) - 1) / uint(p.blockSize);
    if (blockId.x >= blocksX || blockId.y >= blocksY) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);

    int originX = int(blockId.x) * p.blockSize;
    int originY = int(blockId.y) * p.blockSize;

    float bestSAD = INFINITY;
    int2 bestOffset = int2(0, 0);

    for (int dy = -p.searchRadius; dy <= p.searchRadius; dy++) {
        for (int dx = -p.searchRadius; dx <= p.searchRadius; dx++) {
            float sad = 0.0;
            for (int by = 0; by < p.blockSize; by++) {
                for (int bx = 0; bx < p.blockSize; bx++) {
                    float2 curPos = float2(originX + bx, originY + by);
                    float2 refPos = curPos + float2(dx, dy);
                    float curVal = curLuma.sample(s, curPos).r;
                    float refVal = prevLuma.sample(s, refPos).r;
                    sad += fabs(curVal - refVal);
                }
            }
            if (sad < bestSAD) {
                bestSAD = sad;
                bestOffset = int2(dx, dy);
            }
        }
    }

    mvOut.write(float4(float(bestOffset.x), float(bestOffset.y), 0, 1), blockId);
}

// ============================================================================
// Pass 3 — motion-compensated composite. THIS is the datamosh effect.
//
// A real decoder reconstructs a P-frame as:
//     decoded = motionCompensate(reference, mv) + residual
// where `residual` was encoded as (trueCurrentFrame - motionCompensate(the
// encoder's OWN reference, mv)).
//
// Datamoshing happens when the decoder is fed that residual but its
// reference picture buffer holds the WRONG frame (because the real IDR that
// should have reset it was deleted from the stream). This kernel reproduces
// that substitution directly:
//
//   1. `mv` was estimated between this live frame and the PREVIOUS live
//      frame of the same clip — i.e. genuine motion data for clip B.
//   2. `predictedFromB` = motion-compensating clip B's own previous frame.
//      `residual` = currentColor - predictedFromB — the detail clip B's
//      motion alone doesn't explain (new objects, disocclusion, lighting).
//   3. `predictedFromA` = motion-compensating `canvas`, which is clip A's
//      frozen last frame (or an already-corrupted descendant of it) — the
//      reference that was never refreshed with clip B's own I-frame.
//   4. moshed = predictedFromA + residual.
//
// The result is clip A's pixels dragged around by clip B's motion, with
// clip B's genuinely new detail bleeding through wherever the motion model
// doesn't fully explain it — the exact visual signature of real datamosh
// corruption, produced without touching an actual H.264 NAL unit.
// ============================================================================

kernel void motionCompensateComposite(texture2d<float, access::sample> current [[texture(0)]],
                                      texture2d<float, access::sample> prevLive [[texture(1)]],
                                      texture2d<float, access::sample> canvas [[texture(2)]],
                                      texture2d<float, access::read> mv [[texture(3)]],
                                      texture2d<float, access::write> canvasOut [[texture(4)]],
                                      constant CompositeParams &p [[buffer(0)]],
                                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.fullResSize.x || gid.y >= p.fullResSize.y) return;

    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);

    uint2 mvSize = uint2(mv.get_width(), mv.get_height());
    uint2 blockId = min(gid / uint(p.fullResPerBlock), mvSize - uint2(1, 1));
    float2 vector = mv.read(blockId).xy;
    float2 fullResVector = vector * float(p.fullResPerBlock);

    float2 pos = float2(gid);
    float4 currentColor   = current.sample(s, pos);
    float4 predictedFromB = prevLive.sample(s, pos - fullResVector);
    float4 predictedFromA = canvas.sample(s, pos - fullResVector);

    float4 residual = currentColor - predictedFromB;
    float4 moshed = predictedFromA + residual * p.residualGain;

    canvasOut.write(clamp(moshed, 0.0, 1.0), gid);
}

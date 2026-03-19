///////////////////////////////////////////////////////////////////////////////
// hu_tensor.zig — Hounsfield Unit window mapping, callable from Python ctypes
//
// All exported functions use the C ABI so Python ctypes can call them directly.
//
// The core operation for each pixel:
//   1. hu is already a float32 Hounsfield value (conversion done in Python)
//   2. normed  = (hu - hu_low) / (hu_high - hu_low)
//   3. clamped = clamp(normed, 0.0, 1.0)
//   4. output  = @floatCast(clamped) as f16, stored as its u16 bit pattern
//      (ctypes has no f16 type; Python receives u16 and reinterprets as float16)
//
// Build:
//   zig build -Doptimize=ReleaseFast
//   → zig-out/lib/libhu_tensor.so
///////////////////////////////////////////////////////////////////////////////

const std = @import("std");

// ── Single-window mapping ─────────────────────────────────────────────────────

/// Apply a fixed HU window to an array of float32 HU values.
///
/// Parameters:
///   hu      - pointer to float32 array of length n (Hounsfield Units)
///   n       - number of elements (rows * cols, typically 512*512 = 262144)
///   hu_low  - HU value that maps to 0.0
///   hu_high - HU value that maps to 1.0
///   output  - pointer to u16 array of length n; receives float16 bit patterns
///
/// Values below hu_low  → 0.0 (f16)
/// Values above hu_high → 1.0 (f16)
/// Values in [hu_low, hu_high] → linearly mapped to [0.0, 1.0]
export fn apply_window(
    hu: [*]const f32,
    n: usize,
    hu_low: f32,
    hu_high: f32,
    output: [*]u16,
) void {
    const range: f32 = hu_high - hu_low;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const normed: f32 = (hu[i] - hu_low) / range;
        const clamped: f32 = @max(0.0, @min(1.0, normed));
        const as_f16: f16 = @floatCast(clamped);
        output[i] = @bitCast(as_f16);
    }
}

// ── Three-window mapping in one pass ─────────────────────────────────────────

/// Apply three HU windows simultaneously in a single pass over the data.
///
/// Useful when building wide + medium + narrow caches from the same DICOM: the
/// source array is read once from memory and three output tensors are written.
/// On a 512×512 slice this saves ~2/3 of the memory bandwidth vs. three calls.
///
/// Parameters:
///   hu         - float32 HU array, length n
///   n          - number of elements
///   lo0,hi0    - window 0 bounds (e.g. wide:   -1024, 3071)
///   lo1,hi1    - window 1 bounds (e.g. medium:  -200,  200)
///   lo2,hi2    - window 2 bounds (e.g. narrow:    48,   90)
///   out0,out1,out2 - u16 output arrays (float16 bit patterns), length n each
export fn apply_three_windows(
    hu: [*]const f32,
    n: usize,
    lo0: f32, hi0: f32,
    lo1: f32, hi1: f32,
    lo2: f32, hi2: f32,
    out0: [*]u16,
    out1: [*]u16,
    out2: [*]u16,
) void {
    const range0: f32 = hi0 - lo0;
    const range1: f32 = hi1 - lo1;
    const range2: f32 = hi2 - lo2;

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const h = hu[i];

        const n0: f32 = @max(0.0, @min(1.0, (h - lo0) / range0));
        const n1: f32 = @max(0.0, @min(1.0, (h - lo1) / range1));
        const n2: f32 = @max(0.0, @min(1.0, (h - lo2) / range2));

        out0[i] = @bitCast(@as(f16, @floatCast(n0)));
        out1[i] = @bitCast(@as(f16, @floatCast(n1)));
        out2[i] = @bitCast(@as(f16, @floatCast(n2)));
    }
}

// ── Sanity-check export ───────────────────────────────────────────────────────

/// Returns the library version as a u32: major * 10000 + minor * 100 + patch.
/// Call from Python after loading the .so to verify the library loaded correctly.
export fn hu_tensor_version() u32 {
    return 1 * 10000 + 0 * 100 + 0;
}

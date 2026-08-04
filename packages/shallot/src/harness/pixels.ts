// The final-compositor pixel classifier: a masked color-tagged region check over a captured frame.
// Distinct from every upstream rung (mesh/draw counts, timings, error absence) in the GPU evidence ladder
// (testing.md) — none of those prove the tagged content actually reached the composited output, and a
// product gate that stops one rung short can pass over a genuinely blank result (the taste ledger entry
// this closes: green vertex/draw/perf evidence, zero rendered ropes). Pure and capture-source-agnostic:
// feed it bytes from a Playwright canvas screenshot, an in-page `getImageData`, or any other tightly-packed
// RGBA buffer — this module never captures pixels itself, only classifies them.

/**
 * a masked color-tagged region to look for in a captured frame. Channel ranges are inclusive 0..255;
 * `minPixels` and `minSpan` (px, the tagged region's bounding-box footprint) both have to hold — a handful
 * of stray matching pixels isn't a rendered tag.
 *
 * @example
 * ```
 * const probe: PixelProbe = {
 *     name: "rope reaches final compositor",
 *     minPixels: 24,
 *     minSpan: 24,
 *     r: [16, 255],
 *     g: [0, 12],
 *     b: [16, 255],
 * };
 * ```
 */
export interface PixelProbe {
    name: string;
    minPixels: number;
    minSpan: number;
    r: [number, number];
    g: [number, number];
    b: [number, number];
}

/** one {@link probePixels} measurement: how many pixels matched `probe`'s bands, and the matched region's
 *  own bounding-box footprint (`0×0` when nothing matched). */
export interface PixelProbeResult {
    pixels: number;
    width: number;
    height: number;
}

/**
 * count pixels of a tightly-packed RGBA buffer (4 bytes/pixel, row-major) whose channels fall inside
 * `probe`'s bands, and measure the matched region's bounding-box footprint.
 *
 * @example const result = probePixels(rgba, 640, 360, probe);
 */
export function probePixels(
    rgba: Uint8Array | Uint8ClampedArray,
    width: number,
    height: number,
    probe: PixelProbe,
): PixelProbeResult {
    let pixels = 0;
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;
    for (let i = 0; i < rgba.length; i += 4) {
        const r = rgba[i];
        const g = rgba[i + 1];
        const b = rgba[i + 2];
        if (
            r < probe.r[0] ||
            r > probe.r[1] ||
            g < probe.g[0] ||
            g > probe.g[1] ||
            b < probe.b[0] ||
            b > probe.b[1]
        ) {
            continue;
        }
        const p = i / 4;
        const x = p % width;
        const y = Math.floor(p / width);
        pixels++;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
    }
    return {
        pixels,
        width: pixels > 0 ? maxX - minX + 1 : 0,
        height: pixels > 0 ? maxY - minY + 1 : 0,
    };
}

/** whether a {@link probePixels} measurement clears `probe`'s thresholds. */
export function pixelProbePass(result: PixelProbeResult, probe: PixelProbe): boolean {
    return (
        result.pixels >= probe.minPixels && Math.max(result.width, result.height) >= probe.minSpan
    );
}

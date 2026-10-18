import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { gpuApps } from "../../../scripts/gpu.fixture";
import { probeTexture } from "../../engine";
import { DEFAULT_FONT, loadFont } from "./font";
import { TextPlugin } from "./index";
import { SDF_EXPONENT, SDFGenerator, segmentPath } from "./sdf";

setDefaultTimeout(TIMEOUT.gpu);
const SIZE = 96;
const CURVES = 24;
// Two contours ("O", "8"), a hole meeting a stem ("g") and a glyph spanning many chunks of 16.
const GLYPHS = "O8g";
// The reference visits every third row and column; a culling or winding error spans many texels.
const STRIDE = 3;
const subjects = gpuApps(import.meta.path, [{ defaults: false, plugins: [TextPlugin] }]);

/** Each sampled texel's byte from every segment: nearest distance, and nonzero winding of a ray toward +x. */
function reference(path: string, [x0, y0, x1, y1]: [number, number, number, number]): Uint8Array {
    const segments = segmentPath(path, CURVES);
    const maxDistance = Math.max(x1 - x0, y1 - y0) / 2;
    const out = new Uint8Array(SIZE * SIZE);
    for (let row = 0; row < SIZE; row += STRIDE) {
        const py = y0 + ((row + 0.5) / SIZE) * (y1 - y0);
        for (let col = 0; col < SIZE; col += STRIDE) {
            const px = x0 + ((col + 0.5) / SIZE) * (x1 - x0);
            let nearest = maxDistance;
            let winding = 0;
            for (const s of segments) {
                const dx = s.x2 - s.x1;
                const dy = s.y2 - s.y1;
                const lenSq = dx * dx + dy * dy;
                const t =
                    lenSq > 0
                        ? Math.min(1, Math.max(0, ((px - s.x1) * dx + (py - s.y1) * dy) / lenSq))
                        : 0;
                nearest = Math.min(nearest, Math.hypot(px - s.x1 - t * dx, py - s.y1 - t * dy));
                if (s.y1 > py !== s.y2 > py && px < (dx * (py - s.y1)) / dy + s.x1)
                    winding += s.y1 < s.y2 ? 1 : -1;
            }
            const falloff = (1 - Math.min(1, nearest / maxDistance)) ** SDF_EXPONENT * 0.5;
            out[row * SIZE + col] = Math.round((winding !== 0 ? 1 - falloff : falloff) * 255);
        }
    }
    return out;
}

test("each glyph's SDF texels match every segment's nearest distance and nonzero winding within one step", async () => {
    const { world } = subjects()[0];
    const font = await loadFont(DEFAULT_FONT);
    const device = world.gpu.device;
    const output = device.createTexture({
        size: [SIZE * GLYPHS.length, SIZE],
        format: "r8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    world.own(output);
    const generator = new SDFGenerator({
        device,
        sdfSize: SIZE,
        exponent: SDF_EXPONENT,
        curveSubdivisions: CURVES,
    });
    generator.begin(world);
    const glyphs = [...GLYPHS].map((char, i) => {
        const [x0, y0, x1, y1] = font.glyphBounds(char)!;
        const pad = font.unitsPerEm * 0.1;
        const bounds: [number, number, number, number] = [x0 - pad, y0 - pad, x1 + pad, y1 + pad];
        const path = font.glyphPath(char)!;
        generator.add(path, bounds, output, i * SIZE, 0);
        return { char, expected: reference(path, bounds) };
    });
    generator.flush(world);
    const { bytes } = await probeTexture(world, output);
    const texels = new Uint8Array(bytes);
    const width = SIZE * GLYPHS.length;
    for (const [i, { char, expected }] of glyphs.entries()) {
        let worst = 0;
        for (let row = 0; row < SIZE; row += STRIDE)
            for (let col = 0; col < SIZE; col += STRIDE)
                worst = Math.max(
                    worst,
                    Math.abs(texels[row * width + i * SIZE + col] - expected[row * SIZE + col]),
                );
        expect(worst, char).toBeLessThanOrEqual(1);
    }
});

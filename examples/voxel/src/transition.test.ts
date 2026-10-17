import { expect, test } from "bun:test";
import { faces, height } from "./noise";
import {
    capacity,
    chunks,
    chunkX,
    inRing,
    radii,
    ringSlots,
    slotCount,
    type TerrainConfig,
    voxelScale,
} from "./terrain-data";

const config: TerrainConfig = { chunkSize: 8, viewDistance: 4, height: 16, relief: 4 };
test("voxel scale is one inside, zero outside, and decreases smoothly through the annulus", () => {
    const { inner, outer } = radii(config);
    expect(voxelScale(inner - 1, inner, outer)).toBe(1);
    expect(voxelScale(inner, inner, outer)).toBe(1);
    expect(voxelScale((inner + outer) / 2, inner, outer)).toBe(0.5);
    expect(voxelScale(outer, inner, outer)).toBe(0);
    expect(voxelScale(outer + 1, inner, outer)).toBe(0);
    let previous = 1;
    for (let d = inner; d <= outer; d += 0.125) {
        const scale = voxelScale(d, inner, outer);
        expect(scale).toBeLessThanOrEqual(previous);
        expect(scale).toBeGreaterThanOrEqual(0);
        previous = scale;
    }
    expect(1 - voxelScale(inner + 0.001, inner, outer)).toBeLessThan(0.000001);
    expect(voxelScale(outer - 0.001, inner, outer)).toBeLessThan(0.000001);
});

test("a chunk changes face rule when it meets or leaves the transition, not just when its coordinate changes", () => {
    expect(inRing(0, 0, 0, config)).toBe(false);
    expect(inRing(2, 0, 0, config)).toBe(true);
    expect(inRing(4, 4, 0, config)).toBe(false);
    expect(inRing(0, 0, 24, config)).toBe(true);
    expect(inRing(0, 0, 48, config)).toBe(false);
    const { inner, outer } = radii(config);
    for (const viewer of [0, 0.25, 7.75, 2 ** 24 + 0.25]) {
        for (const chunk of chunks(config)) {
            chunk.x = chunkX(chunk.x, viewer, config);
            let shrinking = false;
            for (let x = 0; x < config.chunkSize; x++)
                for (let z = 0; z < config.chunkSize; z++) {
                    const distance = Math.hypot(
                        chunk.x * config.chunkSize + x + 0.5 - viewer,
                        chunk.z * config.chunkSize + z + 0.5,
                    );
                    const scale = voxelScale(distance, inner, outer);
                    shrinking ||= scale > 0 && scale < 1;
                }
            if (shrinking) expect(inRing(chunk.x, chunk.z, viewer, config)).toBe(true);
        }
    }
});

test("the full-face pool's area bound covers every ring assignment across a chunk's fractional motion", () => {
    for (const distance of [1, 2, 3, 4, 12]) {
        const sized = { ...config, viewDistance: distance };
        const loaded = chunks(sized);
        for (let phase = 0; phase < 64; phase++) {
            const viewer = (phase * sized.chunkSize) / 64;
            const count = loaded.filter((chunk) => inRing(chunk.x, chunk.z, viewer, sized)).length;
            expect(count).toBeLessThanOrEqual(ringSlots(sized));
        }
        expect(ringSlots(sized)).toBeLessThanOrEqual(slotCount(sized));
    }
});

test("revealed chunks emit all six faces per solid voxel, within the height-derived capacity", () => {
    for (const chunk of chunks(config)) {
        let voxels = 0;
        for (let x = chunk.x * 8; x < (chunk.x + 1) * 8; x++)
            for (let z = chunk.z * 8; z < (chunk.z + 1) * 8; z++) {
                for (let y = 0; y < config.height; y++) if (y < height(x, z, config)) voxels++;
            }
        expect(faces(chunk.x, chunk.z, config, true)).toBe(voxels * 6);
        expect(faces(chunk.x, chunk.z, config, true)).toBeLessThan(capacity(config, true));
        expect(faces(chunk.x, chunk.z, config)).toBeLessThan(faces(chunk.x, chunk.z, config, true));
    }
});

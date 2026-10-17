import { expect, test } from "bun:test";
import { faces, height, perlin } from "./noise";
import {
    capacity,
    chunks,
    chunkX,
    SPEED,
    slotCount,
    type TerrainConfig,
    translation,
} from "./terrain-data";

const CONFIG: TerrainConfig = { chunkSize: 16, viewDistance: 3, height: 32, relief: 12 };

test("seeded Perlin is continuous at integer lattice seams and preserves fractional voxels beyond 2^24", () => {
    expect(perlin(0, 0)).toBe(0);
    expect(perlin(8, 19)).toBeCloseTo(0.24338162987260148, 12);
    expect(perlin(-7, 43)).toBeCloseTo(-0.34970792323939515, 12);
    expect(perlin(31.999, 17)).toBeCloseTo(perlin(32.001, 17), 3);
    expect(perlin(-0.001, 17)).toBeCloseTo(perlin(0.001, 17), 3);
    expect(perlin(2 ** 24 + 1, 19)).not.toBe(perlin(2 ** 24, 19));
    let low = Infinity,
        high = -Infinity;
    for (let x = -64; x <= 64; x++)
        for (let z = -64; z <= 64; z++) {
            const h = height(x, z, CONFIG);
            low = Math.min(low, h);
            high = Math.max(high, h);
            expect(h).toBeGreaterThanOrEqual(0);
            expect(h).toBeLessThan(CONFIG.height);
        }
    expect(high - low).toBeGreaterThan(8);
});

test("the face oracle agrees with six-neighbour voxel enumeration, including chunk seams", () => {
    const config: TerrainConfig = { chunkSize: 4, height: 16, relief: 4, viewDistance: 1 };
    for (const chunk of chunks(config)) {
        let count = 0;
        for (let x = chunk.x * 4; x < (chunk.x + 1) * 4; x++)
            for (let z = chunk.z * 4; z < (chunk.z + 1) * 4; z++) {
                const solid = (x: number, y: number, z: number) =>
                    y >= 0 && y < height(x, z, config);
                for (let y = 0; y < config.height; y++)
                    if (solid(x, y, z)) {
                        for (const [dx, dy, dz] of [
                            [1, 0, 0],
                            [-1, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0],
                            [0, 0, 1],
                            [0, 0, -1],
                        ]) {
                            if (!solid(x + dx, y + dy, z + dz)) count++;
                        }
                    }
            }
        expect(faces(chunk.x, chunk.z, config)).toBe(count);
    }
    expect(faces(0, 0, { ...config, relief: 0 })).toBe(2 * 4 * 4);
});

test("crossing a chunk boundary drops exactly the trailing column and loads exactly the leading one", () => {
    const pool = chunks(CONFIG);
    const before = new Set(pool.map(({ x, z }) => `${x},${z}`));
    for (const chunk of pool) chunk.x = chunkX(chunk.x, CONFIG.chunkSize + 0.1, CONFIG);
    const after = new Set(pool.map(({ x, z }) => `${x},${z}`));
    const dropped = [...before].filter((key) => !after.has(key));
    const loaded = [...after].filter((key) => !before.has(key));
    expect(dropped.sort()).toEqual(Array.from({ length: 7 }, (_, i) => `-3,${i - 3}`).sort());
    expect(loaded.sort()).toEqual(Array.from({ length: 7 }, (_, i) => `4,${i - 3}`).sort());
});

test("streaming keeps a unique fixed-size chunk/mesh assignment and local translations beyond 2^24 in either direction", () => {
    const pool = chunks(CONFIG);
    for (const viewer of [0, 15.9, 16, 192.3, 2 ** 24 + 13.25, -(2 ** 24) - 7.5]) {
        for (const chunk of pool) {
            chunk.x = chunkX(chunk.x, viewer, CONFIG);
            expect(Math.abs(translation(chunk.x, viewer, CONFIG))).toBeLessThanOrEqual(
                (CONFIG.viewDistance + 1) * CONFIG.chunkSize,
            );
            expect(Math.abs(translation(chunk.z, 0, CONFIG))).toBeLessThanOrEqual(
                CONFIG.viewDistance * CONFIG.chunkSize,
            );
        }
        expect(new Set(pool.map(({ x, z }) => `${x},${z}`)).size).toBe(slotCount(CONFIG));
        expect(pool).toHaveLength(slotCount(CONFIG));
    }
});

test("the fixed seed stays below the slope-derived face capacity over 240 scrolling ticks and beyond 2^24", () => {
    const pool = chunks(CONFIG);
    const check = (x: number, z: number) =>
        expect(faces(x, z, CONFIG)).toBeLessThan(capacity(CONFIG));
    for (const chunk of pool) check(chunk.x, chunk.z);
    let turnovers = 0;
    let viewer = 0;
    for (let tick = 1; tick <= 240; tick++) {
        viewer += SPEED / 60;
        for (const chunk of pool) {
            const x = chunkX(chunk.x, viewer, CONFIG);
            if (x !== chunk.x) {
                chunk.x = x;
                turnovers++;
                check(x, chunk.z);
            }
        }
    }
    expect(turnovers).toBe(49);
    for (const chunk of pool) check(chunkX(chunk.x, 2 ** 24 + 13.25, CONFIG), chunk.z);
});

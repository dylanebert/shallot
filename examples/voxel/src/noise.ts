import { SEED, type TerrainConfig } from "./terrain-data";

function gradient(x: number, z: number, dx: number, dz: number): number {
    let h = (Math.imul(x, 374761393) + Math.imul(z, 668265263) + SEED) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
    switch ((h ^ (h >>> 16)) & 3) {
        case 0:
            return dx;
        case 1:
            return -dx;
        case 2:
            return dz;
        default:
            return -dz;
    }
}
const fade = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
const mix = (a: number, b: number, t: number) => a * (1 - t) + b * t;

export function perlin(x: number, z: number): number {
    const ix = Math.floor(x / 32),
        iz = Math.floor(z / 32);
    const fx = (x - ix * 32) / 32,
        fz = (z - iz * 32) / 32;
    return mix(
        mix(gradient(ix, iz, fx, fz), gradient(ix + 1, iz, fx - 1, fz), fade(fx)),
        mix(gradient(ix, iz + 1, fx, fz - 1), gradient(ix + 1, iz + 1, fx - 1, fz - 1), fade(fx)),
        fade(fz),
    );
}
export function height(x: number, z: number, config: TerrainConfig): number {
    return Math.floor(config.height / 2 + config.relief * perlin(x, z));
}

// Scalar oracle: no producer grid, packed stream or GPU helper enters this count.
export function faces(cx: number, cz: number, config: TerrainConfig): number {
    let count = 0;
    const size = config.chunkSize;
    for (let z = cz * size; z < (cz + 1) * size; z++) {
        for (let x = cx * size; x < (cx + 1) * size; x++) {
            const h = height(x, z, config);
            count +=
                2 +
                Math.max(0, h - height(x + 1, z, config)) +
                Math.max(0, h - height(x - 1, z, config)) +
                Math.max(0, h - height(x, z + 1, config)) +
                Math.max(0, h - height(x, z - 1, config));
        }
    }
    return count;
}

import { height } from "./noise";
import { SEED, type TerrainConfig } from "./terrain-data";

export const HASH_X = 374761393;
export const HASH_Z = 668265263;
export const HASH_MIX = 1274126177;
export const DETAIL_SALT = 0x9e3779b9;
export const DetailKind = { none: 0, grass: 1, amber: 2, cyan: 3, rock: 4, tree: 5 } as const;
export type DetailKind = (typeof DetailKind)[keyof typeof DetailKind];

export function columnHash(x: number, z: number): number {
    let h = (Math.imul(x, HASH_X) + Math.imul(z, HASH_Z) + SEED + DETAIL_SALT) >>> 0;
    h = Math.imul(h ^ (h >>> 13), HASH_MIX) >>> 0;
    return (h ^ (h >>> 16)) >>> 0;
}

export function columnSlope(x: number, z: number, config: TerrainConfig): number {
    const h = height(x, z, config);
    return Math.max(
        Math.abs(h - height(x + 1, z, config)),
        Math.abs(h - height(x - 1, z, config)),
        Math.abs(h - height(x, z + 1, config)),
        Math.abs(h - height(x, z - 1, config)),
    );
}

export function detailKind(x: number, z: number, config: TerrainConfig): DetailKind {
    const hash = columnHash(x, z);
    const slope = columnSlope(x, z, config);
    const h = height(x, z, config);
    if (slope === 0 && (x & 7) === 2 && (z & 7) === 2 && (hash & 3) === 0) return DetailKind.tree;
    if ((slope >= 2 || h >= Math.floor(config.height / 2 + config.relief / 3)) && (hash & 7) === 0)
        return DetailKind.rock;
    if ((hash & 127) === 1) return DetailKind.amber;
    if ((hash & 127) === 2) return DetailKind.cyan;
    if ((hash & 3) === 0) return DetailKind.grass;
    return DetailKind.none;
}

export function detailCount(cx: number, cz: number, config: TerrainConfig): number {
    let count = 0;
    for (let z = cz * config.chunkSize; z < (cz + 1) * config.chunkSize; z++)
        for (let x = cx * config.chunkSize; x < (cx + 1) * config.chunkSize; x++)
            if (detailKind(x, z, config) !== DetailKind.none) count++;
    return count;
}

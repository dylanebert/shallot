export type TerrainConfig = {
    chunkSize: number;
    viewDistance: number;
    height: number;
    relief: number;
};
export const CONFIG: TerrainConfig = { chunkSize: 16, viewDistance: 3, height: 32, relief: 12 };
export const SEED = 173;
export const SPEED = 32;

export function slotCount(config: TerrainConfig): number {
    return (config.viewDistance * 2 + 1) ** 2;
}

export function capacity(config: TerrainConfig): number {
    // Unit-gradient Perlin has |derivative| < 8. Integer heights add at most one
    // face to that slope bound; each column emits top, bottom and four side strips.
    const slope = Math.ceil((8 * config.relief) / 32) + 1;
    return config.chunkSize ** 2 * (2 + 4 * slope);
}

export function chunkX(x: number, viewer: number, config: TerrainConfig): number {
    const center = Math.floor(viewer / config.chunkSize);
    const width = config.viewDistance * 2 + 1;
    return x + Math.ceil((center - config.viewDistance - x) / width) * width;
}

export function translation(coordinate: number, viewer: number, config: TerrainConfig): number {
    return coordinate * config.chunkSize - viewer;
}

export type Chunk = { x: number; z: number };
export function chunks(config: TerrainConfig): Chunk[] {
    const result: Chunk[] = [];
    for (let z = -config.viewDistance; z <= config.viewDistance; z++)
        for (let x = -config.viewDistance; x <= config.viewDistance; x++) result.push({ x, z });
    return result;
}

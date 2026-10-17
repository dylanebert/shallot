export type TerrainConfig = {
    chunkSize: number;
    viewDistance: number;
    height: number;
    relief: number;
};
export const CONFIG: TerrainConfig = { chunkSize: 16, viewDistance: 12, height: 32, relief: 12 };
export const SEED = 173;
export const SPEED = 32;

export function slotCount(config: TerrainConfig): number {
    return (config.viewDistance * 2 + 1) ** 2;
}

export function capacity(config: TerrainConfig, allFaces = false): number {
    if (allFaces) return config.chunkSize ** 2 * Math.ceil(config.height / 2 + config.relief) * 6;
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

export function radii(config: TerrainConfig): { inner: number; outer: number } {
    const outer = config.viewDistance * config.chunkSize;
    return { inner: Math.max(0, outer - config.chunkSize), outer };
}

export function voxelScale(distance: number, inner: number, outer: number): number {
    const t = Math.max(0, Math.min(1, (distance - inner) / (outer - inner)));
    return 1 - t * t * (3 - 2 * t);
}

export function inRing(x: number, z: number, viewer: number, config: TerrainConfig): boolean {
    const size = config.chunkSize;
    const left = x * size - viewer,
        right = left + size;
    const top = z * size,
        bottom = top + size;
    const nearX = Math.max(0, left, -right),
        nearZ = Math.max(0, top, -bottom);
    const farX = Math.max(Math.abs(left), Math.abs(right));
    const farZ = Math.max(Math.abs(top), Math.abs(bottom));
    // One voxel covers a tick of motion while rendering interpolates the prior pose.
    const outer = config.viewDistance * size + 1;
    const inner = Math.max(0, config.viewDistance * size - size - 1);
    return (
        nearX * nearX + nearZ * nearZ <= outer * outer && farX * farX + farZ * farZ >= inner * inner
    );
}

export function ringSlots(config: TerrainConfig): number {
    const { inner, outer } = radii(config);
    // Every square touching the padded annulus lies inside this dilation.
    // Their disjoint area bounds the number of simultaneous full-face chunks.
    const padding = Math.SQRT2 * config.chunkSize + 1;
    const area = Math.PI * ((outer + padding) ** 2 - Math.max(0, inner - padding) ** 2);
    return Math.min(slotCount(config), Math.ceil(area / config.chunkSize ** 2));
}

export type Chunk = { x: number; z: number };
export function chunks(config: TerrainConfig): Chunk[] {
    const result: Chunk[] = [];
    for (let z = -config.viewDistance; z <= config.viewDistance; z++)
        for (let x = -config.viewDistance; x <= config.viewDistance; x++) result.push({ x, z });
    return result;
}

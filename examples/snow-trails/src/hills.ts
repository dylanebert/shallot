import type { Plugin } from "@dylanebert/shallot";

export const FIELD = 24;
export const HILL_SIZE = 65;
export const heights = Float32Array.from({ length: HILL_SIZE * HILL_SIZE }, (_, i) => {
    const x = ((i % HILL_SIZE) / (HILL_SIZE - 1) - 0.5) * FIELD;
    const z = (Math.floor(i / HILL_SIZE) / (HILL_SIZE - 1) - 0.5) * FIELD;
    return 0.018 * (x * x + z * z) + 0.8 * Math.sin(x * 0.4) * Math.cos(z * 0.35) + 0.8;
});
export const LOW = Math.min(...heights);
export const HIGH = Math.max(...heights);
export function groundAt(x: number, z: number) {
    const u = Math.max(0, Math.min(HILL_SIZE - 1.001, (x / FIELD + 0.5) * (HILL_SIZE - 1)));
    const v = Math.max(0, Math.min(HILL_SIZE - 1.001, (z / FIELD + 0.5) * (HILL_SIZE - 1)));
    const a = Math.floor(u),
        b = Math.floor(v),
        dx = u - a,
        dz = v - b;
    return (
        (1 - dz) * ((1 - dx) * heights[b * HILL_SIZE + a]! + dx * heights[b * HILL_SIZE + a + 1]!) +
        dz *
            ((1 - dx) * heights[(b + 1) * HILL_SIZE + a]! +
                dx * heights[(b + 1) * HILL_SIZE + a + 1]!)
    );
}
export const Hills = { create: () => ({ image: null as GPUTexture | null }) };
export const HillsPlugin: Plugin = {
    name: "Hills",
    initialize(world) {
        const image = world.gpu.device.createTexture({
            size: [HILL_SIZE, HILL_SIZE],
            format: "r32float",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        world.own(image);
        world.gpu.device.queue.writeTexture(
            { texture: image },
            heights,
            { bytesPerRow: HILL_SIZE * 4 },
            [HILL_SIZE, HILL_SIZE],
        );
        world.resource(Hills).image = image;
    },
};

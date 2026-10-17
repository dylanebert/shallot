import { expect, test } from "bun:test";
import { chunks, chunkX, radii, type TerrainConfig, voxelScale } from "./terrain-data";

test("wrapped chunks are completely shrunk at both their dropped and newly loaded positions", () => {
    const config: TerrainConfig = { chunkSize: 8, viewDistance: 4, height: 16, relief: 4 };
    const { inner, outer } = radii(config);
    const viewer = config.chunkSize + 0.1;
    let wrapped = 0;
    for (const chunk of chunks(config)) {
        const next = chunkX(chunk.x, viewer, config);
        if (next === chunk.x) continue;
        wrapped++;
        for (const x of [chunk.x, next]) {
            for (let vx = 0; vx < config.chunkSize; vx++)
                for (let vz = 0; vz < config.chunkSize; vz++) {
                    const distance = Math.hypot(
                        x * config.chunkSize + vx + 0.5 - viewer,
                        chunk.z * config.chunkSize + vz + 0.5,
                    );
                    expect(voxelScale(distance, inner, outer)).toBe(0);
                }
        }
    }
    expect(wrapped).toBe(config.viewDistance * 2 + 1);
});

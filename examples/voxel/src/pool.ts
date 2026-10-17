import type { World } from "@dylanebert/shallot";
import type { Mesh } from "@dylanebert/shallot/mesh";
import * as d from "typegpu/data";
import { capacity, type TerrainConfig } from "./terrain-data";

export type Geometry = {
    mesh: Mesh;
    uniform: GPUBuffer;
    input: Int32Array;
    group: GPUBindGroup;
    allFaces: boolean;
};

export function pool(
    world: World,
    config: TerrainConfig,
    counts: GPUBuffer,
    quant: NonNullable<Mesh["quant"]>,
    pipeline: GPUComputePipeline,
    count: number,
): Geometry[] {
    const { device, root } = world.gpu;
    const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = device.createBuffer({ size, usage });
        world.own(buffer);
        return buffer;
    };
    const result: Geometry[] = [];
    const maxFaces = Math.max(capacity(config), capacity(config, true));
    for (let i = 0; i < count; i++) {
        const vertices = allocate(maxFaces * 4 * 16, GPUBufferUsage.STORAGE);
        const position = allocate(maxFaces * 4 * 8, GPUBufferUsage.STORAGE);
        const indices = allocate(maxFaces * 6 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX);
        const voxel = allocate(maxFaces * 4 * 4, GPUBufferUsage.STORAGE);
        const uniform = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        const mesh: Mesh = {
            name: "chunk",
            vertices: root
                .createBuffer(d.arrayOf(d.vec4u, maxFaces * 4), vertices)
                .$usage("storage"),
            position: root
                .createBuffer(d.arrayOf(d.vec2u, maxFaces * 4), position)
                .$usage("storage"),
            indices: root
                .createBuffer(d.arrayOf(d.u32, maxFaces * 6), indices)
                .$usage("storage", "index"),
            attributes: {
                voxel: root.createBuffer(d.arrayOf(d.u32, maxFaces * 4), voxel).$usage("storage"),
            },
            quant,
            indexBase: 0,
            indexCount: maxFaces * 6,
            indirectIndexCount: { buffer: counts, offset: 0 },
            bounds: [
                config.chunkSize / 2,
                config.height / 2,
                config.chunkSize / 2,
                Math.hypot(config.chunkSize, config.height, config.chunkSize) / 2,
            ],
        };
        const group = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [vertices, position, indices, voxel, counts, uniform].map(
                (buffer, binding) => ({ binding, resource: { buffer } }),
            ),
        });
        result.push({
            mesh,
            uniform,
            input: new Int32Array(4),
            group,
            allFaces: false,
        });
    }
    return result;
}

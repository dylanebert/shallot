import { encodePos, encodeUv, type MeshQuant } from "@dylanebert/shallot/mesh";
import { octEncodeNormal } from "@dylanebert/shallot/utils";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { perlin } from "./gpu-noise";
import { capacity, type TerrainConfig } from "./terrain-data";

const layout = tgpu.bindGroupLayout({
    vertices: { storage: d.arrayOf(d.vec4u), access: "mutable" },
    position: { storage: d.arrayOf(d.vec2u), access: "mutable" },
    indices: { storage: d.arrayOf(d.u32), access: "mutable" },
    voxel: { storage: d.arrayOf(d.u32), access: "mutable" },
    counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
    chunk: { uniform: d.vec4i },
});
export function mesher(config: TerrainConfig) {
    if (config.chunkSize > 256 || config.height > 256)
        throw new Error("voxel packed centres require chunkSize and height at most 256");
    const Chunk = config.chunkSize;
    const Capacity = capacity(config);
    const fullCapacity = capacity(config, true);
    const Height = config.height;
    const Relief = config.relief;
    const height = tgpu.fn(
        [d.i32, d.i32],
        d.i32,
    )((x, z) => {
        "use gpu";
        return d.i32(std.floor(Height / 2 + Relief * perlin(x, z)));
    });
    const quantData = new Float32Array([0, 0, 0, 0, Chunk, Height, Chunk, 0, 1, 1, 0, 0]);
    const quant: d.Infer<typeof MeshQuant> = {
        posOffset: d.vec4f(0),
        posScale: d.vec4f(Chunk, Height, Chunk, 0),
        uvScale: d.vec4f(1, 1, 0, 0),
    };
    const quad = tgpu.fn(
        [d.vec3f, d.vec3f, d.vec3f],
        d.Void,
    )((origin, u, v) => {
        "use gpu";
        const index = std.atomicAdd(layout.$.counts[d.u32(layout.$.chunk.z)], 6);
        let limit = d.u32(Capacity * 6);
        if (layout.$.chunk.w !== 0) limit = d.u32(fullCapacity * 6);
        if (index >= limit) return;
        const vertex = d.u32(index / d.u32(6)) * d.u32(4);
        const direction = std.normalize(std.cross(u, v));
        const normal = octEncodeNormal(direction);
        const center = std.floor(origin.add(u.mul(0.5)).add(v.mul(0.5)).sub(direction.mul(0.5)));
        let face = d.u32(0);
        if (direction.x < 0) face = d.u32(1);
        if (direction.y > 0) face = d.u32(2);
        if (direction.y < 0) face = d.u32(3);
        if (direction.z > 0) face = d.u32(4);
        if (direction.z < 0) face = d.u32(5);
        const voxel =
            d.u32(center.x) | (d.u32(center.y) << 8) | (d.u32(center.z) << 16) | (face << 24);
        for (let i = d.u32(0); i < 4; i++) {
            const p = origin
                .add(u.mul(std.select(d.f32(0), d.f32(1), i === 1 || i === 2)))
                .add(v.mul(std.select(d.f32(0), d.f32(1), i >= 2)));
            const packed = encodePos(p, 0, quant);
            layout.$.voxel[vertex + i] = voxel;
            layout.$.position[vertex + i] = d.vec2u(packed);
            layout.$.vertices[vertex + i] = d.vec4u(
                packed.x,
                packed.y,
                normal,
                encodeUv(d.vec2f(0), quant),
            );
        }
        layout.$.indices[index] = vertex;
        layout.$.indices[index + 1] = vertex + 1;
        layout.$.indices[index + 2] = vertex + 2;
        layout.$.indices[index + 3] = vertex;
        layout.$.indices[index + 4] = vertex + 2;
        layout.$.indices[index + 5] = vertex + 3;
    });
    const emit = tgpu
        .computeFn({ workgroupSize: [4, 4], in: { gid: d.builtin.globalInvocationId } })(
            (input) => {
                "use gpu";
                const x = d.i32(input.gid.x);
                const z = d.i32(input.gid.y);
                const wx = layout.$.chunk.x * Chunk + x;
                const wz = layout.$.chunk.y * Chunk + z;
                const h = height(wx, wz);
                const px = d.f32(x);
                const pz = d.f32(z);
                if (layout.$.chunk.w !== 0) {
                    for (let y = d.i32(0); y < h; y++) {
                        const py = d.f32(y);
                        quad(d.vec3f(px, py + 1, pz), d.vec3f(0, 0, 1), d.vec3f(1, 0, 0));
                        quad(d.vec3f(px, py, pz), d.vec3f(1, 0, 0), d.vec3f(0, 0, 1));
                        quad(d.vec3f(px + 1, py, pz), d.vec3f(0, 1, 0), d.vec3f(0, 0, 1));
                        quad(d.vec3f(px, py, pz), d.vec3f(0, 0, 1), d.vec3f(0, 1, 0));
                        quad(d.vec3f(px, py, pz + 1), d.vec3f(1, 0, 0), d.vec3f(0, 1, 0));
                        quad(d.vec3f(px, py, pz), d.vec3f(0, 1, 0), d.vec3f(1, 0, 0));
                    }
                    return;
                }
                quad(d.vec3f(px, d.f32(h), pz), d.vec3f(0, 0, 1), d.vec3f(1, 0, 0));
                quad(d.vec3f(px, 0, pz), d.vec3f(1, 0, 0), d.vec3f(0, 0, 1));
                for (let y = height(wx + 1, wz); y < h; y++)
                    quad(d.vec3f(px + 1, d.f32(y), pz), d.vec3f(0, 1, 0), d.vec3f(0, 0, 1));
                for (let y = height(wx - 1, wz); y < h; y++)
                    quad(d.vec3f(px, d.f32(y), pz), d.vec3f(0, 0, 1), d.vec3f(0, 1, 0));
                for (let y = height(wx, wz + 1); y < h; y++)
                    quad(d.vec3f(px, d.f32(y), pz + 1), d.vec3f(1, 0, 0), d.vec3f(0, 1, 0));
                for (let y = height(wx, wz - 1); y < h; y++)
                    quad(d.vec3f(px, d.f32(y), pz), d.vec3f(0, 1, 0), d.vec3f(1, 0, 0));
            },
        )
        .$name("voxel:emit");
    return { emit, quantData };
}

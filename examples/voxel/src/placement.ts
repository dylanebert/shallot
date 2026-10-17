import type { World } from "@dylanebert/shallot";
import { Xform } from "@dylanebert/shallot/rendering";
import { MeshInstanceInput, registerProducerTable } from "@dylanebert/shallot/standard/rendering";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { detailParts } from "./parts";
import { radii, slotCount, type TerrainConfig } from "./terrain-data";

const Params = d.struct({ center: d.i32, remainder: d.f32 });
const layout = tgpu.bindGroupLayout({
    records: { storage: d.arrayOf(d.vec2u), access: "readonly" },
    counts: { storage: d.arrayOf(d.u32), access: "readonly" },
    chunks: { storage: d.arrayOf(d.vec2i), access: "readonly" },
    parts: { storage: d.arrayOf(d.vec4u), access: "readonly" },
    placements: { storage: d.arrayOf(Xform), access: "mutable" },
    instances: { storage: d.arrayOf(MeshInstanceInput), access: "mutable" },
    count: { storage: d.atomic(d.u32), access: "mutable" },
    params: { uniform: Params },
});
export function placement(
    world: World,
    config: TerrainConfig,
    records: GPUBuffer,
    counts: GPUBuffer,
) {
    const size = config.chunkSize,
        columns = size * size,
        slots = slotCount(config);
    const { inner, outer } = radii(config);
    const kernel = tgpu.computeFn({
        workgroupSize: [64],
        in: { gid: d.builtin.globalInvocationId },
    })(({ gid }) => {
        "use gpu";
        const slot = gid.x / d.u32(columns);
        const j = gid.x % d.u32(columns);
        if (slot >= slots || j >= layout.$.counts[slot]) return;
        const record = layout.$.records[gid.x];
        const packed = record.x;
        const hash = record.y;
        const kind = packed >>> 24;
        const chunk = layout.$.chunks[slot];
        const centre = d.vec3f(
            d.f32((chunk.x - layout.$.params.center) * size) -
                layout.$.params.remainder +
                d.f32(packed & 255) +
                0.5,
            d.f32((packed >>> 8) & 255) - 0.5 - config.height / 2,
            d.f32(chunk.y * size) + d.f32((packed >>> 16) & 255) + 0.5,
        );
        const scale = 1 - std.smoothstep(inner, outer, std.length(centre.xz));
        if (scale <= 0) return;
        let jitter = d.vec2f(0);
        let yaw = d.f32(0);
        if (kind !== 5) {
            jitter = d
                .vec2f(d.f32(hash & 255) / 255 - 0.5, d.f32((hash >>> 8) & 255) / 255 - 0.5)
                .mul(0.35 * scale);
            yaw = (d.f32(hash >>> 16) / 65535) * 6.283185307179586;
        }
        const n = layout.$.parts[kind * 2].w;
        const row = std.atomicAdd(layout.$.count, n);
        for (let p = d.u32(0); p < n; p++) {
            const part = layout.$.parts[kind * 2 + p];
            layout.$.placements[row + p] = Xform({
                pos: centre.add(d.vec3f(jitter.x, 0.5 * scale, jitter.y)),
                quat: d.vec4f(0, std.sin(yaw * 0.5), 0, std.cos(yaw * 0.5)),
                scale: d.vec3f(scale),
            });
            layout.$.instances[row + p] = MeshInstanceInput({
                mesh: part.x,
                materialType: part.y,
                material: part.z,
                alphaMode: 0,
                alphaCutoff: 0.5,
                flags: 0,
                cullBounds: d.vec4f(0, 0, 0, -1),
            });
        }
    });
    const parts = detailParts(world),
        { root, device } = world.gpu;
    const table = registerProducerTable(world, {
        name: "voxel:detail",
        capacity: slots * columns * 2,
        meshes: parts.meshes,
        materials: parts.materials,
    });
    const chunks = root.createBuffer(d.arrayOf(d.vec2i, slots)).$usage("storage");
    const params = root.createBuffer(Params).$usage("uniform");
    const pipe = root.unwrap(root.createComputePipeline({ compute: kernel }).$name("voxel:place"));
    const group = root.unwrap(
        root.createBindGroup(layout, {
            records: root
                .createBuffer(d.arrayOf(d.vec2u, slots * columns), records)
                .$usage("storage"),
            counts: root.createBuffer(d.arrayOf(d.u32, slots), counts).$usage("storage"),
            chunks,
            parts: root
                .createBuffer(d.arrayOf(d.vec4u, parts.rows.length), parts.rows)
                .$usage("storage"),
            placements: table.placements,
            instances: table.meshInstances,
            count: table.count,
            params,
        }),
    );
    const input = new ArrayBuffer(8),
        integers = new Int32Array(input),
        floats = new Float32Array(input);
    const coordinates = new Int32Array(2);
    const descriptor: GPUComputePassDescriptor = { label: "voxel:place" };
    return {
        table,
        chunk(slot: number, x: number, z: number) {
            coordinates[0] = x;
            coordinates[1] = z;
            device.queue.writeBuffer(root.unwrap(chunks), slot * 8, coordinates);
        },
        draw(viewer: number) {
            integers[0] = Math.floor(viewer / size);
            floats[1] = viewer - integers[0] * size;
            device.queue.writeBuffer(root.unwrap(params), 0, input);
            const encoder = world.frameEncoder()!;
            encoder.clearBuffer(root.unwrap(table.count));
            descriptor.timestampWrites = world.gpu.span?.("voxel:place");
            const pass = encoder.beginComputePass(descriptor);
            pass.setPipeline(pipe);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(Math.ceil((slots * columns) / 64));
            pass.end();
        },
    };
}

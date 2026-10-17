import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { DETAIL_SALT, DetailKind, HASH_MIX, HASH_X, HASH_Z } from "./detail";
import type { heightfield } from "./gpu-noise";
import { SEED, type TerrainConfig } from "./terrain-data";

const layout = tgpu.bindGroupLayout({
    records: { storage: d.arrayOf(d.vec2u), access: "mutable" },
    counts: { storage: d.arrayOf(d.atomic(d.u32)), access: "mutable" },
    chunk: { uniform: d.vec4i },
});

export function scatter(config: TerrainConfig, height: ReturnType<typeof heightfield>) {
    const Chunk = config.chunkSize;
    const high = Math.floor(config.height / 2 + config.relief / 3);
    return tgpu
        .computeFn({ workgroupSize: [4, 4], in: { gid: d.builtin.globalInvocationId } })(
            (input) => {
                "use gpu";
                const x = d.i32(input.gid.x);
                const z = d.i32(input.gid.y);
                if (x >= Chunk || z >= Chunk) return;
                const wx = layout.$.chunk.x * Chunk + x;
                const wz = layout.$.chunk.y * Chunk + z;
                const h = height(wx, wz);
                const slope = std.max(
                    std.max(std.abs(h - height(wx + 1, wz)), std.abs(h - height(wx - 1, wz))),
                    std.max(std.abs(h - height(wx, wz + 1)), std.abs(h - height(wx, wz - 1))),
                );
                let hash =
                    d.u32(wx) * d.u32(HASH_X) +
                    d.u32(wz) * d.u32(HASH_Z) +
                    d.u32(SEED) +
                    d.u32(DETAIL_SALT);
                hash = (hash ^ (hash >>> 13)) * d.u32(HASH_MIX);
                hash = hash ^ (hash >>> 16);
                let kind = d.u32(DetailKind.none);
                if (slope === 0 && (wx & 7) === 2 && (wz & 7) === 2 && (hash & 3) === 0)
                    kind = d.u32(DetailKind.tree);
                else if ((slope >= 2 || h >= high) && (hash & 7) === 0)
                    kind = d.u32(DetailKind.rock);
                else if ((hash & 127) === 1) kind = d.u32(DetailKind.amber);
                else if ((hash & 127) === 2) kind = d.u32(DetailKind.cyan);
                else if ((hash & 3) === 0) kind = d.u32(DetailKind.grass);
                if (kind === DetailKind.none) return;
                const slot = d.u32(layout.$.chunk.z);
                const index = std.atomicAdd(layout.$.counts[slot], 1);
                layout.$.records[slot * d.u32(Chunk * Chunk) + index] = d.vec2u(
                    d.u32(x) | (d.u32(h) << 8) | (d.u32(z) << 16) | (kind << 24),
                    hash,
                );
            },
        )
        .$name("voxel:scatter");
}

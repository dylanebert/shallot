// GPU heightfield terrain generation. A TGSL compute pass writes the quantized vertex streams directly,
// one thread per grid vertex: no intermediate density volume, no atomic append — the grid's topology
// (grid.ts) is fixed and data-independent, so each thread owns exactly one output slot
// (`vertexIndex(ix, iz)`, grid.ts) and there's no race to arbitrate. This is voxel's mesher simplified by
// the one fact a heightfield has that a voxel volume doesn't: the vertex/index count is known up front.
//
// Each thread samples {@link flattenedHeightAt} (`flatten.ts` — natural `heightAt` blended toward the
// road network's flattened core, a no-op away from any road) at its own column and its four neighbours
// (a finite-difference normal — the standard heightfield trick: for `y = h(x, z)`, `normal =
// normalize(-∂h/∂x, 1, -∂h/∂z)`), then writes the encoded position + oct-normal directly into the mesh's
// quantized main + position streams (gpu.md rule 6) via the published `encodePos`/`encodeUv`/
// `octEncodeNormal` codecs — the same producer-writes-quantized-storage-directly pattern voxel's
// `emitKernel` uses. Sampling the flattened surface (not bare `heightAt`) at every finite-difference
// neighbour, not just the vertex itself, is what makes the emitted normal reflect the flattened ground
// too — a flat road with an unflattened normal would still light like a slope.

import { Compute } from "@dylanebert/shallot";
import { precompile, precompileScope } from "@dylanebert/shallot/runtime";
import {
    encodePos,
    encodeUv,
    type MeshQuant,
    octEncodeNormal,
} from "@dylanebert/shallot/utils/core";
import tgpu, { type StorageFlag, type TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { flattenedHeightAt, networkBindGroup } from "./flatten";
import { GridPosition, GridVertices, HALF, SPACING, VERTS } from "./grid";
import { GROUND_LEVEL, makePermutation, noiseLayout, PermData, RELIEF } from "./noise";

const WORLD_HALF = ((VERTS - 1) / 2) * SPACING;
const WORLD_EXTENT = (VERTS - 1) * SPACING;

/** the terrain mesh's analytic AABB (grid.ts's world extent × the noise-derived height band), the
 *  {@link MeshQuant} table both the generator (encode) and sear (decode) dequantize against. One source:
 *  a scale change to either the grid spacing or the relief amplitude updates this automatically. */
export const TERRAIN_QUANT: d.Infer<typeof MeshQuant> = {
    posOffset: d.vec4f(-WORLD_HALF, GROUND_LEVEL - RELIEF, -WORLD_HALF, 0),
    posScale: d.vec4f(WORLD_EXTENT, 2 * RELIEF, WORLD_EXTENT, 0),
    uvScale: d.vec4f(0, 0, 0, 0),
};

const WG = 8; // 8×8 = 64 threads per workgroup, one 2D dispatch over the fixed vertex grid
const DISPATCH = { x: Math.ceil(VERTS / WG), y: Math.ceil(VERTS / WG) };

const terrainLayout = tgpu.bindGroupLayout({
    vertices: { storage: GridVertices, access: "mutable" },
    position: { storage: GridPosition, access: "mutable" },
});

const heightKernel = tgpu
    .computeFn({
        workgroupSize: [WG, WG, 1],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const gid = input.gid;
        if (gid.x >= d.u32(VERTS) || gid.y >= d.u32(VERTS)) return;

        const ix = gid.x;
        const iz = gid.y;
        const x = (d.f32(ix) - d.f32(HALF)) * d.f32(SPACING);
        const z = (d.f32(iz) - d.f32(HALF)) * d.f32(SPACING);
        const eps = d.f32(SPACING);

        const y = flattenedHeightAt(x, z);
        const yx0 = flattenedHeightAt(x - eps, z);
        const yx1 = flattenedHeightAt(x + eps, z);
        const yz0 = flattenedHeightAt(x, z - eps);
        const yz1 = flattenedHeightAt(x, z + eps);
        const normal = std.normalize(
            d.vec3f(
                -(yx1 - yx0) / (d.f32(2.0) * eps),
                d.f32(1.0),
                -(yz1 - yz0) / (d.f32(2.0) * eps),
            ),
        );

        const idx = iz * d.u32(VERTS) + ix;
        const octN = octEncodeNormal(normal);
        const uvw = encodeUv(d.vec2f(0, 0), TERRAIN_QUANT);
        const m = encodePos(d.vec3f(x, y, z), d.u32(0), TERRAIN_QUANT);
        terrainLayout.$.vertices[idx] = d.vec4u(m.x, m.y, octN, uvw);
        terrainLayout.$.position[idx] = d.vec2u(m.x, m.y);
    })
    .$name("terrain-height-kernel");

/** the emitted height-kernel WGSL — the device-free structural seam the terrain tests resolve. */
export function heightKernelWgsl(): string {
    return tgpu.resolve([octEncodeNormal, encodePos, encodeUv, flattenedHeightAt, heightKernel], {
        names: "strict",
    });
}

type TerrainRoot = typeof Compute.root;
type TerrainPipeline = ReturnType<TerrainRoot["createComputePipeline"]>;
type PermBuffer = TgpuBuffer<typeof PermData> & StorageFlag;

let pipeline: TerrainPipeline | null = null;
let bindGroup: ReturnType<TerrainRoot["createBindGroup"]> | null = null;
let permRaw: GPUBuffer | null = null;
let warmed = false;
let tail = Promise.resolve();

/** wire the height kernel to the mesh's live vertex/position buffers — called once from `terrain.ts`'s
 *  `warm()`, after the mesh buffers exist. */
export function bindTerrainKernel(
    vertices: TgpuBuffer<typeof GridVertices> & StorageFlag,
    position: TgpuBuffer<typeof GridPosition> & StorageFlag,
): void {
    const { root } = Compute;
    pipeline = root.createComputePipeline({ compute: heightKernel }).$name("terrain-generate");
    bindGroup = root.createBindGroup(terrainLayout, { vertices, position });
    warmed = false;
}

async function run(seed: number): Promise<void> {
    if (!pipeline || !bindGroup) throw new Error("terrain: generate before bindTerrainKernel");
    const activePipeline = pipeline;
    const activeBindGroup = bindGroup;
    const { device, root } = Compute;

    if (!warmed) {
        // `initAsync()` on the returned pipeline pays Dawn's deferred compile under the loading screen
        // — no throwaway perm buffer/bind group/dispatch needed to get there
        const warmLabel = precompileScope("terrain-generate");
        await precompile(warmLabel, () => activePipeline);
        warmed = true;
    }

    const perm = makePermutation(seed);
    const nextRaw = device.createBuffer({
        label: "terrain-perm",
        size: perm.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    try {
        const nextPerm: PermBuffer = root.createBuffer(PermData, nextRaw).$usage("storage");
        device.queue.writeBuffer(nextRaw, 0, perm as Uint32Array<ArrayBuffer>);
        const noiseGroup = root.createBindGroup(noiseLayout, { perm: nextPerm });
        const enc = device.createCommandEncoder({ label: "terrain-generate" });
        const pass = enc.beginComputePass({ label: "terrain-generate" });
        activePipeline
            .with(noiseGroup)
            .with(activeBindGroup)
            .with(networkBindGroup())
            .with(pass)
            .dispatchWorkgroups(DISPATCH.x, DISPATCH.y, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
    } catch (cause) {
        nextRaw.destroy();
        throw cause;
    }
    permRaw?.destroy();
    permRaw = nextRaw;
}

/** regenerate the terrain's height + normal for `seed`, serialized behind the prior call so two
 *  overlapping reseeds (F9 spam, the gate's back-to-back regenerate calls) never interleave dispatches. */
export function generate(seed: number): Promise<void> {
    const result = tail.then(
        () => run(seed),
        () => run(seed),
    );
    tail = result.catch(() => {});
    return result;
}

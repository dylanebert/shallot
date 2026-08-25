// GPU heightmap terrain generation. A TGSL compute pass fills the voxel grid from a layered 2D perlin
// field: a zero-mean multi-octave `fbm2(x,z)` lifts and drops the ground into rolling hills —
// `surface = GROUND_LEVEL + fbm2(x·HFREQ, z·HFREQ)·RELIEF`, solid below the surface, air above. The
// textbook layered-perlin landscape, which reads as recognizable landform where isotropic 3D noise read as
// busy pockets. Writes the same `Voxels.grid` buffer the mesher reads, then dirties it so the next frame
// re-meshes. The visual replacement for the hand-authored patterns; the canonical set stays the
// mesher-correctness gate.

// The grid stays a full 3D density field — the carve brush (edit.ts) sculpts overhangs and caves by hand —
// but the *generated* terrain is a heightmap, deliberately. Auto-generated 3D caves are the "minecraft"
// direction we're not taking here; rolling hills are the simpler, more pleasing initialization.

// Shape: a seeded
// permutation table in a storage buffer + an FBM WGSL chunk + a workgroup dispatch on its own encoder. The
// pure noise primitives (perm table, perlin/fbm TGSL, heightmap knobs, the derived band) live in noise.ts.

import { Compute } from "@dylanebert/shallot";
import { precompile, precompileScope } from "@dylanebert/shallot/runtime";
import tgpu, { type StorageFlag, type TgpuBuffer } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { DENSITY, DIM, GridData, voxelIndex } from "./grid";
import { syncGrid, Voxels } from "./mesher";
import { fbm2, GROUND_LEVEL, HFREQ, makePermutation, noiseLayout, PermData, RELIEF } from "./noise";

export { solidFractionBand } from "./noise";

const WG = 4; // 4³ = 64 threads, matching the mesher's dispatch
const DISPATCH = { x: Math.ceil(DIM.x / WG), y: Math.ceil(DIM.y / WG), z: Math.ceil(DIM.z / WG) };

const densityLayout = tgpu.bindGroupLayout({
    grid: { storage: GridData, access: "mutable" },
});

const densityKernel = tgpu
    .computeFn({
        workgroupSize: [WG, WG, WG],
        in: { gid: d.builtin.globalInvocationId },
    })((input) => {
        "use gpu";
        const gid = input.gid;
        if (gid.x >= d.u32(DIM.x) || gid.y >= d.u32(DIM.y) || gid.z >= d.u32(DIM.z)) return;
        const surface =
            d.f32(GROUND_LEVEL) +
            fbm2(std.mul(d.vec2f(d.f32(gid.x), d.f32(gid.z)), d.f32(HFREQ))) * d.f32(RELIEF);
        const field = surface - d.f32(gid.y);
        densityLayout.$.grid[voxelIndex(gid.x, gid.y, gid.z)] = std.select(
            d.f32(0.0),
            d.f32(DENSITY),
            field > d.f32(0.0),
        );
    })
    .$name("voxel-generate-density");

type GridBuffer = TgpuBuffer<typeof GridData> & StorageFlag;
type PermBuffer = TgpuBuffer<typeof PermData> & StorageFlag;
type DensityRoot = typeof Compute.root;
type DensityPipeline = ReturnType<DensityRoot["createComputePipeline"]>;

interface DensityTarget<Root, Device, RawGrid> {
    root: Root;
    device: Device;
    grid: RawGrid;
}

interface DensityRunnerOps<Root, Device, RawGrid, Pipeline, Grid> {
    pipeline: (target: DensityTarget<Root, Device, RawGrid>) => Pipeline;
    wrapGrid: (target: DensityTarget<Root, Device, RawGrid>) => Grid;
    precompile: (
        target: DensityTarget<Root, Device, RawGrid>,
        pipeline: Pipeline,
        grid: Grid,
    ) => Promise<void>;
    dispatch: (
        target: DensityTarget<Root, Device, RawGrid>,
        pipeline: Pipeline,
        grid: Grid,
        seed: number,
    ) => Promise<void> | void;
}

/** serialize density generation while retaining pipeline identity per active root. @internal */
export function createDensityRunner<Root, Device, RawGrid, Pipeline, Grid>(
    ops: DensityRunnerOps<Root, Device, RawGrid, Pipeline, Grid>,
): (target: DensityTarget<Root, Device, RawGrid>, seed: number) => Promise<void> {
    let active: { root: Root; pipeline: Pipeline; warmed: boolean } | null = null;
    let tail = Promise.resolve();

    const run = async (
        target: DensityTarget<Root, Device, RawGrid>,
        seed: number,
    ): Promise<void> => {
        if (!active || active.root !== target.root) {
            active = { root: target.root, pipeline: ops.pipeline(target), warmed: false };
        }
        const owner = active;
        const grid = ops.wrapGrid(target);
        if (!owner.warmed) {
            await ops.precompile(target, owner.pipeline, grid);
            owner.warmed = true;
        }
        await ops.dispatch(target, owner.pipeline, grid, seed);
    };

    return (target, seed) => {
        const result = tail.then(
            () => run(target, seed),
            () => run(target, seed),
        );
        tail = result.catch(() => {});
        return result;
    };
}

let permRaw: GPUBuffer | null = null;

const runDensity = createDensityRunner<
    DensityRoot,
    GPUDevice,
    GPUBuffer,
    DensityPipeline,
    GridBuffer
>({
    pipeline: ({ root }) =>
        root.createComputePipeline({ compute: densityKernel }).$name("voxel-generate"),
    wrapGrid: ({ root, grid }): GridBuffer => root.createBuffer(GridData, grid).$usage("storage"),
    async precompile(_target, pipeline) {
        // `initAsync()` on the returned pipeline pays Dawn's deferred compile under the loading screen
        // — no throwaway perm buffer/bind group/dispatch needed to get there (that warmed by tripping
        // the same zero-workgroup path this unit's warm idiom replaces)
        const warmLabel = precompileScope("voxel-generate");
        await precompile(warmLabel, () => pipeline);
    },
    dispatch({ root, device }, pipeline, grid, seed) {
        const perm = makePermutation(seed);
        const nextRaw = device.createBuffer({
            label: "voxel-perm",
            size: perm.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        try {
            const nextPerm: PermBuffer = root.createBuffer(PermData, nextRaw).$usage("storage");
            device.queue.writeBuffer(nextRaw, 0, perm as Uint32Array<ArrayBuffer>);
            const noiseGroup = root.createBindGroup(noiseLayout, { perm: nextPerm });
            const group = root.createBindGroup(densityLayout, { grid });
            const enc = device.createCommandEncoder({ label: "voxel-generate" });
            const pass = enc.beginComputePass({ label: "voxel-generate" });
            pipeline
                .with(noiseGroup)
                .with(group)
                .with(pass)
                .dispatchWorkgroups(DISPATCH.x, DISPATCH.y, DISPATCH.z);
            pass.end();
            device.queue.submit([enc.finish()]);
        } catch (cause) {
            nextRaw.destroy();
            throw cause;
        }
        permRaw?.destroy();
        permRaw = nextRaw;
    },
});

/** fill `Voxels.grid` from the heightmap density for `seed`, sync the CPU mirror, then dirty it so the
 *  mesher re-meshes. The pipeline compiles once (baked constants); each call rebuilds the per-seed
 *  permutation table. Runs on its own encoder + submit (decoupled from the frame loop). The sync
 *  (`syncGrid`, a 64 MiB GPU→CPU readback) is part of this call's own contract, not an optional follow-up
 *  — the mesher's chunk allocation is CPU-exact (`facesInChunk`), so a remesh dispatched against a
 *  GPU-only grid has nothing to size chunks from and defers indefinitely (`mesher.ts`'s
 *  `VoxelEmitSystem`). Every caller (boot, reseed, the correctness gate) gets a remeshable grid for free;
 *  `voxel-chunk-streaming` S3 measures whether this cost needs a GPU-count fallback. */
export async function generate(seed: number): Promise<void> {
    if (!Voxels.grid) throw new Error("voxel: generate before the grid buffer exists");
    const { device, root } = Compute;
    await runDensity({ root, device, grid: Voxels.grid }, seed);
    await syncGrid();
    Voxels.dirty = true;
}

/** the emitted density WGSL — the device-free structural seam the voxel tests resolve. */
export function densityWgsl(): string {
    return tgpu.resolve([densityKernel], { names: "strict" });
}

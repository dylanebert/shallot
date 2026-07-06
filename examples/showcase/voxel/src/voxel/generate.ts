// GPU heightmap terrain generation. A compute pass fills the voxel grid from a layered 2D perlin field: a
// zero-mean multi-octave `fbm2(x,z)` lifts and drops the ground into rolling hills —
// `surface = GROUND_LEVEL + fbm2(x·HFREQ, z·HFREQ)·RELIEF`, solid below the surface, air above. The textbook
// layered-perlin landscape, which reads as recognizable landform where isotropic 3D noise read as busy
// pockets. Writes the same `Voxels.grid` buffer the mesher reads, then dirties it so the next frame
// re-meshes. The visual replacement for the hand-authored patterns; the canonical set stays the
// mesher-correctness gate.
//
// The grid stays a full 3D density field — the carve brush (edit.ts) sculpts overhangs and caves by hand —
// but the *generated* terrain is a heightmap, deliberately. Auto-generated 3D caves are the "minecraft"
// direction we're not taking here; rolling hills are the simpler, more pleasing initialization.
//
// Shape: orrstead's generation compute (`orrstead/package/src/generation/{noise,elevation}.ts`) — a seeded
// permutation table in a storage buffer + an FBM WGSL chunk + a workgroup dispatch on its own encoder. The
// pure noise primitives (perm table, perlin/fbm WGSL, heightmap knobs, the derived band) live in noise.ts.

import { Compute } from "@dylanebert/shallot";
import { addressingWgsl, DENSITY, DIM } from "./grid";
import { Voxels } from "./mesher";
import { GROUND_LEVEL, HFREQ, makePermutation, NOISE_WGSL, RELIEF } from "./noise";

export { solidFractionBand } from "./noise";

const WG = 4; // 4³ = 64 threads, matching the mesher's dispatch
const DISPATCH = { x: Math.ceil(DIM.x / WG), y: Math.ceil(DIM.y / WG), z: Math.ceil(DIM.z / WG) };

function densityWgsl(): string {
    return /* wgsl */ `
${addressingWgsl()}

@group(0) @binding(0) var<storage, read> perm: array<u32, 512>;
@group(0) @binding(1) var<storage, read_write> grid: array<f32>;

${NOISE_WGSL}

const HFREQ: f32 = ${HFREQ.toExponential()};
const RELIEF: f32 = ${RELIEF.toExponential()};
const GROUND_LEVEL: f32 = ${GROUND_LEVEL.toExponential()};
const DENSITY: f32 = ${DENSITY.toExponential()};

@compute @workgroup_size(${WG}, ${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= u32(DIM_X) || gid.y >= u32(DIM_Y) || gid.z >= u32(DIM_Z)) { return; }
    let surface = GROUND_LEVEL + fbm2(vec2<f32>(f32(gid.x), f32(gid.z)) * HFREQ) * RELIEF;
    let field = surface - f32(gid.y);
    grid[voxelIndex(gid.x, gid.y, gid.z)] = select(0.0, DENSITY, field > 0.0);
}
`;
}

const gen = {
    pipeline: null as GPUComputePipeline | null,
    perm: null as GPUBuffer | null,
};

/** fill `Voxels.grid` from the heightmap density for `seed`, then dirty it so the mesher re-meshes. The
 *  pipeline compiles once (baked constants); each call rebuilds the per-seed permutation table. Runs on
 *  its own encoder + submit (decoupled from the frame loop, the orrstead generation shape). */
export async function generate(seed: number): Promise<void> {
    if (!Voxels.grid) throw new Error("voxel: generate before the grid buffer exists");
    const { device } = Compute;

    if (!gen.pipeline) {
        const module = device.createShaderModule({ label: "voxel-generate", code: densityWgsl() });
        gen.pipeline = await device.createComputePipelineAsync({
            label: "voxel-generate",
            layout: "auto",
            compute: { module, entryPoint: "main" },
        });
    }

    gen.perm?.destroy();
    const perm = makePermutation(seed);
    gen.perm = device.createBuffer({
        label: "voxel-perm",
        size: perm.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(gen.perm, 0, perm as Uint32Array<ArrayBuffer>);

    const bindGroup = device.createBindGroup({
        label: "voxel-generate",
        layout: gen.pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: gen.perm } },
            { binding: 1, resource: { buffer: Voxels.grid } },
        ],
    });

    const enc = device.createCommandEncoder({ label: "voxel-generate" });
    const pass = enc.beginComputePass({ label: "voxel-generate" });
    pass.setPipeline(gen.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(DISPATCH.x, DISPATCH.y, DISPATCH.z);
    pass.end();
    device.queue.submit([enc.finish()]);

    Voxels.dirty = true;
}

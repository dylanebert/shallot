import {
    Compute,
    mesh,
    type Plugin,
    RenderPlugin,
    type State,
    type System,
} from "@dylanebert/shallot";
import {
    BeginFrameSystem,
    type Draw,
    DrawIndexedIndirect,
    type DrawIndirectBuffer,
    Draws,
    Frame,
    Meshes,
    Render,
} from "@dylanebert/shallot/render/core";
import {
    fsCtxSchema,
    PrepassSystem,
    registerSurface,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "@dylanebert/shallot/sear/core";
import tgpu, { type TgpuBindGroup, type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    integrateKernel,
    PARTICLE_BYTES,
    PARTICLE_COUNT,
    PARTICLE_WORKGROUP,
    Particle,
    ParticleArray,
    type ParticleBuffer,
    particleLayout,
} from "./kernel";

const DISPATCH = Math.ceil(PARTICLE_COUNT / PARTICLE_WORKGROUP);

const SIZE = 0.06;
const TAU = 6.2831853;

// A custom surface's own bindings resolve by name out of `Compute.typed`, so publishing the particle
// buffer under the key this layout declares is the whole wiring between the compute pass that writes
// the particles and the vertex stage that reads them. No CPU round trip, no per-particle entity.
const particlesLayout = surfaceLayout({
    particles: { type: "storage", element: Particle },
});
const particleVaryings = { tint: d.vec3f };
const particlePatch = vsPatchSchema(particleVaryings);

const palette = tgpu.fn(
    [d.f32],
    d.vec3f,
)((t) => {
    "use gpu";
    const phase = d.vec3f(t, t + d.f32(0.33), t + d.f32(0.67));
    return std.add(d.vec3f(0.5), std.mul(d.vec3f(0.5), std.cos(std.mul(phase, d.f32(TAU)))));
});

// One vertex-stage patch per instance: the shared cube's local position is scaled and offset by the
// particle the compute pass wrote for this instance id. `input.iid` indexes the storage buffer directly.
const particleVs = tgpu.fn(
    [VsIn],
    particlePatch,
)((input) => {
    "use gpu";
    const tint = palette(std.fract(d.f32(input.iid) * d.f32(0.6180339887)));
    const pos = particlesLayout.$.particles[input.iid].posSeed.xyz;
    return particlePatch({
        world: d.vec4f(std.add(std.mul(input.localPos, d.f32(SIZE)), pos), d.f32(1)),
        worldNormal: input.worldNormal,
        clip: d.vec4f(0),
        tint,
    });
});

const particleFs = tgpu.fn(
    [fsCtxSchema(particleVaryings)],
    d.vec4f,
)((ctx) => {
    "use gpu";
    return d.vec4f(ctx.tint, d.f32(1));
});

function cubeVertices(): Float32Array {
    const corners: [number, number, number][] = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    const v = new Float32Array(corners.length * 8);
    for (let i = 0; i < corners.length; i++) {
        const [x, y, z] = corners[i];
        const inv = 1 / Math.hypot(x, y, z);
        v.set([x, y, z, 0, x * inv, y * inv, z * inv, 0], i * 8);
    }
    return v;
}

const CUBE_INDICES = new Uint32Array([
    4, 5, 6, 4, 6, 7, 1, 0, 3, 1, 3, 2, 5, 1, 2, 5, 2, 6, 0, 4, 7, 0, 7, 3, 7, 6, 2, 7, 2, 3, 0, 1,
    5, 0, 5, 4,
]);

let particlesRaw: GPUBuffer | null = null;
let particles: ParticleBuffer | null = null;
let argsRaw: GPUBuffer | null = null;
let args: DrawIndirectBuffer | null = null;
let pipeline: TgpuComputePipeline | null = null;
let group: TgpuBindGroup | null = null;
let draw: Draw | null = null;
let dispatches = 0;

/** The particle buffer is only ever *encoded* into the frame's shared encoder; the render plugin
 *  submits it at end of frame. So a readback taken on the same frame as the first dispatch copies a
 *  still-zeroed buffer — the smoke waits for this, which is true only once a dispatch from an earlier
 *  frame has been submitted. */
export function particlesStepped(): boolean {
    return particlesRaw !== null && dispatches > 1;
}

/** The live GPU state, for a consumer that needs the buffer itself: the raw buffer (a readback copies
 *  from it), the typed wrapper the surface's vertex stage resolves by name, and the draw that names
 *  that surface. */
export function particleState(): { raw: GPUBuffer; typed: ParticleBuffer; draw: Draw } | null {
    return particlesRaw && particles && draw ? { raw: particlesRaw, typed: particles, draw } : null;
}

// One dispatch per frame, into the frame's own encoder and before the prepass reads the buffer, so the
// surface always draws the positions this frame's simulation step wrote.
const integrate: System = {
    name: "particles",
    group: "draw",
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update() {
        if (!Render.encoder || !pipeline || !group) return;
        const pass = Render.encoder.beginComputePass({ label: "particles-integrate" });
        pipeline.with(group).with(pass).dispatchWorkgroups(DISPATCH);
        pass.end();
        dispatches++;
    },
};

function build(): void {
    const { device, root } = Compute;
    particlesRaw = device.createBuffer({
        label: "particles",
        size: PARTICLE_BYTES,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    particles = root.createBuffer(ParticleArray, particlesRaw).$usage("storage");
    argsRaw = device.createBuffer({
        label: "particles-draw-args",
        size: d.sizeOf(DrawIndexedIndirect),
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });
    args = root.createBuffer(DrawIndexedIndirect, argsRaw).$usage("indirect");
    const cube = Meshes.get("particleCube");
    if (!cube) throw new Error("gpu-particles: cube mesh not registered");
    // one indirect draw of PARTICLE_COUNT instances, the same cube every time
    args.write({
        indexCount: cube.indexCount,
        instanceCount: PARTICLE_COUNT,
        firstIndex: cube.indexBase,
        baseVertex: 0,
        firstInstance: 0,
    });
    pipeline = root
        .createComputePipeline({ compute: integrateKernel })
        .$name("particles-integrate");
    group = root.createBindGroup(particleLayout, { frame: Frame.buffer, particles });
    draw = {
        name: "particles",
        surface: "particles",
        mesh: "particleCube",
        args: { indirect: args },
    };
    Compute.typed.set("particles", particles);
    Draws.register(draw);
}

// Every State must release its GPU resources on dispose, or a hot reload stacks another buffer set and
// another draw on top of the live one.
function teardown(): void {
    if (draw && Draws.get(draw.name) === draw) Draws.delete(draw.name);
    // only retract this generation's own publication: a reload's build may already have replaced it
    if (particles && Compute.typed.get("particles") === particles)
        Compute.typed.delete("particles");
    particlesRaw?.destroy();
    argsRaw?.destroy();
    particlesRaw = null;
    particles = null;
    argsRaw = null;
    args = null;
    pipeline = null;
    group = null;
    draw = null;
    dispatches = 0;
}

export const ParticlesPlugin: Plugin = {
    name: "Particles",
    dependencies: [RenderPlugin],
    systems: [integrate],
    initialize(state: State) {
        // One static shared cube is instanced by the GPU-owned draw. Particle transforms never enter
        // Part or the Transform firehose: the typed particle record is both the compute output and the
        // vertex-stage input.
        mesh({ name: "particleCube", vertices: cubeVertices(), indices: CUBE_INDICES });
        registerSurface(state, {
            name: "particles",
            layout: particlesLayout,
            varyings: particleVaryings,
            vs: particleVs,
            fs: particleFs,
        });
    },
    warm: build,
    dispose: teardown,
};

export default ParticlesPlugin;

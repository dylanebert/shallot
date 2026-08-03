import {
    Compute,
    mesh,
    type Plugin,
    precompile,
    precompileScope,
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
    FrameGpu,
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
import {
    cleanupFountainGeneration,
    createFountainLifecycle,
    withOwnedFountainBuffers,
} from "./lifecycle";

const DISPATCH = Math.ceil(PARTICLE_COUNT / PARTICLE_WORKGROUP);

const SIZE = 0.06;
const TAU = 6.2831853;

const fountainLayout = surfaceLayout({
    fountainParticles: { type: "storage", element: Particle },
});
const fountainVaryings = { tint: d.vec3f };
const fountainPatch = vsPatchSchema(fountainVaryings);

const palette = tgpu.fn(
    [d.f32],
    d.vec3f,
)((t) => {
    "use gpu";
    const phase = d.vec3f(t, t + d.f32(0.33), t + d.f32(0.67));
    return std.add(d.vec3f(0.5), std.mul(d.vec3f(0.5), std.cos(std.mul(phase, d.f32(TAU)))));
});

const fountainVs = tgpu.fn(
    [VsIn],
    fountainPatch,
)((input) => {
    "use gpu";
    const tint = palette(std.fract(d.f32(input.iid) * d.f32(0.6180339887)));
    const pos = fountainLayout.$.fountainParticles[input.iid].posSeed.xyz;
    return fountainPatch({
        world: d.vec4f(std.add(std.mul(input.localPos, d.f32(SIZE)), pos), d.f32(1)),
        worldNormal: input.worldNormal,
        clip: d.vec4f(0),
        tint,
    });
});

const fountainFs = tgpu.fn(
    [fsCtxSchema(fountainVaryings)],
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

declare global {
    interface Window {
        __fountainGate?: () => Promise<FountainCheck[]>;
    }
}

export interface FountainCheck {
    name: string;
    pass: boolean;
    detail: string;
}

type FountainDraft = { readonly state: State; readonly buffers: GPUBuffer[] };
type FountainOwner = {
    readonly state: State;
    readonly buffers: readonly GPUBuffer[];
    readonly particlesRaw: GPUBuffer;
    readonly particles: ParticleBuffer;
    readonly args: DrawIndirectBuffer;
    readonly pipeline: TgpuComputePipeline;
    readonly group: TgpuBindGroup;
    readonly draw: Draw;
    readonly gate: () => Promise<FountainCheck[]>;
    readonly scope: string;
};

let activeOwner: FountainOwner | null = null;
const stateOwners = new WeakMap<State, FountainOwner>();
const dispatchedOwners = new WeakSet<FountainOwner>();

function cleanupOwner(owner: FountainOwner): void {
    cleanupFountainGeneration(owner, {
        raw: () => Compute.buffers?.get("fountainParticles"),
        typed: () => Compute.typed?.get("fountainParticles") as ParticleBuffer | undefined,
        draw: (name) => Draws.get(name),
        gate: () => (typeof window === "undefined" ? undefined : window.__fountainGate),
        deleteRaw: () => Compute.buffers.delete("fountainParticles"),
        deleteTyped: () => Compute.typed.delete("fountainParticles"),
        deleteDraw: (name) => Draws.delete(name),
        deleteGate: () => {
            delete window.__fountainGate;
        },
        destroy: (buffer) => buffer.destroy(),
        active: () => activeOwner,
        clearActive: () => {
            activeOwner = null;
        },
    });
    if (stateOwners.get(owner.state) === owner) stateOwners.delete(owner.state);
}

async function readParticles(owner: FountainOwner): Promise<Float32Array> {
    const staging = Compute.device.createBuffer({
        label: "fountain-readback",
        size: PARTICLE_BYTES,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    let mapped = false;
    try {
        const enc = Compute.device.createCommandEncoder({ label: "fountain-readback" });
        enc.copyBufferToBuffer(owner.particlesRaw, 0, staging, 0, PARTICLE_BYTES);
        Compute.device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        mapped = true;
        return new Float32Array(staging.getMappedRange().slice(0));
    } finally {
        if (mapped) staging.unmap();
        staging.destroy();
    }
}

function frames(count: number): Promise<void> {
    return new Promise((resolve) => {
        let remaining = count;
        const tick = () => {
            remaining--;
            if (remaining === 0) resolve();
            else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    });
}

async function checkFountain(owner: FountainOwner): Promise<FountainCheck[]> {
    if (activeOwner !== owner) throw new Error("fountain: gate owner is stale");
    const beforeFrame = Compute.frame;
    const before = await readParticles(owner);
    await frames(4);
    const after = await readParticles(owner);
    const indices = [0, PARTICLE_WORKGROUP + 1, PARTICLE_COUNT - 1];
    const stride = d.sizeOf(Particle) / 4;
    const finite = indices.every((i) => {
        const at = i * stride;
        return after[at + 3] === 1 && after.slice(at, at + stride).every(Number.isFinite);
    });
    const sampled = indices.map((i) => Array.from(after.slice(i * stride, i * stride + stride)));
    const progressed = indices.some((i) => {
        const at = i * stride;
        return (
            Math.abs(after[at] - before[at]) +
                Math.abs(after[at + 1] - before[at + 1]) +
                Math.abs(after[at + 2] - before[at + 2]) >
            1e-6
        );
    });
    return [
        {
            name: "typed-publication",
            pass:
                Compute.buffers.get("fountainParticles") === owner.particlesRaw &&
                Compute.typed.get("fountainParticles") === owner.particles &&
                Draws.get(owner.draw.name) === owner.draw,
            detail: "raw and typed particle identities plus the exact draw owner are published",
        },
        {
            name: "multi-workgroup-tail",
            pass: after.length === PARTICLE_BYTES / 4 && finite,
            detail: `sampled particles ${indices.join(", ")} across ${after.length} f32 lanes: ${JSON.stringify(sampled)}`,
        },
        {
            name: "ordered-progression",
            pass: Compute.frame > beforeFrame && progressed,
            detail: `frames ${beforeFrame} -> ${Compute.frame}; sampled positions ${progressed ? "changed" : "stalled"}`,
        },
    ];
}

export const FountainSystem: System = {
    name: "fountain",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update() {
        const owner = activeOwner;
        if (!Render.encoder || !owner) return;
        // Publish the gate only on a frame after this generation first encoded. EndFrame has submitted
        // that prior dispatch by then, so a boot-time browser poll cannot observe a zero-filled buffer.
        if (dispatchedOwners.has(owner) && window.__fountainGate !== owner.gate)
            window.__fountainGate = owner.gate;
        const pass = Render.encoder.beginComputePass({
            label: "fountain-integrate",
            timestampWrites: Compute.span?.("fountain:integrate"),
        });
        owner.pipeline.with(owner.group).with(pass).dispatchWorkgroups(DISPATCH);
        pass.end();
        dispatchedOwners.add(owner);
    },
};

function forceCompile(owner: FountainOwner): TgpuComputePipeline {
    const { device, root } = Compute;
    return withOwnedFountainBuffers<GPUBuffer, TgpuComputePipeline>(
        (own) => {
            const frameRaw = own(
                device.createBuffer({
                    label: "fountain-precompile-frame",
                    size: Math.ceil(d.sizeOf(FrameGpu) / 16) * 16,
                    usage: GPUBufferUsage.UNIFORM,
                }),
            );
            const particlesRaw = own(
                device.createBuffer({
                    label: "fountain-precompile-particles",
                    size: PARTICLE_BYTES,
                    usage: GPUBufferUsage.STORAGE,
                }),
            );
            const frame = root.createBuffer(FrameGpu, frameRaw).$usage("uniform");
            const particles = root.createBuffer(ParticleArray, particlesRaw).$usage("storage");
            const group = root.createBindGroup(particleLayout, { frame, particles });
            const enc = device.createCommandEncoder({ label: owner.scope });
            const pass = enc.beginComputePass({ label: owner.scope });
            owner.pipeline.with(group).with(pass).dispatchWorkgroups(0);
            pass.end();
            device.queue.submit([enc.finish()]);
            return owner.pipeline;
        },
        (buffer) => buffer.destroy(),
    );
}

const lifecycle = createFountainLifecycle<State, FountainDraft, FountainOwner>({
    current: () => activeOwner,
    owned: (state) => stateOwners.get(state),
    draft: (state) => ({ state, buffers: [] }),
    prepare(draft) {
        const { device, root } = Compute;
        const particlesRaw = device.createBuffer({
            label: "fountain-particles",
            size: PARTICLE_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        draft.buffers.push(particlesRaw);
        const particles = root.createBuffer(ParticleArray, particlesRaw).$usage("storage");
        const argsRaw = device.createBuffer({
            label: "fountain-draw-args",
            size: d.sizeOf(DrawIndexedIndirect),
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        draft.buffers.push(argsRaw);
        const args = root.createBuffer(DrawIndexedIndirect, argsRaw).$usage("indirect");
        const cube = Meshes.get("fountainCube");
        if (!cube) throw new Error("fountain: cube mesh not registered");
        args.write({
            indexCount: cube.indexCount,
            instanceCount: PARTICLE_COUNT,
            firstIndex: cube.indexBase,
            baseVertex: 0,
            firstInstance: 0,
        });
        const pipeline = root
            .createComputePipeline({ compute: integrateKernel })
            .$name("fountain-integrate");
        const group = root.createBindGroup(particleLayout, {
            frame: Frame.buffer,
            particles,
        });
        const draw: Draw = {
            name: "fountain",
            surface: "fountain",
            mesh: "fountainCube",
            args: { indirect: args },
        };
        let owner!: FountainOwner;
        const gate = () => checkFountain(owner);
        owner = Object.freeze({
            state: draft.state,
            buffers: Object.freeze([...draft.buffers]),
            particlesRaw,
            particles,
            args,
            pipeline,
            group,
            draw,
            gate,
            scope: precompileScope("fountain-integrate"),
        });
        return owner;
    },
    activate(state, owner) {
        activeOwner = owner;
        stateOwners.set(state, owner);
    },
    cleanupDraft(draft) {
        for (const buffer of draft.buffers) buffer.destroy();
    },
    cleanup: cleanupOwner,
    precompile: (owner, force) => precompile(owner.scope, force),
    force: forceCompile,
    publish(owner) {
        Compute.buffers.set("fountainParticles", owner.particlesRaw);
        Compute.typed.set("fountainParticles", owner.particles);
        Draws.register(owner.draw);
    },
});

export const warmFountain = lifecycle.warm;
export const disposeFountain = lifecycle.dispose;

export const FountainPlugin: Plugin = {
    name: "Fountain",
    dependencies: [RenderPlugin],
    systems: [FountainSystem],
    initialize(state) {
        // One static shared cube is instanced by the GPU-owned draw. Particle transforms never enter Part
        // or the Transform firehose: the typed particle record is both the compute output and VS input.
        mesh({ name: "fountainCube", vertices: cubeVertices(), indices: CUBE_INDICES });
        registerSurface(state, {
            name: "fountain",
            layout: fountainLayout,
            varyings: fountainVaryings,
            vs: fountainVs,
            fs: fountainFs,
        });
    },
    warm: warmFountain,
    dispose: disposeFountain,
};

export default FountainPlugin;

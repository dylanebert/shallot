import { Compute, mesh, type Plugin, RenderPlugin, type System } from "@dylanebert/shallot";
import {
    BeginFrameSystem,
    Draws,
    FRAME_STRUCT_WGSL,
    Frame,
    Meshes,
    Render,
    Surfaces,
} from "@dylanebert/shallot/render/core";
import { PrepassSystem } from "@dylanebert/shallot/sear/core";

// The canonical GPU-driven particle path as a producer: a compute pass integrates
// N particles (launch, gravity, ground collision, recycle) in a single STORAGE buffer,
// and a sear surface draws one cube per particle from it — instanced, indirect, all on
// the GPU. No Part, no Transform: the count never crosses CPU↔GPU, and the only CPU work
// per frame is encoding one dispatch. This is the pure-compute escape hatch the kitchen
// contract leaves open for simulation that writes its own buffers. The ground plane it
// lands on is a plain Part in the scene (main.ts): the happy path alongside the producer.

const COUNT = 50_000;
const WORKGROUP = 64;
const DISPATCH = Math.ceil(COUNT / WORKGROUP);
const PARTICLE_STRIDE = 32; // posSeed: vec4 + vel: vec4

// fountain shape — tuned by hot-reload, not runtime inputs (no live sliders here)
const SIZE = 0.06; // cube edge in world units
const GRAVITY = 9.8;
const UP_MIN = 6.5;
const UP_MAX = 8.0;
const SPREAD = 1.5; // lateral launch speed (cone radius)
const SPAWN_Y = 0.05; // spout height — just above the ground so spawned cubes don't z-fight it
const GROUND_Y = 0.0; // particles recycle when they fall back to this plane

const Fountain = {
    particles: null as unknown as GPUBuffer,
    args: null as unknown as GPUBuffer,
    pipeline: null as GPUComputePipeline | null,
    bindGroup: null as GPUBindGroup | null,
};

// shared hash + jet, used by both the first-frame seed and the per-frame relaunch so the
// fountain's launch distribution has one source of truth
const PARTICLE_WGSL = /* wgsl */ `
const COUNT: u32 = ${COUNT}u;
const GRAVITY: f32 = ${GRAVITY};
const UP_MIN: f32 = ${UP_MIN};
const UP_MAX: f32 = ${UP_MAX};
const SPREAD: f32 = ${SPREAD};
const SPAWN: vec3<f32> = vec3<f32>(0.0, ${SPAWN_Y}, 0.0);
const GROUND_Y: f32 = ${GROUND_Y};
const TAU: f32 = 6.2831853;
const GOLDEN: u32 = 0x9e3779b9u;

struct Particle {
    posSeed: vec4<f32>,
    vel: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;

fn hashU32(x: u32) -> u32 {
    var h = x;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return h;
}

fn rnd(s: u32) -> f32 {
    return f32(hashU32(s)) * (1.0 / 4294967296.0);
}

// upward jet inside a cone: uniform azimuth, sqrt radius for a uniform disk, lerped speed
fn jet(seed: u32) -> vec3<f32> {
    let ang = rnd(seed) * TAU;
    let rad = sqrt(rnd(seed + GOLDEN)) * SPREAD;
    let up = mix(UP_MIN, UP_MAX, rnd(seed + 0x85ebca6bu));
    return vec3<f32>(cos(ang) * rad, up, sin(ang) * rad);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= COUNT) { return; }

    let g = vec3<f32>(0.0, -GRAVITY, 0.0);
    var p = particles[i];
    var pos = p.posSeed.xyz;
    var vel = p.vel.xyz;

    // a freshly-created buffer is zero-filled, so the seed flag (posSeed.w) reads 0 on an
    // un-seeded slot. Place it along its arc at a random fraction of its flight time (closed
    // form, constant gravity) so the fountain is full on frame one; seeding everything at the
    // spout would burst as one synchronized jet.
    if (p.posSeed.w < 0.5) {
        let v0 = jet(i + 0x1234567u);
        let flight = 2.0 * v0.y / GRAVITY; // time to fall back to spout height
        let t = rnd(i * 2u + 1u) * flight; // mid-arc → above the ground
        pos = SPAWN + v0 * t + 0.5 * g * t * t;
        vel = v0 + g * t;
    } else {
        let dt = frame.dt;
        vel += g * dt;
        pos += vel * dt;
        // terminate on the ground, descending only (a particle launches from the spout moving
        // up, so the gate fires only as it arcs back down), then relaunch with a fresh jet
        // seeded by the frame for cycle-to-cycle variety
        if (pos.y <= GROUND_Y && vel.y < 0.0) {
            vel = jet(i ^ (frame.frame * GOLDEN));
            pos = SPAWN;
        }
    }

    particles[i] = Particle(vec4<f32>(pos, 1.0), vec4<f32>(vel, 0.0));
}
`;

// 8-corner unit cube, CCW-outward (sear culls back faces). normals point out of each
// corner, unused by the unlit surface but non-zero so sear's normalize() never sees 0
const CORNERS: [number, number, number][] = [
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
];

// two CCW-outward triangles per face: +Z, -Z, +X, -X, +Y, -Y
// biome-ignore format: one face per line reads as the cube it is
const CUBE_INDICES = new Uint32Array([
    4, 5, 6, 4, 6, 7,
    1, 0, 3, 1, 3, 2,
    5, 1, 2, 5, 2, 6,
    0, 4, 7, 0, 7, 3,
    7, 6, 2, 7, 2, 3,
    0, 1, 5, 0, 5, 4,
]);

function cubeVertices(): Float32Array {
    const v = new Float32Array(CORNERS.length * 8); // stride 8: posXYZ uvX normalXYZ uvY
    for (let i = 0; i < CORNERS.length; i++) {
        const [x, y, z] = CORNERS[i];
        const inv = 1 / Math.hypot(x, y, z);
        v.set([x, y, z, 0, x * inv, y * inv, z * inv, 0], i * 8);
    }
    return v;
}

// Dispatches the integrate pass before sear reads geometry. The surface's vs reads each
// particle's position to place its cube, so the positions are position-determining: sear
// reads them in the prepass, the shadow map, and the color pass, and an emit dropped between
// them would tear new geometry against a stale read (see kitchen.md "System ordering").
// `before: [PrepassSystem]` pins it ahead of every geometry pass.
const FountainSystem: System = {
    name: "fountain",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    async setup() {
        const { device } = Compute;

        Fountain.particles = device.createBuffer({
            label: "fountain-particles",
            size: COUNT * PARTICLE_STRIDE,
            usage: GPUBufferUsage.STORAGE,
        });
        // the surface reads this buffer as array<vec4<f32>>: particle i's position is
        // element 2*i (posSeed.xyz). Published by name for the surface to resolve
        Compute.buffers.set("fountainParticles", Fountain.particles);

        // CPU-known draw: a fixed instance count, one record. Everything is indirect, so the
        // count lives in a buffer rather than literal draw args. firstIndex = the cube's slice
        // base in the shared family buffer (mesh() packs it at RenderPlugin.warm)
        const cube = Meshes.get("fountainCube");
        if (!cube) throw new Error("fountain: cube mesh not registered");
        Fountain.args = device.createBuffer({
            label: "fountain-draw-args",
            size: 20,
            usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(
            Fountain.args,
            0,
            new Uint32Array([cube.indexCount, COUNT, cube.indexBase, 0, 0]),
        );

        Draws.register({
            name: "fountain",
            surface: "fountain",
            mesh: "fountainCube",
            args: { indirect: Fountain.args },
        });

        const module = device.createShaderModule({
            label: "fountain-integrate",
            code: `${FRAME_STRUCT_WGSL}\n${PARTICLE_WGSL}`,
        });
        Fountain.pipeline = await device.createComputePipelineAsync({
            label: "fountain-integrate",
            layout: "auto",
            compute: { module, entryPoint: "main" },
        });
        Fountain.bindGroup = device.createBindGroup({
            label: "fountain-integrate",
            layout: Fountain.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: Frame.buffer } },
                { binding: 1, resource: { buffer: Fountain.particles } },
            ],
        });
    },
    update() {
        if (!Render.encoder || !Fountain.pipeline || !Fountain.bindGroup) return;
        const pass = Render.encoder.beginComputePass({
            label: "fountain-integrate",
            timestampWrites: Compute.span?.("fountain:integrate"),
        });
        pass.setPipeline(Fountain.pipeline);
        pass.setBindGroup(0, Fountain.bindGroup);
        pass.dispatchWorkgroups(DISPATCH);
        pass.end();
    },
};

const FountainPlugin: Plugin = {
    name: "Fountain",
    dependencies: [RenderPlugin],
    systems: [FountainSystem],
    initialize() {
        // a plain static cube in the shared family buffer (registered before warm), and an
        // unlit surface that offsets each cube by its particle position. No eids/transforms
        // bindings, so sear applies no per-instance transform; the vs builds `world` itself
        // from `fountainParticles[iid]`, the producer's own instance buffer. Per-particle color
        // is a low-discrepancy index → cosine palette, carried as a vs→fs varying (constant
        // per cube, so it interpolates exact)
        mesh({ name: "fountainCube", vertices: cubeVertices(), indices: CUBE_INDICES });
        Surfaces.register({
            name: "fountain",
            bindings: {
                fountainParticles: { type: "storage", element: "vec4<f32>" },
            },
            interpolators: { tint: "vec3<f32>" },
            preamble: /* wgsl */ `
                const SIZE: f32 = ${SIZE};
                // Inigo Quilez cosine palette: a smooth spread of hues from one scalar
                fn palette(t: f32) -> vec3<f32> {
                    let d = vec3<f32>(0.0, 0.33, 0.67);
                    return vec3<f32>(0.5) + vec3<f32>(0.5) * cos(6.2831853 * (vec3<f32>(t) + d));
                }
            `,
            vs: /* wgsl */ `
                tint = palette(fract(f32(iid) * 0.6180339887));
                world = vec4<f32>(localPos * SIZE + fountainParticles[iid * 2u].xyz, 1.0);
            `,
            fs: /* wgsl */ `
                col = vec4<f32>(tint, 1.0);
            `,
        });
    },
};

// the manifest references this module by path (`"Fountain": "./src/fountain"`) and imports its default
export default FountainPlugin;

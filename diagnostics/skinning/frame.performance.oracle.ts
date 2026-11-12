import { expect, test } from "bun:test";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import {
    Camera,
    ClearChangeMarksSystem,
    createApp,
    DirectionalLight,
    lookAtRotation,
    Materials,
    Meshes,
    type MeshHandle,
    MeshInstance,
    MeshMaterial,
    type Plugin,
    StandardMaterial,
    StandardRenderer,
    Time,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    evaluateGltfAnimation,
    type GltfNodePose,
    type GltfSceneInstance,
    GltfPlugin,
    loadGltf,
    loadGltfScene,
    poseGltfScene,
    setGltfLeaderPose,
    SkinPalette,
    SkinnedMaterial,
    SkinnedMesh,
    spawnGltfScene,
    parseGlb,
} from "@dylanebert/shallot/gltf";
import {
    attachTexture,
    captureTexture,
    DepthPrepass,
    NormalPrepass,
    xformNormal,
    xformPoint,
} from "@dylanebert/shallot/rendering";
import {
    applyPbrLighting,
    extendedMaterial,
    extensionLayout,
    MaterialPlugin,
    MeshPreprocessSystem,
    MeshRenderPlugin,
    pbrInputFromStandardMaterial,
    StandardRenderingPlugin,
} from "@dylanebert/shallot/standard/rendering";
import { composeMat4, decomposeMat4, multiplyMat4 } from "@dylanebert/shallot/utils";
import { setupGlobals } from "@dylanebert/shallot/webgpu";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import manifest from "../../examples/gltf/assets.json";
import { syncFiles } from "../../examples/gltf/sync";

// bun test ./diagnostics/skinning/frame.performance.oracle.ts
// SKINNING_MODES=unskinned,shipped selects the shipped glTF path and its textured bind-pose baseline.
// The spike modes measure linear blend skinning on one hardware adapter: unskinned
// bind-pose StandardMaterial (baseline), skinning in a StandardMaterial extension's vertex function
// reading a palette storage buffer by eid, and skinning in a compute pass before the draw that the
// extension's vertex function then reads. Sun shadows (four cascades) and the depth and normal
// prepasses are on, so every vertex run of every pass is timed.

const FILE = {
    path: "Models/CesiumMan/glTF-Binary/CesiumMan.glb",
    sha256: "b7001eaeea8254bd44773bcd247e78696d94169388fbb2a1800fc69434e777d9",
};
// 2560 is the largest crowd whose compute-deformed rows fit one default 128 MiB storage binding.
const POPULATIONS = process.env.SKINNING_POPULATIONS?.split(",").map(Number) ?? [640, 1280, 2560];
const ALL_MODES = ["static", "vertex", "compute", "unskinned", "shipped", "followers"] as const;
type Mode = (typeof ALL_MODES)[number];
const MODES: readonly Mode[] = process.env.SKINNING_MODES?.split(",").map((mode) => {
    if (!ALL_MODES.includes(mode as Mode)) throw new Error(`Unknown skinning mode: ${mode}`);
    return mode as Mode;
}) ?? ALL_MODES;
// "all": four-cascade sun shadows and the depth and normal prepasses; "color": the color pass alone,
// so the difference is what the shadow and prepass runs add.
const PASSES = ["all", "color"] as const;
type Passes = (typeof PASSES)[number];
const WARM = 30;
// Serial frames: each waits for its timestamps, timing CPU work and per-pass GPU spans.
const FRAMES = 120;
// Pipelined frames: submitted back to back, timing sustained GPU throughput.
const THROUGHPUT = 240;
// "1080p": 4x MSAA at 1920x1080; "270p": single-sample 480x270, where fragment work shrinks and
// vertex work is a larger share of every pass.
const VIEWS = [
    { name: "1080p", width: 1920, height: 1080, antialias: 1 },
    { name: "270p", width: 480, height: 270, antialias: 0 },
] as const;
type View = (typeof VIEWS)[number];
const SPACING = 1.25;

const ROOT = join(import.meta.dir, "../..");
const ASSETS = join(ROOT, "test-results/skinning-assets");

function median(values: number[]): number {
    return values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)]!;
}

// Deterministic per-instance phase and yaw.
function hash(i: number): number {
    let x = (i + 1) * 0x9e3779b1;
    x ^= x >>> 16;
    x = Math.imul(x, 0x85ebca6b);
    x ^= x >>> 13;
    return (x >>> 0) / 0x100000000;
}

interface Skin {
    nodes: Awaited<ReturnType<typeof loadGltf>>["nodes"];
    animation: Awaited<ReturnType<typeof loadGltf>>["animations"][number];
    duration: number;
    vertices: Float32Array;
    indices: Uint32Array;
    joints: Uint32Array; // packed u8x4 per vertex
    weights: Float32Array; // vec4 per vertex
    bind: Float32Array; // (pos.xyz, 0)(normal.xyz, 0) per vertex, mesh space
    jointNodes: Int32Array; // scene-node index per joint
    inverseBinds: Float32Array[];
    boxes: { joint: number; center: number[]; half: number[] }[];
    meshNode: number;
}

function readFloats(
    json: ReturnType<typeof parseGlb>["json"],
    bin: ArrayBuffer,
    index: number,
): Float32Array {
    const accessor = json.accessors![index]!;
    const view = json.bufferViews![accessor.bufferView!]!;
    const comps = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT2: 4, MAT3: 9, MAT4: 16 }[
        accessor.type as "VEC4"
    ];
    const size = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }[
        accessor.componentType as 5126
    ];
    const stride = view.byteStride ?? comps * size;
    const base = (view.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
    const data = new DataView(bin);
    const out = new Float32Array(accessor.count * comps);
    for (let i = 0; i < accessor.count; i++)
        for (let c = 0; c < comps; c++) {
            const at = base + i * stride + c * size;
            const type = accessor.componentType;
            const scale = accessor.normalized ? (type === 5121 ? 255 : 65535) : 1;
            out[i * comps + c] =
                type === 5126
                    ? data.getFloat32(at, true)
                    : type === 5121
                      ? data.getUint8(at) / scale
                      : data.getUint16(at, true) / scale;
        }
    return out;
}

async function loadSkin(): Promise<Skin> {
    await syncFiles([FILE], new URL(`${manifest.source}${manifest.commit}/`), ASSETS);
    const path = join(ASSETS, FILE.path);
    const bytes = await Bun.file(path).arrayBuffer();
    const { json, bin } = parseGlb(bytes);
    const scene = await loadGltf(Bun.pathToFileURL(path));
    expect(scene.meshes.length).toBe(1);
    expect(json.skins!.length).toBe(1);
    const primitive = json.meshes![0]!.primitives[0]!;
    const positions = readFloats(json, bin!, primitive.attributes.POSITION!);
    const normals = readFloats(json, bin!, primitive.attributes.NORMAL!);
    const rawJoints = readFloats(json, bin!, primitive.attributes.JOINTS_0!);
    const weights = readFloats(json, bin!, primitive.attributes.WEIGHTS_0!);
    const mesh = scene.meshes[0]!;
    const count = positions.length / 3;
    expect(mesh.vertices.length / 8).toBe(count);
    // The decoded mesh keeps accessor vertex order, so influences index the same vertices.
    for (let v = 0; v < count; v++)
        for (let c = 0; c < 3; c++)
            expect(Math.abs(mesh.vertices[v * 8 + c]! - positions[v * 3 + c]!)).toBeLessThan(1e-6);
    const skin = json.skins![0] as { joints: number[]; inverseBindMatrices: number };
    const ibm = readFloats(json, bin!, skin.inverseBindMatrices!);
    const jointNodes = Int32Array.from(skin.joints, (node: number) =>
        scene.nodes.findIndex((n) => n.index === node),
    );
    expect(jointNodes.every((n) => n >= 0)).toBe(true);
    expect(skin.joints.length).toBeLessThanOrEqual(256);
    const joints = new Uint32Array(count);
    const bind = new Float32Array(count * 8);
    const lo = skin.joints.map(() => [Infinity, Infinity, Infinity]);
    const hi = skin.joints.map(() => [-Infinity, -Infinity, -Infinity]);
    for (let v = 0; v < count; v++) {
        let packed = 0;
        for (let k = 0; k < 4; k++) {
            const joint = rawJoints[v * 4 + k]!;
            packed |= joint << (8 * k);
            // SkinnedMeshBounds::from_mesh: each joint's box encloses the vertices it weights above zero.
            if (weights[v * 4 + k]! > 0)
                for (let c = 0; c < 3; c++) {
                    lo[joint]![c] = Math.min(lo[joint]![c]!, positions[v * 3 + c]!);
                    hi[joint]![c] = Math.max(hi[joint]![c]!, positions[v * 3 + c]!);
                }
        }
        joints[v] = packed >>> 0;
        bind.set([positions[v * 3]!, positions[v * 3 + 1]!, positions[v * 3 + 2]!, 0], v * 8);
        bind.set([normals[v * 3]!, normals[v * 3 + 1]!, normals[v * 3 + 2]!, 0], v * 8 + 4);
    }
    const boxes = lo.flatMap((min, joint) =>
        min[0] === Infinity
            ? []
            : [
                  {
                      joint,
                      center: min.map((m, c) => (m + hi[joint]![c]!) / 2),
                      half: min.map((m, c) => (hi[joint]![c]! - m) / 2),
                  },
              ],
    );
    const animation = scene.animations[0]!;
    const duration = Math.max(...animation.samplers.map((s) => s.input[s.input.length - 1]!));
    const meshNode = scene.nodes.findIndex((n) => n.index === scene.instances[0]!.nodeIndex);
    return {
        nodes: scene.nodes,
        animation,
        duration,
        vertices: mesh.vertices,
        indices: mesh.indices,
        joints,
        weights,
        bind,
        jointNodes,
        inverseBinds: Array.from({ length: skin.joints.length }, (_, j) =>
            ibm.slice(j * 16, j * 16 + 16),
        ),
        boxes,
        meshNode: meshNode < 0 ? 0 : meshNode,
    };
}

// Bevy skinning.wesl: skin_model blends the four joint matrices; skin_normals applies the
// blended matrix's inverse transpose.
const skinModel = tgpu.fn(
    [d.mat4x4f, d.mat4x4f, d.mat4x4f, d.mat4x4f, d.vec4f],
    d.mat4x4f,
)((a, b, c, e, w) => {
    "use gpu";
    return std.add(
        std.add(std.mul(w.x, a), std.mul(w.y, b)),
        std.add(std.mul(w.z, c), std.mul(w.w, e)),
    );
});
const skinNormal = tgpu.fn(
    [d.mat4x4f, d.vec3f],
    d.vec3f,
)((m, n) => {
    "use gpu";
    const c0 = m.columns[0].xyz;
    const c1 = m.columns[1].xyz;
    const c2 = m.columns[2].xyz;
    const x = std.cross(c1, c2);
    const y = std.cross(c2, c0);
    const z = std.cross(c0, c1);
    const det = std.dot(c2, z);
    return std.normalize(
        std.add(std.add(std.mul(n.x / det, x), std.mul(n.y / det, y)), std.mul(n.z / det, z)),
    );
});

// StandardMaterial's layouts leave one vertex-stage storage buffer of WebGPU's default eight, so
// each extension packs everything it reads by eid into one array<vec4f>: a header row per eid
// (u32 offsets) followed by its regions.
interface Buffers {
    skinning: GPUBuffer;
    palette: GPUBuffer;
    bind: GPUBuffer;
    influences: GPUBuffer;
    header: number;
}
const Gpu = { create: () => ({ buffers: undefined as Buffers | undefined }) };

const skinningLayout = extensionLayout(Materials, d.struct({ skinPad: d.u32 }), {
    skinning: { type: "storage", element: d.vec4u, visibility: ["vertex"] },
});
const skinningBindings = (world: World) => ({ skinning: world.resource(Gpu).buffers!.skinning });
// Float rows are stored as their bits, read with bitcast.
const jointMatrix = tgpu.fn(
    [d.u32],
    d.mat4x4f,
)((row) => {
    "use gpu";
    return d.mat4x4f(
        std.bitcastU32toF32(skinningLayout.$.skinning[row]),
        std.bitcastU32toF32(skinningLayout.$.skinning[row + 1]),
        std.bitcastU32toF32(skinningLayout.$.skinning[row + 2]),
        std.bitcastU32toF32(skinningLayout.$.skinning[row + 3]),
    );
});

// Header x: the instance's first palette row; y: its mesh's first influence row.
// Influence rows: (packed u8x4 joints, 0, 0, 0), (weights). Palette rows: four columns per joint.
const VertexSkinned = extendedMaterial(Materials, {
    name: "SpikeVertexSkinned",
    layout: skinningLayout,
    defaults: { skinPad: 0 },
    bindings: skinningBindings,
    vertex: (input) => {
        "use gpu";
        const header = skinningLayout.$.skinning[input.eid];
        const row = header.y + input.vidx * 2;
        const j = std.add(std.mul(std.unpack4xU8(skinningLayout.$.skinning[row].x), 4), header.x);
        const m = skinModel(
            jointMatrix(j.x),
            jointMatrix(j.y),
            jointMatrix(j.z),
            jointMatrix(j.w),
            std.bitcastU32toF32(skinningLayout.$.skinning[row + 1]),
        );
        const local = std.mul(m, d.vec4f(input.localPos, 1)).xyz;
        return {
            world: d.vec4f(xformPoint(input.xform, local), 1),
            worldNormal: xformNormal(input.xform, skinNormal(m, input.localNormal)),
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return applyLighting(ctx);
    },
});

// Header x: the instance's first deformed row. Deformed rows: (position bits, pack4x8snorm normal).
const ComputeSkinned = extendedMaterial(Materials, {
    name: "SpikeComputeSkinned",
    layout: skinningLayout,
    defaults: { skinPad: 0 },
    bindings: skinningBindings,
    vertex: (input) => {
        "use gpu";
        const v = skinningLayout.$.skinning[skinningLayout.$.skinning[input.eid].x + input.vidx];
        const n = std.unpack4x8snorm(v.w).xyz;
        return {
            world: d.vec4f(xformPoint(input.xform, std.bitcastU32toF32(v.xyz)), 1),
            worldNormal: xformNormal(input.xform, n),
        };
    },
    fragment: (ctx) => {
        "use gpu";
        return applyLighting(ctx);
    },
});

const applyLighting = (ctx: Parameters<typeof pbrInputFromStandardMaterial>[0]) => {
    "use gpu";
    return applyPbrLighting(pbrInputFromStandardMaterial(ctx));
};

const Influence = d.struct({ joints: d.u32, weights: d.vec4f });
const kernelLayout = tgpu.bindGroupLayout({
    bind: { storage: d.arrayOf(d.vec4f), access: "readonly" },
    influences: { storage: d.arrayOf(Influence), access: "readonly" },
    palette: { storage: d.arrayOf(d.mat4x4f), access: "readonly" },
    deformed: { storage: d.arrayOf(d.vec4u), access: "mutable" },
    params: { uniform: d.vec4u },
});
const skinKernel = tgpu.computeFn({
    workgroupSize: [64],
    in: { gid: d.builtin.globalInvocationId },
})((input) => {
    "use gpu";
    const vertices = kernelLayout.$.params.x;
    const joints = kernelLayout.$.params.y;
    const v = input.gid.x;
    if (v >= vertices) return;
    const instance = input.gid.y;
    const influence = kernelLayout.$.influences[v];
    const j = std.unpack4xU8(influence.joints);
    const offset = instance * joints;
    const m = skinModel(
        kernelLayout.$.palette[offset + j.x],
        kernelLayout.$.palette[offset + j.y],
        kernelLayout.$.palette[offset + j.z],
        kernelLayout.$.palette[offset + j.w],
        influence.weights,
    );
    const p = std.mul(m, d.vec4f(kernelLayout.$.bind[v * 2].xyz, 1)).xyz;
    const n = skinNormal(m, kernelLayout.$.bind[v * 2 + 1].xyz);
    kernelLayout.$.deformed[kernelLayout.$.params.z + instance * vertices + v] = d.vec4u(
        std.bitcastF32toU32(p),
        std.pack4x8snorm(d.vec4f(n, 0)),
    );
});

interface Crowd {
    eids: number[];
    poses: GltfNodePose[][];
    globals: Float32Array[][];
    staging: Float32Array;
    mesh: MeshHandle;
    camera: number;
    instances?: GltfSceneInstance[];
}
const CrowdState = { create: () => ({ value: undefined as Crowd | undefined }) };

function crowdPlugin(
    skin: Skin,
    mode: Mode,
    population: number,
    passes: Passes,
    view: View,
): Plugin {
    const J = skin.inverseBinds.length;
    const V = skin.bind.length / 8;
    return {
        name: "SkinningSpike",
        dependencies: [
            StandardRenderingPlugin,
            MeshRenderPlugin,
            ...(mode === "shipped" || mode === "followers" || mode === "unskinned"
                ? [GltfPlugin]
                : [MaterialPlugin(VertexSkinned), MaterialPlugin(ComputeSkinned)]),
        ],
        async initialize(world) {
            const mesh = world.resource(Meshes).add({
                name: "CesiumMan",
                vertices: skin.vertices,
                indices: skin.indices,
                // Attributes split CesiumMan into its own mesh family, so its absolute vertex index,
                // the vertex function's vidx, starts at zero and indexes the influence rows below.
                attributes: {
                    joints: { element: d.u32, data: skin.joints },
                    weights: { element: d.vec4f, data: skin.weights },
                },
            });
            const values = StandardMaterial({
                baseColor: [0.72, 0.4, 0.05, 1],
                perceptualRoughness: 0.7,
            });
            const material =
                mode === "static" || mode === "shipped" || mode === "followers" || mode === "unskinned"
                    ? world.resource(Materials).add(values)
                    : mode === "vertex"
                      ? world.resource(VertexSkinned).add({ ...values, skinPad: 0 })
                      : world.resource(ComputeSkinned).add({ ...values, skinPad: 0 });
            const shipped = mode === "shipped" || mode === "followers" || mode === "unskinned";
            const gltf = shipped
                ? await loadGltfScene(world, Bun.pathToFileURL(join(ASSETS, FILE.path)))
                : undefined;
            const instances: GltfSceneInstance[] = [];
            if (mode === "followers") {
                for (let i = 0; i < 8; i++) {
                    const leader = spawnGltfScene(world, gltf!);
                    for (const eid of leader.entities) world.remove(eid, MeshMaterial);
                    instances.push(leader);
                }
            }
            const plain = gltf?.primitives.map((primitive) =>
                world.resource(Materials).add(world.resource(SkinnedMaterial).get(primitive.material)!),
            );
            const side = Math.ceil(Math.sqrt(population));
            const meshGlobal = new Float32Array(16);
            const restPoses = evaluateGltfAnimation(skin.nodes, skin.animation, 0);
            const rest = globalsFor(
                skin,
                restPoses,
                skin.nodes.map(() => new Float32Array(16)),
            );
            meshGlobal.set(rest[skin.meshNode]!);
            const eids: number[] = [];
            const placement = new Float32Array(16);
            const composed = new Float32Array(16);
            const trs = new Float32Array(10);
            for (let i = 0; i < population; i++) {
                const x = ((i % side) - (side - 1) / 2) * SPACING;
                const z = (Math.floor(i / side) - (side - 1) / 2) * SPACING;
                const yaw = hash(i * 2) * Math.PI * 2;
                composeMat4(
                    x,
                    0,
                    z,
                    0,
                    Math.sin(yaw / 2),
                    0,
                    Math.cos(yaw / 2),
                    1,
                    1,
                    1,
                    placement,
                );
                if (mode === "static") multiplyMat4(placement, meshGlobal, composed);
                else composed.set(placement);
                decomposeMat4(composed, trs);
                if (gltf) {
                    const instance = spawnGltfScene(world, gltf, {
                        translation: [x, 0, z],
                        rotation: [0, Math.sin(yaw / 2), 0, Math.cos(yaw / 2)],
                    });
                    if (mode === "followers") setGltfLeaderPose(world, instance, instances[i % 8]!);
                    else instances.push(instance);
                    if (mode === "unskinned") {
                        for (let p = 0; p < instance.entities.length; p++) {
                            const eid = instance.entities[p]!;
                            world.remove(eid, SkinnedMesh);
                            multiplyMat4(instance.placement, gltf.primitives[p]!.matrix, composed);
                            decomposeMat4(composed, trs);
                            world.add(eid, Transform, {
                                translation: [trs[0], trs[1], trs[2]],
                                rotation: [trs[3], trs[4], trs[5], trs[6]],
                                scale: [trs[7], trs[8], trs[9]],
                            });
                            world.add(eid, MeshMaterial, plain![p]!);
                        }
                    }
                    eids.push(...instance.entities);
                    continue;
                }
                const eid = world.create();
                world.add(eid, Transform, {
                    translation: [trs[0], trs[1], trs[2]],
                    rotation: [trs[3], trs[4], trs[5], trs[6]],
                    scale: [trs[7], trs[8], trs[9]],
                });
                world.add(eid, MeshInstance, { mesh });
                world.add(eid, MeshMaterial, material);
                eids.push(eid);
            }
            const sun = world.create();
            world.add(sun, Transform, { rotation: lookAtRotation(0, 0, 0, 1, -2, -0.35) });
            world.add(sun, DirectionalLight, {
                illuminance: 6000,
                shadowMapsEnabled: passes === "all" ? 1 : 0,
                numCascades: 4,
                maximumDistance: 200,
            });
            const extent = side * SPACING;
            const camera = world.create();
            world.add(camera, Camera, { far: 1000, antialias: view.antialias });
            world.add(camera, StandardRenderer);
            if (passes === "all") {
                world.add(camera, DepthPrepass);
                world.add(camera, NormalPrepass);
            }
            world.add(camera, Transform, {
                translation: [0, extent * 0.55, extent * 0.85],
                rotation: lookAtRotation(0, extent * 0.55, extent * 0.85, 0, 0, 0),
            });
            attachTexture(world, camera, { width: view.width, height: view.height });
            world.resource(CrowdState).value = {
                eids,
                poses: eids.map(() => evaluateGltfAnimation(skin.nodes, skin.animation, 0)),
                globals: eids.map(() => skin.nodes.map(() => new Float32Array(16))),
                staging: new Float32Array(population * J * 16),
                mesh,
                camera,
                instances: shipped ? instances : undefined,
            };
            if (shipped) return;
            const device = world.gpu.device;
            const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
            const buffer = (label: string, size: number) => {
                const b = device.createBuffer({ label, size: Math.max(16, size), usage });
                world.own(b);
                return b;
            };
            const H = Math.max(...eids) + 1;
            const rows =
                H +
                (mode === "vertex"
                    ? 2 * V + population * J * 4
                    : mode === "compute"
                      ? population * V
                      : 0);
            const buffers: Buffers = {
                skinning: buffer("spike-skinning", rows * 16),
                palette: buffer("spike-palette", mode === "compute" ? population * J * 64 : 16),
                bind: buffer("spike-bind", skin.bind.byteLength),
                influences: buffer("spike-influences", V * 32),
                header: H,
            };
            world.resource(Gpu).buffers = buffers;
            const header = new Uint32Array(H * 4);
            for (let i = 0; i < population; i++) {
                const eid = eids[i]!;
                header[eid * 4] = mode === "compute" ? H + i * V : H + 2 * V + i * J * 4;
                header[eid * 4 + 1] = H;
            }
            device.queue.writeBuffer(buffers.skinning, 0, header);
            const influences = new ArrayBuffer(V * 32);
            for (let v = 0; v < V; v++) {
                new Uint32Array(influences, v * 32, 1)[0] = skin.joints[v]!;
                new Float32Array(influences, v * 32 + 16, 4).set(
                    skin.weights.subarray(v * 4, v * 4 + 4),
                );
            }
            device.queue.writeBuffer(buffers.influences, 0, influences);
            if (mode === "vertex") device.queue.writeBuffer(buffers.skinning, H * 16, influences);
            device.queue.writeBuffer(buffers.bind, 0, skin.bind);
        },
        warm(world) {
            if (mode !== "compute") return;
            const root = world.gpu.root;
            const pipeline = root
                .createComputePipeline({ compute: skinKernel })
                .$name("spike-skin");
            const buffers = world.resource(Gpu).buffers!;
            const params = root
                .createBuffer(d.vec4u, d.vec4u(V, J, buffers.header, 0))
                .$usage("uniform");
            const group = root.createBindGroup(kernelLayout, {
                bind: buffers.bind,
                influences: buffers.influences,
                palette: buffers.palette,
                deformed: buffers.skinning,
                params,
            } as never);
            Object.assign(world.resource(Kernel), {
                pipeline: root.unwrap(pipeline),
                group: root.unwrap(group),
            });
        },
        systems:
            mode === "compute"
                ? [
                      {
                          name: "spike-skin-compute",
                          group: "draw",
                          after: [ClearChangeMarksSystem],
                          before: [MeshPreprocessSystem],
                          update(world) {
                              const kernel = world.resource(Kernel);
                              const pass = world
                                  .frameEncoder()!
                                  .beginComputePass({ label: "spike:skin" });
                              pass.setPipeline(kernel.pipeline!);
                              pass.setBindGroup(0, kernel.group!);
                              pass.dispatchWorkgroups(Math.ceil(V / 64), population);
                              pass.end();
                          },
                      },
                  ]
                : [],
    };
}
const Kernel = {
    create: () => ({
        pipeline: undefined as GPUComputePipeline | undefined,
        group: undefined as GPUBindGroup | undefined,
    }),
};

function globalsFor(skin: Skin, poses: GltfNodePose[], globals: Float32Array[]): Float32Array[] {
    const local = scratch;
    for (let n = 0; n < poses.length; n++) {
        const { translation: t, rotation: r, scale: s } = poses[n]!;
        composeMat4(t[0]!, t[1]!, t[2]!, r[0]!, r[1]!, r[2]!, r[3]!, s[0]!, s[1]!, s[2]!, local);
        const parent = skin.nodes[n]!.parent;
        if (parent !== -1) multiplyMat4(globals[parent]!, local, globals[n]!);
        else globals[n]!.set(local);
    }
    return globals;
}
const scratch = new Float32Array(16);
const joint = new Float32Array(16);

// Stage-1 evaluation and poseGltfScene's parent-first derivation, per instance.
function pose(skin: Skin, crowd: Crowd, time: number): void {
    for (let i = 0; i < crowd.eids.length; i++) {
        const t = (time + hash(i * 2 + 1) * skin.duration) % skin.duration;
        evaluateGltfAnimation(skin.nodes, skin.animation, t, crowd.poses[i]);
        globalsFor(skin, crowd.poses[i]!, crowd.globals[i]!);
    }
}

// Bevy extract_joints_for_skin: joint matrix = joint global × inverse bind, in scene space.
function palette(skin: Skin, crowd: Crowd): void {
    const J = skin.inverseBinds.length;
    for (let i = 0; i < crowd.eids.length; i++) {
        const globals = crowd.globals[i]!;
        for (let j = 0; j < J; j++) {
            multiplyMat4(globals[skin.jointNodes[j]!]!, skin.inverseBinds[j]!, joint);
            crowd.staging.set(joint, (i * J + j) * 16);
        }
    }
}

// Bevy entity_aabb_from_skinned_mesh_bounds with Arvo's transform_aabb, from the palette written
// above; scene space is the entity's local space, so the box's sphere is MeshInstance.cullBounds.
function bounds(world: World, skin: Skin, crowd: Crowd): void {
    const J = skin.inverseBinds.length;
    const cull = world.storage(MeshInstance).cullBounds;
    const m = crowd.staging;
    for (let i = 0; i < crowd.eids.length; i++) {
        let x0 = Infinity,
            y0 = Infinity,
            z0 = Infinity,
            x1 = -Infinity,
            y1 = -Infinity,
            z1 = -Infinity;
        for (const box of skin.boxes) {
            const o = (i * J + box.joint) * 16;
            const [cx, cy, cz] = box.center as [number, number, number];
            const [hx, hy, hz] = box.half as [number, number, number];
            const tx = m[o]! * cx + m[o + 4]! * cy + m[o + 8]! * cz + m[o + 12]!;
            const ty = m[o + 1]! * cx + m[o + 5]! * cy + m[o + 9]! * cz + m[o + 13]!;
            const tz = m[o + 2]! * cx + m[o + 6]! * cy + m[o + 10]! * cz + m[o + 14]!;
            const sx = Math.abs(m[o]!) * hx + Math.abs(m[o + 4]!) * hy + Math.abs(m[o + 8]!) * hz;
            const sy =
                Math.abs(m[o + 1]!) * hx + Math.abs(m[o + 5]!) * hy + Math.abs(m[o + 9]!) * hz;
            const sz =
                Math.abs(m[o + 2]!) * hx + Math.abs(m[o + 6]!) * hy + Math.abs(m[o + 10]!) * hz;
            x0 = Math.min(x0, tx - sx);
            y0 = Math.min(y0, ty - sy);
            z0 = Math.min(z0, tz - sz);
            x1 = Math.max(x1, tx + sx);
            y1 = Math.max(y1, ty + sy);
            z1 = Math.max(z1, tz + sz);
        }
        cull.set(
            crowd.eids[i]!,
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            (z0 + z1) / 2,
            Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2,
        );
    }
}

function writePpm(
    path: string,
    width: number,
    height: number,
    rgba: ArrayLike<number> & { subarray(a: number, b: number): ArrayLike<number> },
): void {
    const rgb = new Uint8Array(width * height * 3);
    for (let i = 0; i < width * height; i++) rgb.set(rgba.subarray(i * 4, i * 4 + 3), i * 3);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, Buffer.concat([Buffer.from(`P6\n${width} ${height}\n255\n`), rgb]));
}

// The CPU-skinned vertices of sampled instances lie inside the cullBounds sphere written this frame.
function expectBoundsEnclose(world: World, skin: Skin, crowd: Crowd): void {
    const J = skin.inverseBinds.length;
    const cull = world.storage(MeshInstance).cullBounds;
    const m = crowd.staging;
    for (let i = 0; i < crowd.eids.length; i += 97) {
        const eid = crowd.eids[i]!;
        const [cx, cy, cz, r] = [
            cull.x.get(eid),
            cull.y.get(eid),
            cull.z.get(eid),
            cull.w.get(eid),
        ];
        expect(r).toBeGreaterThan(0);
        let farthest = 0;
        for (let v = 0; v < skin.bind.length / 8; v++) {
            const p = [skin.bind[v * 8]!, skin.bind[v * 8 + 1]!, skin.bind[v * 8 + 2]!];
            const out = [0, 0, 0];
            for (let k = 0; k < 4; k++) {
                const w = skin.weights[v * 4 + k]!;
                const o = (i * J + ((skin.joints[v]! >>> (8 * k)) & 255)) * 16;
                for (let c = 0; c < 3; c++)
                    out[c]! +=
                        w *
                        (m[o + c]! * p[0]! +
                            m[o + 4 + c]! * p[1]! +
                            m[o + 8 + c]! * p[2]! +
                            m[o + 12 + c]!);
            }
            farthest = Math.max(farthest, Math.hypot(out[0]! - cx, out[1]! - cy, out[2]! - cz));
        }
        expect(farthest).toBeLessThanOrEqual(r * (1 + 1e-5));
    }
}

await setupGlobals();
test("report CesiumMan crowd CPU paths and pipelined GPU frames", async () => {
    const skin = await loadSkin();
    const V = skin.bind.length / 8;
    const J = skin.inverseBinds.length;
    console.info(
        `[skinning] asset=CesiumMan vertices=${V} triangles=${skin.indices.length / 3} joints=${J} joint-boxes=${skin.boxes.length} nodes=${skin.nodes.length} channels=${skin.animation.channels.length} duration-s=${skin.duration.toFixed(3)}`,
    );
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("skinning spike requires a real adapter");
    const { vendor, architecture, device: name, description } = adapter.info;
    console.info(
        `[skinning] adapter vendor=${vendor} architecture=${architecture} device=${name} description=${description} maxStorageBufferBindingSize=${adapter.limits.maxStorageBufferBindingSize}`,
    );
    const device = await adapter.requestDevice({
        requiredFeatures: ["indirect-first-instance", "timestamp-query"],
    });
    device.addEventListener("uncapturederror", (event) => {
        throw new Error(event.error.message);
    });
    const querySet = device.createQuerySet({ type: "timestamp", count: 1024 });
    const resolve = device.createBuffer({
        size: 8192,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
        size: 8192,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    let active = false;
    let slots = 0;
    const labels: string[] = [];
    const createEncoder = device.createCommandEncoder.bind(device);
    Object.defineProperty(device, "createCommandEncoder", {
        configurable: true,
        value: (descriptor?: GPUCommandEncoderDescriptor) => {
            const encoder = createEncoder(descriptor);
            const render = encoder.beginRenderPass.bind(encoder);
            const compute = encoder.beginComputePass.bind(encoder);
            function writes(): GPUComputePassTimestampWrites {
                if (slots + 2 > 1024) throw new Error("frame probe exhausted timestamp slots");
                const start = slots;
                slots += 2;
                return {
                    querySet,
                    beginningOfPassWriteIndex: start,
                    endOfPassWriteIndex: start + 1,
                };
            }
            // Pass labels carry the camera eid; strip it so runs compare by pass.
            Object.defineProperty(encoder, "beginRenderPass", {
                value: (desc: GPURenderPassDescriptor) => {
                    if (active) labels.push((desc.label ?? "render").replace(/\/\d+$/, ""));
                    return render(active ? { ...desc, timestampWrites: writes() } : desc);
                },
            });
            Object.defineProperty(encoder, "beginComputePass", {
                value: (desc?: GPUComputePassDescriptor) => {
                    if (active) labels.push(desc?.label ?? "compute");
                    return compute(active ? { ...desc, timestampWrites: writes() } : desc);
                },
            });
            return encoder;
        },
    });
    const throughput: Record<string, number> = {};
    const paths: Record<string, { median: number; p95: number }> = {};
    try {
        for (const population of POPULATIONS) {
            for (const view of VIEWS) {
                for (const passes of PASSES) {
                    const captures = new Map<Mode, ArrayLike<number>>();
                    for (const mode of MODES) {
                        const app = await createApp({
                            device,
                            adapter,
                            plugins: [crowdPlugin(skin, mode, population, passes, view)],
                        });
                        try {
                            const { world } = app;
                            expect(world.gpu.adapter.class).toBe("real");
                            const crowd = world.resource(CrowdState).value!;
                            const buffers = world.resource(Gpu).buffers!;
                            const cpu = {
                                path: [] as number[],
                                pose: [] as number[],
                                palette: [] as number[],
                                bounds: [] as number[],
                                upload: [] as number[],
                                step: [] as number[],
                            };
                            const spans: number[] = [];
                            const perPass = new Map<string, number[]>();
                            let time = 0;
                            const upload = () => {
                                if (mode === "vertex")
                                    device.queue.writeBuffer(
                                        buffers.skinning,
                                        (buffers.header + 2 * V) * 16,
                                        crowd.staging,
                                    );
                                else if (mode === "compute")
                                    device.queue.writeBuffer(buffers.palette, 0, crowd.staging);
                            };
                            const animate = (measured: boolean) => {
                                time += Time.FIXED_DT;
                                if (mode === "static" || mode === "unskinned") return;
                                const start = performance.now();
                                if (mode === "shipped" || mode === "followers") {
                                    for (let i = 0; i < crowd.instances!.length; i++) {
                                        const instance = crowd.instances![i]!;
                                        const t = (time + hash(i * 2 + 1) * skin.duration) % skin.duration;
                                        poseGltfScene(world, instance, instance.gltf.animations[0]!, t);
                                    }
                                    const elapsed = performance.now() - start;
                                    if (measured) cpu.path.push(elapsed);
                                    return;
                                }
                                pose(skin, crowd, time);
                                const posed = performance.now();
                                palette(skin, crowd);
                                const paletted = performance.now();
                                bounds(world, skin, crowd);
                                const bounded = performance.now();
                                upload();
                                const uploaded = performance.now();
                                if (!measured) return;
                                cpu.path.push(uploaded - start);
                                cpu.pose.push(posed - start);
                                cpu.palette.push(paletted - posed);
                                cpu.bounds.push(bounded - paletted);
                                cpu.upload.push(uploaded - bounded);
                            };
                            // Serial: one frame at a time, each waiting for its own timestamps.
                            for (let frame = 0; frame < WARM + FRAMES; frame++) {
                                const measured = frame >= WARM;
                                animate(measured);
                                slots = 0;
                                labels.length = 0;
                                active = measured;
                                const start = performance.now();
                                try {
                                    world.step(Time.FIXED_DT);
                                } finally {
                                    active = false;
                                }
                                if (!measured) {
                                    await device.queue.onSubmittedWorkDone();
                                    continue;
                                }
                                cpu.step.push(performance.now() - start);
                                const encoder = createEncoder();
                                encoder.resolveQuerySet(querySet, 0, slots, resolve, 0);
                                encoder.copyBufferToBuffer(resolve, 0, readback, 0, slots * 8);
                                device.queue.submit([encoder.finish()]);
                                await readback.mapAsync(GPUMapMode.READ);
                                try {
                                    const times = new BigUint64Array(
                                        readback.getMappedRange(0, slots * 8),
                                    );
                                    let first = times[0]!,
                                        last = times[1]!;
                                    const sums = new Map<string, number>();
                                    for (let i = 0; i < slots; i += 2) {
                                        if (times[i]! < first) first = times[i]!;
                                        if (times[i + 1]! > last) last = times[i + 1]!;
                                        const label = labels[i / 2]!;
                                        sums.set(
                                            label,
                                            (sums.get(label) ?? 0) +
                                                Number(times[i + 1]! - times[i]!) / 1e6,
                                        );
                                    }
                                    spans.push(Number(last - first) / 1e6);
                                    for (const [label, sum] of sums) {
                                        if (!perPass.has(label)) perPass.set(label, []);
                                        perPass.get(label)!.push(sum);
                                    }
                                } finally {
                                    readback.unmap();
                                }
                            }
                            if (mode === "shipped") {
                                const matrices = world.resource(SkinPalette).matrices;
                                const first = world.storage(SkinnedMesh).firstJoint;
                                for (let i = 0; i < crowd.eids.length; i++) {
                                    const offset = first.get(crowd.eids[i]!) * 16;
                                    crowd.staging.set(matrices.subarray(offset, offset + J * 16), i * J * 16);
                                }
                                expectBoundsEnclose(world, skin, crowd);
                                expect(crowd.staging.some((v, i) => Math.abs(v - (i % 16 % 5 === 0 ? 1 : 0)) > 0.01)).toBe(true);
                            } else if (mode === "followers") {
                                expect(crowd.instances).toHaveLength(8);
                                expect(world.resource(SkinPalette).extent).toBe(8 * J);
                                for (const eid of crowd.eids) expect(world.has(eid, SkinnedMesh)).toBe(false);
                            } else if (mode !== "static" && mode !== "unskinned") {
                                expectBoundsEnclose(world, skin, crowd);
                            }
                            const seen = [...perPass.keys()];
                            expect(seen.includes("standard-cascadeshadow")).toBe(passes === "all");
                            expect(seen.includes("standard-prepass")).toBe(passes === "all");
                            expect(seen.includes("spike:skin")).toBe(mode === "compute");
                            // Pipelined: the palette is uploaded and the frame submitted back to back, with
                            // the CPU pose held, so wall time per frame is the GPU's sustained frame time.
                            for (let frame = 0; frame < WARM; frame++) {
                                upload();
                                world.step(Time.FIXED_DT);
                            }
                            await device.queue.onSubmittedWorkDone();
                            const start = performance.now();
                            for (let frame = 0; frame < THROUGHPUT; frame++) {
                                upload();
                                world.step(Time.FIXED_DT);
                            }
                            await device.queue.onSubmittedWorkDone();
                            const frameMs = (performance.now() - start) / THROUGHPUT;
                            throughput[`${population}/${view.name}/${passes}/${mode}`] = frameMs;
                            const capture = await captureTexture(world, crowd.camera);
                            captures.set(mode, capture.rgba);
                            writePpm(
                                join(
                                    ROOT,
                                    `test-results/skinning/${population}-${view.name}-${passes}-${mode}.ppm`,
                                ),
                                capture.width,
                                capture.height,
                                capture.rgba,
                            );
                            const m = (values: number[]) => (values.length ? median(values) : 0);
                            const record = {
                                "gpu-frame-pipelined": frameMs,
                                "gpu-span-serial": m(spans),
                                "cpu-path": m(cpu.path),
                                "cpu-path-p95": cpu.path.toSorted((a, b) => a - b)[Math.ceil(cpu.path.length * 0.95) - 1] ?? 0,
                                "cpu-path/gpu-frame": m(cpu.path) / frameMs,
                                "cpu-path-fits": m(cpu.path) <= frameMs,
                                "cpu-pose": m(cpu.pose),
                                "cpu-palette": m(cpu.palette),
                                "cpu-bounds": m(cpu.bounds),
                                "cpu-upload": m(cpu.upload),
                                "cpu-step": m(cpu.step),
                                ...Object.fromEntries(
                                    [...perPass]
                                        .filter(([, values]) => m(values) >= 0.01)
                                        .map(([label, values]) => [`pass:${label}`, m(values)]),
                                ),
                            };
                            paths[`${population}/${view.name}/${passes}/${mode}`] = {
                                median: m(cpu.path),
                                p95: record["cpu-path-p95"],
                            };
                            console.info(
                                `[skinning] population=${population} view=${view.name} passes=${passes} mode=${mode} vertices-per-pass=${population * V} palette-bytes=${mode === "static" || mode === "unskinned" ? 0 : (mode === "followers" ? 8 : population) * J * 64} deformed-bytes=${mode === "compute" ? population * V * 16 : 0} ${JSON.stringify(record, (_, v) => (typeof v === "number" ? Number(v.toFixed(3)) : v))}`,
                            );
                        } finally {
                            app.dispose();
                        }
                    }
                    // The two skinning paths are independent implementations of one formula: their
                    // images agree but for the compute path's 8-bit normals.
                    if (MODES.includes("shipped") && MODES.includes("unskinned")) {
                        const key = `${population}/${view.name}/${passes}`;
                        const path = paths[`${key}/shipped`]!;
                        const gpu = throughput[`${key}/shipped`]!;
                        console.info(`[skinning-summary] population=${population} view=${view.name} passes=${passes} cpu-path-median-ms=${path.median.toFixed(3)} cpu-path-p95-ms=${path.p95.toFixed(3)} gpu-frame-ms=${gpu.toFixed(3)} cpu/gpu=${(path.median / gpu).toFixed(3)} fits=${path.median <= gpu} shipped-cost-over-unskinned-ms=${(gpu - throughput[`${key}/unskinned`]!).toFixed(3)}`);
                    }
                    if (MODES.includes("followers")) {
                        const key = `${population}/${view.name}/${passes}/followers`;
                        const path = paths[key]!;
                        const gpu = throughput[key]!;
                        console.info(`[skinning-summary] population=${population} leaders=8 mode=followers view=${view.name} passes=${passes} cpu-path-median-ms=${path.median.toFixed(3)} cpu-path-p95-ms=${path.p95.toFixed(3)} gpu-frame-ms=${gpu.toFixed(3)} cpu/gpu=${(path.median / gpu).toFixed(3)} fits=${path.median <= gpu}`);
                    }
                    if (!captures.has("vertex") || !captures.has("compute")) continue;
                    const a = captures.get("vertex")!;
                    const b = captures.get("compute")!;
                    let differing = 0;
                    for (let i = 0; i < a.length; i += 4)
                        for (let c = 0; c < 3; c++)
                            if (Math.abs(a[i + c]! - b[i + c]!) > 32) {
                                differing++;
                                break;
                            }
                    expect(differing / (a.length / 4)).toBeLessThan(1e-3);
                }
                if (!["static", "vertex", "compute"].every((mode) => MODES.includes(mode as Mode))) continue;
                const t = (passes: Passes, mode: Mode) =>
                    throughput[`${population}/${view.name}/${passes}/${mode}`]!;
                const vertexAll = t("all", "vertex") - t("all", "static");
                const vertexColor = t("color", "vertex") - t("color", "static");
                const computeAll = t("all", "compute") - t("all", "static");
                const computeColor = t("color", "compute") - t("color", "static");
                console.info(
                    `[skinning] population=${population} view=${view.name} static frame (ms): all passes=${t("all", "static").toFixed(3)} color pass alone=${t("color", "static").toFixed(3)} shadow+prepass share of the frame=${((t("all", "static") - t("color", "static")) / t("all", "static")).toFixed(3)}`,
                );
                console.info(
                    `[skinning] population=${population} view=${view.name} pipelined skinning cost over static (ms): vertex all=${vertexAll.toFixed(3)} color-only=${vertexColor.toFixed(3)} shadow+prepass share=${((vertexAll - vertexColor) / vertexAll).toFixed(3)}; compute all=${computeAll.toFixed(3)} color-only=${computeColor.toFixed(3)}; compute minus vertex, all passes=${(computeAll - vertexAll).toFixed(3)}`,
                );
            }
        }
        console.info(
            `[skinning] serial=${WARM}+${FRAMES} pipelined=${WARM}+${THROUGHPUT} frames per run. Pipelined frame time is wall time over back-to-back submissions, CPU pose held (world.step stays well under the GPU frame). Serial GPU span is first pass start to last pass end; per-pass sums overlap on a tile-based GPU, so the pipelined ablation (all passes against the color pass alone) attributes the shadow and prepass runs. The palette writeBuffer's GPU copy falls outside passes but inside the pipelined frame. CPU times are serial-frame medians except cpu-path-p95. Shipped cpu-path times the entire phase-shifted poseGltfScene loop; cpu-step includes deformation texture uploads. The path/GPU ratio uses the median, not p95.`,
        );
    } finally {
        querySet.destroy();
        resolve.destroy();
        readback.destroy();
        device.destroy();
        await device.lost;
    }
}, 0);

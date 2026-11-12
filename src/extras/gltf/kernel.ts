import { Meshes, MeshInstance } from "../../core/mesh";
import type { World } from "../../engine/ecs";
import type { GltfAnimation, GltfNode, GltfNodePose } from "./animation";
import { GLTF_WASM_BASE64 } from "./gltf.wasm";
import type { Gltf, GltfSceneInstance } from "./scene";
import { MorphedMesh, SkinnedMesh, SkinPalette } from "./skinning";

interface Kernel {
    memory: WebAssembly.Memory;
    decodeMeshoptAttributes(
        destination: number,
        count: number,
        stride: number,
        source: number,
        sourceLength: number,
        filter: number,
        khr: number,
    ): number;
    decodeMeshoptTriangles(
        destination: number,
        count: number,
        stride: number,
        source: number,
        sourceLength: number,
    ): number;
    decodeMeshoptIndices(
        destination: number,
        count: number,
        stride: number,
        source: number,
        sourceLength: number,
    ): number;
    meshoptAlloc(bytes: number): number;
    meshoptFree(ptr: number, bytes: number): void;
    alloc(count: number): number;
    freeLanes(ptr: number, count: number): void;
    assetNew(ptr: number, count: number): number;
    animationNew(ptr: number, count: number): number;
    view(asset: number, field: number): number;
    destroy(handle: number, kind: number): void;
    pose(asset: number, animation: number, time: number): void;
    evaluate(asset: number, animation: number, time: number): void;
}
let wasm: Kernel | undefined;
function kernel(): Kernel {
    if (!wasm) {
        const bytes = Uint8Array.from(atob(GLTF_WASM_BASE64), (c) => c.charCodeAt(0));
        wasm = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports as unknown as Kernel;
    }
    return wasm;
}
interface MeshoptView {
    count: number;
    byteStride: number;
    mode: string;
    filter?: string;
}
const meshoptFilters = ["NONE", "OCTAHEDRAL", "QUATERNION", "EXPONENTIAL", "COLOR"];

/** Decodes one compressed view into caller-owned bytes; errors name the buffer view. */
export function decodeMeshoptView(
    index: number,
    view: MeshoptView,
    source: Uint8Array,
    output: Uint8Array,
    khr: boolean,
): void {
    const { count, byteStride, mode } = view;
    const filter = meshoptFilters.indexOf(view.filter ?? "NONE");
    if (
        !Number.isInteger(count) ||
        count < 1 ||
        count > 0xffffffff ||
        !Number.isInteger(byteStride) ||
        byteStride < 1 ||
        byteStride > 256 ||
        count * byteStride !== output.length ||
        source.length + output.length > 0x7ffffff8
    ) {
        throw new Error(`bufferView ${index}: meshopt invalid count, stride or byteLength`);
    }
    const k = kernel();
    const bytes = source.length + output.length;
    const ptr = k.meshoptAlloc(bytes);
    const input = ptr + output.length;
    try {
        const heap = new Uint8Array(k.memory.buffer);
        heap.set(source, input);
        let result = -1;
        if (mode === "ATTRIBUTES") {
            result = k.decodeMeshoptAttributes(
                ptr,
                count,
                byteStride,
                input,
                source.length,
                filter,
                khr ? 1 : 0,
            );
        } else if (filter === 0) {
            if (mode === "TRIANGLES")
                result = k.decodeMeshoptTriangles(ptr, count, byteStride, input, source.length);
            if (mode === "INDICES")
                result = k.decodeMeshoptIndices(ptr, count, byteStride, input, source.length);
        }
        if (result) {
            throw new Error(`bufferView ${index}: invalid meshopt stream, mode, stride or filter`);
        }
        output.set(heap.subarray(ptr, ptr + output.length));
    } finally {
        k.meshoptFree(ptr, bytes);
    }
}

const paths = ["translation", "rotation", "scale", "weights"] as const;
function array(data: number[], values: ArrayLike<number>) {
    data.push(values.length);
    for (let i = 0; i < values.length; i++) data.push(values[i]);
}
function nodeData(nodes: readonly GltfNode[]): number[] {
    const data = [nodes.length];
    for (const node of nodes) {
        data.push(node.index, node.parent);
        array(data, [...node.translation, ...node.rotation, ...node.scale, ...node.weights]);
    }
    return data;
}
function upload(data: number[], animation: boolean): number {
    const k = kernel();
    const ptr = k.alloc(data.length);
    try {
        new Float64Array(k.memory.buffer, ptr, data.length).set(data);
        return animation ? k.animationNew(ptr, data.length) : k.assetNew(ptr, data.length);
    } finally {
        k.freeLanes(ptr, data.length);
    }
}
function uploadAnimation(animation: GltfAnimation): number {
    const data = [animation.samplers.length];
    for (const sampler of animation.samplers) {
        data.push(["STEP", "LINEAR", "CUBICSPLINE"].indexOf(sampler.interpolation));
        array(data, sampler.input);
        array(data, sampler.output);
    }
    data.push(animation.channels.length);
    for (const channel of animation.channels)
        data.push(channel.sampler, channel.node, paths.indexOf(channel.path));
    return upload(data, true);
}
interface Asset {
    ptr: number;
    animations: Map<GltfAnimation, number>;
    scratch: Scratch;
}
const assets = new WeakMap<Gltf, Asset>();
const globalCopies = new WeakMap<GltfSceneInstance, Float32Array>();
interface Scratch {
    poses: Float64Array;
    globals: Float32Array;
    input: Uint32Array;
    output: Float32Array;
    weights: Float32Array[];
    palettes: Float32Array[];
    bounds: Float32Array[];
}

export function registerGltfKernel(gltf: Gltf): void {
    const data = nodeData(gltf.nodes);
    data.push(gltf.skins.length);
    for (const skin of gltf.skins) {
        array(data, skin.joints);
        array(data, skin.inverseBindMatrices);
    }
    data.push(gltf.primitives.length);
    for (const primitive of gltf.primitives) {
        data.push(primitive.nodeIndex, primitive.skin);
        array(data, primitive.jointBounds ?? []);
        array(data, primitive.morphReach ?? []);
        array(data, gltf.world.resource(Meshes).get(primitive.mesh)!.bounds ?? [0, 0, 0, 0]);
    }
    const ptr = upload(data, false);
    const asset: Asset = {
        ptr,
        animations: new Map(),
        scratch: {
            poses: new Float64Array(),
            globals: new Float32Array(),
            input: new Uint32Array(),
            output: new Float32Array(),
            weights: [],
            palettes: [],
            bounds: [],
        },
    };
    assets.set(gltf, asset);
    gltf.world.onDispose(() => {
        const k = kernel();
        for (const ptr of asset.animations.values()) k.destroy(ptr, 1);
        k.destroy(asset.ptr, 0);
        assets.delete(gltf);
    });
}
export function initializeGltfGlobals(instance: GltfSceneInstance): void {
    const globals = new Float32Array(instance.gltf.nodes.length * 16);
    for (let i = 0; i < instance.globals.length; i++)
        instance.globals[i] = globals.subarray(i * 16, i * 16 + 16);
    globalCopies.set(instance, globals);
}
function refresh(gltf: Gltf, s: Scratch): void {
    const k = kernel();
    if (s.poses.buffer !== k.memory.buffer) rebuildViews(gltf, s);
}
function rebuildViews(gltf: Gltf, s: Scratch): void {
    const k = kernel();
    const ptr = assets.get(gltf)!.ptr;
    const poseCount = gltf.nodes.reduce((n, node) => n + 10 + node.weights.length, 0);
    const outputCount = gltf.primitives.reduce(
        (n, p) =>
            n +
            4 +
            (p.morphReach?.length ?? 0) +
            (p.skin < 0 ? 0 : gltf.skins[p.skin].joints.length * 16),
        0,
    );
    s.poses = new Float64Array(k.memory.buffer, k.view(ptr, 0), poseCount);
    s.globals = new Float32Array(k.memory.buffer, k.view(ptr, 1), gltf.nodes.length * 16);
    s.input = new Uint32Array(k.memory.buffer, k.view(ptr, 2), gltf.primitives.length * 2);
    s.output = new Float32Array(k.memory.buffer, k.view(ptr, 3), outputCount);
    let offset = 0;
    for (let i = 0; i < gltf.primitives.length; i++) {
        const p = gltf.primitives[i];
        s.bounds[i] = s.output.subarray(offset, offset + 4);
        offset += 4;
        const count = p.morphReach?.length ?? 0;
        s.weights[i] = s.output.subarray(offset, offset + count);
        offset += count;
        const joints = p.skin < 0 ? 0 : gltf.skins[p.skin].joints.length;
        s.palettes[i] = s.output.subarray(offset, offset + joints * 16);
        offset += joints * 16;
    }
}
// Named properties: a keyed load by path name was most of the copy's cost.
function copyPoses(source: Float64Array, poses: GltfNodePose[]): void {
    let o = 0;
    for (let i = 0; i < poses.length; i++) {
        const { translation: t, rotation: r, scale: s, weights: w } = poses[i];
        t[0] = source[o];
        t[1] = source[o + 1];
        t[2] = source[o + 2];
        r[0] = source[o + 3];
        r[1] = source[o + 4];
        r[2] = source[o + 5];
        r[3] = source[o + 6];
        s[0] = source[o + 7];
        s[1] = source[o + 8];
        s[2] = source[o + 9];
        o += 10;
        for (let c = 0; c < w.length; c++) w[c] = source[o++];
    }
}
export function poseKernel(
    world: World,
    instance: GltfSceneInstance,
    animation?: GltfAnimation,
    time = 0,
): void {
    const asset = assets.get(instance.gltf)!;
    const s = asset.scratch;
    refresh(instance.gltf, s);
    const morph = world.storage(MorphedMesh);
    for (let i = 0; i < instance.entities.length; i++) {
        const eid = instance.entities[i];
        s.input[i * 2] = world.has(eid, SkinnedMesh) ? 1 : 0;
        s.input[i * 2 + 1] = world.has(eid, MorphedMesh) ? morph.weightCount.get(eid) + 1 : 0;
    }
    runPose(instance.gltf, animation, time);
    globalCopies.get(instance)!.set(s.globals);
    copyPoses(s.poses, instance.poses);
}
function runPose(gltf: Gltf, animation: GltfAnimation | undefined, time: number): Scratch {
    const asset = assets.get(gltf)!;
    let clip = 0;
    if (animation) {
        clip = asset.animations.get(animation) ?? 0;
        if (!clip) {
            clip = uploadAnimation(animation);
            asset.animations.set(animation, clip);
        }
    }
    const s = asset.scratch;
    kernel().pose(asset.ptr, clip, time);
    refresh(gltf, s);
    return s;
}
/** Borrowed kernel output, overwritten by the next pose; no entity placement is evaluated. */
export function bakeKernelPose(gltf: Gltf, animation: GltfAnimation, time: number): Scratch {
    const s = assets.get(gltf)!.scratch;
    refresh(gltf, s);
    for (let i = 0; i < gltf.primitives.length; i++) {
        const p = gltf.primitives[i];
        s.input[i * 2] = p.skin >= 0 ? 1 : 0;
        s.input[i * 2 + 1] = p.morphReach ? p.morphReach.length + 1 : 0;
    }
    return runPose(gltf, animation, time);
}
export function writeKernelPose(world: World, instance: GltfSceneInstance, i: number): void {
    const s = assets.get(instance.gltf)!.scratch;
    const p = instance.gltf.primitives[i];
    const eid = instance.entities[i];
    if (p.morphReach && world.has(eid, MorphedMesh)) {
        const storage = world.storage(MorphedMesh);
        const count = Math.min(storage.weightCount.get(eid), p.morphReach.length);
        const weights = world.resource(SkinPalette).morph.weights;
        const first = storage.firstWeight.get(eid);
        for (let m = 0; m < count; m++) weights[first + m] = s.weights[i][m];
        if (p.skin < 0) {
            const b = s.bounds[i];
            world.storage(MeshInstance).cullBounds.set(eid, b[0], b[1], b[2], b[3]);
        }
    }
    if (p.skin >= 0 && world.has(eid, SkinnedMesh)) {
        world
            .resource(SkinPalette)
            .matrices.set(s.palettes[i], world.storage(SkinnedMesh).firstJoint.get(eid) * 16);
        const b = s.bounds[i];
        world.storage(MeshInstance).cullBounds.set(eid, b[0], b[1], b[2], b[3]);
    }
}
export function evaluateKernel(
    nodes: readonly GltfNode[],
    animation: GltfAnimation,
    time: number,
    poses: GltfNodePose[],
): void {
    const data = nodeData(nodes);
    data.push(0, 0);
    const k = kernel();
    const asset = upload(data, false);
    let clip = 0;
    try {
        clip = uploadAnimation(animation);
        k.evaluate(asset, clip, time);
        copyPoses(
            new Float64Array(
                k.memory.buffer,
                k.view(asset, 0),
                nodes.reduce((n, node) => n + 10 + node.weights.length, 0),
            ),
            poses,
        );
    } finally {
        if (clip) k.destroy(clip, 1);
        k.destroy(asset, 0);
    }
}

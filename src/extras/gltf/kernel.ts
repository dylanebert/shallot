import { Meshes, MeshInstance } from "../../core/mesh";
import type { World } from "../../engine/ecs";
import type { GltfAnimation, GltfNode, GltfNodePose } from "./animation";
import { GLTF_WASM_BASE64 } from "./gltf.wasm";
import type { Gltf, GltfSceneInstance } from "./scene";
import { MorphedMesh, SkinnedMesh, SkinPalette } from "./skinning";

interface Kernel {
    memory: WebAssembly.Memory;
    alloc(count: number): number;
    freeLanes(ptr: number, count: number): void;
    assetNew(ptr: number, count: number): number;
    animationNew(ptr: number, count: number): number;
    instanceNew(asset: number): number;
    view(instance: number, field: number): number;
    destroy(handle: number, kind: number): void;
    pose(asset: number, instance: number, animation: number, time: number): void;
    evaluate(asset: number, instance: number, animation: number, time: number): void;
}
let wasm: Kernel | undefined;
function kernel(): Kernel {
    if (!wasm) {
        const bytes = Uint8Array.from(atob(GLTF_WASM_BASE64), (c) => c.charCodeAt(0));
        wasm = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports as unknown as Kernel;
    }
    return wasm;
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
    instances: number[];
}
const assets = new WeakMap<Gltf, Asset>();
const instances = new WeakMap<GltfSceneInstance, Scratch>();
interface Scratch {
    ptr: number;
    poses: Float64Array;
    globals: Float32Array;
    input: Uint32Array;
    output: Float32Array;
    globalCopy: Float32Array;
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
    const asset: Asset = { ptr: upload(data, false), animations: new Map(), instances: [] };
    assets.set(gltf, asset);
    gltf.world.onDispose(() => {
        const k = kernel();
        for (const ptr of asset.instances) k.destroy(ptr, 2);
        for (const ptr of asset.animations.values()) k.destroy(ptr, 1);
        k.destroy(asset.ptr, 0);
        assets.delete(gltf);
    });
}
function scratch(instance: GltfSceneInstance): Scratch {
    let s = instances.get(instance);
    if (!s) {
        const asset = assets.get(instance.gltf)!;
        const ptr = kernel().instanceNew(asset.ptr);
        asset.instances.push(ptr);
        const globalCopy = new Float32Array(instance.gltf.nodes.length * 16);
        for (let i = 0; i < instance.globals.length; i++)
            instance.globals[i] = globalCopy.subarray(i * 16, i * 16 + 16);
        s = {
            ptr,
            globalCopy,
            poses: new Float64Array(),
            globals: new Float32Array(),
            input: new Uint32Array(),
            output: new Float32Array(),
            weights: [],
            palettes: [],
            bounds: [],
        };
        instances.set(instance, s);
    }
    refresh(instance.gltf, s);
    return s;
}
function refresh(gltf: Gltf, s: Scratch): void {
    const k = kernel();
    if (s.poses.buffer !== k.memory.buffer) rebuildViews(gltf, s);
}
function rebuildViews(gltf: Gltf, s: Scratch): void {
    const k = kernel();
    const poseCount = gltf.nodes.reduce((n, node) => n + 10 + node.weights.length, 0);
    const outputCount = gltf.primitives.reduce(
        (n, p) =>
            n +
            4 +
            (p.morphReach?.length ?? 0) +
            (p.skin < 0 ? 0 : gltf.skins[p.skin].joints.length * 16),
        0,
    );
    s.poses = new Float64Array(k.memory.buffer, k.view(s.ptr, 0), poseCount);
    s.globals = new Float32Array(k.memory.buffer, k.view(s.ptr, 1), gltf.nodes.length * 16);
    s.input = new Uint32Array(k.memory.buffer, k.view(s.ptr, 2), gltf.primitives.length * 2);
    s.output = new Float32Array(k.memory.buffer, k.view(s.ptr, 3), outputCount);
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
function copyPoses(source: Float64Array, poses: GltfNodePose[]): void {
    let offset = 0;
    for (let i = 0; i < poses.length; i++)
        for (let p = 0; p < paths.length; p++) {
            const value = poses[i][paths[p]];
            for (let c = 0; c < value.length; c++) value[c] = source[offset++];
        }
}
export function poseKernel(
    world: World,
    instance: GltfSceneInstance,
    animation?: GltfAnimation,
    time = 0,
): void {
    const asset = assets.get(instance.gltf)!;
    let clip = 0;
    if (animation) {
        clip = asset.animations.get(animation) ?? 0;
        if (!clip) {
            clip = uploadAnimation(animation);
            asset.animations.set(animation, clip);
        }
    }
    const s = scratch(instance);
    const morph = world.storage(MorphedMesh);
    for (let i = 0; i < instance.entities.length; i++) {
        const eid = instance.entities[i];
        s.input[i * 2] = world.has(eid, SkinnedMesh) ? 1 : 0;
        s.input[i * 2 + 1] = world.has(eid, MorphedMesh) ? morph.weightCount.get(eid) + 1 : 0;
    }
    kernel().pose(asset.ptr, s.ptr, clip, time);
    refresh(instance.gltf, s);
    s.globalCopy.set(s.globals);
    copyPoses(s.poses, instance.poses);
}
export function writeKernelPose(world: World, instance: GltfSceneInstance, i: number): void {
    const s = instances.get(instance)!;
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
    let instance = 0;
    try {
        clip = uploadAnimation(animation);
        instance = k.instanceNew(asset);
        k.evaluate(asset, instance, clip, time);
        copyPoses(
            new Float64Array(
                k.memory.buffer,
                k.view(instance, 0),
                nodes.reduce((n, node) => n + 10 + node.weights.length, 0),
            ),
            poses,
        );
    } finally {
        if (instance) k.destroy(instance, 2);
        if (clip) k.destroy(clip, 1);
        k.destroy(asset, 0);
    }
}

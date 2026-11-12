import type { World } from "../../engine";
import type { GltfAnimation } from "./animation";
import { bakeKernelPose } from "./kernel";
import type { Gltf, GltfSceneInstance } from "./scene";
import {
    BakedFrame,
    type BakedRows,
    LeaderPose,
    MorphedMesh,
    SkinnedMesh,
    SkinPalette,
} from "./skinning";

export interface GltfBake {
    readonly gltf: Gltf;
    /** Samples per second. Frame i samples i / rate. */
    readonly rate: number;
    readonly frameCount: number;
}
const allocations = new WeakMap<GltfBake, number>();
const followers = [LeaderPose];

/** Bakes deformation through the live pose kernel, including morph weights and entity-local bounds.
 * Samples from zero through ceil(max sampler time * rate), inclusive; samplers clamp independently.
 * No neighboring-frame blend or playback is supplied. Rows belong to the bake until release or world
 * disposal. Refuses a nonpositive/nonfinite rate and a bake exceeding the 8192-row palette limit.
 * Animated rigid-node placement still requires poseGltfScene; baking never changes placement.
 */
export function bakeGltfAnimation(
    world: World,
    gltf: Gltf,
    animation: GltfAnimation,
    rate: number,
): GltfBake {
    if (gltf.world !== world) throw new Error("glTF handle belongs to another world");
    if (!Number.isFinite(rate) || rate <= 0)
        throw new Error("glTF bake requires a positive finite rate");
    let end = 0;
    for (const sampler of animation.samplers) end = Math.max(end, sampler.input.at(-1) ?? 0);
    const frameCount = Math.ceil(end * rate) + 1;
    const joints = [0],
        weights = [0];
    for (const p of gltf.primitives) {
        joints.push(joints.at(-1)! + (p.skin < 0 ? 0 : gltf.skins[p.skin].joints.length));
        weights.push(weights.at(-1)! + (p.morphReach?.length ?? 0));
    }
    const jointCount = joints.at(-1)!,
        weightCount = weights.at(-1)!;
    if (
        !Number.isSafeInteger(frameCount) ||
        frameCount * jointCount > 64 * 8192 ||
        frameCount * weightCount > 256 * 8192
    )
        throw new Error(
            `glTF bake "${animation.name}" exceeds WebGPU's default texture dimensions (8192 rows)`,
        );
    const palette = world.resource(SkinPalette);
    const name = `glTF bake "${animation.name}" palette (8192 rows)`;
    const firstJoint = jointCount ? palette.allocate(frameCount * jointCount, name) : 0;
    let firstWeight = 0;
    try {
        firstWeight = weightCount ? palette.morph.allocate(frameCount * weightCount) : 0;
    } catch (error) {
        if (jointCount) palette.release(firstJoint, frameCount * jointCount);
        throw new Error(`glTF bake "${animation.name}": ${error}`);
    }
    const bake: GltfBake = { gltf, rate, frameCount };
    const rows: BakedRows = {
        frameCount,
        firstJoint,
        firstWeight,
        jointCount,
        weightCount,
        joints,
        weights,
        bounds: new Float32Array(frameCount * gltf.primitives.length * 4),
    };
    const id = palette.nextBake++;
    allocations.set(bake, id);
    palette.bakes.set(id, rows);
    try {
        for (let frame = 0; frame < frameCount; frame++) {
            const pose = bakeKernelPose(gltf, animation, frame / rate);
            for (let p = 0; p < gltf.primitives.length; p++) {
                palette.matrices.set(
                    pose.palettes[p],
                    (firstJoint + frame * jointCount + joints[p]) * 16,
                );
                palette.morph.weights.set(
                    pose.weights[p],
                    firstWeight + frame * weightCount + weights[p],
                );
                rows.bounds.set(pose.bounds[p], (frame * gltf.primitives.length + p) * 4);
            }
        }
    } catch (error) {
        releaseGltfBake(world, bake);
        throw error;
    }
    return bake;
}

/** Releases once; attached entities resolve to bind pose and mesh bounds at upload, even after row reuse. */
export function releaseGltfBake(world: World, bake: GltfBake): void {
    if (bake.gltf.world !== world) throw new Error("glTF handle belongs to another world");
    const id = allocations.get(bake)!;
    const palette = world.resource(SkinPalette),
        r = palette.bakes.get(id);
    if (!r) return;
    palette.bakes.delete(id);
    if (r.jointCount) palette.release(r.firstJoint, r.jointCount * bake.frameCount);
    if (r.weightCount) palette.morph.ranges.release(r.firstWeight, r.weightCount * bake.frameCount);
}

/** Selects one whole baked frame, primitive for primitive on the same Gltf, releasing owned ranges.
 * Refuses followers and instances with followers: a Leader Pose leader must own its live pose.
 * Placement stays independent; animated rigid parts stay where spawned unless poseGltfScene is called.
 * A released bake cannot be selected. Updating an attached instance's frame allocates no ranges.
 */
export function setGltfBakedFrame(
    world: World,
    instance: GltfSceneInstance,
    bake: GltfBake,
    frame: number,
): void {
    if (instance.gltf.world !== world || bake.gltf.world !== world)
        throw new Error("glTF handle belongs to another world");
    if (instance.gltf !== bake.gltf) throw new Error("glTF baked frame requires the same Gltf");
    const id = allocations.get(bake)!;
    if (!world.resource(SkinPalette).bakes.has(id)) throw new Error("glTF bake is released");
    if (!Number.isInteger(frame) || frame < 0 || frame >= bake.frameCount)
        throw new Error("glTF baked frame is out of range");
    const storage = world.storage(BakedFrame);
    // setGltfLeaderPose refuses a leader holding BakedFrame and removes it from a follower, so an
    // instance whose deforming entities all hold BakedFrame neither follows nor has followers.
    if (!attached(world, instance)) {
        for (const eid of instance.entities)
            if (world.has(eid, LeaderPose))
                throw new Error("glTF baked frame refuses a Leader Pose follower");
        const leaders = world.storage(LeaderPose).leader;
        for (const eid of world.query(followers))
            if (instance.entities.includes(leaders.get(eid)))
                throw new Error("glTF baked frame refuses an instance with followers");
    }
    for (let p = 0; p < instance.entities.length; p++) {
        const primitive = instance.gltf.primitives[p];
        if (primitive.skin < 0 && !primitive.morphReach) continue;
        const eid = instance.entities[p];
        if (!world.has(eid, BakedFrame)) {
            world.remove(eid, SkinnedMesh);
            world.remove(eid, MorphedMesh);
            world.add(eid, BakedFrame, { bake: id, primitive: p, frame });
        } else {
            storage.bake.set(eid, id);
            storage.frame.set(eid, frame);
        }
    }
}

function attached(world: World, instance: GltfSceneInstance): boolean {
    for (let p = 0; p < instance.entities.length; p++) {
        const primitive = instance.gltf.primitives[p];
        if (
            (primitive.skin >= 0 || primitive.morphReach) &&
            !world.has(instance.entities[p], BakedFrame)
        )
            return false;
    }
    return true;
}

import type { Resource, World } from "@dylanebert/shallot";
import {
    bakeGltfAnimation,
    type Gltf,
    type GltfBake,
    type GltfSceneInstance,
    poseGltfScene,
    releaseGltfBake,
    setGltfBakedFrame,
    setGltfLeaderPose,
    spawnGltfScene,
} from "@dylanebert/shallot/gltf";

export const crowdCount = 2560;
export const leaderCount = 8;
const columns = 64;
const spacing = 1.2;
export type CrowdMode = "live" | "shared" | "baked";
export const Crowd: Resource<{
    model: Gltf | null;
    instances: GltfSceneInstance[];
    offsets: Float64Array;
    bake: GltfBake | null;
    mode: CrowdMode;
    time: number;
    duration: number;
    costs: Record<CrowdMode, number | null>;
}> = {
    create: () => ({
        model: null,
        instances: [],
        offsets: new Float64Array(crowdCount),
        bake: null,
        mode: "live",
        time: 0,
        duration: 0,
        costs: { live: null, shared: null, baked: null },
    }),
};

export function clearCrowd(world: World) {
    const crowd = world.resource(Crowd);
    for (const instance of crowd.instances) {
        for (const eid of instance.entities) world.destroy(eid);
        for (const eid of instance.lightEntities) world.destroy(eid);
    }
    crowd.instances.length = 0;
    if (crowd.bake) releaseGltfBake(world, crowd.bake);
    crowd.bake = null;
    crowd.model = null;
}

export function createCrowd(world: World, model: Gltf) {
    const crowd = world.resource(Crowd);
    crowd.duration = 0;
    for (const sampler of model.animations[0].samplers)
        crowd.duration = Math.max(crowd.duration, sampler.input[sampler.input.length - 1]);
    for (let i = 0; i < crowdCount; i++) crowd.offsets[i] = Math.random() * crowd.duration;
    crowd.time = 0;
    crowd.costs = { live: null, shared: null, baked: null };
    crowd.model = model;
    setCrowdMode(world, "live");
}

export function setCrowdMode(world: World, mode: CrowdMode) {
    const crowd = world.resource(Crowd);
    const model = crowd.model;
    if (!model) return;
    clearCrowd(world);
    crowd.model = model;
    crowd.mode = mode;
    if (mode === "baked") crowd.bake = bakeGltfAnimation(world, model, model.animations[0], 30);
    for (let i = 0; i < crowdCount; i++) {
        const instance = spawnGltfScene(world, model, {
            translation: [
                ((i % columns) - (columns - 1) / 2) * spacing,
                0,
                (Math.floor(i / columns) - (crowdCount / columns - 1) / 2) * spacing,
            ],
        });
        if (mode === "shared" && i >= leaderCount)
            setGltfLeaderPose(world, instance, crowd.instances[i % leaderCount]);
        crowd.instances.push(instance);
    }
    animateCrowd(world, 0);
}

export function crowdBounds(model: Gltf): Gltf["bounds"] {
    const x = ((columns - 1) / 2) * spacing;
    const z = ((crowdCount / columns - 1) / 2) * spacing;
    return {
        min: [model.bounds.min[0] - x, model.bounds.min[1], model.bounds.min[2] - z],
        max: [model.bounds.max[0] + x, model.bounds.max[1], model.bounds.max[2] + z],
    };
}

export function animateCrowd(world: World, delta: number) {
    const crowd = world.resource(Crowd);
    if (!crowd.model) return;
    crowd.time += delta;
    const start = performance.now();
    const count = crowd.mode === "shared" ? leaderCount : crowdCount;
    for (let i = 0; i < count; i++) {
        const time = (crowd.time + crowd.offsets[i]) % crowd.duration;
        if (crowd.bake)
            setGltfBakedFrame(
                world,
                crowd.instances[i],
                crowd.bake,
                Math.round(time * crowd.bake.rate),
            );
        else poseGltfScene(world, crowd.instances[i], crowd.model.animations[0], time);
    }
    const elapsed = performance.now() - start;
    const previous = crowd.costs[crowd.mode];
    crowd.costs[crowd.mode] = previous === null ? elapsed : previous + (elapsed - previous) * 0.1;
}

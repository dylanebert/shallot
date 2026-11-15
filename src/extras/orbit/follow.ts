import {
    GlobalTransformHistoryStartSystem,
    RenderTransformHistoryPlugin,
    readRenderGlobalTransform,
    retainRenderTransformHistory,
} from "../../core/rendering";
import { GlobalTransform, GlobalTransformTickStartSystem, Transform } from "../../core/transform";
import type { Plugin, System, World } from "../../engine";
import { Orbit } from "./index";

const terms = { create: () => [Orbit] };
const pose = { create: () => new Float32Array(12) };
const requests = {
    create(world: World) {
        const cameras = new Map<number, { target: number; release: () => void }>();
        const release = (camera: number) => {
            cameras.get(camera)?.release();
            cameras.delete(camera);
        };
        world.observeMembership(Orbit, (camera, present) => {
            if (!present) release(camera);
        });
        world.observeMembership(Transform, (camera, present) => {
            if (!present) release(camera);
        });
        world.observeMembership(GlobalTransform, (eid, present) => {
            if (present) return;
            release(eid);
            for (const [camera, request] of cameras) if (request.target === eid) release(camera);
        });
        const clear = () => {
            for (const request of cameras.values()) request.release();
            cameras.clear();
        };
        world.onDispose(clear);
        return { cameras, release, clear };
    },
};

function requestTarget(world: World, camera: number, target: number): void {
    const state = world.resource(requests);
    const previous = state.cameras.get(camera);
    if (!target || !world.has(camera, Transform) || !world.has(target, GlobalTransform)) {
        state.release(camera);
    } else if (previous?.target !== target) {
        state.release(camera);
        state.cameras.set(camera, { target, release: retainRenderTransformHistory(world, target) });
    }
}
function updateRequests(world: World): void {
    for (const camera of world.query(world.resource(terms)))
        requestTarget(world, camera, world.storage(Orbit).target.get(camera));
}
const FollowStartSystem: System = {
    name: "orbit-follow-start",
    group: "fixed",
    boundary: "before",
    after: [GlobalTransformTickStartSystem],
    before: [GlobalTransformHistoryStartSystem],
    update: updateRequests,
};
const FollowFrameSystem: System = {
    name: "orbit-follow-frame",
    group: "simulation",
    last: true,
    update: updateRequests,
};
export const OrbitFollowPlugin: Plugin = {
    name: "OrbitFollow",
    dependencies: [RenderTransformHistoryPlugin],
    systems: [FollowStartSystem, FollowFrameSystem],
    recovery(world) {
        const state = world.resource(requests);
        return { snapshot: () => undefined, restore: state.clear };
    },
};

export function followPosition(world: World, camera: number, target: number, axis: number): number {
    requestTarget(world, camera, target);
    const out = world.resource(pose);
    readRenderGlobalTransform(world, target, out);
    return out[axis]!;
}

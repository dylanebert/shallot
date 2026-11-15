import {
    RenderTransformHistory,
    RenderTransformHistoryPlugin,
    readRenderGlobalTransform,
} from "../../core/rendering";
import { GlobalTransformHistoryStartSystem } from "../../core/rendering/global-transform";
import { GlobalTransform, GlobalTransformTickStartSystem } from "../../core/transform";
import type { Plugin, System, World } from "../../engine";
import { Orbit } from "./index";

const terms = { create: () => [Orbit] };
const pose = { create: () => new Float32Array(12) };

const FollowStartSystem: System = {
    name: "orbit-follow-start",
    group: "fixed",
    boundary: "before",
    after: [GlobalTransformTickStartSystem],
    before: [GlobalTransformHistoryStartSystem],
    update(world) {
        for (const eid of world.query(world.resource(terms))) {
            const target = world.storage(Orbit).target.get(eid);
            if (
                target &&
                world.has(target, GlobalTransform) &&
                !world.has(target, RenderTransformHistory)
            )
                world.add(target, RenderTransformHistory);
        }
    },
};

export const OrbitFollowPlugin: Plugin = {
    name: "OrbitFollow",
    dependencies: [RenderTransformHistoryPlugin],
    recovery: "stateless",
    systems: [FollowStartSystem],
};

export function followPosition(world: World, target: number, axis: number): number {
    if (!world.has(target, RenderTransformHistory)) world.add(target, RenderTransformHistory);
    const out = world.resource(pose);
    readRenderGlobalTransform(world, target, out);
    return out[axis]!;
}

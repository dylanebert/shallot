import {
    GlobalTransform,
    GlobalTransformTickEndSystem,
    GlobalTransformTickStartSystem,
    TransformRuntime,
} from "../../core/transform";
import { component, entity, not, type Plugin, type System, vec3, type World } from "../../engine";
import { Orbit } from "./index";

// Camera-local history follows fixed placement without reading GPU presentation state.
const Follow = component("OrbitFollow", { target: entity, previous: vec3, current: vec3 });
const terms = {
    create: () => ({ cameras: [Orbit], removed: [Follow, not(Orbit)], captured: [Orbit, Follow] }),
};

const FollowStartSystem: System = {
    name: "orbit-follow-start",
    group: "fixed",
    boundary: "before",
    after: [GlobalTransformTickStartSystem],
    update(world) {
        for (const eid of world.query(world.resource(terms).removed)) world.remove(eid, Follow);
        const positions = world.storage(GlobalTransform).translation;
        for (const eid of world.query(world.resource(terms).cameras)) {
            const target = world.storage(Orbit).target.get(eid);
            if (!target || !world.has(target, GlobalTransform)) {
                if (world.has(eid, Follow)) world.remove(eid, Follow);
                continue;
            }
            if (!world.has(eid, Follow)) world.add(eid, Follow);
            const history = world.storage(Follow);
            history.target.set(eid, target);
            history.previous.set(
                eid,
                positions.x.get(target),
                positions.y.get(target),
                positions.z.get(target),
            );
        }
    },
};

const FollowEndSystem: System = {
    name: "orbit-follow-end",
    group: "fixed",
    boundary: "after",
    after: [GlobalTransformTickEndSystem],
    update(world) {
        const positions = world.storage(GlobalTransform).translation;
        const history = world.storage(Follow);
        const discontinuities = world.resource(TransformRuntime).discontinuities;
        for (const eid of world.query(world.resource(terms).captured)) {
            const target = history.target.get(eid);
            if (!world.has(target, GlobalTransform)) continue;
            const x = positions.x.get(target),
                y = positions.y.get(target),
                z = positions.z.get(target);
            history.current.set(eid, x, y, z);
            if (discontinuities[target >>> 5] & (1 << (target & 31)))
                history.previous.set(eid, x, y, z);
        }
    },
};

export const OrbitFollowPlugin: Plugin = {
    name: "OrbitFollow",
    recovery: "stateless",
    components: [Follow],
    systems: [FollowStartSystem, FollowEndSystem],
};

export function followPosition(world: World, camera: number, target: number, axis: number): number {
    const offset = target * 4 + axis;
    const positions = world.storage(GlobalTransform).translation;
    const position = positions.column[offset];
    if (!world.has(camera, Follow)) return position;
    const history = world.storage(Follow);
    if (history.target.get(camera) !== target) return position;
    const discontinuities = world.resource(TransformRuntime).discontinuities;
    if (discontinuities[target >>> 5] & (1 << (target & 31))) {
        const x = positions.x.get(target),
            y = positions.y.get(target),
            z = positions.z.get(target);
        history.previous.set(camera, x, y, z);
        history.current.set(camera, x, y, z);
        return position;
    }
    // A frame-authored pose has no matching fixed history on any coordinate.
    if (
        history.current.x.get(camera) !== positions.x.get(target) ||
        history.current.y.get(camera) !== positions.y.get(target) ||
        history.current.z.get(camera) !== positions.z.get(target)
    )
        return position;
    const i = camera * 4 + axis;
    const previous = history.previous.column[i];
    return previous + (position - previous) * world.time.fixedAlpha;
}

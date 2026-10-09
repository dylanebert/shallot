import type { Plugin } from "../../engine";
import {
    type ComponentStorage,
    component,
    type Resource,
    type System,
    vec4,
    type World,
} from "../../engine/ecs";

/** Derived fixed-tick world placement, never authored. Gameplay and physics read these
 * columns; rendering owns interpolated GPU rows in core/rendering. */
export const GlobalTransform = component(
    "GlobalTransform",
    {
        translation: vec4,
        rotation: vec4,
        scale: vec4,
        linearVelocity: vec4,
    },
    {
        defaults: () => ({
            translation: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1, 0],
            linearVelocity: [0, 0, 0, 0],
        }),
    },
);
/** Authored world placement, derived into GlobalTransform by TransformPlugin. There is no hierarchy. */
export const Transform = component(
    "Transform",
    { translation: vec4, rotation: vec4, scale: vec4 },
    {
        defaults: () => ({
            translation: [0, 0, 0, 0],
            rotation: [0, 0, 0, 1],
            scale: [1, 1, 1, 1],
        }),
        requires: [GlobalTransform],
    },
);
/** @internal Fixed placement accessors and discontinuity notifications, with no GPU state. */
export const TransformRuntime: Resource<{
    placement: ComponentStorage<typeof Transform>;
    global: ComponentStorage<typeof GlobalTransform>;
    discontinuities: Uint32Array;
}> = {
    key: Symbol.for("@dylanebert/shallot/transform/runtime"),
    create(world) {
        return {
            placement: world.storage(Transform),
            global: world.storage(GlobalTransform),
            discontinuities: new Uint32Array(1),
        };
    },
};
/** Marks derived placement so a presentation reader discards interpolation across the change.
 * Requires TransformPlugin; an entity without derived placement has nothing to mark. */
export function teleport(world: World, eid: number): void {
    if (!world.has(eid, GlobalTransform)) return;
    const runtime = world.resource(TransformRuntime);
    const word = eid >>> 5;
    if (word >= runtime.discontinuities.length) {
        const next = new Uint32Array(Math.max(word + 1, runtime.discontinuities.length * 2));
        next.set(runtime.discontinuities);
        runtime.discontinuities = next;
    }
    runtime.discontinuities[word] |= 1 << (eid & 31);
}
/** @internal Gather authored placement into fixed world columns without per-row callbacks. */
export function deriveTransforms(world: World): void {
    const runtime = world.resource(TransformRuntime);
    const global = runtime.global,
        source = runtime.placement;
    const pp = source.translation.column,
        pq = source.rotation.column,
        ps = source.scale.column;
    const op = global.translation.column,
        oq = global.rotation.column,
        os = global.scale.column;
    const pd = world.fieldStorage(GlobalTransform, "translation").dirty,
        qd = world.fieldStorage(GlobalTransform, "rotation").dirty,
        sd = world.fieldStorage(GlobalTransform, "scale").dirty;
    const spd = world.fieldStorage(Transform, "translation").dirty,
        sqd = world.fieldStorage(Transform, "rotation").dirty,
        ssd = world.fieldStorage(Transform, "scale").dirty;
    const words = Math.max(spd.length, sqd.length, ssd.length);
    for (let word = 0; word < words; word++) {
        let bits = spd[word] | sqd[word] | ssd[word];
        while (bits !== 0) {
            const mask = bits & -bits;
            bits ^= mask;
            const eid = (word << 5) | (31 - Math.clz32(mask));
            if (!world.has(eid, Transform)) continue;
            const offset = eid * 4;
            let posChanged = false,
                quatChanged = false,
                scaleChanged = false;
            for (let lane = 0; lane < 4; lane++) {
                const j = offset + lane;
                if (!Object.is(op[j], pp[j])) {
                    op[j] = pp[j];
                    posChanged = true;
                }
                if (!Object.is(oq[j], pq[j])) {
                    oq[j] = pq[j];
                    quatChanged = true;
                }
                if (!Object.is(os[j], ps[j])) {
                    os[j] = ps[j];
                    scaleChanged = true;
                }
            }
            if (posChanged) pd[word] |= mask;
            if (quatChanged) qd[word] |= mask;
            if (scaleChanged) sd[word] |= mask;
        }
    }
}
/** Derives initial placement before every ordinary fixed system. */
export const GlobalTransformTickStartSystem: System = {
    group: "fixed",
    name: "global-transform-tick-start",
    boundary: "before",
    update: deriveTransforms,
};
/** Derives completed placement after every ordinary fixed system, including terminal writers. */
export const GlobalTransformTickEndSystem: System = {
    group: "fixed",
    name: "global-transform-tick-end",
    boundary: "after",
    update: deriveTransforms,
};
/** Derives simulation writes before every ordinary draw system. */
export const PrepareGlobalTransformSystem: System = {
    group: "draw",
    name: "prepare-global-transform",
    boundary: "before",
    update: deriveTransforms,
};
/** Owns authored and derived fixed-tick placement and discontinuities, never GPU allocations.
 * Physics and rendering install it as a dependency; placement-only compositions add it explicitly. */
export const TransformPlugin: Plugin = {
    name: "Transform",
    components: [GlobalTransform, Transform],
    systems: [
        GlobalTransformTickStartSystem,
        GlobalTransformTickEndSystem,
        PrepareGlobalTransformSystem,
    ],
    recovery(world) {
        const runtime = world.resource(TransformRuntime);
        return {
            snapshot: () => undefined,
            restore() {
                runtime.discontinuities.fill(0);
                for (const eid of world.query([GlobalTransform])) teleport(world, eid);
            },
        };
    },
};
/** Compose this World's fixed-tick GlobalTransform for CPU camera and query readers. */
export function composeGlobalTransform(world: World, eid: number, out: Float32Array): Float32Array {
    const { translation: pos, rotation: rot, scale } = world.storage(GlobalTransform);
    const px = pos.x.get(eid),
        py = pos.y.get(eid),
        pz = pos.z.get(eid);
    const qx = rot.x.get(eid),
        qy = rot.y.get(eid),
        qz = rot.z.get(eid),
        qw = rot.w.get(eid);
    const sx = scale.x.get(eid),
        sy = scale.y.get(eid),
        sz = scale.z.get(eid);
    const x2 = qx + qx,
        y2 = qy + qy,
        z2 = qz + qz;
    const xx = qx * x2,
        xy = qx * y2,
        xz = qx * z2,
        yy = qy * y2,
        yz = qy * z2,
        zz = qz * z2;
    const wx = qw * x2,
        wy = qw * y2,
        wz = qw * z2;
    out[0] = (1 - yy - zz) * sx;
    out[1] = (xy + wz) * sx;
    out[2] = (xz - wy) * sx;
    out[3] = 0;
    out[4] = (xy - wz) * sy;
    out[5] = (1 - xx - zz) * sy;
    out[6] = (yz + wx) * sy;
    out[7] = 0;
    out[8] = (xz + wy) * sz;
    out[9] = (yz - wx) * sz;
    out[10] = (1 - xx - yy) * sz;
    out[11] = 0;
    out[12] = px;
    out[13] = py;
    out[14] = pz;
    out[15] = 1;
    return out;
}

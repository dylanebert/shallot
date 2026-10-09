import {
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
} from "../../core/physics";
import type { FieldType, ScalarField, Vector4Field, World } from "../../engine";
import type {
    BaseJointConfig,
    DistanceJointConfig,
    MotorJointConfig,
    ParallelJointConfig,
    PrismaticJointConfig,
    RevoluteJointConfig,
    SphericalJointConfig,
    WeldJointConfig,
    WheelJointConfig,
} from "./api/config";

export const FNV_BASIS = 2166136261;
const fold = (h: number, v: number) => Math.imul(h ^ (v >>> 0), 16777619);
const bits = new Float32Array(1);
const uints = new Uint32Array(bits.buffer);
const entries = [
    ["Distance", DistanceJoint],
    ["Filter", FilterJoint],
    ["Motor", MotorJoint],
    ["Parallel", ParallelJoint],
    ["Prismatic", PrismaticJoint],
    ["Revolute", RevoluteJoint],
    ["Spherical", SphericalJoint],
    ["Weld", WeldJoint],
    ["Wheel", WheelJoint],
] as const;
type Config = BaseJointConfig &
    Partial<
        DistanceJointConfig &
            MotorJointConfig &
            ParallelJointConfig &
            PrismaticJointConfig &
            RevoluteJointConfig &
            SphericalJointConfig &
            WeldJointConfig &
            WheelJointConfig
    >;
export interface JointDef {
    kind: (typeof entries)[number][0];
    eid: number;
    a: number;
    b: number;
    config: Config;
}
const bindingsKey = {
    create: (world: World) =>
        entries.map(([kind, component], index) => ({
            kind,
            index,
            terms: [component],
            fields: Object.entries(component).map(([name, type]) => ({
                name,
                type: type as FieldType,
            })),
            storage: world.storage(component) as unknown as Record<
                string,
                ScalarField | Vector4Field
            >,
        })),
};

/** Includes resolved endpoints and reference generations, so recycling an endpoint drops its constraints. */
export function jointSignature(world: World): number {
    let h = FNV_BASIS;
    for (const binding of world.resource(bindingsKey)) {
        for (const eid of world.query(binding.terms)) {
            h = fold(h, binding.index);
            h = fold(h, eid);
            for (const { name, type } of binding.fields) {
                const field = binding.storage[name];
                if (name === "a" || name === "b") {
                    h = fold(h, (field as ScalarField).get(eid));
                    const ref = field.column[eid];
                    h = fold(h, ref);
                    h = fold(h, Math.floor(ref / 2 ** 32));
                } else {
                    for (let lane = 0; lane < type.lanes; lane++) {
                        bits[0] = field.column[eid * type.lanes + lane];
                        h = fold(h, uints[0]);
                    }
                }
            }
        }
    }
    return h;
}

/** Composes the two authored vec4 columns per frame; no pose or mass-derived values enter a definition. */
export function jointDefs(world: World): JointDef[] {
    const out: JointDef[] = [];
    for (const binding of world.resource(bindingsKey)) {
        for (const eid of world.query(binding.terms)) {
            const values: Record<string, unknown> = {};
            for (const { name, type } of binding.fields) {
                if (
                    name === "a" ||
                    name === "b" ||
                    name.startsWith("localAnchor") ||
                    name.startsWith("localRotation")
                )
                    continue;
                const field = binding.storage[name];
                if (type.lanes === 1) {
                    const value = (field as ScalarField).get(eid);
                    values[name] =
                        name.startsWith("enable") || name === "collideConnected"
                            ? value !== 0
                            : value;
                } else {
                    const vector = field as Vector4Field;
                    const xyz = {
                        x: vector.x.get(eid),
                        y: vector.y.get(eid),
                        z: vector.z.get(eid),
                    };
                    values[name] =
                        name === "targetRotation" ? { v: xyz, s: vector.w.get(eid) } : xyz;
                }
            }
            for (const suffix of ["A", "B"]) {
                const p = binding.storage[`localAnchor${suffix}`] as Vector4Field;
                const q = binding.storage[`localRotation${suffix}`] as Vector4Field;
                values[`localFrame${suffix}`] = {
                    p: { x: p.x.get(eid), y: p.y.get(eid), z: p.z.get(eid) },
                    q: {
                        v: { x: q.x.get(eid), y: q.y.get(eid), z: q.z.get(eid) },
                        s: q.w.get(eid),
                    },
                };
            }
            values.userData = eid;
            out.push({
                kind: binding.kind,
                eid,
                a: (binding.storage.a as ScalarField).get(eid),
                b: (binding.storage.b as ScalarField).get(eid),
                config: values as Config,
            });
        }
    }
    // Snapshots share these values.
    for (const def of out) deepFreeze(def);
    Object.freeze(out);
    return out;
}
function deepFreeze(value: object): void {
    for (const child of Object.values(value)) {
        if (child !== null && typeof child === "object") deepFreeze(child);
    }
    Object.freeze(value);
}

import {
    Body,
    DistanceJoint,
    FilterJoint,
    MotorJoint,
    ParallelJoint,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    SphericalJoint,
    WeldJoint,
    WheelJoint,
} from "../../core/physics";
import type { Component, FieldType, ScalarField, Vector4Field, World } from "../../engine";
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
export const JOINT_KIND_COUNT = entries.length;
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
/** One component's fields, their mark indices, dirty sets and storage, resolved once per World. */
export interface FieldBinding {
    index: number;
    component: Component;
    fields: { name: string; type: FieldType }[];
    fieldIndices: Map<string, number>;
    dirty: ReturnType<World["fieldStorage"]>[];
    storage: Record<string, ScalarField | Vector4Field>;
}
export interface JointBinding extends FieldBinding {
    kind: JointDef["kind"];
}
function fieldBinding(world: World, component: Component, index: number): FieldBinding {
    const fields = Object.entries(component).map(([name, type]) => ({
        name,
        type: type as FieldType,
    }));
    return {
        index,
        component,
        fields,
        fieldIndices: new Map(fields.map(({ name }, fieldIndex) => [name, fieldIndex])),
        dirty: fields.map(({ name }) => world.fieldStorage(component, name)),
        storage: world.storage(component) as unknown as Record<string, ScalarField | Vector4Field>,
    };
}
/** Whether `mask`, a candidate's field marks, includes the named field. */
export function fieldMarked(binding: FieldBinding, mask: number, name: string): boolean {
    const index = binding.fieldIndices.get(name);
    return (
        index !== undefined && (mask === 0xffffffff || (index < 32 && (mask & (1 << index)) !== 0))
    );
}
const bindingsKey = {
    create: (world: World): JointBinding[] =>
        entries.map(([kind, component], index) => ({
            kind,
            ...fieldBinding(world, component, index),
        })),
};

export function jointBindings(world: World): readonly JointBinding[] {
    return world.resource(bindingsKey);
}

const bodyBindingKey = { create: (world: World) => fieldBinding(world, Body, 0) };
export function bodyBinding(world: World): FieldBinding {
    return world.resource(bodyBindingKey);
}

const shapeBindingKey = { create: (world: World) => fieldBinding(world, Shape, 0) };
export function shapeBinding(world: World): FieldBinding {
    return world.resource(shapeBindingKey);
}

/** Composes one joint definition only when membership or an endpoint requires creation. */
export function jointDef(world: World, binding: JointBinding, eid: number): JointDef {
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
                name.startsWith("enable") || name === "collideConnected" ? value !== 0 : value;
        } else {
            const vector = field as Vector4Field;
            const xyz = {
                x: vector.x.get(eid),
                y: vector.y.get(eid),
                z: vector.z.get(eid),
            };
            values[name] = name === "targetRotation" ? { v: xyz, s: vector.w.get(eid) } : xyz;
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
    values.userData = world.ref(eid);
    return {
        kind: binding.kind,
        eid,
        a: (binding.storage.a as ScalarField).get(eid),
        b: (binding.storage.b as ScalarField).get(eid),
        config: values as Config,
    };
}

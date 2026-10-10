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
export interface JointBinding {
    kind: JointDef["kind"];
    index: number;
    component: Component;
    fields: { name: string; type: FieldType }[];
    fieldIndices: Map<string, number>;
    dirty: ReturnType<World["fieldStorage"]>[];
    storage: Record<string, ScalarField | Vector4Field>;
}
const bindingsKey = {
    create: (world: World): JointBinding[] =>
        entries.map(([kind, component], index) => {
            const fields = Object.entries(component).map(([name, type]) => ({
                name,
                type: type as FieldType,
            }));
            return {
                kind,
                index,
                component,
                fields,
                fieldIndices: new Map(fields.map(({ name }, fieldIndex) => [name, fieldIndex])),
                dirty: fields.map(({ name }) => world.fieldStorage(component, name)),
                storage: world.storage(component) as unknown as Record<
                    string,
                    ScalarField | Vector4Field
                >,
            };
        }),
};

export function jointBindings(world: World): readonly JointBinding[] {
    return world.resource(bindingsKey);
}

export interface BodyBinding {
    index: number;
    component: Component;
    fields: { name: string; type: FieldType }[];
    fieldIndices: Map<string, number>;
    dirty: ReturnType<World["fieldStorage"]>[];
    storage: Record<string, ScalarField | Vector4Field>;
}
const bodyBindingKey = {
    create: (world: World): BodyBinding => {
        const fields = Object.entries(Body).map(([name, type]) => ({
            name,
            type: type as FieldType,
        }));
        return {
            index: 0,
            component: Body,
            fields,
            fieldIndices: new Map(fields.map(({ name }, index) => [name, index])),
            dirty: fields.map(({ name }) => world.fieldStorage(Body, name)),
            storage: world.storage(Body) as unknown as Record<string, ScalarField | Vector4Field>,
        };
    },
};
export function bodyBinding(world: World): BodyBinding {
    return world.resource(bodyBindingKey);
}

export interface ShapeBinding {
    index: number;
    component: Component;
    fields: { name: string; type: FieldType }[];
    fieldIndices: Map<string, number>;
    dirty: ReturnType<World["fieldStorage"]>[];
    storage: Record<string, ScalarField | Vector4Field>;
}
const shapeBindingKey = {
    create: (world: World): ShapeBinding => {
        const fields = Object.entries(Shape).map(([name, type]) => ({
            name,
            type: type as FieldType,
        }));
        return {
            index: 0,
            component: Shape,
            fields,
            fieldIndices: new Map(fields.map(({ name }, index) => [name, index])),
            dirty: fields.map(({ name }) => world.fieldStorage(Shape, name)),
            storage: world.storage(Shape) as unknown as Record<string, ScalarField | Vector4Field>,
        };
    },
};
export function shapeBinding(world: World): ShapeBinding {
    return world.resource(shapeBindingKey);
}

/** Composes one joint definition only when membership or an endpoint requires creation. */
export function jointDef(binding: JointBinding, eid: number): JointDef {
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
    values.userData = eid;
    return {
        kind: binding.kind,
        eid,
        a: (binding.storage.a as ScalarField).get(eid),
        b: (binding.storage.b as ScalarField).get(eid),
        config: values as Config,
    };
}

import type { EntityRef } from "../../engine";
import type { Vec3 } from "./api";
import * as layout from "./kernel/contact-layout";
import type { Kernel } from "./kernel/kernel";

export interface PhysicsManifoldPoint {
    anchorA: Vec3;
    anchorB: Vec3;
    separation: number;
    baseSeparation: number;
    normalImpulse: number;
    totalNormalImpulse: number;
    normalVelocity: number;
    featureId: number;
    triangleIndex: number;
    persisted: boolean;
}

export interface PhysicsManifold {
    points: PhysicsManifoldPoint[];
    normal: Vec3;
    twistImpulse: number;
    frictionImpulse: Vec3;
    rollingImpulse: Vec3;
}

export interface PhysicsContactData {
    shapeA: EntityRef;
    shapeB: EntityRef;
    manifolds: PhysicsManifold[];
}

export function readContactManifolds(k: Kernel, pointer: number, count: number): PhysicsManifold[] {
    const f = new Float32Array(k.memory.buffer, pointer, count * layout.MANIFOLD_STRIDE);
    const u = new Uint32Array(f.buffer, f.byteOffset, f.length);
    const i = new Int32Array(f.buffer, f.byteOffset, f.length);
    const vec = (offset: number): Vec3 => ({ x: f[offset]!, y: f[offset + 1]!, z: f[offset + 2]! });
    return Array.from({ length: count }, (_, m) => {
        const o = m * layout.MANIFOLD_STRIDE;
        return {
            normal: vec(o + layout.M_NORMAL),
            twistImpulse: f[o + layout.M_TWIST]!,
            frictionImpulse: vec(o + layout.M_FRICTION),
            rollingImpulse: vec(o + layout.M_ROLLING),
            points: Array.from({ length: i[o + layout.M_POINT_COUNT]! }, (_, p) => {
                const base = o + layout.M_POINTS + p * layout.POINT_STRIDE;
                return {
                    anchorA: vec(base + layout.P_ANCHOR_A),
                    anchorB: vec(base + layout.P_ANCHOR_B),
                    separation: f[base + layout.P_SEPARATION]!,
                    baseSeparation: f[base + layout.P_BASE_SEPARATION]!,
                    normalImpulse: f[base + layout.P_NORMAL_IMPULSE]!,
                    totalNormalImpulse: f[base + layout.P_TOTAL_NORMAL_IMPULSE]!,
                    normalVelocity: f[base + layout.P_NORMAL_VELOCITY]!,
                    featureId: u[base + layout.P_FEATURE_ID]!,
                    triangleIndex: i[base + layout.P_TRIANGLE_INDEX]!,
                    persisted: (u[base + layout.P_PERSISTED]! & 0xff) !== 0,
                };
            }),
        };
    });
}

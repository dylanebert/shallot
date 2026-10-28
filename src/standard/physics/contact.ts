import type { EntityRef } from "../../engine";
import type { Vec3 } from "./api";
import * as layout from "./kernel/contact-layout";
import type { Kernel } from "./kernel/kernel";
import { liveShapeEntity, type ShapeEntity } from "./world/shape-entities";
import type { WorldState } from "./world/world";

export interface PhysicsManifoldPoint {
    /** World-space offset from body A's center of mass, in meters. */
    anchorA: Vec3;
    /** World-space offset from body B's center of mass, in meters. */
    anchorB: Vec3;
    /** Signed separation in meters; negative means penetration. */
    separation: number;
    /** Cached separation in meters used for contact recycling. */
    baseSeparation: number;
    /** Normal impulse in kg·m/s from the final substep. */
    normalImpulse: number;
    /** Sum of normal impulses in kg·m/s across the completed step's substeps. */
    totalNormalImpulse: number;
    /** Relative normal velocity in m/s before solving; negative means approaching. */
    normalVelocity: number;
    /** Unitless feature identifier matching a point between steps. */
    featureId: number;
    /** Mesh or height-field triangle index, or -1 when neither participates. */
    triangleIndex: number;
    /** Whether the point existed in the preceding step. */
    persisted: boolean;
}

export interface PhysicsManifold {
    /** Number of active contact points, including speculative points with positive separation. */
    pointCount: number;
    /** Retained point storage; only the first `pointCount` entries describe this step. */
    points: PhysicsManifoldPoint[];
    /** World-space unit normal pointing from returned shape A toward returned shape B. */
    normal: Vec3;
    /** Central friction angular impulse about the normal, in kg·m²/s. */
    twistImpulse: number;
    /** World-space central friction linear impulse, in kg·m/s. */
    frictionImpulse: Vec3;
    /** World-space rolling-resistance angular impulse, in kg·m²/s. */
    rollingImpulse: Vec3;
}

export interface PhysicsContactData {
    /** Shape A's generation-aware identity in solver order, not argument order. */
    shapeA: EntityRef;
    /** Shape B's generation-aware identity in solver order, not argument order. */
    shapeB: EntityRef;
    /** Compound child identifying the contact within this pair; 0 for a non-compound pair. */
    childIndex: number;
    /** Number of current manifolds from the most recently completed fixed physics step. */
    manifoldCount: number;
    /** Retained manifold storage; only the first `manifoldCount` entries describe this step. */
    manifolds: PhysicsManifold[];
}

const vector = (): Vec3 => ({ x: 0, y: 0, z: 0 });
function point(): PhysicsManifoldPoint {
    return {
        anchorA: vector(),
        anchorB: vector(),
        separation: 0,
        baseSeparation: 0,
        normalImpulse: 0,
        totalNormalImpulse: 0,
        normalVelocity: 0,
        featureId: 0,
        triangleIndex: -1,
        persisted: false,
    };
}

/** Per-world scratch, retaining manifold and point records across narrower reads. */
export class ContactDataBuffer {
    readonly data: PhysicsContactData = {
        shapeA: 0 as EntityRef,
        shapeB: 0 as EntityRef,
        childIndex: 0,
        manifoldCount: 0,
        manifolds: [],
    };
    readonly first: ShapeEntity = { shape: 0 as EntityRef, body: 0 as EntityRef };
    readonly second: ShapeEntity = { shape: 0 as EntityRef, body: 0 as EntityRef };
    f = new Float32Array(0);
    u = new Uint32Array(0);
    i = new Int32Array(0);

    read(
        k: Kernel,
        pointer: number,
        childIndex: number,
        state: WorldState,
    ): PhysicsContactData | null {
        if (this.f.buffer !== k.memory.buffer) {
            this.f = new Float32Array(k.memory.buffer);
            this.u = new Uint32Array(k.memory.buffer);
            this.i = new Int32Array(k.memory.buffer);
        }
        const header = k.worldQueryResultPtr() >>> 2;
        const count = this.u[header + 2]!;
        if (
            !liveShapeEntity(state.shapeEntities, this.u[header]!, this.first) ||
            !liveShapeEntity(state.shapeEntities, this.u[header + 1]!, this.second)
        )
            return null;
        const data = this.data;
        data.shapeA = this.first.shape;
        data.shapeB = this.second.shape;
        data.childIndex = childIndex;
        for (let m = 0; m < count; m++) {
            let manifold = data.manifolds[m];
            if (!manifold) {
                manifold = {
                    pointCount: 0,
                    points: [],
                    normal: vector(),
                    twistImpulse: 0,
                    frictionImpulse: vector(),
                    rollingImpulse: vector(),
                };
                data.manifolds[m] = manifold;
            }
            const o = (pointer >>> 2) + m * layout.MANIFOLD_STRIDE;
            this.vec(manifold.normal, o + layout.M_NORMAL);
            manifold.twistImpulse = this.f[o + layout.M_TWIST]!;
            this.vec(manifold.frictionImpulse, o + layout.M_FRICTION);
            this.vec(manifold.rollingImpulse, o + layout.M_ROLLING);
            const pointCount = this.i[o + layout.M_POINT_COUNT]!;
            for (let p = 0; p < pointCount; p++) {
                const record = manifold.points[p] ?? (manifold.points[p] = point());
                const base = o + layout.M_POINTS + p * layout.POINT_STRIDE;
                this.vec(record.anchorA, base + layout.P_ANCHOR_A);
                this.vec(record.anchorB, base + layout.P_ANCHOR_B);
                record.separation = this.f[base + layout.P_SEPARATION]!;
                record.baseSeparation = this.f[base + layout.P_BASE_SEPARATION]!;
                record.normalImpulse = this.f[base + layout.P_NORMAL_IMPULSE]!;
                record.totalNormalImpulse = this.f[base + layout.P_TOTAL_NORMAL_IMPULSE]!;
                record.normalVelocity = this.f[base + layout.P_NORMAL_VELOCITY]!;
                record.featureId = this.u[base + layout.P_FEATURE_ID]!;
                record.triangleIndex = this.i[base + layout.P_TRIANGLE_INDEX]!;
                record.persisted = (this.u[base + layout.P_PERSISTED]! & 0xff) !== 0;
            }
            manifold.pointCount = pointCount;
        }
        data.manifoldCount = count;
        return data;
    }

    private vec(out: Vec3, offset: number): void {
        out.x = this.f[offset]!;
        out.y = this.f[offset + 1]!;
        out.z = this.f[offset + 2]!;
    }
}

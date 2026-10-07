import type { World } from "../../../engine";
import type { SurfaceMaterial } from "../common/types";
import { kernel } from "../kernel/kernel";
import { KernelViews } from "../kernel/views";
import type { WorldState } from "../world/world";
import type { Manifold } from "./contact";

import {
    DIR_STRIDE,
    DIR_COUNT,
    DIR_BLOCK,
    DIR_FLAGS,
    DIR_FRICTION,
    DIR_RESTITUTION,
    DIR_ROLLING_RESISTANCE,
    DIR_TANGENT_VELOCITY,
    MANIFOLD_STRIDE,
    M_NORMAL,
    M_TWIST,
    M_FRICTION,
    M_ROLLING,
    M_POINT_COUNT,
    M_POINTS,
    POINT_STRIDE,
    P_ANCHOR_A,
    P_ANCHOR_B,
    P_SEPARATION,
    P_BASE_SEPARATION,
    P_NORMAL_IMPULSE,
    P_TOTAL_NORMAL_IMPULSE,
    P_NORMAL_VELOCITY,
    P_FEATURE_ID,
    P_TRIANGLE_INDEX,
    P_PERSISTED,
} from "../kernel/contact-layout";
export {
    DIR_STRIDE,
    MANIFOLD_STRIDE,
    DIR_BLOCK,
    M_POINT_COUNT,
    M_POINTS,
    POINT_STRIDE,
} from "../kernel/contact-layout";

/** Current views of a world's contact directory and stable manifold blocks. */
export class ManifoldStore extends KernelViews {
    readonly worldId: number;
    private _layout = new Uint32Array(0);
    dirF = new Float32Array(0);
    dirU = new Uint32Array(0);
    poolF = new Float32Array(0);
    poolU = new Uint32Array(0);
    poolI = new Int32Array(0);
    constructor(ecsState: World | undefined, worldId: number) {
        super(ecsState);
        this.worldId = worldId;
        this.guardViews();
    }
    override captureCheckpoint() {
        return null;
    }
    override restoreCheckpoint(_state: unknown): void {}
    freeSlot(id: number): void {
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        k.freeManifolds(id);
        k.freeMeshCache(id);
    }
    protected deriveViews(): void {
        const k = kernel(this.ecsState);
        const cap = k.contactRecordCapacity(this.worldId);
        k.bodySetActiveWorld(this.worldId);
        const buf = k.memory.buffer;
        const pointer = k.manifoldLayoutPtr();
        if (this._layout.buffer !== buf || this._layout.byteOffset !== pointer)
            this._layout = new Uint32Array(buf, pointer, 1);
        const address = this._layout[0];
        if (
            this.dirU.buffer === buf &&
            this.dirU.byteOffset === address &&
            this.dirU.length === cap * DIR_STRIDE
        )
            return;
        this.dirF = new Float32Array(buf, address, cap * DIR_STRIDE);
        this.dirU = new Uint32Array(buf, address, cap * DIR_STRIDE);
        this.poolF = new Float32Array(buf);
        this.poolU = new Uint32Array(buf);
        this.poolI = new Int32Array(buf);
    }
}

export function writeContactMaterial(
    dirF: Float32Array,
    id: number,
    material: Pick<
        SurfaceMaterial,
        "friction" | "restitution" | "rollingResistance" | "tangentVelocity"
    >,
): void {
    const o = id * DIR_STRIDE;
    dirF[o + DIR_FRICTION] = material.friction;
    dirF[o + DIR_RESTITUTION] = material.restitution;
    dirF[o + DIR_ROLLING_RESISTANCE] = material.rollingResistance;
    dirF[o + DIR_TANGENT_VELOCITY] = material.tangentVelocity.x;
    dirF[o + DIR_TANGENT_VELOCITY + 1] = material.tangentVelocity.y;
    dirF[o + DIR_TANGENT_VELOCITY + 2] = material.tangentVelocity.z;
}
/** Independent snapshot of a contact's mixed material. */
export function readContactMaterial(dirF: Float32Array, id: number) {
    const o = id * DIR_STRIDE;
    return {
        friction: dirF[o + DIR_FRICTION],
        restitution: dirF[o + DIR_RESTITUTION],
        rollingResistance: dirF[o + DIR_ROLLING_RESISTANCE],
        tangentVelocity: {
            x: dirF[o + DIR_TANGENT_VELOCITY],
            y: dirF[o + DIR_TANGENT_VELOCITY + 1],
            z: dirF[o + DIR_TANGENT_VELOCITY + 2],
        },
    };
}
export function contactHit(dirU: Uint32Array, id: number): boolean {
    return (dirU[id * DIR_STRIDE + DIR_FLAGS] & 0x00000002) !== 0;
}
export function contactPointCount(
    dirU: Uint32Array,
    poolU: Uint32Array,
    id: number,
    count: number,
): number {
    const base = dirU[id * DIR_STRIDE + DIR_BLOCK] >>> 2;
    let points = 0;
    for (let i = 0; i < count; ++i) points += poolU[base + i * MANIFOLD_STRIDE + M_POINT_COUNT];
    return points;
}
export function contactTotalImpulse(world: WorldState, id: number): number {
    const store = world.manifoldStore,
        u = store.dirU,
        p = store.poolU,
        f = store.poolF;
    const count = u[id * DIR_STRIDE + DIR_COUNT];
    const base = u[id * DIR_STRIDE + DIR_BLOCK] >>> 2;
    let impulse = 0;
    for (let m = 0; m < count; ++m) {
        const o = base + m * MANIFOLD_STRIDE;
        for (let point = 0; point < p[o + M_POINT_COUNT]; ++point)
            impulse = Math.fround(
                impulse + f[o + M_POINTS + point * POINT_STRIDE + P_TOTAL_NORMAL_IMPULSE],
            );
    }
    return impulse;
}
/** Independent manifold snapshots; no object holds an address into kernel storage. */
export function readContactManifolds(world: WorldState, id: number): Manifold[] {
    const store = world.manifoldStore;
    const u = store.dirU;
    const count = u[id * DIR_STRIDE + DIR_COUNT];
    const base = u[id * DIR_STRIDE + DIR_BLOCK] >>> 2;
    const f = store.poolF,
        p = store.poolU,
        s = store.poolI;
    const vector = (o: number) => ({ x: f[o], y: f[o + 1], z: f[o + 2] });
    const out: Manifold[] = [];
    for (let i = 0; i < count; ++i) {
        const o = base + i * MANIFOLD_STRIDE;
        const pointCount = p[o + M_POINT_COUNT];
        const manifold: Manifold = {
            normal: vector(o + M_NORMAL),
            frictionImpulse: vector(o + M_FRICTION),
            twistImpulse: f[o + M_TWIST],
            rollingImpulse: vector(o + M_ROLLING),
            pointCount,
            points: [],
        };
        for (let j = 0; j < pointCount; ++j) {
            const q = o + M_POINTS + j * POINT_STRIDE;
            manifold.points.push({
                anchorA: vector(q + P_ANCHOR_A),
                anchorB: vector(q + P_ANCHOR_B),
                separation: f[q + P_SEPARATION],
                baseSeparation: f[q + P_BASE_SEPARATION],
                normalImpulse: f[q + P_NORMAL_IMPULSE],
                totalNormalImpulse: f[q + P_TOTAL_NORMAL_IMPULSE],
                normalVelocity: f[q + P_NORMAL_VELOCITY],
                featureId: p[q + P_FEATURE_ID],
                triangleIndex: s[q + P_TRIANGLE_INDEX],
                persisted: (p[q + P_PERSISTED] & 0xff) !== 0,
            });
        }
        out.push(manifold);
    }
    return out;
}
export function createManifoldStore(world: World | undefined, worldId: number): ManifoldStore {
    return new ManifoldStore(world, worldId);
}

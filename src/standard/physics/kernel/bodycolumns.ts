import type { World } from "../../../engine";
// Views over the kernel's resident b3BodyState and b3BodySim arrays. Flags are inline in state.
//
// Each World owns its columns. A reserve can reallocate a column without growing memory, so
// refresh after grow-capable calls checks both the column offsets and the memory buffer.

import { SetType } from "../common/constants";
import type { Mat3, Vec3, WorldTransform } from "../common/records";
import type { BodySim } from "../world/body";
import type { WorldState } from "../world/world";
import { BODY_RECORD_STRIDE } from "./bodyrecords";
import { BODY_ID, CENTER0, FLAGS, MIN_EXTENT, SIM_STRIDE, STATE_STRIDE } from "./columns";
import { kernel } from "./kernel";
import { KernelViews } from "./views";

// BODY_LAYOUT header indices (bodies.rs); slots 2, 3 and 5 are unallocated.
export const B_STATE = 0;
const B_SIM = 1;
export const B_FLAGS = 4;
const B_SYNC_EID = 8;
const B_SYNC_POS = 9;
const B_SYNC_QUAT = 10;
const B_SYNC_VEL = 11;
const B_RECORD = 13;
export const N_BODY = 14;

type MovedRows = { eids: Uint32Array; pos: Float32Array; quat: Float32Array; vel: Float32Array };

/** Null-lane identity records the region holds past `bodyCap` — one per thread, since the wide
 * gather/scatter writes the running worker's record (bodies.rs `IDENT_RECORDS`). */
export const IDENT_RECORDS = 8;

/** Typed-array views over the kernel's resident body columns. One per world. */
export class BodyStore extends KernelViews {
    readonly worldId: number;
    constructor(ecsState: World | undefined, worldId: number) {
        super(ecsState);
        this.worldId = worldId;
        this.guardViews();
    }

    /** Resident state column (`STATE_STRIDE` f32 per body), re-derived after growth. */
    stateF = new Float32Array(0);
    /** Integer view of state; each record's flags occupy word 13. */
    flagsU = new Uint32Array(0);
    recordF = new Float32Array(0);
    recordU = new Uint32Array(0);
    memoryF = new Float32Array(0);
    memoryU = new Uint32Array(0);
    /** Resident b3BodySim records in Box3D field order. */
    simF = new Float32Array(0);
    /** The same sim bytes viewed as u32, for the integer `bodyId`/`flags` slots. */
    simU = new Uint32Array(0);
    syncCount = 0;
    #syncRanges = new Map<number, MovedRows>();
    // The held layout header view the column views are derived from.
    private _layout = new Uint32Array(0);

    /** Re-derive the column views over the current region. No-op before the kernel first reserves
     * bodies (the region has zero capacity), and when the buffer, layout offsets and capacity are those
     * the views were derived at, so a steady step mints no typed-array views. */
    protected deriveViews(): void {
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        const cap = k.bodyCap();
        if (cap === 0) return;
        const buf = k.memory.buffer;
        if (this.memoryF.buffer !== buf || this.memoryF.byteLength !== buf.byteLength) {
            this.memoryF = new Float32Array(buf);
            this.memoryU = new Uint32Array(buf);
        }
        const ptr = k.bodyLayoutPtr();
        if (this._layout.buffer !== buf || this._layout.byteOffset !== ptr)
            this._layout = new Uint32Array(buf, ptr, N_BODY);
        const layout = this._layout;
        if (
            this.stateF.buffer === buf &&
            this.stateF.byteOffset === layout[B_STATE] &&
            this.stateF.length === cap * STATE_STRIDE &&
            this.flagsU.byteOffset === layout[B_FLAGS] &&
            this.simF.byteOffset === layout[B_SIM] &&
            this.recordU.byteOffset === layout[B_RECORD]
        )
            return;
        this.stateF = new Float32Array(buf, layout[B_STATE], cap * STATE_STRIDE);
        this.flagsU = new Uint32Array(buf, layout[B_FLAGS], cap * STATE_STRIDE);
        this.recordF = new Float32Array(buf, layout[B_RECORD], cap * BODY_RECORD_STRIDE);
        this.recordU = new Uint32Array(buf, layout[B_RECORD], cap * BODY_RECORD_STRIDE);
        this.simF = new Float32Array(buf, layout[B_SIM], cap * SIM_STRIDE);
        this.simU = new Uint32Array(buf, layout[B_SIM], cap * SIM_STRIDE);
        this.#syncRanges.clear();
    }

    override captureCheckpoint() {
        return { syncCount: this.syncCount };
    }

    override restoreCheckpoint(state: unknown): void {
        this.syncCount = (state as ReturnType<BodyStore["captureCheckpoint"]>).syncCount;
        this.#syncRanges.clear();
        this.#setColumns.clear();
    }

    /** Stable views of the kernel's compact, ECS-tagged moved rows. */
    movedRows(): MovedRows {
        this.refreshViews();
        const count = this.syncCount;
        const cached = this.#syncRanges.get(count);
        if (cached) return cached;
        const buf = this.stateF.buffer;
        const layout = this._layout;
        const rows = {
            eids: new Uint32Array(buf, layout[B_SYNC_EID], count),
            pos: new Float32Array(buf, layout[B_SYNC_POS], count * 4),
            quat: new Float32Array(buf, layout[B_SYNC_QUAT], count * 4),
            vel: new Float32Array(buf, layout[B_SYNC_VEL], count * 4),
        };
        this.#syncRanges.set(count, rows);
        return rows;
    }

    readonly #setColumns = new Map<
        number,
        {
            layout: Uint32Array;
            simF: Float32Array;
            simU: Uint32Array;
        }
    >();
    simColumns(set: number) {
        if (set === SetType.Awake) return this;
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        const ptr = k.solverSetLayout(set);
        const old = this.#setColumns.get(set);
        const layout =
            old?.layout.buffer === k.memory.buffer && old.layout.byteOffset === ptr
                ? old.layout
                : new Uint32Array(k.memory.buffer, ptr, 6);
        const count = k.solverSetBodyCount(set);
        if (
            old &&
            old.simF.buffer === k.memory.buffer &&
            old.simF.byteOffset === layout[1] &&
            old.simF.length >= count * SIM_STRIDE
        )
            return old;
        const columns = {
            layout,
            simF: new Float32Array(k.memory.buffer, layout[1], count * SIM_STRIDE),
            simU: new Uint32Array(k.memory.buffer, layout[1], count * SIM_STRIDE),
        };
        this.#setColumns.set(set, columns);
        return columns;
    }
}

/** Create an empty body store for a new world. Its views are derived on the first refresh. */
export function createBodyStore(world: World | undefined, worldId: number): BodyStore {
    return new BodyStore(world, worldId);
}

export function setSimField<K extends keyof BodySim>(
    world: WorldState,
    ref: number,
    field: K,
    value: BodySim[K],
): void {
    const store = world.bodyStore;
    switch (field) {
        case "center": {
            const v = value as Vec3;
            const f = store.memoryF;
            const o = simOffset(world, ref);
            f[o + 7] = v.x;
            f[o + 8] = v.y;
            f[o + 9] = v.z;
            return;
        }
        case "center0": {
            const v = value as Vec3;
            const f = store.memoryF;
            const o = simOffset(world, ref) + CENTER0;
            f[o] = v.x;
            f[o + 1] = v.y;
            f[o + 2] = v.z;
            return;
        }
        case "localCenter": {
            const v = value as Vec3;
            const f = store.memoryF;
            const o = simOffset(world, ref);
            f[o + 17] = v.x;
            f[o + 18] = v.y;
            f[o + 19] = v.z;
            return;
        }
        case "force": {
            const v = value as Vec3;
            const sf = store.memoryF;
            const so = simOffset(world, ref);
            sf[so + 20] = v.x;
            sf[so + 21] = v.y;
            sf[so + 22] = v.z;
            return;
        }
        case "torque": {
            const v = value as Vec3;
            const sf = store.memoryF;
            const so = simOffset(world, ref);
            sf[so + 23] = v.x;
            sf[so + 24] = v.y;
            sf[so + 25] = v.z;
            return;
        }
        case "invMass": {
            const v = value as number;
            store.memoryF[simOffset(world, ref) + 26] = v;
            return;
        }
        case "invInertiaLocal": {
            const m = value as Mat3;
            const f = store.memoryF;
            const o = simOffset(world, ref) + 27;
            f[o] = m.cx.x;
            f[o + 1] = m.cx.y;
            f[o + 2] = m.cx.z;
            f[o + 3] = m.cy.x;
            f[o + 4] = m.cy.y;
            f[o + 5] = m.cy.z;
            f[o + 6] = m.cz.x;
            f[o + 7] = m.cz.y;
            f[o + 8] = m.cz.z;
            return;
        }
        case "invInertiaWorld": {
            const m = value as Mat3;
            const f = store.memoryF;
            const o = simOffset(world, ref) + 36;
            f[o] = m.cx.x;
            f[o + 1] = m.cx.y;
            f[o + 2] = m.cx.z;
            f[o + 3] = m.cy.x;
            f[o + 4] = m.cy.y;
            f[o + 5] = m.cy.z;
            f[o + 6] = m.cz.x;
            f[o + 7] = m.cz.y;
            f[o + 8] = m.cz.z;
            return;
        }
        case "minExtent": {
            const v = value as number;
            store.memoryF[simOffset(world, ref) + MIN_EXTENT] = v;
            return;
        }
        case "maxExtent": {
            const v = value as Vec3;
            const f = store.memoryF;
            const o = simOffset(world, ref);
            f[o + 46] = v.x;
            f[o + 47] = v.y;
            f[o + 48] = v.z;
            return;
        }
        case "linearDamping": {
            const v = value as number;
            store.memoryF[simOffset(world, ref) + 49] = v;
            return;
        }
        case "angularDamping": {
            const v = value as number;
            store.memoryF[simOffset(world, ref) + 50] = v;
            return;
        }
        case "gravityScale": {
            const v = value as number;
            store.memoryF[simOffset(world, ref) + 51] = v;
            return;
        }
        case "bodyId": {
            const v = value as number;
            store.memoryU[simOffset(world, ref) + BODY_ID] = v;
            return;
        }
        case "flags": {
            const v = value as number;
            store.memoryU[simOffset(world, ref) + FLAGS] = v;
            return;
        }
        default:
            throw new Error("unknown body sim field");
    }
}

export function readSimTransform(
    world: WorldState,
    ref: number,
    out: WorldTransform,
): WorldTransform {
    const sf = world.bodyStore.memoryF;
    const so = simOffset(world, ref);
    out.p.x = sf[so];
    out.p.y = sf[so + 1];
    out.p.z = sf[so + 2];
    out.q.v.x = sf[so + 3];
    out.q.v.y = sf[so + 4];
    out.q.v.z = sf[so + 5];
    out.q.s = sf[so + 6];
    return out;
}

export function readSimCenter(world: WorldState, ref: number, out: Vec3): Vec3 {
    const f = world.bodyStore.memoryF;
    const o = simOffset(world, ref) + 7;
    out.x = f[o];
    out.y = f[o + 1];
    out.z = f[o + 2];
    return out;
}

export function readSimLocalCenter(world: WorldState, ref: number, out: Vec3): Vec3 {
    const f = world.bodyStore.memoryF;
    const o = simOffset(world, ref) + 17;
    out.x = f[o];
    out.y = f[o + 1];
    out.z = f[o + 2];
    return out;
}

export function readStateLinearVelocity(world: WorldState, ref: number, out: Vec3): Vec3 {
    const f = world.bodyStore.stateF;
    const o = ref * STATE_STRIDE + 0;
    out.x = f[o];
    out.y = f[o + 1];
    out.z = f[o + 2];
    return out;
}

export function readStateAngularVelocity(world: WorldState, ref: number, out: Vec3): Vec3 {
    const f = world.bodyStore.stateF;
    const o = ref * STATE_STRIDE + 3;
    out.x = f[o];
    out.y = f[o + 1];
    out.z = f[o + 2];
    return out;
}

export function bodySimSlot(set: number, index: number): number {
    return set === SetType.Awake ? index : (set + 1) * 4294967296 + index;
}
export function bodySimIndex(ref: number): number {
    return ref % 4294967296;
}

export function bodySimSet(ref: number): number {
    return ref < 4294967296 ? SetType.Awake : Math.floor(ref / 4294967296) - 1;
}

export function readSimInvInertiaLocal(world: WorldState, ref: number, out: Mat3): Mat3 {
    return readSimMatrix(world, ref, 27, out);
}
export function readSimInvInertiaWorld(world: WorldState, ref: number, out: Mat3): Mat3 {
    return readSimMatrix(world, ref, 36, out);
}
function readSimMatrix(world: WorldState, ref: number, offset: number, out: Mat3): Mat3 {
    const f = world.bodyStore.memoryF,
        o = simOffset(world, ref) + offset;
    out.cx.x = f[o];
    out.cx.y = f[o + 1];
    out.cx.z = f[o + 2];
    out.cy.x = f[o + 3];
    out.cy.y = f[o + 4];
    out.cy.z = f[o + 5];
    out.cz.x = f[o + 6];
    out.cz.y = f[o + 7];
    out.cz.z = f[o + 8];
    return out;
}

export function simBodyId(world: WorldState, ref: number): number {
    if (ref < 0) return -ref - 1;
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k.solverSetBodyId(bodySimSet(ref), bodySimIndex(ref));
}

export function simFlags(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryU[simOffset(world, ref) + FLAGS];
}

function simOffset(world: WorldState, ref: number): number {
    const k = kernel(world.ecsState);
    return (
        (ref < 0
            ? k.bodyColumnPtr(world.worldId, -ref - 1, B_SIM)
            : k.simColumnPtr(world.worldId, bodySimSet(ref), bodySimIndex(ref), B_SIM)) >>> 2
    );
}

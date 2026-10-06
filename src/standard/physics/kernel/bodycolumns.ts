import type { World } from "../../../engine";
import { BodyField, bodyField } from "./bodyrecords";
// The persistent body region (kernel/src/bodies.rs) — the awake body columns held resident across
// steps in each World's allocations: velocity/delta `state` + `flags` and the
// integrate/finalize `sim`/`fin`/`sim2` fields. The solver runs directly over these columns
// (the kernel phases alias `LAYOUT[STATE]`/`LAYOUT[SIM]`/etc here), so a step no longer marshals the
// body in and reads it back out — the column is the single source of truth. This module owns the
// grow-only sizing (the region tracks the total-body high-water so it never shrinks with the churny
// awake set), body-column field access, and the record migration that mirrors the
// JS-array swap-remove when an awake body leaves the set.
//
// Each World owns its columns. A reserve can reallocate a column without growing memory, so
// refresh after grow-capable calls checks both the column offsets and the memory buffer.

import { SetType } from "../common/constants";
import type { Mat3, Quat, Vec3, WorldTransform } from "../common/math";
import type { BodySim, BodyState } from "../world/body";
import type { WorldState } from "../world/world";
import {
    FIN_STRIDE,
    MOVE_STRIDE,
    S2_BODY_ID,
    S2_CENTER0,
    S2_FLAGS,
    S2_HEAD_SHAPE,
    S2_MAX_ANGULAR_VELOCITY,
    S2_MIN_EXTENT,
    S2_ROTATION0,
    SIM_STRIDE,
    SIM2_STRIDE,
    STATE_STRIDE,
} from "./columns";
import { kernel } from "./kernel";
import { KernelViews } from "./views";

// BODY_LAYOUT header indices (bodies.rs), in memory order: world, sim, fin, finOut, flags, sim2.
export const B_STATE = 0;
const B_SIM = 1;
const B_FIN = 2;
export const B_FLAGS = 4;
const B_SIM2 = 5;
const B_MOVE = 6;
const B_SYNC_EID = 8;
const B_SYNC_POS = 9;
const B_SYNC_QUAT = 10;
const B_SYNC_VEL = 11;
const B_RECORD = 13;
export const N_BODY = 14;
const BODY_RECORD_STRIDE = 29;
export const CONTINUOUS_STRIDE = 18;

type MovedRows = { eids: Uint32Array; pos: Float32Array; quat: Float32Array; vel: Float32Array };

/** Null-lane identity records the region holds past `bodyCap` — one per thread, since the wide
 * gather/scatter writes the running worker's record (bodies.rs `IDENT_RECORDS`). */
export const IDENT_RECORDS = 8;

/** @returns the smallest power-of-two capacity ≥ `need`, at least 16 (amortizes region grows). */
function growCap(need: number): number {
    let cap = 16;
    while (cap < need) cap *= 2;
    return cap;
}

/**
 * Size the persistent body region to hold `bodyCount` bodies (the total-body high-water). Grows the
 * kernel columns only when the count exceeds the current capacity. @returns true if they grew;
 * callers refresh views after growth.
 */
export function reserveBodies(world: World | undefined, bodyCount: number): boolean {
    return kernel(world).reserveBodies(growCap(bodyCount)) !== 0;
}

/**
 * Typed-array views over the resident `state` + `flags` columns, plus the initial-write + record
 * initialization the body lifecycle needs. One per world.
 */
export class BodyStore extends KernelViews {
    readonly worldId: number;
    constructor(ecsState: World | undefined, worldId: number) {
        super(ecsState);
        this.worldId = worldId;
        this.guardViews();
    }

    /** Resident state column (`STATE_STRIDE` f32 per body), re-derived after growth. */
    stateF = new Float32Array(0);
    /** Resident flags column (one u32 per body), the sidecar paired with `state`. */
    flagsU = new Uint32Array(0);
    recordF = new Float32Array(0);
    recordU = new Uint32Array(0);
    memoryF = new Float32Array(0);
    memoryU = new Uint32Array(0);
    /** Resident sim column (`SIM_STRIDE` f32 per body) — the integrate/finalize `BodySim` fields the
     * kernel gathers. Finalize also indexes it raw. */
    simF = new Float32Array(0);
    /** Resident fin column (`FIN_STRIDE` f32 per body) — the pose-finalize geometric fields (center,
     * localCenter, maxExtent, transform.p). */
    finF = new Float32Array(0);
    /** Resident sim2 column (`SIM2_STRIDE` f32 per body) — the `BodySim` fields the kernel never
     * gathers (rotation0, center0, minExtent, maxAngularVelocity, bodyId, flags). */
    sim2F = new Float32Array(0);
    /** The same sim2 bytes viewed as u32, for the integer `bodyId`/`flags` slots. */
    sim2U = new Uint32Array(0);
    /** Retained kernel body-move records: body index, generation, fellAsleep. */
    moveU = new Uint32Array(0);
    continuousF = new Float32Array(0);
    continuousU = new Uint32Array(0);
    #continuousCount = 0;
    syncCount = 0;
    #syncRanges = new Map<number, MovedRows>();
    // The held layout header view the column views are derived from.
    private _layout = new Uint32Array(0);

    /** Re-derive the column views over the current region. No-op before the first `reserveBodies` (the
     * region has zero capacity), and when the buffer, layout offsets and capacity are those the views were
     * derived at, so a steady step mints no typed-array views. */
    protected deriveViews(): void {
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        const cap = k.bodyCap();
        if (cap === 0) return;
        this.refreshContinuous();
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
            this.finF.byteOffset === layout[B_FIN] &&
            this.sim2F.byteOffset === layout[B_SIM2] &&
            this.moveU.byteOffset === layout[B_MOVE] &&
            this.recordU.byteOffset === layout[B_RECORD]
        )
            return;
        this.stateF = new Float32Array(buf, layout[B_STATE], cap * STATE_STRIDE);
        this.flagsU = new Uint32Array(buf, layout[B_FLAGS], cap);
        this.recordF = new Float32Array(buf, layout[B_RECORD], cap * BODY_RECORD_STRIDE);
        this.recordU = new Uint32Array(buf, layout[B_RECORD], cap * BODY_RECORD_STRIDE);
        this.simF = new Float32Array(buf, layout[B_SIM], cap * SIM_STRIDE);
        this.finF = new Float32Array(buf, layout[B_FIN], cap * FIN_STRIDE);
        this.sim2F = new Float32Array(buf, layout[B_SIM2], cap * SIM2_STRIDE);
        this.sim2U = new Uint32Array(buf, layout[B_SIM2], cap * SIM2_STRIDE);
        this.moveU = new Uint32Array(buf, layout[B_MOVE], cap * MOVE_STRIDE);
        this.#syncRanges.clear();
    }

    override captureCheckpoint() {
        return { syncCount: this.syncCount };
    }

    override restoreCheckpoint(state: unknown): void {
        this.syncCount = (state as ReturnType<BodyStore["captureCheckpoint"]>).syncCount;
        this.#continuousCount = 0;
        this.#syncRanges.clear();
        this.#setColumns.clear();
        this.#setResults.clear();
    }

    refreshContinuous(count = this.#continuousCount): void {
        this.#continuousCount = count;
        const k = kernel(this.ecsState);
        const buf = k.memory.buffer;
        const ptr = k.continuousPtr();
        const length = count * CONTINUOUS_STRIDE;
        if (
            this.continuousF.buffer === buf &&
            this.continuousF.byteOffset === ptr &&
            this.continuousF.length === length
        )
            return;
        this.continuousF = new Float32Array(buf, ptr, length);
        this.continuousU = new Uint32Array(buf, ptr, length);
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

    readonly #setResults = new Map<number, Uint32Array>();
    moveResult(ptr: number, length: number): Uint32Array {
        const buffer = kernel(this.ecsState).memory.buffer;
        const old = this.#setResults.get(ptr);
        if (old?.buffer === buffer && old.length === length) return old;
        const result = new Uint32Array(buffer, ptr, length);
        this.#setResults.set(ptr, result);
        return result;
    }
    forgetSet(set: number): void {
        this.#setColumns.delete(set);
    }

    readonly #setColumns = new Map<
        number,
        {
            layout: Uint32Array;
            simF: Float32Array;
            finF: Float32Array;
            sim2F: Float32Array;
            sim2U: Uint32Array;
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
            old.finF.byteOffset === layout[2] &&
            old.sim2F.byteOffset === layout[5] &&
            old.simF.length >= count * SIM_STRIDE
        )
            return old;
        const columns = {
            layout,
            simF: new Float32Array(k.memory.buffer, layout[1], count * SIM_STRIDE),
            finF: new Float32Array(k.memory.buffer, layout[2], count * FIN_STRIDE),
            sim2F: new Float32Array(k.memory.buffer, layout[5], count * SIM2_STRIDE),
            sim2U: new Uint32Array(k.memory.buffer, layout[5], count * SIM2_STRIDE),
        };
        this.#setColumns.set(set, columns);
        return columns;
    }

    /** Marshal a plain `BodyState` into the resident column at record `i` — the object→view write on a
     * body entering the awake set (create / wake / transfer), not a per-step cost. */
    writeState(i: number, s: BodyState): void {
        const f = this.stateF;
        const o = i * STATE_STRIDE;
        f[o] = s.linearVelocity.x;
        f[o + 1] = s.linearVelocity.y;
        f[o + 2] = s.linearVelocity.z;
        f[o + 3] = s.angularVelocity.x;
        f[o + 4] = s.angularVelocity.y;
        f[o + 5] = s.angularVelocity.z;
        f[o + 6] = s.deltaPosition.x;
        f[o + 7] = s.deltaPosition.y;
        f[o + 8] = s.deltaPosition.z;
        f[o + 9] = s.deltaRotation.v.x;
        f[o + 10] = s.deltaRotation.v.y;
        f[o + 11] = s.deltaRotation.v.z;
        f[o + 12] = s.deltaRotation.s;
        this.flagsU[i] = s.flags;
    }

    /** Initialize a sim from a body definition, or copy an existing column row in the kernel. */
    writeSim(i: number, s: BodySim | number): void {
        const destination = i;
        const target = this.simColumns(bodySimSet(i));
        i = bodySimIndex(i);
        if (typeof s === "number") {
            kernel(this.ecsState).solverSetCopyBody(
                bodySimSet(s),
                bodySimIndex(s),
                bodySimSet(destination),
                i,
            );
            return;
        }
        const sf = target.simF;
        const so = i * SIM_STRIDE;
        sf[so] = s.invMass;
        sf[so + 1] = s.gravityScale;
        sf[so + 2] = s.linearDamping;
        sf[so + 3] = s.angularDamping;
        const force = s.force;
        sf[so + 4] = force.x;
        sf[so + 5] = force.y;
        sf[so + 6] = force.z;
        const torque = s.torque;
        sf[so + 7] = torque.x;
        sf[so + 8] = torque.y;
        sf[so + 9] = torque.z;
        const invLocal = s.invInertiaLocal;
        sf[so + 10] = invLocal.cx.x;
        sf[so + 11] = invLocal.cx.y;
        sf[so + 12] = invLocal.cx.z;
        sf[so + 13] = invLocal.cy.x;
        sf[so + 14] = invLocal.cy.y;
        sf[so + 15] = invLocal.cy.z;
        sf[so + 16] = invLocal.cz.x;
        sf[so + 17] = invLocal.cz.y;
        sf[so + 18] = invLocal.cz.z;
        const invWorld = s.invInertiaWorld;
        sf[so + 19] = invWorld.cx.x;
        sf[so + 20] = invWorld.cx.y;
        sf[so + 21] = invWorld.cx.z;
        sf[so + 22] = invWorld.cy.x;
        sf[so + 23] = invWorld.cy.y;
        sf[so + 24] = invWorld.cy.z;
        sf[so + 25] = invWorld.cz.x;
        sf[so + 26] = invWorld.cz.y;
        sf[so + 27] = invWorld.cz.z;
        const q = s.transform.q;
        sf[so + 28] = q.v.x;
        sf[so + 29] = q.v.y;
        sf[so + 30] = q.v.z;
        sf[so + 31] = q.s;

        const ff = target.finF;
        const fo = i * FIN_STRIDE;
        const center = s.center;
        ff[fo] = center.x;
        ff[fo + 1] = center.y;
        ff[fo + 2] = center.z;
        const localCenter = s.localCenter;
        ff[fo + 3] = localCenter.x;
        ff[fo + 4] = localCenter.y;
        ff[fo + 5] = localCenter.z;
        const maxExtent = s.maxExtent;
        ff[fo + 6] = maxExtent.x;
        ff[fo + 7] = maxExtent.y;
        ff[fo + 8] = maxExtent.z;
        const p = s.transform.p;
        ff[fo + 9] = p.x;
        ff[fo + 10] = p.y;
        ff[fo + 11] = p.z;

        const s2f = target.sim2F;
        const s2o = i * SIM2_STRIDE;
        const rotation0 = s.rotation0;
        s2f[s2o + S2_ROTATION0] = rotation0.v.x;
        s2f[s2o + S2_ROTATION0 + 1] = rotation0.v.y;
        s2f[s2o + S2_ROTATION0 + 2] = rotation0.v.z;
        s2f[s2o + S2_ROTATION0 + 3] = rotation0.s;
        const center0 = s.center0;
        s2f[s2o + S2_CENTER0] = center0.x;
        s2f[s2o + S2_CENTER0 + 1] = center0.y;
        s2f[s2o + S2_CENTER0 + 2] = center0.z;
        s2f[s2o + S2_MIN_EXTENT] = s.minExtent;
        s2f[s2o + S2_MAX_ANGULAR_VELOCITY] = s.maxAngularVelocity;
        target.sim2U[s2o + S2_BODY_ID] = s.bodyId;
        target.sim2U[s2o + S2_FLAGS] = s.flags;
    }

    /** Write the head of the body's shape list into record `i`'s sim2 lane — the entry point the
     * in-kernel finalize refit walks the shape column from (shapes.rs). `NULL_INDEX` (-1) wraps to the
     * kernel's `NULL_SHAPE` sentinel through the u32 view. */
    writeHeadShape(i: number, headShapeId: number): void {
        this.sim2U[i * SIM2_STRIDE + S2_HEAD_SHAPE] = headShapeId;
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
            const ff = store.memoryF;
            const fo = simOffset(world, ref, 2);
            ff[fo] = v.x;
            ff[fo + 1] = v.y;
            ff[fo + 2] = v.z;
            return;
        }
        case "center0": {
            const v = value as Vec3;
            const s2 = store.memoryF;
            const o = simOffset(world, ref, 5) + S2_CENTER0;
            s2[o] = v.x;
            s2[o + 1] = v.y;
            s2[o + 2] = v.z;
            return;
        }
        case "localCenter": {
            const v = value as Vec3;
            const ff = store.memoryF;
            const fo = simOffset(world, ref, 2);
            ff[fo + 3] = v.x;
            ff[fo + 4] = v.y;
            ff[fo + 5] = v.z;
            return;
        }
        case "force": {
            const v = value as Vec3;
            const sf = store.memoryF;
            const so = simOffset(world, ref, 1);
            sf[so + 4] = v.x;
            sf[so + 5] = v.y;
            sf[so + 6] = v.z;
            return;
        }
        case "torque": {
            const v = value as Vec3;
            const sf = store.memoryF;
            const so = simOffset(world, ref, 1);
            sf[so + 7] = v.x;
            sf[so + 8] = v.y;
            sf[so + 9] = v.z;
            return;
        }
        case "invMass": {
            const v = value as number;
            store.memoryF[simOffset(world, ref, 1)] = v;
            return;
        }
        case "invInertiaLocal": {
            const m = value as Mat3;
            const f = store.memoryF;
            const o = simOffset(world, ref, 1) + 10;
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
            const o = simOffset(world, ref, 1) + 19;
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
            store.memoryF[simOffset(world, ref, 5) + S2_MIN_EXTENT] = v;
            return;
        }
        case "maxExtent": {
            const v = value as Vec3;
            const ff = store.memoryF;
            const fo = simOffset(world, ref, 2);
            ff[fo + 6] = v.x;
            ff[fo + 7] = v.y;
            ff[fo + 8] = v.z;
            return;
        }
        case "maxAngularVelocity": {
            const v = value as number;
            store.memoryF[simOffset(world, ref, 5) + S2_MAX_ANGULAR_VELOCITY] = v;
            return;
        }
        case "linearDamping": {
            const v = value as number;
            store.memoryF[simOffset(world, ref, 1) + 2] = v;
            return;
        }
        case "angularDamping": {
            const v = value as number;
            store.memoryF[simOffset(world, ref, 1) + 3] = v;
            return;
        }
        case "gravityScale": {
            const v = value as number;
            store.memoryF[simOffset(world, ref, 1) + 1] = v;
            return;
        }
        case "bodyId": {
            const v = value as number;
            store.memoryU[simOffset(world, ref, 5) + S2_BODY_ID] = v;
            return;
        }
        case "flags": {
            const v = value as number;
            store.memoryU[simOffset(world, ref, 5) + S2_FLAGS] = v;
            return;
        }
        default:
            throw new Error("unknown body sim field");
    }
}

export function setStateField<K extends keyof BodyState>(
    world: WorldState,
    ref: number,
    field: K,
    value: BodyState[K],
): void {
    const store = world.bodyStore;
    const i = ref;
    switch (field) {
        case "linearVelocity": {
            const v = value as Vec3;
            const f = store.stateF;
            const o = i * STATE_STRIDE;
            f[o] = v.x;
            f[o + 1] = v.y;
            f[o + 2] = v.z;
            return;
        }
        case "angularVelocity": {
            const v = value as Vec3;
            const f = store.stateF;
            const o = i * STATE_STRIDE;
            f[o + 3] = v.x;
            f[o + 4] = v.y;
            f[o + 5] = v.z;
            return;
        }
        case "deltaPosition": {
            const v = value as Vec3;
            const f = store.stateF;
            const o = i * STATE_STRIDE;
            f[o + 6] = v.x;
            f[o + 7] = v.y;
            f[o + 8] = v.z;
            return;
        }
        case "deltaRotation": {
            const q = value as Quat;
            const f = store.stateF;
            const o = i * STATE_STRIDE;
            f[o + 9] = q.v.x;
            f[o + 10] = q.v.y;
            f[o + 11] = q.v.z;
            f[o + 12] = q.s;
            return;
        }
        case "flags": {
            const v = value as number;
            store.flagsU[i] = v;
            return;
        }
        default:
            throw new Error("unknown body state field");
    }
}

export function readSimTransform(
    world: WorldState,
    ref: number,
    out: WorldTransform,
): WorldTransform {
    const sf = world.bodyStore.memoryF,
        ff = world.bodyStore.memoryF;
    const so = simOffset(world, ref, 1),
        fo = simOffset(world, ref, 2);
    out.p.x = ff[fo + 9];
    out.p.y = ff[fo + 10];
    out.p.z = ff[fo + 11];
    out.q.v.x = sf[so + 28];
    out.q.v.y = sf[so + 29];
    out.q.v.z = sf[so + 30];
    out.q.s = sf[so + 31];
    return out;
}

export function readSimCenter(world: WorldState, ref: number, out: Vec3): Vec3 {
    const f = world.bodyStore.memoryF;
    const o = simOffset(world, ref, 2) + 0;
    out.x = f[o];
    out.y = f[o + 1];
    out.z = f[o + 2];
    return out;
}

export function readSimLocalCenter(world: WorldState, ref: number, out: Vec3): Vec3 {
    const f = world.bodyStore.memoryF;
    const o = simOffset(world, ref, 2) + 3;
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

export function writeSimTransform(world: WorldState, ref: number, t: WorldTransform): void {
    const sf = world.bodyStore.memoryF;
    const ff = world.bodyStore.memoryF;
    const so = simOffset(world, ref, 1);
    const fo = simOffset(world, ref, 2);
    ff[fo + 9] = t.p.x;
    ff[fo + 10] = t.p.y;
    ff[fo + 11] = t.p.z;
    sf[so + 28] = t.q.v.x;
    sf[so + 29] = t.q.v.y;
    sf[so + 30] = t.q.v.z;
    sf[so + 31] = t.q.s;
}

export function writeSimRotation0(world: WorldState, ref: number, q: Quat): void {
    const s2 = world.bodyStore.memoryF;
    const o = simOffset(world, ref, 5) + S2_ROTATION0;
    s2[o] = q.v.x;
    s2[o + 1] = q.v.y;
    s2[o + 2] = q.v.z;
    s2[o + 3] = q.s;
}
/** Append an awake body at its local index and initialize its columns. */
export function residentPush(
    world: WorldState,
    initState: BodyState,
    initSim: BodySim | number,
    headShapeId: number,
): void {
    const store = world.bodyStore;
    const i = kernel(world.ecsState).solverSetBodyAppend(SetType.Awake);
    store.writeState(i, initState);
    store.writeSim(i, initSim);
    store.writeHeadShape(i, headShapeId);
}

/**
 * Re-write an awake body's headShapeId lane after its shape list changed (shape create / destroy). A
 * body outside the awake set has no resident record — its lane is written when it enters one, from the
 * body's then-current `headShapeId`.
 */
export function syncHeadShape(world: WorldState, body: number): void {
    if (bodyField(world, body, BodyField.setIndex) !== SetType.Awake) return;
    world.bodyStore.refreshViews();
    world.bodyStore.writeHeadShape(
        bodyField(world, body, BodyField.localIndex),
        bodyField(world, body, BodyField.headShapeId),
    );
}

/** Swap-remove an awake row; return the body moved into the hole, or NULL_INDEX. */
export function residentRemove(world: WorldState, index: number): number {
    return kernel(world.ecsState).solverSetRemoveBody(SetType.Awake, index) | 0;
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
    return readSimMatrix(world, ref, 10, out);
}
export function readSimInvInertiaWorld(world: WorldState, ref: number, out: Mat3): Mat3 {
    return readSimMatrix(world, ref, 19, out);
}
function readSimMatrix(world: WorldState, ref: number, offset: number, out: Mat3): Mat3 {
    const f = world.bodyStore.memoryF,
        o = simOffset(world, ref, 1) + offset;
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

export function simInvMass(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryF[simOffset(world, ref, 1)];
}

export function simMinExtent(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryF[simOffset(world, ref, 5) + S2_MIN_EXTENT];
}

export function simMaxAngularVelocity(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryF[simOffset(world, ref, 5) + S2_MAX_ANGULAR_VELOCITY];
}

export function simLinearDamping(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryF[simOffset(world, ref, 1) + 2];
}

export function simAngularDamping(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryF[simOffset(world, ref, 1) + 3];
}

export function simGravityScale(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryF[simOffset(world, ref, 1) + 1];
}

export function simBodyId(world: WorldState, ref: number): number {
    if (ref < 0) return -ref - 1;
    const k = kernel(world.ecsState);
    k.bodySetActiveWorld(world.worldId);
    return k.solverSetBodyId(bodySimSet(ref), bodySimIndex(ref));
}

export function simFlags(world: WorldState, ref: number): number {
    const store = world.bodyStore;
    return store.memoryU[simOffset(world, ref, 5) + S2_FLAGS];
}

export function addSimForce(world: WorldState, ref: number, v: Vec3): void {
    const f = world.bodyStore.memoryF;
    const o = simOffset(world, ref, 1) + 4;
    f[o] = Math.fround(f[o] + v.x);
    f[o + 1] = Math.fround(f[o + 1] + v.y);
    f[o + 2] = Math.fround(f[o + 2] + v.z);
}
export function addSimTorque(world: WorldState, ref: number, v: Vec3): void {
    const f = world.bodyStore.memoryF;
    const o = simOffset(world, ref, 1) + 7;
    f[o] = Math.fround(f[o] + v.x);
    f[o + 1] = Math.fround(f[o + 1] + v.y);
    f[o + 2] = Math.fround(f[o + 2] + v.z);
}
export function addSimForceTorque(world: WorldState, ref: number, force: Vec3, point: Vec3): void {
    const columns = world.bodyStore;
    const fo = simOffset(world, ref, 2);
    const x = Math.fround(point.x - columns.memoryF[fo]);
    const y = Math.fround(point.y - columns.memoryF[fo + 1]);
    const z = Math.fround(point.z - columns.memoryF[fo + 2]);
    const f = columns.memoryF;
    const o = simOffset(world, ref, 1) + 7;
    f[o] = Math.fround(f[o] + Math.fround(Math.fround(y * force.z) - Math.fround(z * force.y)));
    f[o + 1] = Math.fround(
        f[o + 1] + Math.fround(Math.fround(z * force.x) - Math.fround(x * force.z)),
    );
    f[o + 2] = Math.fround(
        f[o + 2] + Math.fround(Math.fround(x * force.y) - Math.fround(y * force.x)),
    );
}

export function readSimMaxExtent(world: WorldState, ref: number, out: Vec3): Vec3 {
    const f = world.bodyStore.memoryF;
    const o = simOffset(world, ref, 2) + 6;
    out.x = f[o];
    out.y = f[o + 1];
    out.z = f[o + 2];
    return out;
}

function simOffset(world: WorldState, ref: number, column: number): number {
    const k = kernel(world.ecsState);
    return (
        (ref < 0
            ? k.bodyColumnPtr(world.worldId, -ref - 1, column)
            : k.simColumnPtr(world.worldId, bodySimSet(ref), bodySimIndex(ref), column)) >>> 2
    );
}

// The persistent body region (kernel/src/bodies.rs) — the awake body columns held resident across
// steps, first in the kernel's linear memory: velocity/delta `state` + `flags` (4a.2) and the
// integrate/finalize `sim`/`fin`/`sim2` fields (4a.3). The solver runs directly over these columns
// (the kernel phases alias `LAYOUT[STATE]`/`LAYOUT[SIM]`/etc here), so a step no longer marshals the
// body in and reads it back out — the column is the single source of truth. This module owns the
// grow-only sizing (the region tracks the total-body high-water so it never shrinks with the churny
// awake set), the offset-backed `BodyState`/`BodySim` views, and the record migration that mirrors the
// JS-array swap-remove when an awake body leaves the set.
//
// A region grow relocates the manifold + geometry regions above it (kernel-side, in place) and, like
// any `memory.grow`, detaches every typed-array view — so the store re-derives its arrays after every
// grow-capable kernel call (the discipline the callers follow). Detach is not the only signal: a
// body-grow relocation without a page grow would leave views attached but pointing at stale bytes, so
// refresh keys off the call, never off detachment.
//
// The region is a singleton (one kernel, one linear memory for every world); the kernel owns the
// authoritative capacity (`bodyCap`), so no TS mirror is needed. Resident state makes interleaved
// stepping of two live worlds corrupt the shared region, so a single-live-world guard (below) throws
// when a world steps after another has taken the region over; sequential worlds keep working.

import { NULL_INDEX } from "./array";
import type { Body, BodySim, BodyState } from "./body";
import {
    FIN_STRIDE,
    S2_BODY_ID,
    S2_CENTER0,
    S2_FLAGS,
    S2_HEAD_SHAPE,
    S2_MAX_ANGULAR_VELOCITY,
    S2_MIN_EXTENT,
    S2_ROTATION0,
    SIM_STRIDE,
    SIM2_LIVE,
    SIM2_STRIDE,
    STATE_LIVE,
    STATE_STRIDE,
    writeMat3,
} from "./columns";
import { SetType } from "./core";
import { kernel, sharedBytes } from "./kernel";
import type { Mat3, Quat, Vec3, WorldTransform } from "./math";
import type { WorldState } from "./world";

// BODY_LAYOUT header indices (bodies.rs), in memory order: state, sim, fin, finOut, flags, sim2.
export const B_STATE = 0;
const B_SIM = 1;
const B_FIN = 2;
export const B_FLAGS = 4;
const B_SIM2 = 5;
export const N_BODY = 6;

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
 * kernel region — relocating the manifold + geometry regions above it in place — only when the count
 * exceeds the current capacity. @returns true if the region grew (the caller must refresh any views
 * over the relocated regions, including the body store's).
 */
export function reserveBodies(bodyCount: number): boolean {
    return kernel().reserveBodies(growCap(bodyCount)) !== 0;
}

/**
 * Typed-array views over the resident `state` + `flags` columns, plus the initial-write + record
 * migration the awake-set lifecycle needs. One per world; the awake set's `bodyStates` array holds
 * `ResidentBodyState` views over this store.
 */
export class BodyStore {
    /** Resident state column (`STATE_STRIDE` f32 per body). Re-derived after every grow (`memory.grow`
     * detaches it, and a body-region relocation shifts the bytes even without a page grow). */
    stateF = new Float32Array(0);
    /** Resident flags column (one u32 per body), the sidecar paired with `state`. */
    flagsU = new Uint32Array(0);
    /** Resident sim column (`SIM_STRIDE` f32 per body) — the integrate/finalize `BodySim` fields the
     * kernel gathers. Backs the awake `ResidentBodySim` view; finalize also indexes it raw. */
    simF = new Float32Array(0);
    /** Resident fin column (`FIN_STRIDE` f32 per body) — the pose-finalize geometric fields (center,
     * localCenter, maxExtent, transform.p). */
    finF = new Float32Array(0);
    /** Resident sim2 column (`SIM2_STRIDE` f32 per body) — the `BodySim` fields the kernel never
     * gathers (rotation0, center0, minExtent, maxAngularVelocity, bodyId, flags). */
    sim2F = new Float32Array(0);
    /** The same sim2 bytes viewed as u32, for the integer `bodyId`/`flags` slots. */
    sim2U = new Uint32Array(0);
    /** Memory size the views were derived at, on the shared (multithreaded) path; 0 single-threaded,
     * where detachment is the signal instead. See `stale`. */
    bytes = 0;

    /** Whether a `memory.grow` has happened since the views were derived — the guard for the reads a
     * mid-loop grow (the narrowphase's manifold `alloc`) can strand. Single-threaded that grow detaches
     * every view (length 0). A shared memory never detaches, so the shared path compares the memory's
     * size against the size the views were derived at (`sharedBytes`, kernel.ts). */
    get stale(): boolean {
        return this.simF.length === 0 || (this.bytes !== 0 && this.bytes !== sharedBytes());
    }

    /** Re-derive the column views over the current region. Cheap — a handful of typed-array
     * constructions, no copy. No-op before the first `reserveBodies` (the region has zero capacity). */
    refreshViews(): void {
        const k = kernel();
        const cap = k.bodyCap();
        if (cap === 0) return;
        const buf = k.memory.buffer;
        this.bytes = sharedBytes();
        const layout = new Uint32Array(buf, k.bodyLayoutPtr(), N_BODY);
        this.stateF = new Float32Array(buf, layout[B_STATE], cap * STATE_STRIDE);
        this.flagsU = new Uint32Array(buf, layout[B_FLAGS], cap);
        this.simF = new Float32Array(buf, layout[B_SIM], cap * SIM_STRIDE);
        this.finF = new Float32Array(buf, layout[B_FIN], cap * FIN_STRIDE);
        this.sim2F = new Float32Array(buf, layout[B_SIM2], cap * SIM2_STRIDE);
        this.sim2U = new Uint32Array(buf, layout[B_SIM2], cap * SIM2_STRIDE);
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

    /** Move a resident state record (the 13 live state fields + flags) from `from` to `to`, mirroring
     * the JS-array swap-remove compaction when an awake body leaves the set (destroy / sleep / transfer). */
    migrate(from: number, to: number): void {
        this.stateF.copyWithin(
            to * STATE_STRIDE,
            from * STATE_STRIDE,
            from * STATE_STRIDE + STATE_LIVE,
        );
        this.flagsU[to] = this.flagsU[from];
    }

    /** Marshal a `BodySim` into the resident sim/fin/sim2 columns at record `i` — the object→view write
     * on a body entering the awake set (create / wake / transfer), not a per-step cost. `s` may be a
     * plain `BodySim` (from a sleeping/static set) or already a view (reads its getters either way).
     * Field order mirrors read_sim / read_fin (kernel/src/body.rs) + the sim2 offsets (columns.ts). */
    writeSim(i: number, s: BodySim): void {
        const sf = this.simF;
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
        writeMat3(sf, so + 10, s.invInertiaLocal);
        writeMat3(sf, so + 19, s.invInertiaWorld);
        const q = s.transform.q;
        sf[so + 28] = q.v.x;
        sf[so + 29] = q.v.y;
        sf[so + 30] = q.v.z;
        sf[so + 31] = q.s;

        const ff = this.finF;
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

        const s2f = this.sim2F;
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
        this.sim2U[s2o + S2_BODY_ID] = s.bodyId;
        this.sim2U[s2o + S2_FLAGS] = s.flags;
    }

    /** Write the head of the body's shape list into record `i`'s sim2 lane — the entry point the
     * in-kernel finalize refit walks the shape column from (shapes.rs). `NULL_INDEX` (-1) wraps to the
     * kernel's `NULL_SHAPE` sentinel through the u32 view. */
    writeHeadShape(i: number, headShapeId: number): void {
        this.sim2U[i * SIM2_STRIDE + S2_HEAD_SHAPE] = headShapeId;
    }

    /** Move a resident sim record (sim + fin + sim2) from `from` to `to`, mirroring the `bodySims`
     * swap-remove alongside {@link migrate}'s state move when an awake body leaves the set. The sim2
     * copy spans `SIM2_LIVE`, which includes the headShapeId lane — it rides the migration. */
    migrateSim(from: number, to: number): void {
        this.simF.copyWithin(to * SIM_STRIDE, from * SIM_STRIDE, from * SIM_STRIDE + SIM_STRIDE);
        this.finF.copyWithin(to * FIN_STRIDE, from * FIN_STRIDE, from * FIN_STRIDE + FIN_STRIDE);
        this.sim2F.copyWithin(to * SIM2_STRIDE, from * SIM2_STRIDE, from * SIM2_STRIDE + SIM2_LIVE);
    }
}

/** Create an empty body store for a new world. Its views are derived on the first refresh. */
export function createBodyStore(): BodyStore {
    return new BodyStore();
}

/**
 * A `BodyState` (b3BodyState) backed by a record in the resident `state`/`flags` columns rather than a
 * plain object. Getters return fresh `Vec3`/`Quat` (matching the plain-object semantics the joint
 * solver + API expect); setters write into the column. Reads go through `store.stateF`/`flagsU` on
 * every access so a grow's re-derivation is transparent. The view is **index-fixed**: it reads record
 * `i`, so after an awake-set swap-remove migrates a body into `i` this view reads that new body — the
 * awake set fetches views fresh by `body.localIndex`, never caching one across a migration. Offsets
 * mirror `read_state`/`write_state` in kernel/src/body.rs.
 */
class ResidentBodyState implements BodyState {
    private readonly _o: number;
    constructor(
        private readonly _s: BodyStore,
        private readonly _i: number,
    ) {
        this._o = _i * STATE_STRIDE;
    }
    get linearVelocity(): Vec3 {
        const f = this._s.stateF;
        const o = this._o;
        return { x: f[o], y: f[o + 1], z: f[o + 2] };
    }
    set linearVelocity(v: Vec3) {
        const f = this._s.stateF;
        const o = this._o;
        f[o] = v.x;
        f[o + 1] = v.y;
        f[o + 2] = v.z;
    }
    get angularVelocity(): Vec3 {
        const f = this._s.stateF;
        const o = this._o;
        return { x: f[o + 3], y: f[o + 4], z: f[o + 5] };
    }
    set angularVelocity(v: Vec3) {
        const f = this._s.stateF;
        const o = this._o;
        f[o + 3] = v.x;
        f[o + 4] = v.y;
        f[o + 5] = v.z;
    }
    get deltaPosition(): Vec3 {
        const f = this._s.stateF;
        const o = this._o;
        return { x: f[o + 6], y: f[o + 7], z: f[o + 8] };
    }
    set deltaPosition(v: Vec3) {
        const f = this._s.stateF;
        const o = this._o;
        f[o + 6] = v.x;
        f[o + 7] = v.y;
        f[o + 8] = v.z;
    }
    get deltaRotation(): Quat {
        const f = this._s.stateF;
        const o = this._o;
        return { v: { x: f[o + 9], y: f[o + 10], z: f[o + 11] }, s: f[o + 12] };
    }
    set deltaRotation(q: Quat) {
        const f = this._s.stateF;
        const o = this._o;
        f[o + 9] = q.v.x;
        f[o + 10] = q.v.y;
        f[o + 11] = q.v.z;
        f[o + 12] = q.s;
    }
    get flags(): number {
        return this._s.flagsU[this._i];
    }
    set flags(v: number) {
        this._s.flagsU[this._i] = v;
    }
}

/**
 * A `BodySim` (b3BodySim) backed by record `i` of the resident sim/fin/sim2 columns rather than a plain
 * object (4a.3), so no per-step marshal runs. Vector/quaternion/matrix getters return fresh objects
 * (plain-object semantics — in-place sub-field writes on the caller side become whole-field setter
 * assignments); scalar getters return the raw slot (no allocation). Reads go through `store.simF` etc
 * on every access so a grow's re-derivation is transparent. Index-fixed like {@link ResidentBodyState}:
 * fetch fresh by `body.localIndex`, never cache across a migration. Offsets mirror read_sim / read_fin
 * (kernel/src/body.rs) + the sim2 layout (columns.ts). `finalize` also reads/writes these columns raw,
 * bypassing the view, to stay zero-alloc in its per-body hot loop.
 */
class ResidentBodySim implements BodySim {
    private readonly _so: number;
    private readonly _fo: number;
    private readonly _s2o: number;
    constructor(
        private readonly _s: BodyStore,
        i: number,
    ) {
        this._so = i * SIM_STRIDE;
        this._fo = i * FIN_STRIDE;
        this._s2o = i * SIM2_STRIDE;
    }
    get transform(): WorldTransform {
        const sf = this._s.simF;
        const ff = this._s.finF;
        const so = this._so;
        const fo = this._fo;
        return {
            p: { x: ff[fo + 9], y: ff[fo + 10], z: ff[fo + 11] },
            q: { v: { x: sf[so + 28], y: sf[so + 29], z: sf[so + 30] }, s: sf[so + 31] },
        };
    }
    set transform(t: WorldTransform) {
        const sf = this._s.simF;
        const ff = this._s.finF;
        const so = this._so;
        const fo = this._fo;
        ff[fo + 9] = t.p.x;
        ff[fo + 10] = t.p.y;
        ff[fo + 11] = t.p.z;
        sf[so + 28] = t.q.v.x;
        sf[so + 29] = t.q.v.y;
        sf[so + 30] = t.q.v.z;
        sf[so + 31] = t.q.s;
    }
    get center(): Vec3 {
        const ff = this._s.finF;
        const fo = this._fo;
        return { x: ff[fo], y: ff[fo + 1], z: ff[fo + 2] };
    }
    set center(v: Vec3) {
        const ff = this._s.finF;
        const fo = this._fo;
        ff[fo] = v.x;
        ff[fo + 1] = v.y;
        ff[fo + 2] = v.z;
    }
    get rotation0(): Quat {
        const s2 = this._s.sim2F;
        const o = this._s2o + S2_ROTATION0;
        return { v: { x: s2[o], y: s2[o + 1], z: s2[o + 2] }, s: s2[o + 3] };
    }
    set rotation0(q: Quat) {
        const s2 = this._s.sim2F;
        const o = this._s2o + S2_ROTATION0;
        s2[o] = q.v.x;
        s2[o + 1] = q.v.y;
        s2[o + 2] = q.v.z;
        s2[o + 3] = q.s;
    }
    get center0(): Vec3 {
        const s2 = this._s.sim2F;
        const o = this._s2o + S2_CENTER0;
        return { x: s2[o], y: s2[o + 1], z: s2[o + 2] };
    }
    set center0(v: Vec3) {
        const s2 = this._s.sim2F;
        const o = this._s2o + S2_CENTER0;
        s2[o] = v.x;
        s2[o + 1] = v.y;
        s2[o + 2] = v.z;
    }
    get localCenter(): Vec3 {
        const ff = this._s.finF;
        const fo = this._fo;
        return { x: ff[fo + 3], y: ff[fo + 4], z: ff[fo + 5] };
    }
    set localCenter(v: Vec3) {
        const ff = this._s.finF;
        const fo = this._fo;
        ff[fo + 3] = v.x;
        ff[fo + 4] = v.y;
        ff[fo + 5] = v.z;
    }
    get force(): Vec3 {
        const sf = this._s.simF;
        const so = this._so;
        return { x: sf[so + 4], y: sf[so + 5], z: sf[so + 6] };
    }
    set force(v: Vec3) {
        const sf = this._s.simF;
        const so = this._so;
        sf[so + 4] = v.x;
        sf[so + 5] = v.y;
        sf[so + 6] = v.z;
    }
    get torque(): Vec3 {
        const sf = this._s.simF;
        const so = this._so;
        return { x: sf[so + 7], y: sf[so + 8], z: sf[so + 9] };
    }
    set torque(v: Vec3) {
        const sf = this._s.simF;
        const so = this._so;
        sf[so + 7] = v.x;
        sf[so + 8] = v.y;
        sf[so + 9] = v.z;
    }
    get invMass(): number {
        return this._s.simF[this._so];
    }
    set invMass(v: number) {
        this._s.simF[this._so] = v;
    }
    get invInertiaLocal(): Mat3 {
        return readMat3(this._s.simF, this._so + 10);
    }
    set invInertiaLocal(m: Mat3) {
        writeMat3(this._s.simF, this._so + 10, m);
    }
    get invInertiaWorld(): Mat3 {
        return readMat3(this._s.simF, this._so + 19);
    }
    set invInertiaWorld(m: Mat3) {
        writeMat3(this._s.simF, this._so + 19, m);
    }
    get minExtent(): number {
        return this._s.sim2F[this._s2o + S2_MIN_EXTENT];
    }
    set minExtent(v: number) {
        this._s.sim2F[this._s2o + S2_MIN_EXTENT] = v;
    }
    get maxExtent(): Vec3 {
        const ff = this._s.finF;
        const fo = this._fo;
        return { x: ff[fo + 6], y: ff[fo + 7], z: ff[fo + 8] };
    }
    set maxExtent(v: Vec3) {
        const ff = this._s.finF;
        const fo = this._fo;
        ff[fo + 6] = v.x;
        ff[fo + 7] = v.y;
        ff[fo + 8] = v.z;
    }
    get maxAngularVelocity(): number {
        return this._s.sim2F[this._s2o + S2_MAX_ANGULAR_VELOCITY];
    }
    set maxAngularVelocity(v: number) {
        this._s.sim2F[this._s2o + S2_MAX_ANGULAR_VELOCITY] = v;
    }
    get linearDamping(): number {
        return this._s.simF[this._so + 2];
    }
    set linearDamping(v: number) {
        this._s.simF[this._so + 2] = v;
    }
    get angularDamping(): number {
        return this._s.simF[this._so + 3];
    }
    set angularDamping(v: number) {
        this._s.simF[this._so + 3] = v;
    }
    get gravityScale(): number {
        return this._s.simF[this._so + 1];
    }
    set gravityScale(v: number) {
        this._s.simF[this._so + 1] = v;
    }
    get bodyId(): number {
        return this._s.sim2U[this._s2o + S2_BODY_ID];
    }
    set bodyId(v: number) {
        this._s.sim2U[this._s2o + S2_BODY_ID] = v;
    }
    get flags(): number {
        return this._s.sim2U[this._s2o + S2_FLAGS];
    }
    set flags(v: number) {
        this._s.sim2U[this._s2o + S2_FLAGS] = v;
    }
}

/** Read a Mat3 out of `col` at `o` in the kernel's row order (cx, cy, cz), matching read_sim (body.rs). */
function readMat3(col: Float32Array, o: number): Mat3 {
    return {
        cx: { x: col[o], y: col[o + 1], z: col[o + 2] },
        cy: { x: col[o + 3], y: col[o + 4], z: col[o + 5] },
        cz: { x: col[o + 6], y: col[o + 7], z: col[o + 8] },
    };
}

/**
 * Push a body entering the awake set: marshal its initial `state`/`sim` into the resident columns at the
 * new record and append the `ResidentBodyState`/`ResidentBodySim` views to `bodyStates`/`bodySims`. The
 * new index equals `bodyStates.length` (the awake set keeps `bodyStates`, `bodySims` lockstep by localIndex).
 * `headShapeId` is the body's (not the sim's) — the marshal-in write of the shape-list lane; the shape
 * lifecycle patches it after (`syncHeadShape`).
 */
export function residentPush(
    store: BodyStore,
    bodyStates: BodyState[],
    bodySims: BodySim[],
    initState: BodyState,
    initSim: BodySim,
    headShapeId: number,
): void {
    const i = bodyStates.length;
    store.writeState(i, initState);
    store.writeSim(i, initSim);
    store.writeHeadShape(i, headShapeId);
    bodyStates.push(new ResidentBodyState(store, i));
    bodySims.push(new ResidentBodySim(store, i));
}

/**
 * Re-write an awake body's headShapeId lane after its shape list changed (shape create / destroy). A
 * body outside the awake set has no resident record — its lane is written when it enters one, from the
 * body's then-current `headShapeId`.
 */
export function syncHeadShape(world: WorldState, body: Body): void {
    if (body.setIndex !== SetType.Awake) return;
    world.bodyStore.refreshViews();
    world.bodyStore.writeHeadShape(body.localIndex, body.headShapeId);
}

/**
 * Remove the awake body at `index` from the resident columns: migrate the last record (state + sim) into
 * the freed slot and drop the tail views. Views are index-fixed, so the tail is dropped —
 * `bodyStates[index]`/`bodySims[index]` stay and now read the migrated body. @returns the migrated
 * body's id (so the caller updates its `localIndex`), or `NULL_INDEX` if the removed body was the tail.
 */
export function residentRemove(
    store: BodyStore,
    bodyStates: BodyState[],
    bodySims: BodySim[],
    index: number,
): number {
    const last = bodyStates.length - 1;
    let movedBodyId = NULL_INDEX;
    if (index !== last) {
        store.migrate(last, index);
        store.migrateSim(last, index);
        movedBodyId = bodySims[index].bodyId;
    }
    bodyStates.pop();
    bodySims.pop();
    return movedBodyId;
}

// --- single-live-world guard ----------------------------------------------------------------
// The resident region (and the singleton manifold store) hold one world's live state at a time.
// Interleaving two live worlds' steps corrupts the shared region, so ownership transfers on step:
// stepping world B while A owns the region evicts A; A is then only broken if it steps again, which
// throws. Sequential worlds — a world used then abandoned before the next is stepped, the fixture /
// test / sample shape — never re-step the evicted one, so they keep working.

let owner: object | null = null;
const evicted = new WeakSet<object>();

/** Claim the resident region for `token` (a world) at step entry. Throws if `token` was evicted by a
 * later world taking the region over — its resident body state is gone. */
export function claimResident(token: object): void {
    if (owner === token) return;
    if (evicted.has(token)) {
        throw new Error(
            "tumble: this world's physics state was overwritten by another world; two live worlds " +
                "cannot be stepped interleaved (destroy one before stepping the other)",
        );
    }
    if (owner !== null) evicted.add(owner);
    owner = token;
}

/** Release the resident region on world destroy, so a later world can claim it without eviction. */
export function releaseResident(token: object): void {
    if (owner === token) owner = null;
    evicted.delete(token);
}

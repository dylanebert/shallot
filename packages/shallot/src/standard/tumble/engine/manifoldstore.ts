// The persistent contact-manifold store — the warm-start state that survives across steps, held
// column-resident in the kernel's linear memory (kernel/src/manifolds.rs) instead of as JS objects on
// each contact. TS owns the allocator + lifecycle because the mesh narrowphase is TS and the convex one
// is the kernel (3c.3), so allocation can't live on one side of the FFI; the kernel just reads/writes
// manifold data at the offsets this store hands out.
//
// Two columns, mirroring box3d's `b3Contact.manifolds` heap array:
//   - directory: one record per contactId (material row + block descriptor), indexed directly.
//   - pool: the variable manifold records. Each contact owns a contiguous run of `manifoldCount`
//     records; size-class free lists (keyed by count) recycle the runs, exactly like box3d's
//     `b3AllocateManifolds`. Points inline in a manifold record (points-per-manifold ≤ 4), so only the
//     manifold count per contact is variable — no separate point pool.
//
// The strides MIRROR kernel/src/manifolds.rs; the wasm layout is the contract.

import type { Manifold, ManifoldPoint } from "./contact";
import { kernel } from "./kernel";
import { f32, type Mat3, mat3, type Quat, type Transform, type Vec3, vec3 } from "./math";

/** f32/u32 slots per directory record (DIR_STRIDE in manifold_abi.rs): the solver's per-step row —
 * friction, restitution, rollingResistance, tangentVelocity(3), flags, manifoldCount, manifoldBase,
 * indexA, indexB, hit (0..11) — plus the convex narrowphase's persistent GJK/SAT cache union (12..21,
 * `DIR_CACHE`) and the in-kernel recycle loop's cached relative pose (22..36, `DIR_CACHED_*`), both
 * folded here to share the directory's contactId key + grow-in-place lifecycle. TS never reads/writes
 * the recycle tail — the kernel recycle pass owns it (4b.3c). */
export const DIR_STRIDE = 37;
/** First slot of the convex cache union within a directory record (kernel `DIR_CACHE`). */
const DIR_CACHE = 12;
// The recycle record (kernel `DIR_CACHED_*`): the cached relative pose the recycle test reads/writes.
// The kernel recycle pass owns it for a kernel contact; the TS path mirrors it for a would-be-kernel
// contact temporarily off the kernel path (readRecyclePose/writeRecyclePose).
const DIR_CACHED_ROT_A = 22; // q4
const DIR_CACHED_ROT_B = 26; // q4
const DIR_CACHED_REL_POSE = 30; // p3 + q4
/** Cache union words (10): the wider SimplexCache (metric, count, indexA[4], indexB[4]) overlaps the
 * narrower SatCache. Zeroed on `freeSlot` so a recycled contactId starts cold (box3d's create-zero). */
const CACHE_WORDS = 10;
/** f32/u32 slots per manifold record (MANIFOLD_STRIDE in manifold_abi.rs) — matches b3Manifold
 * (67 f32): header(11) + 4 inline points(14 each). */
export const MANIFOLD_STRIDE = 67;

// Directory record slots (DIR_STRIDE). The material row (0..6) + body sim indices (9,10) are written
// per step by `writeContactRow` (the narrowphase → solver handoff the kernel gathers through); the
// block descriptor (7,8) by `alloc`; the hit flag (11) by the kernel `store`, read back by TS.
const DIR_FRICTION = 0;
const DIR_RESTITUTION = 1;
const DIR_ROLLING_RESISTANCE = 2;
const DIR_TANGENT_VELOCITY = 3; // 3..5
const DIR_FLAGS = 6;
const DIR_MANIFOLD_COUNT = 7;
const DIR_MANIFOLD_BASE = 8;
const DIR_INDEX_A = 9;
const DIR_INDEX_B = 10;
const DIR_HIT = 11;

// Manifold record header slots (within a MANIFOLD_STRIDE record), then the four inline point records.
const M_NORMAL = 0; // 0..2
const M_FRICTION = 3; // 3..5
const M_TWIST = 6;
const M_ROLLING = 7; // 7..9
const M_POINT_COUNT = 10;
const M_POINTS = 11; // first point record
const POINT_STRIDE = 14;
// Point record slots (relative to the point's start).
const P_ANCHOR_A = 0; // 0..2
const P_ANCHOR_B = 3; // 3..5
const P_SEPARATION = 6;
const P_BASE_SEPARATION = 7;
const P_NORMAL_IMPULSE = 8;
const P_TOTAL_NORMAL_IMPULSE = 9;
const P_NORMAL_VELOCITY = 10;
const P_FEATURE_ID = 11; // u32
const P_TRIANGLE_INDEX = 12; // i32 (NULL_INDEX = -1)
const P_PERSISTED = 13; // u32 (0/1)

// MANIFOLD_LAYOUT header indices (manifolds.rs), in memory order.
const DIR = 0;
const POOL = 1;
const N_MANIFOLD = 2;

/** @returns the smallest power-of-two capacity ≥ `need`, at least 16 (amortizes region grows). */
function growCap(need: number): number {
    let cap = 16;
    while (cap < need) cap *= 2;
    return cap;
}

// Scratch for the raw per-point walks below; never live across calls.
const walkNormal: Vec3 = { x: 0, y: 0, z: 0 };
const walkA: Vec3 = { x: 0, y: 0, z: 0 };
const walkB: Vec3 = { x: 0, y: 0, z: 0 };

/**
 * The persistent manifold store for one world. Holds the allocator bookkeeping (block free lists +
 * bump high-water) and the current wasm-region capacities, and re-derives its column views whenever the
 * region grows (or `memory.grow` elsewhere detaches them).
 */
export class ManifoldStore {
    // Current wasm-region capacities (directory records / pool manifold records).
    private _dirCap = 0;
    private _poolCap = 0;
    // Required capacities, updated as contacts + blocks come and go; the region reserves up to these.
    private _needDir = 0;
    private _poolTop = 0;
    // Size-class free lists: _freeLists[count] = stack of free block bases (element index into the pool).
    private _freeLists = new Map<number, number[]>();
    // Each live contact's manifold block (the allocator's source of truth; the wasm directory is the
    // materialized copy the kernel reads).
    private _blocks = new Map<number, { base: number; count: number }>();
    // Manifold views cached per block base. Views are pure (store, offset) pairs and a base's size
    // class never changes (fresh blocks bump the top; freed blocks recycle within their count's free
    // list), so a block's view array is built once and reused for every contact that lands on it —
    // the speculative touching/not-touching flap allocates nothing after first touch.
    private _viewCache = new Map<number, Manifold[]>();

    /** Directory column, aliased through both views (material row is f32, the meta tail is u32). */
    dirF = new Float32Array(0);
    dirU = new Uint32Array(0);
    /** Manifold pool, aliased three ways: header/point floats are f32, featureId/persisted/pointCount
     * are u32, and the signed triangleIndex (NULL_INDEX = -1) reads through the i32 view. */
    poolF = new Float32Array(0);
    poolU = new Uint32Array(0);
    poolI = new Int32Array(0);
    // Set when a mid-narrowphase `alloc` grew the region (which shifts the geometry region after it);
    // the step drains it into `world.geometryDirty` so geometry re-uploads before the next narrowphase.
    grew = false;

    /** Track a directory slot for a contact (b3CreateContact). Grows the directory capacity if the id
     * is past the current high-water; the block itself is allocated later, on first touch. */
    ensureSlot(contactId: number): void {
        if (contactId + 1 > this._needDir) this._needDir = contactId + 1;
    }

    /** Release a contact's slot on destroy (b3DestroyContact): free its block, zero its manifold count,
     * and cold its convex GJK/SAT cache so a recycled contactId starts fresh (box3d zeroes the cache union
     * at create; the column equivalent is to clear it on release — a not-touching contact keeps its cache
     * via `clear`, only a *destroyed* one drops it). A recycled contactId within the existing directory
     * range doesn't grow the region, so `flush` never re-zeros its slot — this is the only cold path for it. */
    freeSlot(contactId: number): void {
        this.clear(contactId);
        const o = contactId * DIR_STRIDE + DIR_CACHE;
        if (o + CACHE_WORDS <= this.dirU.length) {
            for (let k = 0; k < CACHE_WORDS; ++k) this.dirU[o + k] = 0;
        }
    }

    /**
     * Allocate a contiguous run of `count` manifold records for `contactId`, recycling a freed block of
     * the same size class or bumping the pool high-water. Frees any existing block first (a contact
     * whose cluster count changed reallocates, mirroring box3d). @returns the block base (element index).
     */
    allocBlock(contactId: number, count: number): number {
        this.freeBlock(contactId);
        const free = this._freeLists.get(count);
        let base: number;
        if (free !== undefined && free.length > 0) {
            base = free.pop() as number;
        } else {
            base = this._poolTop;
            this._poolTop += count;
        }
        this._blocks.set(contactId, { base, count });
        return base;
    }

    /** Return a contact's block to its size-class free list (no-op if it holds none). */
    freeBlock(contactId: number): void {
        const b = this._blocks.get(contactId);
        if (b === undefined) return;
        let free = this._freeLists.get(b.count);
        if (free === undefined) {
            free = [];
            this._freeLists.set(b.count, free);
        }
        free.push(b.base);
        this._blocks.delete(contactId);
    }

    /**
     * Reserve the wasm region up to the required capacities (growing + memmoving the pool in place, so
     * live blocks keep their offsets) and re-derive the column views. @returns true if the region grew,
     * so the caller re-uploads the geometry region that sits after it and shifted.
     */
    flush(): boolean {
        // Reserve on the first call even when empty: the region must exist before the first geometry
        // upload, which lays the geo pools right after it (a hull can be created — and its geometry
        // uploaded — before the first contact, so the region can't wait for a contact to size it, or geo
        // would land at heap_base and the later-reserved directory would overlap it).
        if (this._dirCap > 0 && this._needDir <= this._dirCap && this._poolTop <= this._poolCap) {
            return false;
        }
        const oldDirCap = this._dirCap;
        this._dirCap = growCap(Math.max(this._needDir, this._dirCap));
        this._poolCap = growCap(Math.max(this._poolTop, this._poolCap));
        kernel().reserveManifolds(this._dirCap, this._poolCap);
        this.refreshViews();
        // Cold the convex GJK/SAT cache of every newly-reserved directory record. The wasm kernel is a
        // singleton shared across worlds, so a fresh region reuses another world's (or an abandoned
        // contact's) linear memory; a stale SAT/simplex cache holds indices for a different hull and
        // faults the narrowphase. This zeros [oldDirCap, dirCap) — the whole directory on the first
        // reserve, only the grown tail afterwards (existing contacts keep their warm caches).
        for (let cid = oldDirCap; cid < this._dirCap; ++cid) {
            const o = cid * DIR_STRIDE + DIR_CACHE;
            for (let w = 0; w < CACHE_WORDS; ++w) this.dirU[o + w] = 0;
        }
        return true;
    }

    /** Re-derive the column views over the current region (after any `memory.grow`, which detaches every
     * view). Cheap — a handful of typed-array constructions, no copy. No-op before the first reserve. */
    refreshViews(): void {
        if (this._dirCap === 0) return;
        const k = kernel();
        const buf = k.memory.buffer;
        const layout = new Uint32Array(buf, k.manifoldLayoutPtr(), N_MANIFOLD);
        this.dirF = new Float32Array(buf, layout[DIR], this._dirCap * DIR_STRIDE);
        this.dirU = new Uint32Array(buf, layout[DIR], this._dirCap * DIR_STRIDE);
        this.poolF = new Float32Array(buf, layout[POOL], this._poolCap * MANIFOLD_STRIDE);
        this.poolU = new Uint32Array(buf, layout[POOL], this._poolCap * MANIFOLD_STRIDE);
        this.poolI = new Int32Array(buf, layout[POOL], this._poolCap * MANIFOLD_STRIDE);
    }

    /**
     * Allocate `contactId`'s manifold block for `count` records (recycling a freed same-size block or
     * bumping the pool), write the block descriptor into the directory, growing + re-deriving views in
     * place if the pool overflowed, and return column-backed `Manifold` views over the fresh block. The
     * narrowphase writes its manifolds through these; the data is the persistent warm-start state.
     */
    alloc(contactId: number, count: number): Manifold[] {
        const base = this.allocBlock(contactId, count);
        if (this._poolTop > this._poolCap) {
            this._poolCap = growCap(this._poolTop);
            kernel().reserveManifolds(this._dirCap, this._poolCap);
            this.refreshViews();
            this.grew = true;
        }
        const dir = contactId * DIR_STRIDE;
        this.dirU[dir + DIR_MANIFOLD_COUNT] = count;
        this.dirU[dir + DIR_MANIFOLD_BASE] = base;
        return this.views(contactId, count, base);
    }

    /** Column-backed `Manifold` views over a contact's already-allocated block, cached per block base
     * (callers treat the returned array as immutable). */
    views(
        contactId: number,
        count: number,
        base = this._blocks.get(contactId)?.base ?? 0,
    ): Manifold[] {
        let out = this._viewCache.get(base);
        if (out === undefined) {
            out = new Array(count);
            for (let i = 0; i < count; ++i) {
                out[i] = new ManifoldView(this, (base + i) * MANIFOLD_STRIDE);
            }
            this._viewCache.set(base, out);
        }
        return out;
    }

    /** Free a contact's block and zero its directory manifold count (b3-empty manifold set). */
    clear(contactId: number): void {
        this.freeBlock(contactId);
        if (contactId * DIR_STRIDE < this.dirU.length) {
            this.dirU[contactId * DIR_STRIDE + DIR_MANIFOLD_COUNT] = 0;
        }
    }

    /**
     * Write a contact's per-step directory row — the material + body sim indices the solver gathers,
     * zeroing the hit flag — before the solve. The block descriptor (manifoldCount/manifoldBase) is
     * written separately by `alloc` during the narrowphase. `indexA`/`indexB` may be `NULL_INDEX`
     * (-1), which lands as `0xFFFFFFFF` on the u32 write (= the kernel's `NULL_INDEX`).
     */
    writeContactRow(
        contactId: number,
        friction: number,
        restitution: number,
        rollingResistance: number,
        tangentVelocity: Vec3,
        flags: number,
        indexA: number,
        indexB: number,
    ): void {
        const o = contactId * DIR_STRIDE;
        this.dirF[o + DIR_FRICTION] = friction;
        this.dirF[o + DIR_RESTITUTION] = restitution;
        this.dirF[o + DIR_ROLLING_RESISTANCE] = rollingResistance;
        this.dirF[o + DIR_TANGENT_VELOCITY] = tangentVelocity.x;
        this.dirF[o + DIR_TANGENT_VELOCITY + 1] = tangentVelocity.y;
        this.dirF[o + DIR_TANGENT_VELOCITY + 2] = tangentVelocity.z;
        this.dirU[o + DIR_FLAGS] = flags;
        this.dirU[o + DIR_INDEX_A] = indexA;
        this.dirU[o + DIR_INDEX_B] = indexB;
        this.dirU[o + DIR_HIT] = 0;
    }

    /** @returns true if the kernel `store` flagged a hit event for this contact this step. */
    hit(contactId: number): boolean {
        return this.dirU[contactId * DIR_STRIDE + DIR_HIT] !== 0;
    }

    /**
     * Load a contact's cached relative pose (last full narrowphase) out of the directory recycle record
     * into the given objects. The kernel recycle pass owns this record for a dynamic-dynamic convex
     * contact; when such a contact temporarily runs the TS path (a partner is sleeping) it reads the
     * directory here, so the pose stays consistent across the kernel↔TS transition. Out-params (zero-alloc).
     */
    readRecyclePose(contactId: number, rotA: Quat, rotB: Quat, relPose: Transform): void {
        const o = contactId * DIR_STRIDE;
        const f = this.dirF;
        rotA.v.x = f[o + DIR_CACHED_ROT_A];
        rotA.v.y = f[o + DIR_CACHED_ROT_A + 1];
        rotA.v.z = f[o + DIR_CACHED_ROT_A + 2];
        rotA.s = f[o + DIR_CACHED_ROT_A + 3];
        rotB.v.x = f[o + DIR_CACHED_ROT_B];
        rotB.v.y = f[o + DIR_CACHED_ROT_B + 1];
        rotB.v.z = f[o + DIR_CACHED_ROT_B + 2];
        rotB.s = f[o + DIR_CACHED_ROT_B + 3];
        relPose.p.x = f[o + DIR_CACHED_REL_POSE];
        relPose.p.y = f[o + DIR_CACHED_REL_POSE + 1];
        relPose.p.z = f[o + DIR_CACHED_REL_POSE + 2];
        relPose.q.v.x = f[o + DIR_CACHED_REL_POSE + 3];
        relPose.q.v.y = f[o + DIR_CACHED_REL_POSE + 4];
        relPose.q.v.z = f[o + DIR_CACHED_REL_POSE + 5];
        relPose.q.s = f[o + DIR_CACHED_REL_POSE + 6];
    }

    /** Store a contact's cached relative pose into the directory recycle record — the TS-path mirror of
     * the kernel recycle pass's pose-cache write, keeping the directory current for the next step
     * whichever path processes the contact then. See {@link readRecyclePose}. */
    writeRecyclePose(contactId: number, rotA: Quat, rotB: Quat, relPose: Transform): void {
        const o = contactId * DIR_STRIDE;
        const f = this.dirF;
        f[o + DIR_CACHED_ROT_A] = rotA.v.x;
        f[o + DIR_CACHED_ROT_A + 1] = rotA.v.y;
        f[o + DIR_CACHED_ROT_A + 2] = rotA.v.z;
        f[o + DIR_CACHED_ROT_A + 3] = rotA.s;
        f[o + DIR_CACHED_ROT_B] = rotB.v.x;
        f[o + DIR_CACHED_ROT_B + 1] = rotB.v.y;
        f[o + DIR_CACHED_ROT_B + 2] = rotB.v.z;
        f[o + DIR_CACHED_ROT_B + 3] = rotB.s;
        f[o + DIR_CACHED_REL_POSE] = relPose.p.x;
        f[o + DIR_CACHED_REL_POSE + 1] = relPose.p.y;
        f[o + DIR_CACHED_REL_POSE + 2] = relPose.p.z;
        f[o + DIR_CACHED_REL_POSE + 3] = relPose.q.v.x;
        f[o + DIR_CACHED_REL_POSE + 4] = relPose.q.v.y;
        f[o + DIR_CACHED_REL_POSE + 5] = relPose.q.v.z;
        f[o + DIR_CACHED_REL_POSE + 6] = relPose.q.s;
    }

    // Raw per-point walks over a contact's resident manifolds — the narrowphase's hot loops, run on
    // the pool columns directly so no view getters (which return fresh Vec3s) are touched. The f32
    // expression trees are op-identical to the view-based loops they replaced. `count` is the
    // caller's `contact.manifoldCount` — the JS-side truth; the directory count can be stale for a
    // contact that stopped touching without a `clear` (the mesh not-touching path), so only the
    // block base is read from the directory (valid whenever count > 0, written by `alloc`).

    /** Shift every point's anchors from body origin to center of mass (b3UpdateContact tail):
     * anchorA -= centerA, anchorB -= centerB. */
    shiftAnchors(contactId: number, count: number, centerA: Vec3, centerB: Vec3): void {
        const base = this.dirU[contactId * DIR_STRIDE + DIR_MANIFOLD_BASE];
        const f = this.poolF;
        const u = this.poolU;
        for (let m = 0; m < count; ++m) {
            const mo = (base + m) * MANIFOLD_STRIDE;
            const pc = u[mo + M_POINT_COUNT];
            for (let p = 0; p < pc; ++p) {
                const po = mo + M_POINTS + p * POINT_STRIDE;
                f[po + P_ANCHOR_A] = f32(f[po + P_ANCHOR_A] - centerA.x);
                f[po + P_ANCHOR_A + 1] = f32(f[po + P_ANCHOR_A + 1] - centerA.y);
                f[po + P_ANCHOR_A + 2] = f32(f[po + P_ANCHOR_A + 2] - centerA.z);
                f[po + P_ANCHOR_B] = f32(f[po + P_ANCHOR_B] - centerB.x);
                f[po + P_ANCHOR_B + 1] = f32(f[po + P_ANCHOR_B + 1] - centerB.y);
                f[po + P_ANCHOR_B + 2] = f32(f[po + P_ANCHOR_B + 2] - centerB.z);
            }
        }
    }

    /** Cache every point's separation for the next recycle test: baseSeparation = separation. */
    rebaseSeparations(contactId: number, count: number): void {
        const base = this.dirU[contactId * DIR_STRIDE + DIR_MANIFOLD_BASE];
        const f = this.poolF;
        const u = this.poolU;
        for (let m = 0; m < count; ++m) {
            const mo = (base + m) * MANIFOLD_STRIDE;
            const pc = u[mo + M_POINT_COUNT];
            for (let p = 0; p < pc; ++p) {
                const po = mo + M_POINTS + p * POINT_STRIDE;
                f[po + P_BASE_SEPARATION] = f[po + P_SEPARATION];
            }
        }
    }

    /** The recycle-success separation update (tryRecycle's per-point loop): with the incremental body
     * rotations `matrixA`/`matrixB` and center delta `dc`, separation = baseSeparation +
     * dot(dc + (matrixB·anchorB − matrixA·anchorA), normal), and every point marks persisted. */
    recycleSeparations(
        contactId: number,
        count: number,
        matrixA: Mat3,
        matrixB: Mat3,
        dc: Vec3,
    ): void {
        const base = this.dirU[contactId * DIR_STRIDE + DIR_MANIFOLD_BASE];
        const f = this.poolF;
        const u = this.poolU;
        for (let m = 0; m < count; ++m) {
            const mo = (base + m) * MANIFOLD_STRIDE;
            walkNormal.x = f[mo + M_NORMAL];
            walkNormal.y = f[mo + M_NORMAL + 1];
            walkNormal.z = f[mo + M_NORMAL + 2];
            const pc = u[mo + M_POINT_COUNT];
            for (let p = 0; p < pc; ++p) {
                const po = mo + M_POINTS + p * POINT_STRIDE;
                walkA.x = f[po + P_ANCHOR_A];
                walkA.y = f[po + P_ANCHOR_A + 1];
                walkA.z = f[po + P_ANCHOR_A + 2];
                mat3.mulVOut(matrixA, walkA, walkA);
                walkB.x = f[po + P_ANCHOR_B];
                walkB.y = f[po + P_ANCHOR_B + 1];
                walkB.z = f[po + P_ANCHOR_B + 2];
                mat3.mulVOut(matrixB, walkB, walkB);
                vec3.subOut(walkB, walkA, walkB);
                vec3.addOut(dc, walkB, walkB);
                f[po + P_SEPARATION] = f32(f[po + P_BASE_SEPARATION] + vec3.dot(walkB, walkNormal));
                u[po + P_PERSISTED] = 1;
            }
        }
    }
}

/**
 * A `ManifoldPoint` (b3ManifoldPoint) backed by an inline point record in the store pool rather than a
 * plain object. Vec3 getters return fresh objects (matching the plain-object semantics the narrowphase
 * expects); setters write into the pool. Reads go through `store.pool*` each access so a `memory.grow`
 * re-derivation is transparent. Offsets mirror the point record in kernel/src/manifolds.rs.
 */
class ManifoldPointView implements ManifoldPoint {
    constructor(
        private readonly _s: ManifoldStore,
        private readonly _o: number,
    ) {}
    get anchorA(): Vec3 {
        const f = this._s.poolF;
        const o = this._o + P_ANCHOR_A;
        return { x: f[o], y: f[o + 1], z: f[o + 2] };
    }
    set anchorA(v: Vec3) {
        const f = this._s.poolF;
        const o = this._o + P_ANCHOR_A;
        f[o] = v.x;
        f[o + 1] = v.y;
        f[o + 2] = v.z;
    }
    get anchorB(): Vec3 {
        const f = this._s.poolF;
        const o = this._o + P_ANCHOR_B;
        return { x: f[o], y: f[o + 1], z: f[o + 2] };
    }
    set anchorB(v: Vec3) {
        const f = this._s.poolF;
        const o = this._o + P_ANCHOR_B;
        f[o] = v.x;
        f[o + 1] = v.y;
        f[o + 2] = v.z;
    }
    get separation(): number {
        return this._s.poolF[this._o + P_SEPARATION];
    }
    set separation(v: number) {
        this._s.poolF[this._o + P_SEPARATION] = v;
    }
    get baseSeparation(): number {
        return this._s.poolF[this._o + P_BASE_SEPARATION];
    }
    set baseSeparation(v: number) {
        this._s.poolF[this._o + P_BASE_SEPARATION] = v;
    }
    get normalImpulse(): number {
        return this._s.poolF[this._o + P_NORMAL_IMPULSE];
    }
    set normalImpulse(v: number) {
        this._s.poolF[this._o + P_NORMAL_IMPULSE] = v;
    }
    get totalNormalImpulse(): number {
        return this._s.poolF[this._o + P_TOTAL_NORMAL_IMPULSE];
    }
    set totalNormalImpulse(v: number) {
        this._s.poolF[this._o + P_TOTAL_NORMAL_IMPULSE] = v;
    }
    get normalVelocity(): number {
        return this._s.poolF[this._o + P_NORMAL_VELOCITY];
    }
    set normalVelocity(v: number) {
        this._s.poolF[this._o + P_NORMAL_VELOCITY] = v;
    }
    get featureId(): number {
        return this._s.poolU[this._o + P_FEATURE_ID];
    }
    set featureId(v: number) {
        this._s.poolU[this._o + P_FEATURE_ID] = v;
    }
    get triangleIndex(): number {
        return this._s.poolI[this._o + P_TRIANGLE_INDEX];
    }
    set triangleIndex(v: number) {
        this._s.poolI[this._o + P_TRIANGLE_INDEX] = v;
    }
    get persisted(): boolean {
        return this._s.poolU[this._o + P_PERSISTED] !== 0;
    }
    set persisted(v: boolean) {
        this._s.poolU[this._o + P_PERSISTED] = v ? 1 : 0;
    }
}

/**
 * A `Manifold` (b3Manifold) backed by a pool record. The four inline point views are built once (the
 * live prefix is `pointCount`); header getters/setters read/write the pool. Offsets mirror the manifold
 * record header in kernel/src/manifolds.rs.
 */
class ManifoldView implements Manifold {
    readonly points: ManifoldPoint[];
    constructor(
        private readonly _s: ManifoldStore,
        private readonly _o: number,
    ) {
        this.points = [
            new ManifoldPointView(_s, _o + M_POINTS),
            new ManifoldPointView(_s, _o + M_POINTS + POINT_STRIDE),
            new ManifoldPointView(_s, _o + M_POINTS + 2 * POINT_STRIDE),
            new ManifoldPointView(_s, _o + M_POINTS + 3 * POINT_STRIDE),
        ];
    }
    get normal(): Vec3 {
        const f = this._s.poolF;
        const o = this._o + M_NORMAL;
        return { x: f[o], y: f[o + 1], z: f[o + 2] };
    }
    set normal(v: Vec3) {
        const f = this._s.poolF;
        const o = this._o + M_NORMAL;
        f[o] = v.x;
        f[o + 1] = v.y;
        f[o + 2] = v.z;
    }
    get frictionImpulse(): Vec3 {
        const f = this._s.poolF;
        const o = this._o + M_FRICTION;
        return { x: f[o], y: f[o + 1], z: f[o + 2] };
    }
    set frictionImpulse(v: Vec3) {
        const f = this._s.poolF;
        const o = this._o + M_FRICTION;
        f[o] = v.x;
        f[o + 1] = v.y;
        f[o + 2] = v.z;
    }
    get twistImpulse(): number {
        return this._s.poolF[this._o + M_TWIST];
    }
    set twistImpulse(v: number) {
        this._s.poolF[this._o + M_TWIST] = v;
    }
    get rollingImpulse(): Vec3 {
        const f = this._s.poolF;
        const o = this._o + M_ROLLING;
        return { x: f[o], y: f[o + 1], z: f[o + 2] };
    }
    set rollingImpulse(v: Vec3) {
        const f = this._s.poolF;
        const o = this._o + M_ROLLING;
        f[o] = v.x;
        f[o + 1] = v.y;
        f[o + 2] = v.z;
    }
    get pointCount(): number {
        return this._s.poolU[this._o + M_POINT_COUNT];
    }
    set pointCount(v: number) {
        this._s.poolU[this._o + M_POINT_COUNT] = v;
    }
}

/** Create an empty manifold store for a new world. */
export function createManifoldStore(): ManifoldStore {
    return new ManifoldStore();
}

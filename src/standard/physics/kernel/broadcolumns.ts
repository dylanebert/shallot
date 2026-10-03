import type { World } from "../../../engine";
// The persistent broad-phase region (kernel/src/broad.rs) — three dynamic-tree node pools, shape-pair
// membership arrays and joint-filtered body pairs. The kernel queries them without a per-step marshal.
// This store owns views of the resident tree headers, pools, pairs, body filters and moves.
// Tree operations run in the kernel; the pair table uses the resident TS views.
//
// A region grow (or any `memory.grow` elsewhere) detaches every typed-array view, so the store follows
// the shared-memory view-refresh discipline: it re-derives the views from the kernel layout header
// and writes them straight back into the DynamicTree / HashSet structs, so every tree/table op reads a
// current view. `refreshViews` is called at the top of the pair-finding pass and after every grow —
// never per-iteration (that would reintroduce churn).

import type { HashSet } from "../collision/table";
import type { WorldState } from "../world/world";
import { kernel } from "./kernel";
import type { DynamicTree } from "./treecolumns";

/** u32/f32 slots per dynamic-tree node — mirrors tree.rs and broad.rs. */
const TREE_STRIDE = 12;
/** Three trees, three pair columns, body filters, moves and three moved bitsets. */
const N_BROAD = 11;

const EMPTY_F = new Float32Array(0);
const EMPTY_I = new Int32Array(0);
const EMPTY_U = new Uint32Array(0);

/**
 * The resident broad-phase region's TS-side view manager. One per world. Holds references to the three
 * dynamic trees and the pair set so a refresh can rewrite their column views in place, and to the world
 * so a grow can refresh the sibling stores a `memory.grow` detached.
 */
export class BroadStore {
    readonly ecsState: World | undefined;

    readonly worldId: number;
    constructor(ecsState: World | undefined, worldId: number) {
        this.ecsState = ecsState;
        this.worldId = worldId;
    }

    /** The three dynamic trees (static / kinematic / dynamic), set at broad-phase creation. */
    trees: DynamicTree[] = [];
    /** The pair set, set at broad-phase creation. */
    set: HashSet | null = null;
    /** The owning world, set once the world is fully constructed (sibling-store refresh on a grow). */
    world: WorldState | null = null;
    moveData = EMPTY_I;
    moveState = EMPTY_U;
    movedBits: Uint32Array[] = [EMPTY_U, EMPTY_U, EMPTY_U];
    /** `memory.buffer.byteLength` at the last refresh — catches a `memory.grow` (single-thread detach or
     * shared-memory tail extension). */
    private _lastLen = -1;
    /** The kernel's layout generation catches column reallocation without a memory grow. */
    private _lastGen = -1;
    private _genPtr = 0;
    private _gen = EMPTY_U;
    initialization = { claimed: false, movesInitialized: false };

    /** Initialize this World's native broad-phase metadata on first use. */
    initialize(): void {
        const world = this.world;
        if (this.initialization.claimed || world === null) return;
        this.initialization.claimed = true;
        this.growBodyFilters(world.bodyFilters.capacity);
        world.bodyFilters.data.fill(0);
    }

    /** Refresh only if the region moved or memory grew since the last refresh. O(1) when fresh (a
     * function call + a byteLength read), so it can guard every broad-phase read/mutate entry point
     * without reintroducing churn. */
    refreshIfStale(): void {
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        if (k.memory.buffer.byteLength === this._lastLen && this._gen[0] === this._lastGen) return;
        this.refreshViews();
    }

    /** Re-derive the column views over the current region and write them into the tree/set/filter structs.
     * Cheap — a handful of typed-array constructions, no copy. */
    refreshViews(): void {
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        const buf = k.memory.buffer;
        this._lastLen = buf.byteLength;
        if (this._genPtr === 0) this._genPtr = k.broadGenPtr();
        this._gen = new Uint32Array(buf, this._genPtr, 1);
        this._lastGen = this._gen[0];
        const layout = new Uint32Array(buf, k.broadLayoutPtr(), N_BROAD);

        for (let i = 0; i < 3; ++i) {
            const t = this.trees[i];
            if (t === undefined) continue;
            const cap = k.broadTreeCap(i);
            if (cap === 0) {
                t.nf = EMPTY_F;
                t.ni = EMPTY_I;
                continue;
            }
            if (this.initialization.claimed) {
                const initial = !t.residentState;
                const previous = t.state;
                t.state = new Int32Array(buf, layout[i], 6);
                if (initial) t.state.set(previous);
                t.residentState = true;
            }
            t.nf = new Float32Array(buf, layout[i] + 24, cap * TREE_STRIDE);
            t.ni = new Int32Array(buf, layout[i] + 24, cap * TREE_STRIDE);
        }

        const moveCapacity = k.broadTreeCap(0) + k.broadTreeCap(1) + k.broadTreeCap(2);
        if (this.initialization.claimed && moveCapacity !== 0) {
            this.moveState = new Uint32Array(buf, layout[7], 1);
            if (!this.initialization.movesInitialized) {
                k.broadClearMoves();
                this.initialization.movesInitialized = true;
            }
            this.moveData = new Int32Array(buf, layout[7] + 4, moveCapacity);
            for (let i = 0; i < 3; i++)
                this.movedBits[i] = new Uint32Array(
                    buf,
                    layout[8 + i],
                    Math.ceil(k.broadTreeCap(i) / 32),
                );
        } else {
            this.moveState = EMPTY_U;
            this.moveData = EMPTY_I;
            for (let i = 0; i < 3; i++) this.movedBits[i] = EMPTY_U;
        }
        const filter = this.world?.bodyFilters;
        if (this.initialization.claimed && filter !== undefined && filter.capacity !== 0) {
            filter.data = new Uint32Array(buf, layout[6], 1 + 3 * filter.capacity);
        }

        const s = this.set;
        if (s !== null) {
            const setCap = k.broadSetCap();
            if (setCap === 0) {
                s.keyHi = EMPTY_U;
                s.keyLo = EMPTY_U;
                s.hashes = EMPTY_U;
            } else {
                s.keyHi = new Uint32Array(buf, layout[3], setCap);
                s.keyLo = new Uint32Array(buf, layout[4], setCap);
                s.hashes = new Uint32Array(buf, layout[5], setCap);
            }
        }
    }

    reserveTreeWork(depth: number, words: number): number {
        const k = kernel(this.ecsState);
        const before = k.memory.buffer.byteLength;
        const ptr = k.reserveTreeWork(depth, words) >>> 0;
        this.refreshIfStale();
        if (before !== k.memory.buffer.byteLength && this.world) {
            this.world.manifoldStore.refreshViews();
            this.world.bodyStore.refreshViews();
            this.world.shapeStore.refreshViews();
        }
        return ptr;
    }

    /** Grow tree pool `i` to `nodeCapacity` nodes (grow-only), refreshing all views afterward. */
    growTree(i: number, nodeCapacity: number): void {
        this.reserve(
            i === 0 ? nodeCapacity : 0,
            i === 1 ? nodeCapacity : 0,
            i === 2 ? nodeCapacity : 0,
            0,
        );
    }

    /** Grow the pair-set arrays to `setCap` slots (grow-only), refreshing all views afterward. */
    growSet(setCap: number): void {
        this.reserve(0, 0, 0, setCap);
    }

    growBodyFilters(capacity: number): void {
        this.reserve(0, 0, 0, 0, capacity);
    }

    // Reallocation can move columns without growing memory; derive views after every reserve.
    private reserve(capS: number, capK: number, capD: number, setCap: number, filterCap = 0): void {
        if (this.world !== null) this.initialize();
        else this.initialization.claimed = true;
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        const grew = k.reserveBroad(capS, capK, capD, setCap, filterCap) !== 0;
        this.refreshViews();
        if (grew) {
            const w = this.world;
            if (w !== null) {
                w.manifoldStore.refreshViews();
                w.bodyStore.refreshViews();
                w.shapeStore.refreshViews();
            }
        }
    }
}

/** Create an empty broad store for a new world. Its trees + set are registered by `createBroadPhase`. */
export function createBroadStore(world: World | undefined, worldId: number): BroadStore {
    return new BroadStore(world, worldId);
}

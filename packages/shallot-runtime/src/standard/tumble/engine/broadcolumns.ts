// The persistent broad-phase region (kernel/src/broad.rs) — the three dynamic-tree node pools plus the
// pair-set membership arrays, held resident in the kernel's linear memory so the in-kernel pair query +
// tree rebuild (3d) run over them without a per-step marshal. This store owns the TS views over the six
// sub-columns and the grow-only reservation policy; the tree/table algorithms mutate the columns
// through the views it hands back (`src/tree.ts`, `src/table.ts`).
//
// A region grow (or any `memory.grow` elsewhere) detaches every typed-array view, so the store follows
// the shared-memory view-refresh discipline: it re-derives all six views from the kernel layout header
// and writes them straight back into the DynamicTree / HashSet structs, so every tree/table op reads a
// current view. `refreshViews` is called at the top of the pair-finding pass and after every grow —
// never per-iteration (that would reintroduce churn).

import { kernel } from "./kernel";
import type { HashSet } from "./table";
import type { DynamicTree } from "./tree";
import type { WorldState } from "./world";

/** u32/f32 slots per dynamic-tree node — mirrors `STRIDE` in `src/tree.ts` + `TREE_STRIDE` in broad.rs. */
const TREE_STRIDE = 12;
/** Broad layout header size (3 tree pools + keyHi/keyLo/hashes). */
const N_BROAD = 6;

const EMPTY_F = new Float32Array(0);
const EMPTY_I = new Int32Array(0);
const EMPTY_U = new Uint32Array(0);

/**
 * The resident broad-phase region's TS-side view manager. One per world. Holds references to the three
 * dynamic trees and the pair set so a refresh can rewrite their column views in place, and to the world
 * so a grow can refresh the sibling stores a `memory.grow` detached.
 */
export class BroadStore {
    /** The three dynamic trees (static / kinematic / dynamic), set at broad-phase creation. */
    trees: DynamicTree[] = [];
    /** The pair set, set at broad-phase creation. */
    set: HashSet | null = null;
    /** The owning world, set once the world is fully constructed (sibling-store refresh on a grow). */
    world: WorldState | null = null;
    /** `memory.buffer.byteLength` at the last refresh — catches a `memory.grow` (single-thread detach or
     * shared-memory tail extension). */
    private _lastLen = -1;
    /** The kernel's broad-layout generation at the last refresh — catches a relocation (a region below
     * grew within committed pages, shifting this region's offsets without a `memory.grow`). */
    private _lastGen = -1;

    /** Refresh only if the region moved or memory grew since the last refresh. O(1) when fresh (a
     * function call + a byteLength read), so it can guard every broad-phase read/mutate entry point
     * without reintroducing churn. */
    refreshIfStale(): void {
        const k = kernel();
        if (k.broadGen() === this._lastGen && k.memory.buffer.byteLength === this._lastLen) return;
        this.refreshViews();
    }

    /** Re-derive all six column views over the current region and write them into the tree/set structs.
     * Cheap — a handful of typed-array constructions, no copy. */
    refreshViews(): void {
        const k = kernel();
        const buf = k.memory.buffer;
        this._lastLen = buf.byteLength;
        this._lastGen = k.broadGen();
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
            t.nf = new Float32Array(buf, layout[i], cap * TREE_STRIDE);
            t.ni = new Int32Array(buf, layout[i], cap * TREE_STRIDE);
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

    // Reserve the region (grow-only per column) and re-derive the views. The growing column needs its
    // view rebuilt even when the region did not grow (a fresh world reuses the singleton's larger stale
    // capacity, so its first reserve is a no-op — but the tree still needs a view over it). A real grow
    // additionally `memory.grow`s, detaching every sibling store's views; refresh those too.
    private reserve(capS: number, capK: number, capD: number, setCap: number): void {
        const grew = kernel().reserveBroad(capS, capK, capD, setCap) !== 0;
        this.refreshViews();
        if (grew) {
            const w = this.world;
            if (w !== null) {
                w.manifoldStore.refreshViews();
                w.bodyStore.refreshViews();
                w.shapeStore.refreshViews();
                w.fatAabbStore.refreshViews();
            }
        }
    }
}

/** Create an empty broad store for a new world. Its trees + set are registered by `createBroadPhase`. */
export function createBroadStore(): BroadStore {
    return new BroadStore();
}

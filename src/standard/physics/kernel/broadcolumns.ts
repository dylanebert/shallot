import type { World } from "../../../engine";
// The persistent broad-phase region (kernel/src/broad.rs) — three dynamic-tree node pools, shape-pair
// membership arrays. The kernel queries them without a per-step marshal.
// This store owns views of the resident tree headers, pools, pairs and moves.
// Tree, pair-table and move operations run in the kernel.
//
// A region grow (or any `memory.grow` elsewhere) detaches every typed-array view, so the store follows
// the shared-memory view-refresh discipline: it re-derives the views from the kernel layout header
// and writes them straight back into the DynamicTree structs, so every tree op reads a
// current view. `refreshViews` is called at the top of the pair-finding pass and after every grow —
// never per-iteration (that would reintroduce churn).

import type { WorldState } from "../world/world";
import { kernel } from "./kernel";
import type { DynamicTree } from "./treecolumns";
import { KernelViews } from "./views";

/** u32/f32 slots per dynamic-tree node — mirrors tree.rs and broad.rs. */
const TREE_STRIDE = 12;
/** Three trees, three pair columns, moves and three moved bitsets. */
const N_BROAD = 10;

const EMPTY_F = new Float32Array(0);
const EMPTY_I = new Int32Array(0);
const EMPTY_U = new Uint32Array(0);

/**
 * The resident broad-phase region's TS-side view manager. One per world. Holds references to the three
 * dynamic trees and World so a refresh can rebind their column views.
 */
export class BroadStore extends KernelViews {
    readonly worldId: number;
    constructor(ecsState: World | undefined, worldId: number) {
        super(ecsState);
        this.worldId = worldId;
        this.guardViews();
    }

    /** The three dynamic trees (static / kinematic / dynamic), set at broad-phase creation. */
    trees: DynamicTree[] = [];
    /** The owning world, set once the world is fully constructed (sibling-store refresh on a grow). */
    world: WorldState | null = null;
    moveData = EMPTY_I;
    moveState = EMPTY_U;
    movedBits: Uint32Array[] = [EMPTY_U, EMPTY_U, EMPTY_U];
    initialization = { claimed: false, movesInitialized: false };
    #layout = EMPTY_U;

    override captureCheckpoint() {
        return { ...this.initialization };
    }
    override restoreCheckpoint(state: unknown): void {
        Object.assign(this.initialization, state);
    }

    /** Initialize this World's native broad-phase metadata on first use. */
    initialize(): void {
        const world = this.world;
        if (this.initialization.claimed || world === null) return;
        this.initialization.claimed = true;
    }

    /** Select this World for native operations; rebind views only when the shared key changed. */
    override refreshIfStale(): void {
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        this.ensureViews();
    }

    /** Re-derive the column views over the current region and write them into the tree structs.
     * Cheap — a handful of typed-array constructions, no copy. */
    protected deriveViews(): void {
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        const buf = k.memory.buffer;
        const ptr = k.broadLayoutPtr();
        if (this.#layout.buffer !== buf || this.#layout.byteOffset !== ptr)
            this.#layout = new Uint32Array(buf, ptr, N_BROAD);
        const layout = this.#layout;

        for (let i = 0; i < 3; ++i) {
            const t = this.trees[i];
            if (t === undefined) continue;
            const cap = k.broadTreeCap(i);
            if (cap === 0) {
                t.nf = EMPTY_F;
                t.ni = EMPTY_I;
                continue;
            }
            const initial = this.initialization.claimed && !t.residentState && this.world === null;
            if (this.initialization.claimed) {
                const previous = t.state;
                if (t.state.buffer !== buf || t.state.byteOffset !== layout[i])
                    t.state = new Int32Array(buf, layout[i], 6);
                if (initial) t.state.set(previous);
                t.residentState = true;
            }
            t.nodeCapacity = !this.initialization.claimed || initial ? 0 : cap;
            if (
                t.nf.buffer !== buf ||
                t.nf.byteOffset !== layout[i] + 24 ||
                t.nf.length !== cap * TREE_STRIDE
            ) {
                t.nf = new Float32Array(buf, layout[i] + 24, cap * TREE_STRIDE);
                t.ni = new Int32Array(buf, layout[i] + 24, cap * TREE_STRIDE);
            }
        }

        const moveCapacity = k.broadTreeCap(0) + k.broadTreeCap(1) + k.broadTreeCap(2);
        if (this.initialization.claimed && moveCapacity !== 0) {
            if (this.moveState.buffer !== buf || this.moveState.byteOffset !== layout[6])
                this.moveState = new Uint32Array(buf, layout[6], 1);
            const initial = !this.initialization.movesInitialized && this.world === null;
            if (initial) this.moveState[0] = 0;
            this.initialization.movesInitialized = true;
            if (
                this.moveData.buffer !== buf ||
                this.moveData.byteOffset !== layout[6] + 4 ||
                this.moveData.length !== moveCapacity
            )
                this.moveData = new Int32Array(buf, layout[6] + 4, moveCapacity);
            for (let i = 0; i < 3; i++) {
                const old = this.movedBits[i];
                const length = Math.ceil(k.broadTreeCap(i) / 32);
                if (old.buffer !== buf || old.byteOffset !== layout[7 + i] || old.length !== length)
                    this.movedBits[i] = new Uint32Array(buf, layout[7 + i], length);
                if (initial) this.movedBits[i].fill(0);
            }
        } else {
            this.moveState = EMPTY_U;
            this.moveData = EMPTY_I;
            for (let i = 0; i < 3; i++) this.movedBits[i] = EMPTY_U;
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

    // Reallocation can move columns without growing memory; derive views after every reserve.
    private reserve(capS: number, capK: number, capD: number, setCap: number): void {
        if (this.world !== null) this.initialize();
        else this.initialization.claimed = true;
        const k = kernel(this.ecsState);
        k.bodySetActiveWorld(this.worldId);
        const grew = k.reserveBroad(capS, capK, capD, setCap) !== 0;
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

/** Create an empty broad store for a new world. Its trees are registered by `createBroadPhase`. */
export function createBroadStore(world: World | undefined, worldId: number): BroadStore {
    return new BroadStore(world, worldId);
}

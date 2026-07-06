import { type Plugin, type State, type System, slab, vec4 } from "@dylanebert/shallot";

// The CPU-submission burn knob: the per-frame JS residual a heavy scene still pays after GPU-driven
// rendering dissolves the draw-call overhead — ECS query iteration + slab writes. A fixed pool of churn
// entities, over which the `submission` system runs `getSubmission()` query-and-write passes per frame:
// each pass iterates the cached `[Churn]` query and writes every member's slab (marking its dirty bit, so
// the SlabSystem flush packs it — the real CPU→GPU slab-write path). Work = POOL × passes, so the knob
// scales the per-frame JS smoothly. The CPU-bound sibling of the GPU `load` (../load) / `bandwidth`
// (../bandwidth) burns: an induce-half whose cost lands on its OWN per-system profiler span
// (`Submission/submission`, the {@link SUBMISSION_SYSTEM} key), so the stress atom proves the induced
// submission cost is attributable and paces evenly. `setSubmission(0)` makes it a no-op.
//
// "entities × systems × slab writes" (roadmap) are the axes of the residual; the pass count scales their
// product — each pass is one system's worth of query iteration + per-entity slab writes, so a high pass
// count stands in for many systems churning a large scene.

// the churn pool: entities carrying ONLY the slab-backed Churn component (no Part/Transform), so they're
// pure CPU+slab churn and never touch the GPU render axes — the compute/bandwidth/gpu-mem axes don't see
// them. Sized so ~1k passes saturate the CPU side on a fast desktop (each pass ≈ POOL × (query step + vec4
// write) ≈ 0.04 ms on lovelace), the way the compute knob's iters/lane ceiling does the GPU.
const POOL = 4096;

/** per-entity churn data — a `slab(vec4)` so each write marks a dirty bit and the SlabSystem flush packs
 *  it (the real CPU→GPU slab path). Nothing reads the GPU mirror; the write + flush IS the stressor. */
export const Churn = { data: slab(vec4) };

// the system's profiler label is `${pluginName}/${system.name}` (scheduler) — the one source of truth the
// submission gate reads its induced per-system CPU span from.
export const SUBMISSION_SYSTEM = "Submission/submission";

// hoisted terms array — stable element identity hits the query cache's no-alloc fast path (query.ts), so
// the per-frame query resolution allocates nothing; only the `for…of` iterator does (the honest residual).
const TERMS = [Churn];

let passes = 0;
// the spawned churn-entity count, not the eids — the `submission` system drives work through the cached
// `[Churn]` query, never an eid list, so caching the eids would be both dead storage and the module-cached-
// identity anti-pattern (ecs.md). Reset + re-counted per build by warm.
let poolCount = 0;

/** set the per-frame query-and-write pass count — the CPU-time inflation level. ≤0 makes the pass a no-op. */
export function setSubmission(n: number): void {
    passes = Math.max(0, n | 0);
}

/** the active per-frame pass count. */
export function getSubmission(): number {
    return passes;
}

/** the live churn-pool entity count (POOL once warm) — the per-pass query length the reporter shows. */
export function submissionPool(): number {
    return poolCount;
}

/** grow or shrink the live churn pool to `target` entities — the entity-count lever the memory axis
 *  ramps to prove per-entity zero-allocation on the slab-write path (allocPerFrame flat vs pool size).
 *  Spawns or destroys Churn members to match; snapshots the live set when shrinking, so no module
 *  cached eids (ecs.md). The default is POOL once warm; restore it after a ramp. */
export function setSubmissionPool(state: State, target: number): void {
    const want = Math.max(0, target | 0);
    while (poolCount < want) {
        state.add(state.create(), Churn);
        poolCount++;
    }
    if (poolCount > want) {
        const live = [...state.query([Churn])];
        for (let i = want; i < live.length; i++) state.destroy(live[i]);
        poolCount = want;
    }
}

export const SubmissionPlugin: Plugin = {
    name: "Submission",
    components: { Churn },

    warm(state: State) {
        poolCount = 0;
        for (let i = 0; i < POOL; i++) {
            state.add(state.create(), Churn);
            poolCount++;
        }
    },

    systems: [
        {
            name: "submission",
            group: "draw",
            annotations: { mode: "always" },
            update(state) {
                if (passes <= 0) return;
                for (let p = 0; p < passes; p++) {
                    for (const eid of state.query(TERMS)) {
                        // a finite, bounded write — nothing reads it, but the slab.set side effect (array +
                        // dirty bit) keeps the loop from being dead-code-eliminated
                        const v = eid * 1e-4 + p * 1e-3;
                        Churn.data.set(eid, v, v, v, v);
                    }
                }
            },
        } satisfies System,
    ],
};

/** reset the pass count + pool counter — call from a scenario's `dispose`. The churn entities live in the
 *  State (warm-spawned), torn down with it; this just clears the module-scope knob. */
export function disposeSubmission(): void {
    passes = 0;
    poolCount = 0;
}

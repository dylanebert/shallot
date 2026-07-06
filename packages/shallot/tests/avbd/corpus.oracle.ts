import { describe, expect, test } from "bun:test";
import {
    BAND_ENERGY_EXCESS,
    BAND_PENETRATION,
    BAND_SETTLE,
    BAND_TUNNEL,
    bandState,
    CORPUS,
    energy,
    GROUND_TOP,
    type Scene,
    toBand,
} from "./corpus";
import { makeSolver, step } from "./solver";

// The Phase-4.6 topology stability corpus, oracle tier (scratch.md "AVBD rebuild" → Phase 4.6).
// The gym `pile` scenario is one jittered grid; this is the band-gated corpus across distinct
// topologies (corpus.ts) so stability problems have somewhere to surface. Here it runs through the
// f64 oracle — the reference for what "stable" looks like: every scene settles, energy never
// exceeds the drop supremum, penetration converges to the margin rest, nothing tunnels. The gym
// `pile` scenario runs the identical scenes + band; a scene steady here but wobbling there is a
// GPU bug, not a band that's wrong (physics.md "the oracle is not the suspect").
//
// The band is the gate-ladder's statistical band (scratch.md "Gate ladder"), never a long-horizon
// trajectory match — chaotic settling can't bit-match. Bounds are derived (corpus.ts BAND_*) and
// the oracle sits an order inside each (measured values in-line).

const G = -10;

// run one scene through the oracle for its horizon, returning the trajectory-worst band quantities
// + the settled-state band. Energy non-increasing is a per-frame invariant (E0 is the drop
// supremum); settle + penetration are read at rest.
function run(sc: Scene, iterations: number) {
    const bodies = sc.bodies();
    const s = makeSolver(bodies, { iterations, gravity: G, layer: "warmstart" });
    const E0 = energy(bodies.map(toBand), G);
    let finite = true;
    let maxExcess = -Infinity;
    let worstBottom = Infinity;
    for (let f = 0; f < sc.frames; f++) {
        step(s);
        const band = s.bodies.map(toBand);
        const bs = bandState(band);
        if (!bs.finite) finite = false;
        maxExcess = Math.max(maxExcess, (energy(band, G) - E0) / Math.abs(E0));
        worstBottom = Math.min(worstBottom, bs.minBottom);
    }
    const settled = bandState(s.bodies.map(toBand));
    return { finite, maxExcess, worstBottom, settled };
}

// asserted at the ROBUSTNESS iters=10 (the validation tier, decoupled from the iters=6 production ship —
// iters is a perf/robustness tradeoff, validate high / ship low) and a below-that iters=8 stress — graceful
// degradation: every topology must stay stable + settle with fewer-than-validation iterations. iters=8,
// not lower, because the 8:1-mass-ratio mixed-sizes bridge needs iters≥6 to converge (the iteration
// floor for high mass ratios, same family as the tall-chain floor — physics.md "f32 precision").
describe.each([10, 8])("AVBD oracle — topology stability corpus (iters=%i)", (iterations) => {
    for (const sc of CORPUS) {
        test(`${sc.name}: settles, energy non-increasing, bounded penetration, no tunnel`, () => {
            const r = run(sc, iterations);
            // no NaN/Inf anywhere on the trajectory (the bedrock — an instability goes non-finite)
            expect(r.finite).toBe(true);
            // E(t) ≤ E0 (the drop supremum) — a dissipative implicit solver only loses energy; the
            // penalty spring loads on impact but the band counts only KE+PE, so excess stays ≤ 0
            // (measured), and a real injection is O(1). 1e-2 is the margin between the two.
            expect(r.maxExcess).toBeLessThan(BAND_ENERGY_EXCESS);
            // comes to rest — residual creep well under the 10 m/s fall speed (measured ≤ 0.017)
            expect(r.settled.maxSpeed).toBeLessThan(BAND_SETTLE);
            // converged to a non-overlapping rest — overlap at the margin+mg/k floor (measured ≤ 0.019)
            expect(r.settled.maxPenetration).toBeLessThan(BAND_PENETRATION);
            // no tunnel-through: even a hard topple's transient corner dip (measured ≥ ground − 0.11)
            // stays well above a true escape through the ground (a tunneling body's bottom goes ≪ 0)
            expect(r.worstBottom).toBeGreaterThan(GROUND_TOP - BAND_TUNNEL);
        });
    }
});

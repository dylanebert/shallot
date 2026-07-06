import { describe, expect, test } from "bun:test";
import { type Cand, reduceManifold } from "./collide";
import type { Vec3 } from "./math";
import gold from "./reduce-gold.json";

// The 4.8.1 manifold-reduction crux gate. `reduceManifold` (collide.ts) is a port of Jolt's
// `PruneContactPoints`; reduce-gold.json comes from the verbatim-Jolt extract
// (reference/avbd-demo3d/reduce-gold.cpp → gen-reduce-gold.ts), an INDEPENDENT implementation, so
// agreement is a real cross-check (not the oracle vs itself). The extract is f64 like the oracle and
// the clouds are asymmetric (no argmax ties), so the kept SET is exact — match it order-independently
// (the oracle sorts by feature key afterward; only which points survive matters here).

interface GoldPoint {
    xA: [number, number, number];
    xB: [number, number, number];
}
interface GoldCase {
    name: string;
    axis: [number, number, number];
    com: [number, number, number];
    points: GoldPoint[];
    keep: number[];
}

const key = (p: Vec3): string => p.map((x) => x.toFixed(9)).join(",");

describe("contact-manifold reduction vs verbatim-Jolt gold", () => {
    for (const cfg of gold.cases as GoldCase[]) {
        test(cfg.name, () => {
            const cands: Cand[] = cfg.points.map((p, i) => ({
                feature: i,
                xA: p.xA as Vec3,
                xB: p.xB as Vec3,
            }));
            const sel = reduceManifold(cands, cfg.axis as Vec3, cfg.com as Vec3);

            // Jolt keeps 2-4; the oracle must keep the same count and the same physical points.
            expect(sel.length).toBe(cfg.keep.length);
            const got = new Set(sel.map((c) => key(c.xA)));
            const want = new Set(cfg.keep.map((i) => key(cfg.points[i].xA as Vec3)));
            expect(got).toEqual(want);
        });
    }
});

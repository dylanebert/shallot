// the per-plugin reload-conformance sweep (testing.md "Reload tier"): a live host that rebuilds a
// State (a rejected swap's fallback, a restore) re-runs every plugin's lifecycle against the SAME
// module-level singletons (ecs.md "Reload-safety"). The mechanism tests pin the core seams (ids,
// swap, rebuild, serialize); this harness pins each plugin's module-scope state — two identical
// build→step→dispose passes must produce the same observable State, so a registry that
// double-registers or a warm that doubles its derived spawns goes red here. The browser end-to-end
// is the survive-reload flow at examples/flows/survive-reload/ (`bun run flows`), which rebuilds through a real page
// reload; this roster is the sole per-plugin conformance coverage.
//
// This is the default-tier SENTINEL (`coding.md` Suite speed: a gate leaving the default suite
// leaves a cheap sentinel): the pipeline-compiling arms — Render, Part, Sear, Glaze, Lines, Sprite,
// Skin, Physics, Character, Player, both toggle arms, and the SkinPlugin+GltfPlugin pair — were
// promoted to the by-path tier `conformance.tier.ts` because their real GPU pipeline-compile cost
// (render arms measured 4.6–6.3 s, readings 5062–6279 ms) straddled the 5000 ms per-file cap
// (`tests/test-cap.ts`). The physics arms (Physics/Character/Player) compile the AVBD solver's
// compute pipeline set on the device — their headless cost is measured in
// `tests/avbd/headless.tier.ts`'s header. What stays here is the
// cheap half of the roster (Project, Mirror, Input, Slab+Transforms, Orbit, Animation) plus the seeded
// non-idempotent red arm — the sentinel that discriminates: a split can silently drop the population
// a red-capable arm needs, so the red arm stays to prove the harness still catches a violation.

import { describe, expect, test } from "bun:test";
import { type Plugin, sparse, u32 } from "../src";
import { conform, isPipelineCompiling, roster } from "./conformance-roster";

describe("reload conformance", () => {
    // the harness itself catches a violation: a module-level registry initialize fails to clear,
    // so the second build's warm derives one entity per accumulated entry — the doubling shape
    test("a seeded non-idempotent plugin goes red", async () => {
        const ledger: number[] = []; // module-scope registry, never cleared — the violation
        const Bad = { n: sparse(u32) };
        const BadPlugin: Plugin = {
            name: "bad",
            components: { Bad },
            initialize: () => {
                ledger.push(1);
            },
            warm: (s) => {
                for (const _ of ledger) s.add(s.create(), Bad);
            },
        };
        const violations = await conform({ plugins: [BadPlugin] });
        expect(violations.length).toBeGreaterThan(0);
        expect(violations.join("\n")).toContain("counts");
    });

    for (const [name, entry] of Object.entries(roster)) {
        if (isPipelineCompiling(entry)) continue;
        test(`${name} rebuilds idempotently`, async () => {
            expect(await conform(entry)).toEqual([]);
        });
    }
});

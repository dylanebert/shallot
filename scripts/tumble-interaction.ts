import { skipReason, teardownBridge, verify } from "./verify";

// `bun run tumble:interaction` — the standing interaction + visual gate for the tumble gym sample host
// (spec tumble-inline stage 5). A thin wrapper over the shipped gate exactly like bench:tumble / recipes /
// flows: it drives `shallot verify examples/gym --query scenario=<slug>` on a real device, reusing one
// bridge session, and reads the probe's Verdict checks.
//
// Two things the gold oracle can't cover, proven on a real device:
//   • a REAL pointer drag — genuine PointerEvents through the app's InputPlugin → the scenario's own grab —
//     moves a body (`grab-drag`), and a light held-still grab does not launch a settled stack
//     (`grab-no-launch`, the stage-3 grab-regression bound);
//   • the derived visual layer is present — the native solid layer materialized instanced Parts
//     (`visual-solids`) and the debug-draw walk produced joint gizmos (`visual-gizmos`);
//   • the per-sample overlay channel is present — a sample's `render()` demonstration layer put nonzero
//     labels ON-SCREEN (post-cull, `overlay-labels`, on events-joint-break's six threshold labels).
// Visuals and input never feed the gold oracle; these are additive checks on the same run.
//
//   ⚠ ONE bridge session at a time. On WSL the bridge is a SINGLE shared host browser; never run this
//   alongside another `bun bench`, `bun run bench:tumble`, `bun run flows`, or `bun run recipes`.

const GYM = "examples/gym";

// Enough warmup frames for the scene to settle before the probe drives it (the pyramid's no-launch bound
// needs a settled stack); a short measure window after — the probe, not the timing, is the signal here.
const WARMUP = 180;
const FRAMES = 6;

interface Probe {
    slug: string;
    /** the probe opts (`--query`) this run passes past the scenario/warmup/frames scaffolding. */
    query: string[];
    /** the checks (beyond the always-present `gold`) this run must surface and pass. */
    checks: string[];
}

// `visual-draws` is the RENDER-level guard (`tumble-probe.ts`): it reads the Part pack's own `drawArgs` back
// and asserts every distinct solid mesh actually DRAWS on the GPU at the gold framing. `visual-solids` proves
// the Part entities EXIST (CPU); a GPU-side drop (a cull regression, a degenerate bound, a NaN in the pack)
// leaves those entities untouched while nothing draws — the "floor disappears while every gate stays green"
// class. This runs on the operator's own device, so a hardware-specific drop this bridge can't reproduce
// still trips it there. The three probed scenes all frame their whole scene at gold defaults (verified: every
// solid mesh on-screen), so the drawing-pair count equals the distinct-mesh count.
const PROBES: Probe[] = [
    // a settled stack: a light held-still grab must not launch it, and a drag lifts a box clear.
    {
        slug: "stacking-box-pyramid",
        query: ["interact=drag", "draws=1"],
        checks: ["grab-no-launch", "grab-drag", "visual-draws"],
    },
    // a jointed body + gizmo layer: a drag moves a hinged link, and the joints render solids + gizmos.
    {
        slug: "joints-pendulum",
        query: ["interact=grab", "visual=1", "draws=1"],
        checks: ["grab-drag", "visual-solids", "visual-gizmos", "visual-draws"],
    },
    // the manual-pass scene (a solid vanishing under interaction): a grab drives the scene while the visual
    // invariants guard the layer. `visual-solids` pins the derivation TOTAL (Part entity per world solid);
    // `visual-draws` pins the GPU draw (every solid mesh actually rasterizes) — the render-level half the CPU
    // check can't see. grab-drag is NOT gated here: the bridge deck never settles (measured baseline
    // self-motion ~0.7 m > a drag's ~0.4 m), so the beat-baseline drag metric is unreliable on it — the grab
    // still runs to exercise interaction, but the visual invariants are the reliable guard for this scene.
    {
        slug: "joints-bridge",
        query: ["interact=drag", "visual=1", "draws=1"],
        checks: ["visual-solids", "visual-gizmos", "visual-draws"],
    },
    // the demonstration-layer scene: six boxes hang from rising-threshold breakable joints, each labelled
    // with its threshold ("cut" once broken) + a live load/broken HUD line — the `render()` overlay the
    // first port dropped. `overlay-labels` reads the overlay channel's own output back and asserts nonzero
    // ON-SCREEN (post-cull) labels — emit-count alone stays green on a broken projection. No `visual=1` (the joints break + vanish by the measure window, so `visual-gizmos`
    // is vacuously 0 here) and no grab-drag (the boxes fall + settle mid-run — an unreliable window); the
    // render-level `visual-draws` (every solid mesh rasterizes) + the overlay presence are the reliable guards.
    {
        slug: "events-joint-break",
        query: ["draws=1", "overlay=1"],
        checks: ["visual-draws", "overlay-labels"],
    },
];

async function runProbe(p: Probe): Promise<boolean> {
    console.log(`\n--- ${p.slug} (${p.query.join(", ")}) ---`);
    const query = [`scenario=${p.slug}`, `warmup=${WARMUP}`, `frames=${FRAMES}`, ...p.query];
    const result = await verify(GYM, [
        ...query.flatMap((q) => ["--query", q]),
        "--timeout",
        "120000",
    ]);
    // Gate on gold + the entry's OWN declared checks — each probe is self-describing about what it requires,
    // so a scene where an incidental check is unreliable (joints-bridge grab-drag) gates only on the checks
    // that matter for it. A null result / setup error is a hard fail (no scene ran).
    if (!result || result.error) {
        console.log(`  ✗ verify failed: ${result?.error ?? "no result"}`);
        return false;
    }
    let ok = true;
    // gold must still hold — a probe run also asserts the scenario's gold (visuals/input never feed it).
    const gold = result.verdict?.checks?.find((c) => c.name === "gold");
    if (!gold?.ok) {
        console.log(`  ✗ gold check missing or failed`);
        ok = false;
    }
    for (const name of p.checks) {
        const c = result?.verdict?.checks?.find((x) => x.name === name);
        if (!c?.ok) {
            console.log(`  ✗ ${name}: ${c?.detail ?? "missing"}`);
            ok = false;
        } else {
            console.log(`  ✓ ${name}: ${c.detail ?? ""}`);
        }
    }
    console.log(ok ? `PASS: ${p.slug}` : `FAIL: ${p.slug}`);
    return ok;
}

async function main(): Promise<void> {
    const args = process.argv.slice(2);
    if (args.includes("--help") || args.includes("-h")) {
        console.log(`Usage: bun run tumble:interaction [--only <slug>]

Drives the tumble gym host's interaction + visual probe through \`shallot verify\` on a real device,
reusing one bridge session. Display-gated (native hardware / the WSL host bridge). ONE bridge session at a
time — never run concurrent with another bench / bench:tumble / flows / recipes.

Options:
  --only <slug>   Run a single probe by its scenario slug (${PROBES.map((p) => p.slug).join(", ")})`);
        process.exit(0);
    }

    const skip = skipReason();
    if (skip) {
        console.log(`bun run tumble:interaction needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    const onlyIdx = args.indexOf("--only");
    const only = onlyIdx !== -1 ? args[onlyIdx + 1] : undefined;
    const list = only ? PROBES.filter((p) => p.slug === only) : PROBES;
    if (only && list.length === 0) {
        console.error(
            `no interaction probe "${only}" — one of: ${PROBES.map((p) => p.slug).join(", ")}`,
        );
        process.exit(2);
    }

    console.log(`Running ${list.length} tumble interaction probe(s) over one browser session...`);
    let allPass = true;
    try {
        for (const p of list) allPass = (await runProbe(p)) && allPass;
    } finally {
        await teardownBridge();
    }

    if (!allPass) {
        console.error("\nFAIL: tumble interaction gate red");
        process.exit(1);
    }
    console.log("\nPASS: tumble interaction gate green");
    process.exit(0);
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});

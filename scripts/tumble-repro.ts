import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { skipReason } from "./verify";
import { type Bridge, start as startBridge } from "./wsl-bridge";

// `bun run tumble:repro` — the trusted-input floor-vanish repro + standing gate (spec tumble-inline stage 6b).
// A thin bun orchestrator over the node driver (scripts/tumble-repro-driver.mjs): it owns the WSL→Windows
// bridge lifecycle exactly like bench:tumble, then spawns the driver under NODE (Bun's Playwright client
// hangs on the bridge — wsl-bridge.ts fact 2) with the bridge's `--connect` ws endpoint. The driver boots the
// gym vite server, connects to the host's real-GPU browser, and drives a bridge plank with browser-trusted
// `page.mouse` flicks (one-frame cursor jumps) until any draw pair's drawn count drops below its derivation
// count or a pose/transform goes non-finite — the loss the dispatched-event probe never reproduced.
//
// Two modes:
//   • default — the F1 diagnostic: escalating drag violence, first-break report (exits 0 either way).
//   • --gate  — the F3 STANDING GATE: two real-device asserts over one bridge session. (1) a trusted-input
//     violent reversing whip must fling NO pair out of frustum under the grab-energy cap (tumble-grab.ts
//     driveGrab) — a smoke; the DETERMINISTIC cap red→green + regression is the headless whip-cap unit test
//     (examples/gym/src/tumble-pilot.test.ts, in `bun run test:gym`). (2) `--inject far` must STILL drop
//     pairs (it bypasses the grab entirely, proving the frustum-cull detector still detects — the
//     discriminating half). Exits nonzero if either fails.
//
//   ⚠ ONE bridge session at a time. On WSL the bridge is a SINGLE shared host browser (scripts/wsl-bridge.ts);
//   never run this alongside `bun bench`, `bun run bench:tumble`, `bun run tumble:interaction`, `bun run
//   flows`, or `bun run recipes`.

const isWSL = process.platform === "linux" && existsSync("/proc/sys/fs/binfmt_misc/WSLInterop");
const DRIVER = resolve(import.meta.dir, "tumble-repro-driver.mjs");

interface Snapshot {
    drawing: number;
    meshes: number;
    bodyCount?: number;
    partCount?: number;
    nonFinite: { source: string; index: number; values: number[] }[];
}
interface Attempt {
    level: string;
    magnitude: number;
    aim: { clientX: number; clientY: number } | null;
    bursts: { j: number; target: number[]; snap: Snapshot | null }[];
    broke: boolean;
}
interface DriverResult {
    scenario?: string;
    connect?: boolean;
    dpr?: number;
    hardware?: string;
    baseline?: Snapshot;
    attempts?: Attempt[];
    reproduced: boolean;
    breach?: {
        kind: string;
        drawing: number;
        meshes: number;
        nonFinite: Snapshot["nonFinite"];
    } | null;
    firstBreakFrame?: number | null;
    deep?: {
        frustumFinite: boolean;
        frustumExtreme: number;
        pairs: { pair: number; mesh: string; gpuCount: number; cpuVisible: number }[];
        flung: { eid: number; mesh: string; pos: number[] }[];
        mismatch: { pair: number; mesh: string; gpuCount: number; cpuVisible: number }[];
    } | null;
    watcherDump?: unknown;
    inject?: {
        requested: string;
        injected: boolean;
        dump: unknown;
        deep?: DriverResult["deep"];
        screenshot: string | null;
        snapshot?: Snapshot | null;
    };
    recipe?: RecipeResult;
    artifacts?: (string | null)[];
    pageErrors?: string[];
    error?: string;
}

// F1′ — the sustained-downward-drag pixel-level repro (one driver run = one parameter row).
interface RecipeResult {
    speed: string;
    holdMs: number;
    depth: string;
    pitch: string;
    fired: boolean;
    cam0?: { pitch: number; yaw: number; distance: number };
    minAnchorY?: number | null;
    anchorWentBelowGround?: boolean;
    samples?: unknown[];
    breach?: {
        surface: string;
        kind: string;
        patches: number;
        drawArgsGreen: boolean;
        drawing?: number;
        meshes?: number;
        nonFinite: number;
        refLum: number[];
        atLum: number[];
        anchor?: number[];
        anchorBelowGround?: boolean | null;
        plank?: number[];
        atShot?: string | null;
    };
    recovery?: { breached: boolean }[];
}

async function runDriver(connect: string, extra: string[]): Promise<DriverResult | null> {
    const outDir = resolve(import.meta.dir, "..", "node_modules/.cache/tumble-repro");
    const args = [
        "node",
        DRIVER,
        "--out",
        outDir,
        ...(connect ? ["--connect", connect] : []),
        ...extra,
    ];
    const proc = Bun.spawn(args, {
        cwd: resolve(import.meta.dir, ".."),
        stdout: "pipe",
        stderr: "inherit",
    });
    const stdout = await new Response(proc.stdout).text();
    await proc.exited;
    // the driver prints exactly one JSON line on stdout (its diagnostics go to stderr, which we inherit).
    let parsed: DriverResult | null = null;
    for (const line of stdout.split("\n")) {
        const t = line.trim();
        if (!t.startsWith("{")) continue;
        try {
            const o = JSON.parse(t);
            if (o && typeof o.reproduced === "boolean") parsed = o as DriverResult;
        } catch {}
    }
    return parsed;
}

function report(r: DriverResult): void {
    const bar = "=".repeat(64);
    console.log(`\n${bar}`);
    console.log(`  tumble floor-vanish repro — joints-bridge, trusted page.mouse input`);
    console.log(bar);
    console.log(`  hardware: ${r.hardware ?? "unknown"}   dpr: ${r.dpr}   connect: ${r.connect}`);
    if (r.inject) {
        const fired =
            !!(r.inject.dump as { frame?: number } | null)?.frame != null && !!r.inject.dump;
        console.log(bar);
        console.log(
            `  inject=${r.inject.requested}: injected=${r.inject.injected}, watcher ${fired ? "FIRED" : "did NOT fire"}`,
        );
        if (r.inject.dump) console.log(`      dump: ${JSON.stringify(r.inject.dump)}`);
        const d = r.inject.deep;
        if (d) {
            console.log(
                `  LAYER BISECTION — frustumFinite: ${d.frustumFinite}  mismatch (render-boundary loss): ${d.mismatch.length}  flung: ${d.flung.length}`,
            );
            console.log(`  per-pair (slot 0) after inject:`);
            for (const p of d.pairs) {
                const flag = p.gpuCount < p.cpuVisible ? "  ← LOSS" : "";
                console.log(
                    `      ${p.mesh.padEnd(14)} gpuCount ${String(p.gpuCount).padStart(4)}  cpuVisible ${String(p.cpuVisible).padStart(4)}${flag}`,
                );
            }
        }
        if (r.inject.screenshot) console.log(`      screenshot: ${r.inject.screenshot}`);
        console.log(bar + "\n");
        return;
    }
    if (r.baseline) {
        console.log(
            `  baseline: ${r.baseline.drawing} drawing / ${r.baseline.meshes} mesh pairs · ${r.baseline.nonFinite.length} non-finite`,
        );
    }
    for (const a of r.attempts ?? []) {
        const verdict = a.broke ? "BROKE" : a.aim ? "clean" : "no-aim";
        const last = a.bursts.at(-1)?.snap;
        const tail = last
            ? ` (last: ${last.drawing}/${last.meshes} draw, ${last.nonFinite.length} nf)`
            : "";
        console.log(
            `  ${a.level.padEnd(26)} mag ${a.magnitude.toFixed(2)}  ${a.bursts.length} bursts → ${verdict}${tail}`,
        );
    }
    console.log(bar);
    if (r.reproduced && r.breach) {
        console.log(
            `  *** REPRODUCED: ${r.breach.kind} — ${r.breach.drawing} of ${r.breach.meshes} pairs drawing`,
        );
        if (r.breach.nonFinite?.length) {
            console.log(`      non-finite: ${JSON.stringify(r.breach.nonFinite.slice(0, 4))}`);
        }
        if (r.firstBreakFrame != null) console.log(`      first break frame: ${r.firstBreakFrame}`);
        if (r.deep) {
            const d = r.deep;
            console.log(bar);
            console.log(
                `  LAYER BISECTION — frustumFinite: ${d.frustumFinite}  frustumExtreme: ${d.frustumExtreme.toExponential(2)}`,
            );
            console.log(
                `  mismatch (GPU dropped what CPU says visible = render-boundary loss): ${d.mismatch.length}`,
            );
            for (const m of d.mismatch) {
                console.log(
                    `      ${m.mesh.padEnd(12)} gpuCount ${m.gpuCount}  cpuVisible ${m.cpuVisible}`,
                );
            }
            console.log(`  per-pair (slot 0):`);
            for (const p of d.pairs) {
                const flag = p.gpuCount < p.cpuVisible ? "  ← LOSS" : "";
                console.log(
                    `      ${p.mesh.padEnd(12)} gpuCount ${String(p.gpuCount).padStart(4)}  cpuVisible ${String(p.cpuVisible).padStart(4)}${flag}`,
                );
            }
            if (d.flung.length)
                console.log(`  flung bodies (|pos| > 50): ${JSON.stringify(d.flung.slice(0, 8))}`);
        }
    } else {
        console.log(
            `  not reproduced — the trusted-input escalation did not break the invariant this run`,
        );
    }
    if (r.artifacts?.length) {
        console.log(`  artifacts:`);
        for (const p of r.artifacts) if (p) console.log(`    ${p}`);
    }
    if (r.pageErrors?.length) {
        console.log(`  page errors:`);
        for (const e of r.pageErrors) console.log(`    ${e.split("\n")[0]}`);
    }
    console.log(bar + "\n");
}

// The standing gate (spec 6b/F3): the two asserts over one bridge session. Owns the bridge lifecycle, runs the
// driver twice, prints a verdict per check, exits 0 only if both hold. See the header + `--help`.
async function runGate(): Promise<void> {
    const skip = skipReason();
    if (skip) {
        console.log(`bun run tumble:repro --gate needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }
    const bar = "=".repeat(64);
    let bridge: Bridge | null = null;
    let whip: DriverResult | null = null;
    let far: DriverResult | null = null;
    try {
        const connect = isWSL ? (bridge = await startBridge()).connectUrl : "";
        whip = await runDriver(connect, ["--gate"]); // assert: no fling (the grab-energy cap holds)
        far = await runDriver(connect, ["--inject", "far"]); // assert: the frustum-cull detector still fires
    } finally {
        if (bridge) await bridge.teardown().catch(() => {});
    }

    // (1) the trusted-input whip must not fling any pair out of frustum.
    const whipOk = !!whip && !whip.error && whip.reproduced === false;
    // (2) --inject far bypasses the grab, so its multi-pair loss must persist: some pairs culled (drawing <
    //     meshes), others surviving (drawing > 0) — the detector proving it still detects.
    const snap = far?.inject?.snapshot ?? null;
    const farOk = !!far && !far.error && !!snap && snap.drawing > 0 && snap.drawing < snap.meshes;

    console.log(`\n${bar}`);
    console.log(`  tumble floor-vanish standing gate — joints-bridge, trusted page.mouse input`);
    console.log(bar);
    console.log(`  hardware: ${whip?.hardware ?? far?.hardware ?? "unknown"}`);
    console.log(
        `  ${whipOk ? "✓" : "✗"} no-fling: trusted whip flick-out/back kept every pair drawing` +
            (whip?.reproduced
                ? ` — FLUNG (${whip.breach?.drawing}/${whip.breach?.meshes} pairs)`
                : ""),
    );
    console.log(
        `  ${farOk ? "✓" : "✗"} detector: --inject far still drops pairs` +
            (snap ? ` (${snap.drawing} of ${snap.meshes} drawing)` : " — no snapshot"),
    );
    console.log(bar + "\n");

    if (whip?.error) console.error(`whip driver error: ${whip.error}`);
    if (far?.error) console.error(`inject-far driver error: ${far.error}`);
    if (!whipOk || !farOk) {
        console.error("FAIL: tumble floor-vanish gate red");
        process.exit(1);
    }
    console.log("PASS: tumble floor-vanish gate green");
    process.exit(0);
}

// The F1′ recipe sweep (spec 6b/F1′): owns one bridge session and runs the user's exact gesture — a sustained
// slow downward drag of a central plank carrying the grab handle below the ground plane, held there — across a
// parameter table (drag speed × hold duration × below-edge depth × camera pitch), asserting at the PIXEL layer.
// Stops on the first pixel breach (whole static surface goes black), reporting the key bit: pixels broken WITH
// drawArgs green (render-side corruption) vs drawArgs dropped (cull layer). Exits 0 — this is a diagnostic
// repro, not a gate.
async function runRecipeSweep(): Promise<void> {
    const skip = skipReason();
    if (skip) {
        console.log(
            `bun run tumble:repro --recipe drag-below needs native hardware (${skip}). Skipping.`,
        );
        process.exit(0);
    }
    const table = [
        { speed: "slow", hold: 3500, depth: "below", pitch: "default" },
        { speed: "med", hold: 2500, depth: "below", pitch: "default" },
        { speed: "slow", hold: 4000, depth: "below", pitch: "steep" },
        { speed: "med", hold: 2500, depth: "bottom", pitch: "steep" },
        { speed: "slow", hold: 3000, depth: "bottom", pitch: "default" },
    ];
    const bar = "=".repeat(78);
    let bridge: Bridge | null = null;
    const rows: { row: (typeof table)[number]; r: DriverResult | null }[] = [];
    try {
        const connect = isWSL ? (bridge = await startBridge()).connectUrl : "";
        for (const row of table) {
            console.error(`\n[recipe] attempt ${JSON.stringify(row)}`);
            const r = await runDriver(connect, [
                "--recipe",
                "drag-below",
                "--speed",
                row.speed,
                "--hold",
                String(row.hold),
                "--depth",
                row.depth,
                "--pitch",
                row.pitch,
            ]);
            rows.push({ row, r });
            if (r?.recipe?.fired) break; // first pixel breach — stop and report
        }
    } finally {
        if (bridge) await bridge.teardown().catch(() => {});
    }

    console.log(`\n${bar}`);
    console.log(
        `  tumble floor-vanish F1′ — sustained downward central-plank drag, PIXEL-level assert`,
    );
    console.log(bar);
    console.log(`  hardware: ${rows.find((x) => x.r?.hardware)?.r?.hardware ?? "unknown"}`);
    console.log(
        `  ${"speed".padEnd(6)} ${"pitch".padEnd(8)} ${"depth".padEnd(7)} ${"hold".padEnd(6)} ${"minAnchorY".padEnd(11)} ${"belowGnd".padEnd(9)} verdict`,
    );
    for (const { row, r } of rows) {
        const rec = r?.recipe;
        const verdict = r?.error
            ? `ERROR: ${r.error}`
            : rec?.fired
              ? `*** PIXEL BREACH (${rec.breach?.surface} ${rec.breach?.kind}, drawArgsGreen=${rec.breach?.drawArgsGreen})`
              : "no breach";
        const minY = rec?.minAnchorY != null ? rec.minAnchorY.toFixed(2) : "—";
        console.log(
            `  ${row.speed.padEnd(6)} ${row.pitch.padEnd(8)} ${row.depth.padEnd(7)} ${String(row.hold).padEnd(6)} ${minY.padEnd(11)} ${String(rec?.anchorWentBelowGround ?? "—").padEnd(9)} ${verdict}`,
        );
    }
    console.log(bar);

    const hit = rows.find((x) => x.r?.recipe?.fired);
    if (hit?.r?.recipe?.breach) {
        const b = hit.r.recipe.breach;
        console.log(
            `  *** REPRODUCED — pixel breach on the ${b.surface} surface (${b.kind}, ${b.patches} patches)`,
        );
        console.log(
            `  THE KEY BIT: drawArgs ${b.drawArgsGreen ? "GREEN — render-side corruption" : "DROPPED — cull/count layer"}`,
        );
        console.log(`      drawing ${b.drawing}/${b.meshes} pairs, ${b.nonFinite} non-finite`);
        console.log(`      refLum ${JSON.stringify(b.refLum)} → atLum ${JSON.stringify(b.atLum)}`);
        console.log(
            `      grab anchor ${JSON.stringify(b.anchor?.map((v) => Number(v.toFixed(2))))} belowGround=${b.anchorBelowGround}`,
        );
        console.log(`      plank ${JSON.stringify(b.plank?.map((v) => Number(v.toFixed(2))))}`);
        const recov = hit.r.recipe.recovery;
        if (recov)
            console.log(
                `      recovery after release: ${recov.map((x) => (x.breached ? "×" : "✓")).join("")}`,
            );
    } else {
        console.log(`  not reproduced across ${rows.length} parameter rows`);
        const below = rows.filter((x) => x.r?.recipe?.anchorWentBelowGround);
        console.log(
            `      ${below.length}/${rows.length} attempts carried the grab handle below the ground plane`,
        );
    }
    for (const { r } of rows) {
        if (r?.artifacts?.length)
            for (const p of r.artifacts) if (p) console.log(`      artifact: ${p}`);
    }
    console.log(bar + "\n");
    process.exit(0);
}

async function main(): Promise<void> {
    const argv = process.argv.slice(2);
    if (argv.includes("--help") || argv.includes("-h")) {
        console.log(`Usage: bun run tumble:repro [--gate] [--recipe drag-below] [--inject nan|inf|far] [--dpr <n>]

Drives joints-bridge with browser-trusted page.mouse flicks over the WSL→host real-GPU bridge. Display-gated
(native hardware / the WSL host bridge). ONE bridge session at a time — never run concurrent with another
bench / bench:tumble / tumble:interaction / flows / recipes.

Modes:
  (default)          Diagnostic escalation (F1): five levels of growing drag violence, first-break report.
  --recipe drag-below  The F1′ pixel-level repro: the user's exact gesture — a sustained slow downward drag of
                     a CENTRAL plank carrying the grab handle below the ground plane, held there, then continued
                     — swept across drag speed × hold × depth × camera pitch over one bridge session. Asserts at
                     the PIXEL layer (reference patches over the static ground + end posts), because a shadow /
                     tonemap black-out keeps every drawArgs count intact. Reports the key bit on a breach: pixels
                     broken WITH drawArgs green (render-side corruption) vs drawArgs dropped (the cull layer).
  --gate             STANDING GATE (spec 6b/F3): two real-device asserts over one bridge session —
                     (1) a trusted-input violent reversing whip must NOT fling any pair out of frustum under
                         the grab-energy cap (a smoke; the deterministic cap red→green is the headless
                         whip-cap unit test in \`bun run test:gym\`); (2) --inject far must STILL produce the
                         multi-pair loss (it bypasses the grab, proving the frustum-cull detector still
                         detects — the discriminating half). Exits nonzero if either fails.

Options:
  --inject nan|inf   Red-first self-test: poison a world body pose and prove the auto-dump watcher fires.
  --inject far       Synthetic F2 proof: displace every dynamic body to a finite far position and show the
                     traced mechanism — a correct multi-pair frustum cull (mismatch 0), not a poison.
  --dpr <n>          Device pixel ratio for the browser context (default 1.5).`);
        process.exit(0);
    }

    if (argv.includes("--gate")) return runGate();
    if (argv.includes("--recipe")) return runRecipeSweep();

    const skip = skipReason();
    if (skip) {
        console.log(`bun run tumble:repro needs native hardware (${skip}). Skipping.`);
        process.exit(0);
    }

    const extra: string[] = [];
    const injectIdx = argv.indexOf("--inject");
    if (injectIdx !== -1 && argv[injectIdx + 1]) extra.push("--inject", argv[injectIdx + 1]);
    const dprIdx = argv.indexOf("--dpr");
    if (dprIdx !== -1 && argv[dprIdx + 1]) extra.push("--dpr", argv[dprIdx + 1]);

    let bridge: Bridge | null = null;
    let result: DriverResult | null = null;
    try {
        const connect = isWSL ? (bridge = await startBridge()).connectUrl : "";
        result = await runDriver(connect, extra);
    } finally {
        if (bridge) await bridge.teardown().catch(() => {});
    }

    if (!result) {
        console.error("\nrepro driver produced no result (crashed before reporting)");
        process.exit(1);
    }
    report(result);
    if (result.error) {
        console.error(`driver error: ${result.error}`);
        process.exit(1);
    }
    process.exit(0);
}

main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
});

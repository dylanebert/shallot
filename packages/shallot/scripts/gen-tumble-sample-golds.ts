// PROVENANCE RECORD — the mint ran once and can't run again: tumble.js is retired and its checkout is
// gone, so the committed golds at tests/tumble/samples/ are frozen truth (see that directory's README).
// Kept as the recipe record; without the source checkout it errors honestly below.
//
// Mints the per-sample gold trajectories that verify a ported physics scenario reproduces its source
// sample bit-exact. The tumble.js sample corpus (../tumble.js/samples) is the reference implementation:
// each sample subclasses `Sample`, builds a world in build(), and steps headless on bun with no GPU.
// This runs every registered sample at knob defaults, single-threaded, and records for each:
//   - the initial body-state snapshot (transform + velocity, in the hash's id order),
//   - the per-step world-state hash for STEPS steps (the engine's own `hashWorldState`), and
//   - metadata: camera pose (framed AABB → orbit pose) and the declarative knob schema.
//
// The sample SOURCE comes from tumble.js, but the hashes are produced by SHALLOT's inlined engine
// (`src/standard/tumble/engine`, not tumble.js's own copy): a Bun resolver aliases the samples'
// `import ... from "tumble.js"` to the shipping engine barrel. A gym scenario (spec stage 3+) that
// reproduces build() through `Tumble.world` runs the same engine, so it reproduces these hashes
// bit-exact — any authoring divergence (wrong axis, wrong joint, wrong shape) mismatches at the first
// divergent step.
//
// Each sample mints in its OWN child process (a fresh wasm kernel per gold). Two reasons: the kernel is
// a process-wide singleton whose grow-only resident regions carry a high-water across sequential
// destroy/create, which eventually traps mid-run (queryPairs, ~7 worlds in); and a pristine kernel is
// exactly what a single-scenario gym replay gets, so the gold matches that by construction. Hashes are
// reuse-invariant regardless (verified identical in-process vs isolated) — isolation removes the trap,
// not a nondeterminism.
//
// Output lands in tests/tumble/samples/ (one <slug>.json per sample + index.json + README). The output
// is deterministic — no timestamps — so a double mint is byte-identical. Absent the corpus (a plain
// shallot checkout with no kex workspace around it), it errors honestly, mirroring gen-tumble-fixtures.
//
// Usage: bun run scripts/gen-tumble-sample-golds.ts   (from packages/shallot)

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { plugin } from "bun";

const STEPS = 600; // 10 s at 60 Hz — long enough to pin behavior phases that fire late: a breakable
// joint's break, a chaotic divergence (the intermediate-axis flip), a settling stack, a motor reversal.
// A short horizon only pins the opening transient, leaving anything past it unverified. The hash is
// bit-exact every step, so a static authoring error (mass/shape/pose) already diverges at step 0; the
// horizon exists for the dynamics-driven errors that surface only once the scene has run for a while.
// 41 × 600 hex hashes still stay trivially committable.

const pkgRoot = resolve(import.meta.dir, "..");
const shallotRoot = resolve(pkgRoot, "..", "..");
const tumbleRoot = resolve(shallotRoot, "..", "tumble.js");
const samplesDir = resolve(tumbleRoot, "samples");
const sampleBase = resolve(samplesDir, "src", "sample.ts");
const sampleIndex = resolve(samplesDir, "src", "samples", "index.ts");
const enginePath = resolve(pkgRoot, "src", "standard", "tumble", "engine", "index.ts");
const bodyPath = resolve(pkgRoot, "src", "standard", "tumble", "engine", "body.ts");
const outDir = resolve(pkgRoot, "tests", "tumble", "samples");

if (!existsSync(sampleIndex)) {
    console.error(`tumble.js sample corpus missing: ${sampleIndex}`);
    console.error(
        "expected the tumble.js checkout at ../tumble.js beside the shallot repo (the kex workspace layout: kex/tumble.js, sibling of kex/shallot). It is the sample corpus this mint reads.",
    );
    process.exit(1);
}

// Route the samples' `import "tumble.js"` to the shipping engine barrel, so the golds are minted with
// the exact engine a gym scenario reproduces them against (a virtual re-export module — Bun's runtime
// onResolve mangles a bare file path into a bad file: URL, so redirect through a namespace instead).
plugin({
    name: "tumble-engine-alias",
    setup(build) {
        build.onResolve({ filter: /^tumble\.js$/ }, () => ({
            path: "virtual:tumble-engine",
            namespace: "tumble-alias",
        }));
        build.onLoad({ filter: /.*/, namespace: "tumble-alias" }, () => ({
            contents: `export * from ${JSON.stringify(enginePath)};`,
            loader: "ts",
        }));
    },
});

// Samples that can't mint a gold, by sample name → reason. Recorded in index.json + the README rather
// than worked around (the source is read-only). `Benchmark / Rain` streams ragdoll columns onto a
// static torus/grid mesh field; the scalar engine hangs inside a single step once the second column
// lands (~step 40), well short of any horizon this mint could choose. It is a throughput benchmark,
// not an authoring-correctness scene, so its absence doesn't weaken the oracle.
const SKIP: Record<string, string> = {
    Rain: "engine hangs inside a step once a second ragdoll column lands on the mesh field (~step 40), short of any horizon. Throughput benchmark, not an authoring scene.",
};

function slugify(category: string, name: string): string {
    return `${category}-${name}`
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "-")
        .replace(/^-+|-+$/g, "");
}

const { defaultContext, sampleEntries } = await import(sampleBase);
const { registerAllSamples } = await import(sampleIndex);
const { getBodySim, getBodyState } = await import(bodyPath);
registerAllSamples();
const entries = sampleEntries();

const workerFlag = process.argv.indexOf("--sample");
if (workerFlag !== -1) {
    await mintOne(Number(process.argv[workerFlag + 1]));
} else {
    await drive();
}

// --- worker: mint one sample into its own file, in a pristine kernel ------------------------------

async function mintOne(idx: number): Promise<void> {
    const { hashWorldState, init, threads } = await import(enginePath);
    await init({ threads: 0 }); // single-thread; the gold contract is thread-count-independent
    if (threads() !== 1) {
        console.error(`expected single-thread mint, got threads()=${threads()}`);
        process.exit(1);
    }

    const entry = entries[idx];
    const ctx = defaultContext();
    const sample = entry.create(ctx); // create() also calls reset() → build() → frames the camera
    const cam = ctx.camera;

    const gold = {
        slug: slugify(entry.category, entry.name),
        category: entry.category,
        name: entry.name,
        description: entry.description,
        timeStep: ctx.hertz > 0 ? 1 / ctx.hertz : 0,
        subStepCount: ctx.subStepCount,
        gravity: [ctx.gravity.x, ctx.gravity.y, ctx.gravity.z],
        enableSleep: ctx.enableSleep,
        enableContinuous: ctx.enableContinuous,
        stepCount: STEPS,
        // Orbit pose the sample framed to its scene AABB (build → Sample.frame → Camera.frame). These
        // targets fully determine the pose; the gym host frames its camera from them.
        camera: {
            pivot: [cam.pivot.x, cam.pivot.y, cam.pivot.z],
            yaw: cam.yaw,
            pitch: cam.pitch,
            radius: cam.radius,
            fov: cam.fov,
            near: cam.near,
            far: cam.far,
        },
        knobs: sample.knobs(),
        bodyCount: liveBodies(sample.world.state).length,
        initial: dumpBodies(sample.world.state),
        hashes: [] as string[],
    };
    for (let i = 0; i < STEPS; ++i) {
        sample.step();
        gold.hashes.push(toHex(hashWorldState(sample.world.state)));
    }
    writeFileSync(resolve(outDir, `${gold.slug}.json`), `${JSON.stringify(gold, null, 2)}\n`);
    sample.destroy();
}

// --- driver: enumerate, spawn one worker per sample, write the index + README ---------------------

async function drive(): Promise<void> {
    mkdirSync(outDir, { recursive: true });
    // Drop prior gold JSON first so a dropped or re-slugged sample leaves no orphan.
    for (const f of readdirSync(outDir)) {
        if (f.endsWith(".json")) rmSync(resolve(outDir, f));
    }

    const index: { slug: string; category: string; name: string; description: string }[] = [];
    const exceptions: { category: string; name: string; reason: string }[] = [];
    for (let i = 0; i < entries.length; ++i) {
        const e = entries[i];
        if (e.name in SKIP) {
            exceptions.push({ category: e.category, name: e.name, reason: SKIP[e.name] });
            continue;
        }
        const slug = slugify(e.category, e.name);
        const r = spawnSync("bun", ["run", import.meta.path, "--sample", String(i)], {
            cwd: pkgRoot,
            stdio: ["ignore", "ignore", "inherit"],
        });
        if (r.status !== 0 || !existsSync(resolve(outDir, `${slug}.json`))) {
            console.error(`[gen-tumble-sample-golds] sample ${i} (${e.category}/${e.name}) failed`);
            process.exit(r.status ?? 1);
        }
        index.push({ slug, category: e.category, name: e.name, description: e.description });
    }
    if (index.length + exceptions.length !== entries.length) {
        console.error(
            `[gen-tumble-sample-golds] accounted ${index.length} golds + ${exceptions.length} exceptions != ${entries.length} registered`,
        );
        process.exit(1);
    }

    const tumbleSha =
        spawnSync("git", ["rev-parse", "HEAD"], {
            cwd: tumbleRoot,
            encoding: "utf8",
        }).stdout?.trim() ?? "unknown";
    writeFileSync(
        resolve(outDir, "index.json"),
        `${JSON.stringify(
            {
                source: "tumble.js/samples",
                sourceCommit: tumbleSha,
                engine: "shallot src/standard/tumble/engine",
                stepCount: STEPS,
                registeredCount: entries.length,
                goldCount: index.length,
                samples: index,
                exceptions,
            },
            null,
            2,
        )}\n`,
    );
    writeFileSync(
        resolve(outDir, "README.md"),
        readme(tumbleSha, index.length, entries.length, exceptions),
    );
    console.log(
        `[gen-tumble-sample-golds] minted ${index.length} golds (+${exceptions.length} exceptions) of ${entries.length} registered -> ${outDir}`,
    );
}

// --- helpers --------------------------------------------------------------------------------------

function toHex(h: bigint): string {
    return `0x${h.toString(16).padStart(16, "0")}`;
}

// The engine's WorldState / Body are module-internal; this dev script types them structurally as any.
function liveBodies(state: any): any[] {
    return state.bodies.filter((b: any, i: number) => b.id === i);
}

type BodyDump = { p: number[]; q: number[]; v?: number[]; w?: number[] };

// Every live body's transform + velocity in the id order hashWorldState walks — the same shape the C
// fixtures' state dumps use, so an initial-construction divergence reads directly against the gold.
function dumpBodies(state: any): BodyDump[] {
    const out: BodyDump[] = [];
    for (const body of liveBodies(state)) {
        const sim = getBodySim(state, body);
        const dump: BodyDump = {
            p: [sim.transform.p.x, sim.transform.p.y, sim.transform.p.z],
            q: [sim.transform.q.v.x, sim.transform.q.v.y, sim.transform.q.v.z, sim.transform.q.s],
        };
        const bs = getBodyState(state, body);
        if (bs !== null) {
            dump.v = [bs.linearVelocity.x, bs.linearVelocity.y, bs.linearVelocity.z];
            dump.w = [bs.angularVelocity.x, bs.angularVelocity.y, bs.angularVelocity.z];
        }
        out.push(dump);
    }
    return out;
}

function readme(
    tumbleSha: string,
    count: number,
    registered: number,
    exceptions: { category: string; name: string; reason: string }[],
): string {
    const exceptionRows =
        exceptions.length === 0
            ? "None.\n"
            : `${exceptions.map((e) => `- **${e.category} / ${e.name}** — ${e.reason}`).join("\n")}\n`;
    return `# tumble sample golds — source-faithful physics oracle

The ${count} \`*.json\` files here are per-sample **gold trajectories**: the ground truth a ported gym
scenario must reproduce bit-exact. Each is one tumble.js sample run headless at knob defaults, recording
the initial body snapshot, the per-step world-state hash, and the sample's camera pose + knob schema.
The corpus registers ${registered} samples; ${count} mint a gold, ${exceptions.length} are excepted (below).

A gym scenario (spec \`tumble-inline\` stage 3+) reproduces a sample's \`build()\` through the escape hatch
(\`Tumble.world\`) and replays it against the same engine these golds were minted with. Both sides are the
same engine, so only authoring can differ — a wrong axis, wrong joint, or wrong shape mismatches the hash
at the first divergent step. That is the oracle the earlier examples port lacked.

## Provenance

- **Sample corpus:** \`tumble.js/samples\` at commit \`${tumbleSha}\` — the reference implementation.
- **Minting engine:** shallot's inlined \`src/standard/tumble/engine\` (box3d pin \`29bf523\`), **not**
  tumble.js's own engine copy. The samples' \`import "tumble.js"\` is aliased to the shipping engine barrel
  during the mint, so the hashes are exactly what a gym replay against \`Tumble.world\` produces.
- **Hash:** the engine's own \`hashWorldState\` (FNV-1a over every live body's transform + velocity), the
  same function the C fixtures compare against — so a ported scenario compares identically on both sides.

## Mint recipe

\`\`\`
bun run scripts/gen-tumble-sample-golds.ts     # from packages/shallot
\`\`\`

- **Defaults** (\`defaultContext()\`): 60 Hz (timeStep 1/60), 4 substeps, gravity (0, -10, 0), sleep on,
  continuous on, every sample at its declared knob defaults — no interaction (no mouse grab).
- **Threads:** single-thread (\`init({ threads: 0 })\`). The gold contract is thread-count-independent by
  construction; a gym replay at any thread count reproduces it.
- **Step count:** ${STEPS} steps (10 s at 60 Hz). The hash is bit-exact every step, so a static authoring
  error diverges at step 0; the horizon is what pins the behavior phases that fire late — a break
  threshold, a motor reversal, a settling stack, a chaotic divergence — while keeping the golds trivially
  committable.
- **Isolation:** one child process per sample — a fresh wasm kernel per gold. The kernel is a
  process-wide singleton whose grow-only regions carry a high-water across sequential worlds and trap
  after several; a pristine kernel is also what a single-scenario gym replay gets. Output is
  timestamp-free, so a double mint is byte-identical.

Regenerating requires the tumble.js checkout at \`../tumble.js\` (the kex workspace layout). Absent it, the
script errors honestly. These golds are committed, so an outside shallot checkout needs neither the corpus
nor a mint to run the ported scenarios' oracle.

## Exceptions

Samples that can't mint a gold (recorded, not worked around — the source is read-only):

${exceptionRows}
## Layout

- \`<slug>.json\` — one per minted sample: \`{ slug, category, name, description, timeStep, subStepCount,
  gravity, enableSleep, enableContinuous, stepCount, camera, knobs, bodyCount, initial, hashes }\`.
- \`index.json\` — the manifest: source commit, engine, step count, \`registeredCount\` / \`goldCount\`, every
  minted sample's slug + category + name + description (registration order), and \`exceptions\`.
`;
}

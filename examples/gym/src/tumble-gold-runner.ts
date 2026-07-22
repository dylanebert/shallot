// One gold-test job — a single tumble world — run in an isolated child process (spec tumble-inline stage
// 4). The spawner (`tumble-golds.test.ts`) launches this once per job, `bun run tumble-gold-runner.ts
// --slug <slug> [--knob <index>]`, ~4 at a time:
//   --slug X              replay entry X's gold bit-exact vs its committed trajectory
//   --slug X --knob N     step entry X at knob point N, asserting finite positions (boundedness only —
//                         the gold trajectory exists at defaults, so a knob run is bounded-checked, never
//                         gold-checked)
//
// One world per process is correctness, not tidiness. The tumble kernel is a process-wide wasm singleton
// whose grow-only regions carry a high-water across sequential worlds; after several rich worlds in one
// process `queryPairs` traps `unreachable`, and below that threshold stepping can silently lose determinism
// (a gold that matches in isolation diverges given a different set of worlds ahead of it) — spec Residue
// "sequential-world kernel trap". Even a single entry's oracle + its 2-3 knob probes exceeded the trap for
// shape-soup (4 rich worlds), so each is its own child; a fresh kernel per world is exactly what the gold
// mint used (`scripts/gen-tumble-sample-golds.ts`), making registry order irrelevant to every verdict.
//
// On any failure — including an engine trap — this prints one diagnostic line naming the entry, the check,
// and (for a gold divergence) the first divergent step + got/expected, then exits nonzero; that line is all
// a queue agent needs. Not shipped surface: gym is unpublished, so this stays under examples/gym/src/.

import { type Body, init, World } from "@dylanebert/shallot/tumble/core";
import { goldParams, runOracle } from "./tumble-oracle";
import { type GoldEntry, goldRegistry } from "./tumble-registry";

function fail(message: string): never {
    console.error(message);
    process.exit(1);
}

// Replay the committed gold bit-exact (fresh ST kernel, one world, destroyed on exit).
async function runGold(entry: GoldEntry): Promise<void> {
    const result = await runOracle(entry.gold, entry.build, entry.update);
    if (!result.pass) {
        fail(
            `[${entry.slug}] gold diverged at step ${result.step}: got ${result.got}, expected ${result.expected}`,
        );
    }
    if (result.steps !== entry.gold.stepCount) {
        fail(`[${entry.slug}] gold ran ${result.steps} steps, expected ${entry.gold.stepCount}`);
    }
}

// Step one knob point to completion, asserting every tracked body stays finite. Bodies are tracked via a
// `createBody` proxy — the only way to read positions generically across an arbitrary entry without
// widening `SampleBuild` or reaching into engine internals (`World`/`Body` are the escape-hatch surface).
async function runKnob(entry: GoldEntry, index: number): Promise<void> {
    const point = entry.knobPoints?.[index];
    if (!point) fail(`[${entry.slug}] no knob point at index ${index}`);
    await init({ threads: 0 });
    const params = { ...goldParams(entry.gold), ...point };
    const bodies: Body[] = [];
    const raw = new World({
        gravity: {
            x: entry.gold.gravity[0],
            y: entry.gold.gravity[1],
            z: entry.gold.gravity[2],
        },
        enableSleep: entry.gold.enableSleep,
        enableContinuous: entry.gold.enableContinuous,
    });
    const world = new Proxy(raw, {
        get(target, prop) {
            const value = Reflect.get(target, prop, target);
            if (prop === "createBody" && typeof value === "function") {
                return (...args: unknown[]) => {
                    const body = (value as (...a: unknown[]) => Body).apply(target, args);
                    bodies.push(body);
                    return body;
                };
            }
            return typeof value === "function" ? value.bind(target) : value;
        },
    });
    try {
        entry.build(world, params);
        for (let i = 0; i < entry.gold.stepCount; i++) {
            entry.update?.(world, params, entry.gold.timeStep, i);
            world.step(entry.gold.timeStep, entry.gold.subStepCount);
        }
        for (const body of bodies) {
            const p = body.getPosition();
            if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) {
                fail(
                    `[${entry.slug}] non-finite position at ${JSON.stringify(point)}: ${JSON.stringify(p)}`,
                );
            }
        }
    } finally {
        raw.destroy();
    }
}

const args = process.argv;
const slug = args[args.indexOf("--slug") + 1];
if (!slug || slug.startsWith("--")) fail("[tumble-gold-runner] missing --slug <slug>");
const entry = goldRegistry.find((e) => e.slug === slug);
if (!entry) fail(`[tumble-gold-runner] no registry entry for slug ${slug}`);

const knobFlag = args.indexOf("--knob");
const knobIndex = knobFlag === -1 ? -1 : Number(args[knobFlag + 1]);

// Wrap so an engine trap (an uncaught RuntimeError, not a `fail`) still surfaces the entry + check, not a
// bare stack a queue agent can't attribute.
try {
    if (knobIndex >= 0) await runKnob(entry, knobIndex);
    else await runGold(entry);
} catch (err) {
    const where = knobIndex >= 0 ? `knob ${knobIndex}` : "gold";
    fail(`[${entry.slug}] ${where} threw: ${err instanceof Error ? err.message : String(err)}`);
}
process.exit(0);

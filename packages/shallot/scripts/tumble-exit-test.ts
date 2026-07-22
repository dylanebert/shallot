// Standalone-lifecycle gate for the tumble MT pool: a script that inits the multithreaded kernel, steps
// a scene, and ends must exit on its own — no `shutdown()`. Parked pool workers pin the host event loop,
// so without the boot-time `unref` (engine/pool.ts) the process hangs here; with it, it exits cleanly.
// The consumer also prints the resolved thread count so a silent single-thread fallback (which would exit
// trivially and prove nothing) fails the gate. Run under both bun and node, since the worker-ref
// behaviour is runtime-specific.
//
//   bun run scripts/tumble-exit-test.ts   (from packages/shallot)

import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

const PKG_ROOT = resolve(import.meta.dir, "..");
const ENGINE = resolve(PKG_ROOT, "src/standard/tumble/engine/index.ts");

// node can't run the .ts engine source, so bundle a self-contained consumer that inits the auto path,
// steps a small pyramid, prints the resolved thread count, then falls off the end with no shutdown. The
// bundle inlines the base64 wasm + the lazy shared-kernel chunk a real install ships; both runtimes run
// the same file.
const dir = mkdtempSync(resolve(tmpdir(), "tumble-exit-"));
const consumer = resolve(dir, "consumer.ts");
const bundle = resolve(dir, "consumer.mjs");
await Bun.write(
    consumer,
    `import { init, threads, World, BodyType, makeBoxHull } from ${JSON.stringify(ENGINE)};
await init();
const w = new World({ gravity: { x: 0, y: -10, z: 0 } });
const g = w.createBody({ position: { x: 0, y: -1, z: 0 } });
g.createHull({}, makeBoxHull(50, 1, 50));
for (let i = 0; i < 20; i++) for (let j = i; j < 20; j++) {
    const b = w.createBody({ type: BodyType.Dynamic, position: { x: (j - i) * 1.05 - 10, y: 1 + i, z: 0 } });
    b.createHull({ density: 100 }, makeBoxHull(0.5, 0.5, 0.5));
}
for (let s = 0; s < 60; s++) w.step(1 / 60, 4);
console.log("DONE threads=" + threads());
`,
);

console.log("bundling consumer…");
const built = spawnSync("bun", ["build", consumer, "--target", "node", "--outfile", bundle], {
    encoding: "utf8",
});
if (built.status !== 0) {
    console.error(built.stdout, built.stderr);
    rmSync(dir, { recursive: true, force: true });
    process.exit(1);
}

const RUNTIMES: Array<[string, string[]]> = [
    ["node", [bundle]],
    ["bun", [bundle]],
];

const fails: string[] = [];
for (const [rt, args] of RUNTIMES) {
    const t0 = Date.now();
    // `timeout` turns a hang into a SIGTERM kill: status is null and signal is set, which we treat as a
    // failure. A clean run returns status 0 well under it.
    const r = spawnSync(rt, args, { encoding: "utf8", timeout: 15_000 });
    const ms = Date.now() - t0;
    const out = `${r.stdout ?? ""}${r.stderr ?? ""}`.trim();
    const hung = r.signal != null || r.status === null;
    const clean = !hung && r.status === 0;
    const mt = /DONE threads=(\d+)/.exec(out);
    const threaded = mt != null && Number(mt[1]) > 1;
    const ok = clean && threaded;
    console.log(
        `  ${ok ? "✓" : "✗"} ${rt}: ${hung ? "HANG" : `exit ${r.status}`} in ${ms}ms — ${mt ? mt[0] : "no thread report"}`,
    );
    if (!ok) {
        fails.push(rt);
        if (out) console.log(out.replace(/^/gm, "      "));
    }
}

rmSync(dir, { recursive: true, force: true });

if (fails.length) {
    console.error(`\nFAIL: ${fails.join(", ")} did not exit cleanly with a live pool`);
    process.exit(1);
}
console.log(
    "\nPASS: inits, steps, and exits on its own (no shutdown), multithreaded on both runtimes",
);

// Runs the tumble engine's heavy bit-exact fixture tier (`*.fixture.ts`). Bun only auto-discovers
// `.test`/`.spec` files, so fixture files must be passed as explicit paths — which also keeps them out
// of the fast `bun test` tier. The scene hashes are C-generated truth committed at tests/tumble/fixtures/;
// TUMBLE_THREADS selects the thread count (unset/0 = single-thread, n = n threads, `auto` = the default
// resolved path), so the same suite gates ST, t2, t8, and the auto path.
//
// Usage: bun run scripts/run-tumble-fixtures.ts   (from packages/shallot)

import { spawnSync } from "node:child_process";
import { Glob } from "bun";

const files = [...new Glob("src/**/*.fixture.ts").scanSync(".")].map((f) => `./${f}`);
if (files.length === 0) {
    console.error("[test:fixture] no fixture files found under src/");
    process.exit(1);
}

const r = spawnSync("bun", ["test", ...files], { stdio: "inherit" });
process.exit(r.status ?? 1);

// Runs the physics engine's heavy bit-exact fixture tier (`*.fixture.ts`). Bun only auto-discovers
// `.test`/`.spec` files, so fixture files must be passed as explicit paths — which also keeps them out
// of the fast `bun test` tier. The scene hashes are C-generated truth committed at tests/physics/fixtures/;
// SHALLOT_PHYSICS_THREADS selects the thread count (unset/0 = single-thread, n = n threads, `auto` = the default
// resolved path), so the same suite gates ST, t2, t8, and the auto path.
//
// Usage: bun run rust/physics/scripts/run-fixtures.ts   (from the repo root)

import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { Glob } from "bun";

const owner = resolve(import.meta.dir, "../../..");
const files = [...new Glob("src/standard/physics/**/*.fixture.ts").scanSync(owner)].map(
    (f) => `./${f}`,
);
if (files.length === 0) {
    console.error("[test:fixture] no fixture files found under src/");
    process.exit(1);
}

const r = spawnSync("bun", ["test", ...files], { cwd: owner, stdio: "inherit" });
process.exit(r.status ?? 1);

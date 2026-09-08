// Remove the tarball-only projections prepack.ts / build-tooling.ts generated: the examples projection
// and the compiled tooling exports (dist/vite.js, dist/harness-browser.js). Both ship in the tarball but
// are gitignored and regenerated on every pack, so neither belongs in the working tree — leaving the
// examples projection would make `tsc`/biome process 20 duplicate recipe copies on every `bun check`, and
// leaving dist/ would let it silently drift stale against the source it was compiled from.

import { rmSync } from "node:fs";
import { resolve } from "node:path";
import { $ } from "bun";

rmSync(resolve(import.meta.dir, "../examples"), { recursive: true, force: true });
rmSync(resolve(import.meta.dir, "../dist"), { recursive: true, force: true });
for (const path of ["bin", "src/project", "src/harness/browser.ts", "rust/window", "assets"]) {
    rmSync(resolve(import.meta.dir, "..", path), { recursive: true, force: true });
}
await $`bun ../shallot-runtime/scripts/project.ts`.cwd(resolve(import.meta.dir, ".."));

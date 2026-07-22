// Remove the tarball-only context projection `prepack.ts` generated. It ships in the tarball but is
// gitignored and regenerated on every pack, so it never belongs in the working tree — leaving it would
// make `tsc`/biome process 20 duplicate recipe copies on every `bun check`.

import { rmSync } from "node:fs";
import { resolve } from "node:path";

rmSync(resolve(import.meta.dir, "../examples"), { recursive: true, force: true });

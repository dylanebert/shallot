/*
# Receding-plane minification captures

Renders the generated receding plane of `src/standard/rendering/receding-plane.fixture.ts` through
StandardMaterial on the hardware sampling path (default sampler) and the manual path (repeat), at a
native 1280×720 and a fixed 320×180 target with the same camera, UVs and antialias off. Writes PNGs
and `report.json` under `test-results/texture-minification/<label>/` and prints a summary: the adapter,
the largest row-to-row contrast jump, a mip-generation check against a CPU box filter and the share
of pixels whose selected level matches the level the geometry predicts.

```sh
bun diagnostics/texture-minification/run.ts [label]
```

Set `SHALLOT_GLTF` to a shallot-gltf checkout to also decode an equivalent
generated glTF (plane, repeat sampler, unspecified filters) and compare its render with direct
registration byte for byte.
*/

import { shallot } from "@dylanebert/shallot/bun";
import { plugin } from "bun";

plugin(shallot({ root: import.meta.dir }));
await import("./capture");

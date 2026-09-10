// collision-overlap-box — stage-4 gym twin: the upstream `OverlapBox` sample
// (`samples/src/samples/collision.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold. The circling `overlapAABB` query + HUD count (`render()`) is outside the gold contract —
// it only feeds debug-draw output, never mutates the world — so only `build()` ports.

import goldJson from "../../../../src/standard/physics/samples/collision-overlap-box.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { buildOverlapBox, renderOverlapBox } from "../physics-overlap-box";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildOverlapBox,
        render: renderOverlapBox,
    }),
);

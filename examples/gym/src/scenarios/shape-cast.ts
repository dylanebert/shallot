// collision-shape-cast — stage-4 gym twin: the upstream `ShapeCast` sample
// (`samples/src/samples/collision.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold. The swept-sphere `castShape` sweep + HUD draw (`render()`) is outside the gold contract —
// it only feeds debug-draw output, never mutates the world — so only `build()` ports.

import goldJson from "../../../../src/standard/physics/samples/collision-shape-cast.json";
import { register } from "../gym";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";
import { buildShapeCast, renderShapeCast } from "../physics-shape-cast";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildShapeCast,
        render: renderShapeCast,
    }),
);

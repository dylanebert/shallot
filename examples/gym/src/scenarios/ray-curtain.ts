// collision-ray-curtain — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `RayCurtain` sample
// (`samples/src/samples/collision.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold. The sweeping `castRayClosest` curtain + HUD line/point draw (`render()`) is outside the
// gold contract — it only feeds debug-draw output, never mutates the world — so only `build()` ports.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/collision-ray-curtain.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { buildRayCurtain, renderRayCurtain } from "../tumble-ray-curtain";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildRayCurtain,
        render: renderRayCurtain,
    }),
);

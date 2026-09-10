// compound-tile-floor — stage-4 gym twin: the upstream `TileFloor` sample
// (`samples/src/samples/compound.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer.

import goldJson from "../../../../src/standard/physics/samples/compound-tile-floor.json";
import { register } from "../gym";
import { buildTileFloor } from "../physics-compound-tile-floor";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildTileFloor,
    }),
);

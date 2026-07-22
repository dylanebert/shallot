// joints-bridge — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Bridge` sample
// (`samples/src/samples/joints.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/joints-bridge.json";
import { register } from "../gym";
import { buildBridge } from "../tumble-bridge";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildBridge,
    }),
);

// stacking-dominoes — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Dominoes` sample
// (`samples/src/samples/stacks.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The `rings` knob
// picks the concentric ring count kicked into a toppling chain reaction.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/stacking-dominoes.json";
import { register } from "../gym";
import { buildDominoes } from "../tumble-dominoes";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildDominoes }));

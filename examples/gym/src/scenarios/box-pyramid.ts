// stacking-box-pyramid — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `BoxPyramid` sample
// (`samples/src/samples/stacks.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The `rows` knob picks
// the pyramid's base row count.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/stacking-box-pyramid.json";
import { register } from "../gym";
import { buildBoxPyramid } from "../tumble-box-pyramid";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildBoxPyramid }));

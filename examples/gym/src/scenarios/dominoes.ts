// stacking-dominoes — stage-4 gym twin: the upstream `Dominoes` sample
// (`samples/src/samples/stacks.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The `rings` knob
// picks the concentric ring count kicked into a toppling chain reaction.

import goldJson from "../../../../src/standard/physics/samples/stacking-dominoes.json";
import { register } from "../gym";
import { buildDominoes } from "../physics-dominoes";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildDominoes }));

// stacking-arch — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Arch` sample
// (`samples/src/samples/stacks.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. No knobs.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/stacking-arch.json";
import { register } from "../gym";
import { buildArch } from "../tumble-arch";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildArch }));

// stacking-arch — stage-4 gym twin: the upstream `Arch` sample
// (`samples/src/samples/stacks.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. No knobs.

import goldJson from "../../../../src/standard/physics/samples/stacking-arch.json";
import { register } from "../gym";
import { buildArch } from "../physics-arch";
import type { SampleGold } from "../physics-oracle";
import { sampleScenario } from "../physics-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildArch }));

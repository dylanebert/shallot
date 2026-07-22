// events-hit — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `HitEvents` sample
// (`samples/src/samples/events.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/events-hit.json";
import { register } from "../gym";
import { buildHitEvents, renderHitEvents } from "../tumble-hit";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildHitEvents,
        render: renderHitEvents,
    }),
);

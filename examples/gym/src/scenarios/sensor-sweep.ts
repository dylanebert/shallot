// events-sensor-sweep — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `SensorSweep` sample
// (`samples/src/samples/events.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold and rendered via the source-faithful debug-draw + mouse-grab layer. The sine-driven
// vertical sweep lives in `update()`.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/events-sensor-sweep.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";
import { buildSensorSweep, renderSensorSweep, updateSensorSweep } from "../tumble-sensor-sweep";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildSensorSweep,
        update: updateSensorSweep,
        render: renderSensorSweep,
    }),
);

// joints-paddle — the tumble sample host's pilot scenario (spec tumble-inline stage 3): the Paddle sample
// ported through the escape hatch, verified bit-exact against its committed gold, rendered with the
// source-faithful debug-draw layer + mouse-grab. It replaces the dropped powered-rotor recipe as the
// motor-joint verification home (the red-first oracle proof lives in `tumble-pilot.test.ts`).

import goldJson from "../../../../packages/shallot/tests/tumble/samples/joints-paddle.json";
import { register } from "../gym";
import type { SampleGold } from "../tumble-oracle";
import { buildPaddle } from "../tumble-paddle";
import { sampleScenario } from "../tumble-sample";

register(sampleScenario({ gold: goldJson as unknown as SampleGold, build: buildPaddle }));

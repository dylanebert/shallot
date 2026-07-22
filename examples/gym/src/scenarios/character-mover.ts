// character-mover — stage-4 gym twin (spec tumble-inline stage 4): the tumble.js `Character` sample
// (`samples/src/samples/character.ts`) ported through the escape hatch, verified bit-exact against its
// committed gold. A self-driven kinematic capsule mover patrols an arena on the plane solver — the drive
// lives in `update()`, shoving dynamic crates it leans on.

import goldJson from "../../../../packages/shallot/tests/tumble/samples/character-mover.json";
import { register } from "../gym";
import {
    buildCharacterMover,
    renderCharacterMover,
    updateCharacterMover,
} from "../tumble-character-mover";
import type { SampleGold } from "../tumble-oracle";
import { sampleScenario } from "../tumble-sample";

register(
    sampleScenario({
        gold: goldJson as unknown as SampleGold,
        build: buildCharacterMover,
        update: updateCharacterMover,
        render: renderCharacterMover,
    }),
);

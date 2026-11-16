import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import {
    actions,
    composition,
    trappedTick,
} from "../../../scripts/determinism/composition.fixture";
import { GAMEPLAY_HASH } from "../../../scripts/determinism/golden.fixture";
import { GlobalTransformHistory } from "../../core/rendering/global-transform";
import { GlobalTransform } from "../../core/transform";

setDefaultTimeout(TIMEOUT.node);

test.each([1, 2, 4])(
    "built-in fixed composition matches the gameplay golden with %i workers without implementation-approximated Math",
    async (workerCount) => {
        const app = await composition(workerCount);
        try {
            for (const _ of actions) trappedTick(app.world);
            expect(app.world.time.fixedTick).toBe(actions.length);
            expect(app.world.storage(GlobalTransform).translation.z.get(app.player)).not.toBe(0);
            expect(app.world.resource(GlobalTransformHistory).cpuRequests.has(app.player)).toBe(
                true,
            );
            expect(app.artifact).toBe("shared");
            expect(app.workers).toBe(workerCount);
            expect(app.hash()).toBe(GAMEPLAY_HASH);
        } finally {
            app.dispose();
        }
    },
);

import { expect, setDefaultTimeout, test } from "bun:test";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { GlobalTransformHistory } from "../../core/rendering/global-transform";
import { GlobalTransform } from "../../core/transform";
import { actions, composition, trappedTick } from "./determinism.fixture";

setDefaultTimeout(TIMEOUT.node);

test("built-in fixed composition steps the recorded stream without implementation-approximated Math", async () => {
    const app = await composition();
    try {
        for (const _ of actions) trappedTick(app.world);
        expect(app.world.time.fixedTick).toBe(actions.length);
        expect(app.world.storage(GlobalTransform).translation.z.get(app.player)).not.toBe(0);
        expect(app.world.resource(GlobalTransformHistory).cpuRequests.has(app.player)).toBe(true);
        expect(app.hash()).toBeTypeOf("bigint");
    } finally {
        app.dispose();
    }
});

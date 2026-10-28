import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, GlobalTransform } from "@dylanebert/shallot";
import { PhysicsWorldDefinition, setTransform } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { ContinuousSamplePlugin, ContinuousState } from "../src/continuous";
import { sceneSwitcher } from "../src/sample";
import { changeControl, controlValues } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

test("Mesh Drop fall-through pauses stepping; Generate keeps the pause and picker rebuild clears it", async () => {
    const app = await createApp({ defaults: false, plugins: [ContinuousSamplePlugin] });
    try {
        app.world.resource(ContinuousState).seed = 12345;
        const select = sceneSwitcher(app.world);
        const bodies = select("meshDrop");
        app.world.tick();
        const state = app.world.resource(ContinuousState);
        setTransform(app.world, bodies[0]!, [0, -3, 0], [0, 0, 0, 1]);
        app.world.tick();
        expect(state.paused).toBe(true);
        expect(state.failure).toBe(true);
        expect(app.world.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(false);
        changeControl(app.world, "meshDrop", bodies, controlValues("meshDrop"), "Generate", 1);
        app.world.tick();
        const height = app.world.storage(GlobalTransform).translation.y.get(state.bodies[0]!);
        app.world.tick();
        expect(app.world.storage(GlobalTransform).translation.y.get(state.bodies[0]!)).toBe(height);
        expect(state.paused).toBe(true);
        select("meshDrop");
        expect(app.world.resource(ContinuousState).paused).toBe(false);
        expect(app.world.resource(PhysicsWorldDefinition).enableAutomaticStep).toBe(true);
        app.world.tick();
    } finally {
        app.dispose();
    }
});

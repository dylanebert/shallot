import { build, devices, pressKey, Time, Transform, TransformsPlugin } from "@dylanebert/shallot";
import { check } from "@dylanebert/shallot/harness/check";
import { Controlled, Drive } from "./drive";

check(
    "respond-to-input moves through the public device producer",
    {
        claim: "respond-to-input presses W through the public producer seam and moves its production transform",
    },
    async () => {
        const app = await build({ defaults: false, plugins: [TransformsPlugin] });
        const state = app.state;
        for (const system of Drive.systems ?? []) state.addSystem(system, Drive.name);
        const eid = state.create();
        state.add(eid, Controlled);
        state.add(eid, Transform);
        pressKey(state, "KeyW");
        state.step(Time.FIXED_DT);
        if (!devices(state).keys.held.has("KeyW")) throw new Error("W was not produced");
        if (Transform.pos.z.get(eid) >= 0)
            throw new Error("production transform did not move forward");
        app.dispose();
    },
);

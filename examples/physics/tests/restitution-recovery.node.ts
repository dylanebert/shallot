import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { RestitutionPlugin, RestitutionState } from "../src/restitution";
import { sceneSwitcher } from "../src/sample";

setDefaultTimeout(TIMEOUT.node);
test("leaving a played Restitution scene clears its gameplay before another scene steps", async () => {
    const app = await createApp({ defaults: false, plugins: [RestitutionPlugin] });
    try {
        const select = sceneSwitcher(app.world);
        select("restitutionPropagation");
        for (let i = 0; i < 240; i++) app.world.tick();
        expect(app.world.resource(RestitutionState).apex.some((height) => height > 0)).toBe(true);
        select("boxStack");
        expect(app.world.resource(RestitutionState)).toEqual(RestitutionState.create());
        app.world.tick();
        select("bounceImpulse");
        for (let i = 0; i < 240; i++) app.world.tick();
        expect(app.world.resource(RestitutionState).hit).toBe(true);
        select("boxStack");
        expect(app.world.resource(RestitutionState)).toEqual(RestitutionState.create());
        app.world.tick();
    } finally {
        app.dispose();
    }
});

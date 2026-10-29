import { expect, test } from "bun:test";
import { createApp, GlobalTransform } from "@dylanebert/shallot";
import { setLinearVelocity } from "@dylanebert/shallot/standard/physics";
import { IssueState, IssuesPlugin, issueScenes } from "./issues";
import { changeControl, controlValues, createScene } from "./scenes";

test("a grounded upward spike records its location once, recovers, and resets through the scene control", async () => {
    const app = await createApp({ defaults: false, plugins: [IssuesPlugin] });
    try {
        const bodies = createScene(app.world, "ghostCollisions"),
            e = bodies[0]!;
        app.world.tick();
        setLinearVelocity(app.world, e, { x: 0, y: 0.9, z: 0 });
        app.world.tick();
        const state = app.world.resource(IssueState),
            pose = app.world.storage(GlobalTransform);
        expect(state.launches).toBe(1);
        expect(state.maxLaunchSpeed).toBe(pose.linearVelocity.y.get(e));
        expect(state.markers).toEqual([
            [pose.translation.x.get(e), pose.translation.y.get(e), pose.translation.z.get(e)],
        ]);
        expect(state.wasLaunched).toBe(true);
        const image = app.world.snapshot(),
            recorded = structuredClone(state);
        app.world.tick();
        expect(state.launches).toBe(1);
        changeControl(
            app.world,
            "ghostCollisions",
            bodies,
            controlValues("ghostCollisions"),
            "Reset Counters",
            1,
        );
        expect(state.launches).toBe(0);
        expect(state.maxLaunchSpeed).toBe(0);
        expect(state.markers).toEqual([]);
        app.world.restore(image);
        expect(state).toEqual(recorded);
        expect(issueScenes.ghostCollisions.controls.Launches.read(app.world)).toBe(1);
        expect(issueScenes.ghostCollisions.controls.Worst.read(app.world)).toBe(
            recorded.maxLaunchSpeed,
        );
    } finally {
        app.dispose();
    }
});

import { expect, test } from "bun:test";
import { Body, BodyType, Shape } from "../../core/physics";
import { GlobalTransform } from "../../core/transform";
import { createApp, Time } from "../../engine";
import { StandardPhysicsPlugin, setTargetTransform } from ".";

test("a dynamic box turns with a rotating kinematic platform", async () => {
    const app = await createApp({ defaults: false, plugins: [StandardPhysicsPlugin] });
    try {
        const world = app.world;
        const platform = world.create();
        world.add(platform, Body, { type: BodyType.Kinematic, position: [0, 0, 0, 0] });
        world.add(platform, Shape, { scale: [5, 0.25, 5, 0], friction: 1 });
        const rider = world.create();
        world.add(rider, Body, { type: BodyType.Dynamic, position: [0.5, 0.75, 0, 0] });
        world.add(rider, Shape, { scale: [0.5, 0.5, 0.5, 0], friction: 1 });
        world.tick();

        for (let tick = 1; tick <= 2; tick++) {
            const angle = 1.2 * tick * Time.FIXED_DT;
            setTargetTransform(
                world,
                platform,
                [0, 0, 0],
                [0, Math.sin(angle / 2), 0, Math.cos(angle / 2)],
            );
            world.tick();
        }

        const pose = world.storage(GlobalTransform);
        const rotation = pose.rotation;
        const yaw = 2 * Math.atan2(rotation.y.get(rider), rotation.w.get(rider));
        expect(yaw).toBeGreaterThan(0.01);
        expect(pose.translation.y.get(rider)).toBeCloseTo(0.75, 1);
    } finally {
        app.dispose();
    }
});

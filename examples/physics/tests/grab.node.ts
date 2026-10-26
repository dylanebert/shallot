import { expect, setDefaultTimeout, test } from "bun:test";
import { createApp, GlobalTransform, StandardPhysicsPlugin } from "@dylanebert/shallot";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { EventSamplePlugin } from "../src/events";
import { Grab, pick, release } from "../src/grab";
import { createScene, type Scene, scenes } from "../src/scenes";

setDefaultTimeout(TIMEOUT.node);

for (const scene of [
    "revolute",
    "prismatic",
    "wheel",
    "parallel",
    "filter",
    "motor",
    "joint",
    "sensor",
    "contact",
    "hit",
] as Scene[]) {
    test(`${scene} can be picked from its camera and released without a GPU`, async () => {
        let bodies: number[] = [];
        const app = await createApp({
            defaults: false,
            plugins: [StandardPhysicsPlugin, EventSamplePlugin],
            setup(world) {
                bodies = createScene(world, scene);
            },
        });
        try {
            const world = app.world;
            world.tick();
            world.tick();
            const body = bodies[0]!;
            const view = scenes[scene].camera;
            const yaw = (view.yaw * Math.PI) / 180;
            const pitch = (view.pitch * Math.PI) / 180;
            const origin = [
                view.target[0]! + view.distance * Math.cos(pitch) * Math.sin(yaw),
                view.target[1]! + view.distance * Math.sin(pitch),
                view.target[2]! + view.distance * Math.cos(pitch) * Math.cos(yaw),
            ];
            const position = world.storage(GlobalTransform).translation;
            const target = [position.x.get(body), position.y.get(body), position.z.get(body)];
            const grab = world.resource(Grab);
            Object.assign(grab.origin, { x: origin[0], y: origin[1], z: origin[2] });
            Object.assign(grab.translation, {
                x: 2 * (target[0]! - origin[0]!),
                y: 2 * (target[1]! - origin[1]!),
                z: 2 * (target[2]! - origin[2]!),
            });
            pick(world);
            expect(grab.body).toBe(body);
            expect(grab.mouse).toBeGreaterThan(0);
            expect(grab.joint).toBeGreaterThan(0);
            release(world);
            expect([grab.body, grab.mouse, grab.joint]).toEqual([0, 0, 0]);
        } finally {
            app.dispose();
        }
    });
}

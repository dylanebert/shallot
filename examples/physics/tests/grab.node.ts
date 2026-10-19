import { expect, setDefaultTimeout, test } from "bun:test";
import {
    createApp,
    GlobalTransform,
    hashPhysics,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot";
import { setTargetTransform } from "@dylanebert/shallot/standard/physics";
import { TIMEOUT } from "@dylanebert/shallot/testing";
import { EventSamplePlugin } from "../src/events";
import { Grab, pick, release } from "../src/grab";
import { createScene, type Scene, scenes } from "../src/scenes";

import { sampleGold } from "./gold";

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

test("grab, moving kinematic target and release equal native Sample::MouseDown on every step", async () => {
    const gold = sampleGold(await Bun.file(new URL("./grab.gold.txt", import.meta.url)).text());
    const app = await createApp({
        defaults: false,
        plugins: [StandardPhysicsPlugin],
        setup(world) {
            createScene(world, "boxStack");
        },
    });
    try {
        const world = app.world;
        const grab = world.resource(Grab);
        for (let i = 0; i < gold.length; i++) {
            if (i === 1) {
                Object.assign(grab.origin, { x: 0, y: 25.75, z: 10 });
                Object.assign(grab.translation, { x: 0, y: 0, z: -20 });
                pick(world);
                expect(grab.body).not.toBe(0);
            }
            if (i > 1 && i < 31) grab.point[0] = Math.fround(Math.fround(0.05) * (i - 1));
            if (i === 31) release(world);
            if (grab.mouse) setTargetTransform(world, grab.mouse, grab.point, [0, 0, 0, 1]);
            world.tick();
            expect(hashPhysics(world), `step ${i} hash`).toBe(gold[i]!.hash);
        }
    } finally {
        app.dispose();
    }
});

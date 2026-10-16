import { expect, test } from "bun:test";
import {
    createApp,
    Sprite,
    Text,
    Transform,
    TransformPlugin,
    type World,
} from "@dylanebert/shallot";
import {
    Car,
    CYCLE_SECONDS,
    Delivery,
    DRIVE_SECONDS,
    DriveCar,
    EXIT_SECONDS,
    ROAD_LENGTH,
    ROAD_START,
} from "./delivery";

function check() {
    return createApp({
        defaults: false,
        plugins: [
            {
                name: "DeliveryCheck",
                components: [Car, Text, Sprite],
                dependencies: [TransformPlugin],
                systems: [DriveCar],
                initialize(world) {
                    const scene = world.resource(Delivery);
                    scene.car = world.create();
                    world.add(scene.car, Transform, { scale: [1.1, 0.32, 0.65] });
                    world.add(scene.car, Car);
                    scene.parts.push({ eid: scene.car, offset: [0, 0.32, 1] });
                    const cab = world.create();
                    world.add(cab, Transform);
                    scene.parts.push({ eid: cab, offset: [-0.1, 0.62, 1] });
                    for (const key of ["label", "bar", "backing", "icon"] as const) {
                        scene[key] = world.create();
                        world.add(scene[key], Transform);
                    }
                    world.storage(Transform).translation.set(scene.icon, 3, 3.1, -2);
                    world.add(scene.label, Text);
                    world.add(scene.bar, Sprite);
                    DriveCar.update(world);
                },
            },
        ],
    });
}
function advance(world: World, seconds: number) {
    world.setTimeScale(seconds * 60);
    world.step(1 / 60);
    world.setTimeScale(1);
}

test("the bar's fill follows the car's progress along the road", async () => {
    const app = await check();
    try {
        const world = app.world,
            scene = world.resource(Delivery);
        for (let second = 0; second <= DRIVE_SECONDS; second++) {
            advance(world, second ? 1 : 0);
            const x = world.storage(Transform).translation.x.get(scene.car);
            const progress = world.storage(Car).progress.get(scene.car);
            const fill = world.storage(Sprite).fillAmount.get(scene.bar);
            expect(progress).toBeCloseTo(second / DRIVE_SECONDS, 6);
            expect(fill).toBeCloseTo((x - ROAD_START) / ROAD_LENGTH, 6);
            expect(fill).toBe(progress);
            for (const eid of [scene.bar, scene.backing, scene.label])
                expect(world.storage(Transform).translation.x.get(eid)).toBe(x);
        }
    } finally {
        app.dispose();
    }
});

test("the trail tip touches the lower rear edge, never the inside of the car", async () => {
    const app = await check();
    try {
        const world = app.world,
            scene = world.resource(Delivery);
        for (let i = 0; i < 3; i++) {
            const x = world.storage(Transform).translation.x.get(scene.car);
            expect(scene.trailEnd[0]).toBeCloseTo(x - 0.55, 6);
            expect(scene.trailEnd[1]).toBeCloseTo(0.16, 6);
            expect(scene.trailEnd[2]).toBe(1);
            expect(scene.trailStart[1]).toBe(scene.trailEnd[1]);
            advance(world, DRIVE_SECONDS / 2);
        }
    } finally {
        app.dispose();
    }
});

test("the car stops in front of the building while the icon animates", async () => {
    const app = await check();
    try {
        const world = app.world,
            scene = world.resource(Delivery);
        advance(world, DRIVE_SECONDS);
        for (let i = 0; i < 4; i++) {
            expect(world.storage(Transform).translation.x.get(scene.car)).toBe(3);
            expect(world.storage(Transform).translation.z.get(scene.car)).toBe(1);
            expect(world.storage(Car).progress.get(scene.car)).toBe(1);
            expect(world.storage(Sprite).fillAmount.get(scene.bar)).toBe(1);
            advance(world, 0.5);
        }
    } finally {
        app.dispose();
    }
});

test("the icon exits in 0.35 seconds with Penner easeInBack, overshooting without moving its centre", async () => {
    const app = await check();
    try {
        const world = app.world,
            scene = world.resource(Delivery);
        const transform = world.storage(Transform);
        advance(world, DRIVE_SECONDS);
        expect(transform.scale.x.get(scene.icon)).toBe(1);
        advance(world, EXIT_SECONDS * 0.4);
        expect(transform.translation.y.get(scene.icon)).toBeCloseTo(3.1, 6);
        for (const axis of [transform.scale.x, transform.scale.y, transform.scale.z])
            expect(axis.get(scene.icon)).toBeCloseTo(1.09935168, 6);
        advance(world, EXIT_SECONDS * 0.4);
        expect(transform.scale.x.get(scene.icon)).toBeCloseTo(0.70580224, 6);
        advance(world, EXIT_SECONDS * 0.2 + 0.001);
        for (const axis of [transform.scale.x, transform.scale.y, transform.scale.z])
            expect(axis.get(scene.icon)).toBe(0);
        expect(transform.translation.x.get(scene.icon)).toBe(3);
        expect(transform.translation.y.get(scene.icon)).toBeCloseTo(3.1, 6);
        expect(transform.translation.z.get(scene.icon)).toBe(-2);
    } finally {
        app.dispose();
    }
});

test("the scene then restarts as at spawn: car at start, icon whole, bar empty", async () => {
    const app = await check();
    try {
        const world = app.world,
            scene = world.resource(Delivery);
        const transform = world.storage(Transform);
        const snapshot = () => ({
            positions: [
                scene.car,
                ...scene.parts.map((part) => part.eid),
                scene.label,
                scene.backing,
                scene.bar,
                scene.icon,
            ].map((eid) => [
                transform.translation.x.get(eid),
                transform.translation.y.get(eid),
                transform.translation.z.get(eid),
            ]),
            iconScale: [
                transform.scale.x.get(scene.icon),
                transform.scale.y.get(scene.icon),
                transform.scale.z.get(scene.icon),
            ],
            fill: world.storage(Sprite).fillAmount.get(scene.bar),
            progress: world.storage(Car).progress.get(scene.car),
            trailEnd: [...scene.trailEnd],
        });
        const spawn = snapshot();
        expect(spawn.fill).toBe(0);
        expect(spawn.iconScale).toEqual([1, 1, 1]);
        expect(transform.translation.x.get(scene.car)).toBe(ROAD_START);
        advance(world, CYCLE_SECONDS - 0.25);
        expect(transform.scale.x.get(scene.icon)).toBe(0);
        advance(world, 0.25);
        expect(snapshot()).toEqual(spawn);
        advance(world, CYCLE_SECONDS);
        expect(snapshot()).toEqual(spawn);
    } finally {
        app.dispose();
    }
});

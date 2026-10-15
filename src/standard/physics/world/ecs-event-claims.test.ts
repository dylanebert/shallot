import { expect, test } from "bun:test";
import {
    Body,
    BodyType,
    DistanceJoint,
    PhysicsPlugin,
    Shape,
    ShapeKind,
    WeldJoint,
} from "../../../core/physics";
import { GlobalTransform } from "../../../core/transform";
import { type EntityRef, Time, World } from "../../../engine";
import { PhysicsEvents, PhysicsWorldDefinition, StandardPhysicsPlugin, setAwake } from "../index";

type Names = Map<EntityRef, string>;
const zero = { x: 0, y: 0, z: 0 };

async function physicsWorld(): Promise<World> {
    const world = new World();
    for (const component of PhysicsPlugin.components!)
        world.registry.register(component, PhysicsPlugin.name);
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    const recovery = StandardPhysicsPlugin.recovery!;
    if (recovery === "stateless") throw new Error("physics must declare recovery");
    world.registerRecovery(StandardPhysicsPlugin.name, recovery(world));
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);
    world.resource(PhysicsWorldDefinition).gravity = zero;
    return world;
}

function sphere(
    world: World,
    names: Names,
    name: string,
    body: Record<string, unknown>,
    shape: Record<string, unknown> = {},
): number {
    const eid = world.create();
    world.add(eid, Body, body as never);
    world.add(eid, Shape, {
        kind: ShapeKind.Sphere,
        sphereCenter: [0, 0, 0],
        radius: 1,
        ...shape,
    } as never);
    names.set(world.ref(eid), name);
    return eid;
}

function refName(world: World, names: Names, reference: EntityRef): string {
    return names.get(reference) ?? `dead:${world.resolve(reference)}`;
}

function eventSummary(world: World, names: Names) {
    const events = world.resource(PhysicsEvents);
    const pair = (event: { shapeA: EntityRef; shapeB: EntityRef }) => [
        refName(world, names, event.shapeA),
        refName(world, names, event.shapeB),
    ];
    return {
        contactBegin: events.contactBegin.records
            .slice(0, events.contactBegin.count)
            .map((event) => [...pair(event), event.normalImpulse]),
        contactEnd: events.contactEnd.records
            .slice(0, events.contactEnd.count)
            .map((event) => [...pair(event), event.normalImpulse]),
        hit: events.contactHit.records
            .slice(0, events.contactHit.count)
            .map((event) => [
                ...pair(event),
                event.point,
                event.normal,
                event.approachSpeed,
                event.userMaterialIdA.toString(),
                event.userMaterialIdB.toString(),
            ]),
        sensorBegin: events.sensorBegin.records
            .slice(0, events.sensorBegin.count)
            .map((event) => [
                refName(world, names, event.sensorShape),
                refName(world, names, event.visitorShape),
            ]),
        sensorEnd: events.sensorEnd.records
            .slice(0, events.sensorEnd.count)
            .map((event) => [
                refName(world, names, event.sensorShape),
                refName(world, names, event.visitorShape),
            ]),
        joints: events.joints.records
            .slice(0, events.joints.count)
            .map((event) => refName(world, names, event.joint)),
    };
}

function nonemptyEvents(world: World, names: Names) {
    return Object.fromEntries(
        Object.entries(eventSummary(world, names)).filter(([, records]) => records.length !== 0),
    );
}

async function dispose(world: World): Promise<void> {
    await StandardPhysicsPlugin.dispose!(world);
    world.dispose();
}

test("PhysicsEvents order contact output across body type, disable, enable and destruction", async () => {
    const world = await physicsWorld();
    const names: Names = new Map();
    try {
        const a = sphere(world, names, "a", { type: BodyType.Static }, { enableContactEvents: 1 });
        const b = sphere(
            world,
            names,
            "b",
            { type: BodyType.Dynamic, position: [0, 1.5, 0, 0] },
            { enableContactEvents: 1 },
        );
        const steps: unknown[] = [];
        const tick = (name: string) => {
            world.step(Time.FIXED_DT);
            steps.push([name, nonemptyEvents(world, names)]);
        };
        tick("touch");
        world.storage(Body).type.set(a, BodyType.Dynamic);
        tick("static-to-dynamic");
        world.storage(Body).isEnabled.set(b, 0);
        tick("disable");
        world.storage(Body).isEnabled.set(b, 1);
        tick("enable");
        world.destroy(b);
        tick("destroy");
        expect(steps).toEqual([
            ["touch", { contactBegin: [["a", "b", 50265.484375]] }],
            [
                "static-to-dynamic",
                {
                    contactBegin: [["b", "a", 25132.740234375]],
                    contactEnd: [["a", "b", 0]],
                },
            ],
            ["disable", { contactEnd: [["b", "a", 0]] }],
            ["enable", { contactBegin: [["a", "b", 25132.7421875]] }],
            ["destroy", { contactEnd: [["a", "b", 0]] }],
        ]);
    } finally {
        await dispose(world);
    }
});

test("PhysicsEvents preserve contact and sensor output across filter and event-flag toggles", async () => {
    const world = await physicsWorld();
    const names: Names = new Map();
    try {
        const ground = sphere(
            world,
            names,
            "ground",
            { type: BodyType.Static },
            { enableContactEvents: 1 },
        );
        const visitor = sphere(
            world,
            names,
            "visitor",
            { type: BodyType.Dynamic, position: [0, 1.5, 0, 0] },
            { enableContactEvents: 1, enableSensorEvents: 1 },
        );
        const sensor = world.create();
        world.add(sensor, Shape, {
            body: ground,
            kind: ShapeKind.Sphere,
            sphereCenter: [0, 0, 0],
            radius: 3,
            isSensor: 1,
            enableSensorEvents: 1,
        });
        names.set(world.ref(sensor), "sensor");
        const steps: unknown[] = [];
        const tick = (name: string) => {
            world.step(Time.FIXED_DT);
            steps.push([name, nonemptyEvents(world, names)]);
        };
        tick("touches");
        world.storage(Shape).filterMaskLow.set(visitor, 0);
        world.storage(Shape).filterMaskHigh.set(visitor, 0);
        tick("filter-off");
        world.storage(Shape).filterMaskLow.set(visitor, 0xffffffff);
        world.storage(Shape).filterMaskHigh.set(visitor, 0xffffffff);
        tick("filter-on");
        world.storage(Shape).enableSensorEvents.set(visitor, 0);
        tick("visitor-events-off");
        world.storage(Shape).enableSensorEvents.set(visitor, 1);
        world.storage(Shape).enableSensorEvents.set(sensor, 0);
        tick("sensor-events-off");
        world.storage(Shape).enableSensorEvents.set(sensor, 1);
        tick("sensor-events-on");
        world.storage(Shape).enableContactEvents.set(ground, 0);
        world.storage(Shape).enableContactEvents.set(visitor, 0);
        world.storage(Shape).enableHitEvents.set(visitor, 1);
        world.storage(Shape).filterCategoryLow.set(visitor, 4);
        tick("contact-off-hit-on");
        world.storage(Shape).enableContactEvents.set(ground, 1);
        world.storage(Shape).enableContactEvents.set(visitor, 1);
        world.storage(Shape).enableHitEvents.set(visitor, 0);
        tick("contact-on-hit-off");
        expect(steps).toEqual([
            [
                "touches",
                {
                    contactBegin: [["ground", "visitor", 50265.484375]],
                    sensorBegin: [["sensor", "visitor"]],
                },
            ],
            [
                "filter-off",
                {
                    contactEnd: [["ground", "visitor", 0]],
                    sensorEnd: [["sensor", "visitor"]],
                },
            ],
            [
                "filter-on",
                {
                    contactBegin: [["ground", "visitor", 50265.48046875]],
                    sensorBegin: [["sensor", "visitor"]],
                },
            ],
            ["visitor-events-off", { sensorEnd: [["sensor", "visitor"]] }],
            ["sensor-events-off", {}],
            ["sensor-events-on", { sensorBegin: [["sensor", "visitor"]] }],
            ["contact-off-hit-on", { contactEnd: [["ground", "visitor", 0]] }],
            ["contact-on-hit-off", {}],
        ]);
    } finally {
        await dispose(world);
    }
});

test("PhysicsEvents keep ordered contacts across repeated collideConnected toggles and sleep", async () => {
    const world = await physicsWorld();
    const names: Names = new Map();
    try {
        const a = sphere(world, names, "a", { type: BodyType.Dynamic }, { enableContactEvents: 1 });
        const b = sphere(
            world,
            names,
            "b",
            { type: BodyType.Dynamic, position: [1.5, 0, 0, 0] },
            { enableContactEvents: 1 },
        );
        const joint = world.create();
        world.add(joint, DistanceJoint, { a, b, collideConnected: 1 });
        names.set(world.ref(joint), "joint");
        const steps: unknown[] = [];
        const tick = (name: string) => {
            world.step(Time.FIXED_DT);
            steps.push([name, nonemptyEvents(world, names)]);
        };
        tick("collision-on");
        world.storage(DistanceJoint).collideConnected.set(joint, 0);
        tick("collision-off");
        world.storage(DistanceJoint).collideConnected.set(joint, 0);
        tick("collision-off-again");
        world.storage(DistanceJoint).collideConnected.set(joint, 1);
        tick("collision-on-again");
        world.storage(DistanceJoint).collideConnected.set(joint, 1);
        tick("collision-on-repeat");
        setAwake(world, b, false);
        tick("sleep");
        setAwake(world, b, true);
        tick("wake");
        expect(steps).toEqual([
            ["collision-on", { contactBegin: [["b", "a", 1138877]] }],
            ["collision-off", { contactEnd: [["b", "a", 0]] }],
            ["collision-off-again", {}],
            ["collision-on-again", { contactBegin: [["b", "a", 228614.375]] }],
            ["collision-on-repeat", {}],
            ["sleep", {}],
            ["wake", {}],
        ]);
    } finally {
        await dispose(world);
    }
});

test("PhysicsEvents and GlobalTransform preserve the ordered contact, hit, joint and move sequence", async () => {
    const world = await physicsWorld();
    const names: Names = new Map();
    try {
        const definition = world.resource(PhysicsWorldDefinition);
        definition.gravity = { x: 0, y: -10, z: 0 };
        definition.hitEventThreshold = 0;
        sphere(
            world,
            names,
            "ground",
            { type: BodyType.Static, position: [0, -0.5, 0, 0] },
            { sphereCenter: [0, 0, 0], radius: 0.5, enableContactEvents: 1 },
        );
        const falling = sphere(
            world,
            names,
            "falling",
            { type: BodyType.Dynamic, position: [0, 0.8, 0, 0], linearVelocity: [0, -10, 0, 0] },
            {
                sphereCenter: [0, 0, 0],
                radius: 0.5,
                density: 1,
                enableContactEvents: 1,
                enableHitEvents: 1,
                userMaterialIdLow: 0x76543210,
                userMaterialIdHigh: 0xfedcba98,
            },
        );
        const anchor = sphere(
            world,
            names,
            "anchor",
            { type: BodyType.Static, position: [5, 2, 0, 0] },
            { sphereCenter: [0, 0, 0], radius: 0.5 },
        );
        const bob = sphere(
            world,
            names,
            "bob",
            { type: BodyType.Dynamic, position: [5, 2, 0, 0] },
            { sphereCenter: [0, 0, 0], radius: 0.5, density: 1 },
        );
        const joint = world.create();
        world.add(joint, WeldJoint, { a: anchor, b: bob, forceThreshold: 0, torqueThreshold: 0 });
        names.set(world.ref(joint), "weld");

        const positions = world.storage(GlobalTransform).translation;
        const last = new Map<number, number[]>([
            [falling, [0, 1, 0]],
            [bob, [5, 2, 0]],
        ]);
        const steps: unknown[] = [];
        for (let tick = 0; tick < 4; tick++) {
            world.step(Time.FIXED_DT);
            const summary = eventSummary(world, names);
            const records: unknown[] = [];
            for (const event of summary.contactBegin) records.push(["contactBegin", ...event]);
            for (const event of summary.contactEnd) records.push(["contactEnd", ...event]);
            for (const event of summary.hit) records.push(["hit", ...event]);
            for (const joint of summary.joints) records.push(["joint", joint]);
            const moves: unknown[] = [];
            for (const [eid, name] of [
                [falling, "falling"],
                [bob, "bob"],
            ] as const) {
                const position = [positions.x.get(eid), positions.y.get(eid), positions.z.get(eid)];
                if (position.some((value, i) => value !== last.get(eid)![i])) {
                    moves.push(["move", name, position]);
                    last.set(eid, position);
                }
            }
            steps.push([tick, [...records, ...moves]]);
        }
        expect(steps).toEqual([
            [
                0,
                [
                    ["joint", "weld"],
                    ["move", "falling", [0, 0.6315972208976746, 0]],
                    ["move", "bob", [5, 1.9999514818191528, 0]],
                ],
            ],
            [
                1,
                [
                    ["joint", "weld"],
                    ["move", "falling", [0, 0.46041661500930786, 0]],
                    ["move", "bob", [5, 1.9999364614486694, 0]],
                ],
            ],
            [
                2,
                [
                    ["contactBegin", "ground", "falling", 12.912276268005371],
                    [
                        "hit",
                        "ground",
                        "falling",
                        { x: 0, y: -0.012165412306785583, z: 0 },
                        { x: 0, y: 1, z: 0 },
                        10.333335876464844,
                        "0",
                        "18364758544493064720",
                    ],
                    ["joint", "weld"],
                    ["move", "falling", [0, 0.4756692349910736, 0]],
                    ["move", "bob", [5, 1.999931812286377, 0]],
                ],
            ],
            [
                3,
                [
                    ["joint", "weld"],
                    ["move", "falling", [0, 0.48594605922698975, 0]],
                    ["move", "bob", [5, 1.9999302625656128, 0]],
                ],
            ],
        ]);
    } finally {
        await dispose(world);
    }
});

test("PhysicsEvents order moving, sleeping, destroyed and continuous sensor visitors", async () => {
    const world = await physicsWorld();
    const names: Names = new Map();
    try {
        const definition = world.resource(PhysicsWorldDefinition);
        definition.enableContinuous = true;
        const sensor = sphere(
            world,
            names,
            "sensor",
            { type: BodyType.Static },
            { sphereCenter: [0, 0, 0], radius: 2, isSensor: 1, enableSensorEvents: 1 },
        );
        const moving = sphere(
            world,
            names,
            "moving",
            { type: BodyType.Dynamic, position: [-4, 0, 0, 0], linearVelocity: [2, 0, 0, 0] },
            { sphereCenter: [0, 0, 0], radius: 0.5, density: 1, enableSensorEvents: 1 },
        );
        const sleeping = sphere(
            world,
            names,
            "sleeping",
            { type: BodyType.Dynamic, position: [0, 1, 0, 0], isAwake: 0 },
            { sphereCenter: [0, 0, 0], radius: 0.25, density: 1, enableSensorEvents: 1 },
        );
        const bullet = sphere(
            world,
            names,
            "bullet",
            {
                type: BodyType.Dynamic,
                position: [-10, -1, 0, 0],
                linearVelocity: [1200, 0, 0, 0],
                isBullet: 1,
            },
            { sphereCenter: [0, 0, 0], radius: 0.1, density: 1, enableSensorEvents: 1 },
        );
        const sequence: unknown[] = [];
        const events = world.resource(PhysicsEvents);
        for (let step = 0; step < 240; step++) {
            if (step === 20) world.destroy(sleeping);
            if (step === 200) world.destroy(sensor);
            world.step(Time.FIXED_DT);
            for (const [kind, list] of [
                ["begin", events.sensorBegin],
                ["end", events.sensorEnd],
            ] as const) {
                for (const event of list.records.slice(0, list.count))
                    sequence.push([
                        step,
                        kind,
                        refName(world, names, event.sensorShape),
                        refName(world, names, event.visitorShape),
                    ]);
            }
        }
        expect(sequence).toEqual([
            [0, "begin", "sensor", "sleeping"],
            [1, "begin", "sensor", "bullet"],
            [2, "end", "sensor", "bullet"],
            [20, "end", "sensor", "sleeping"],
            [44, "begin", "sensor", "moving"],
            [195, "end", "sensor", "moving"],
        ]);
        expect([moving, bullet].map((eid) => names.get(world.ref(eid)))).toEqual([
            "moving",
            "bullet",
        ]);
    } finally {
        await dispose(world);
    }
});

test("PhysicsEvents retain the solved begin impulse after contact destruction before the next delivery", async () => {
    const world = await physicsWorld();
    const names: Names = new Map();
    try {
        const definition = world.resource(PhysicsWorldDefinition);
        definition.gravity = { x: 0, y: -10, z: 0 };
        definition.enableContinuous = false;
        sphere(
            world,
            names,
            "ground",
            { type: BodyType.Static, position: [0, -0.5, 0, 0] },
            { sphereCenter: [0, 0, 0], radius: 0.5 },
        );
        const falling = sphere(
            world,
            names,
            "falling",
            { type: BodyType.Dynamic, position: [0, 3, 0, 0] },
            { sphereCenter: [0, 0, 0], radius: 0.5, enableContactEvents: 1 },
        );
        const events = world.resource(PhysicsEvents);
        let tick = 0;
        while (events.contactBegin.count === 0 && tick < 120) {
            world.step(Time.FIXED_DT);
            tick++;
        }
        expect(tick - 1).toBe(43);
        world.destroy(falling);
        const begin = events.contactBegin.records[0];
        const impulse = begin?.normalImpulse;
        expect(impulse).toBe(11963.703125);
        expect(events.contactBegin.records[0]).toBe(begin);
        expect(events.contactBegin.records[0].normalImpulse).toBe(impulse);
        world.step(Time.FIXED_DT);
        expect(events.contactEnd.count).toBe(1);
        expect(events.contactEnd.records[0].normalImpulse).toBe(0);
        expect(world.resolve(events.contactEnd.records[0].shapeB)).toBe(0);
    } finally {
        await dispose(world);
    }
});

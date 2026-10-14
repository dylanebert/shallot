import { World } from "../../src/engine";
import { BodyType, defaultShapeDef, defaultSurfaceMaterial } from "../../src/standard/physics/common/types";
import { EventKind, eventCount, eventId } from "../../src/standard/physics/kernel/eventbuffers";
import { kernel, init, shutdown } from "../../src/standard/physics/kernel/kernel";
import { createTestBody, createTestWorld, destroyTestWorld, stepTestWorld, testShapeDef } from "../../src/standard/physics/testing";
import { createCompound } from "../../src/standard/physics/shapes/compound";
import { createCompoundShape, createHullShape, createSphereShape } from "../../src/standard/physics/shapes/shape";
import { createDistanceJoint, defaultDistanceJointDef } from "../../src/standard/physics/solver/distanceJoint";
import { defaultJointDef } from "../../src/standard/physics/solver/joint";
import { worldSetCustomFilterCallback, worldSetPreSolveCallback, worldCounters } from "../../src/standard/physics/world/world";
import { getSensorData } from "../../src/standard/physics/shapes/shape";
import { hashWorldState } from "../../src/standard/physics/world/hash";
import { makeBoxHull } from "../../src/standard/physics/shapes/build-hull";
import type { WorldState } from "../../src/standard/physics/world/world";

const owner = new World();
const count = Number(process.argv[2]);
await init(owner, { threads: count === 1 ? 0 : count });
const noCallback = process.argv[3] === "no-callback-pressure";
const acceptPressure = process.argv[3] === "accept-pressure" || noCallback;
const pressure = process.argv[3] === "pressure" || acceptPressure;
const world = createTestWorld(
    { enableSleep: false, ...(pressure ? { gravity: { x: 0, y: 0, z: 0 } } : {}) },
    owner,
);
let calls = 0;
let callbackStep = 0;
const float = new Float32Array(1);
const integer = new Uint32Array(float.buffer);
const bits = (value: number) => {
    float[0] = value;
    return integer[0];
};
const decision = (value: boolean) => {
    if (!world.locked) throw new Error("collision callback ran outside the locked step");
    if (process.env.CALLBACK_OPPOSITE === "1") value = !value;
    return value;
};
const pre = process.argv[3] === "pre";
if (pre)
    worldSetPreSolveCallback(world, (a, b, point, normal) => {
        calls++;
        console.log(
            `${callbackStep} P ${a.index1} ${a.generation} ${b.index1} ${b.generation} ${bits(point.x)} ${bits(point.y)} ${bits(point.z)} ${bits(normal.x)} ${bits(normal.y)} ${bits(normal.z)}`,
        );
        const answer =
            !(((point.x > 1 && point.x < 3) || (point.x > 19 && point.x < 21)) &&
                point.y < 1 &&
                normal.y > 0.9);
        return decision(answer);
    });
else if (!noCallback)
    worldSetCustomFilterCallback(world, (a, b) => {
        calls++;
        console.log(`${callbackStep} F ${a.index1} ${a.generation} ${b.index1} ${b.generation}`);
        const answer = acceptPressure
            ? true
            : pressure
              ? a.index1 > 98 && b.index1 > 98
              : a.index1 !== 4 && b.index1 !== 4 && a.index1 !== 6 && b.index1 !== 6;
        return (
            decision(answer) &&
            getSensorData(world, a.index1 - 1).length === 0 &&
            getSensorData(world, b.index1 - 1).length === 0
        );
    });

const shape = {
    enableCustomFiltering: !pre,
    enablePreSolveEvents: pre,
    enableContactEvents: true,
    enableHitEvents: true,
    enableSensorEvents: true,
};
const body = (def: Parameters<typeof createTestBody>[1] = {}) => createTestBody(world, def);
const sphere = (
    def: Parameters<typeof createTestBody>[1],
    shapeDef: Record<string, unknown>,
    geometry: { center: { x: number; y: number; z: number }; radius: number },
) => {
    const id = body(def);
    createSphereShape(world, id, testShapeDef(shapeDef), geometry);
    return id;
};
const hull = (
    def: Parameters<typeof createTestBody>[1],
    shapeDef: Record<string, unknown>,
    data: ReturnType<typeof makeBoxHull>,
) => {
    const id = body(def);
    createHullShape(world, id, testShapeDef(shapeDef), data);
    return id;
};

function eventTouches(state: WorldState, kind: number) {
    const k = kernel(state.ecsState);
    return Array.from({ length: eventCount(state, kind) }, (_, i) => ({
        shapeA: eventId(state, kind, i, 0).index1,
        shapeB: eventId(state, kind, i, 2).index1,
        contact: eventId(state, kind, i, 4).index1,
        normalImpulse: kind === EventKind.ContactBegin ? k.eventFloat(state.worldId, kind, i, 7) : 0,
    }));
}
function hitEvents(state: WorldState) {
    const k = kernel(state.ecsState);
    return Array.from({ length: eventCount(state, EventKind.ContactHit) }, (_, i) => ({
        shapeA: eventId(state, EventKind.ContactHit, i, 0).index1,
        shapeB: eventId(state, EventKind.ContactHit, i, 2).index1,
        contact: eventId(state, EventKind.ContactHit, i, 4).index1,
        point: {
            x: k.eventFloat(state.worldId, EventKind.ContactHit, i, 7),
            y: k.eventFloat(state.worldId, EventKind.ContactHit, i, 8),
            z: k.eventFloat(state.worldId, EventKind.ContactHit, i, 9),
        },
        normal: {
            x: k.eventFloat(state.worldId, EventKind.ContactHit, i, 10),
            y: k.eventFloat(state.worldId, EventKind.ContactHit, i, 11),
            z: k.eventFloat(state.worldId, EventKind.ContactHit, i, 12),
        },
        approachSpeed: k.eventFloat(state.worldId, EventKind.ContactHit, i, 13),
        userMaterialIdA:
            BigInt(k.eventWord(state.worldId, EventKind.ContactHit, i, 14) >>> 0) |
            (BigInt(k.eventWord(state.worldId, EventKind.ContactHit, i, 15) >>> 0) << 32n),
        userMaterialIdB:
            BigInt(k.eventWord(state.worldId, EventKind.ContactHit, i, 16) >>> 0) |
            (BigInt(k.eventWord(state.worldId, EventKind.ContactHit, i, 17) >>> 0) << 32n),
    }));
}
function sensorTouches(state: WorldState, kind: number) {
    return Array.from({ length: eventCount(state, kind) }, (_, i) => ({
        sensor: eventId(state, kind, i, 0).index1,
        visitor: eventId(state, kind, i, 2).index1,
    }));
}

try {
    if (pressure) {
        for (let i = 0; i < (acceptPressure ? 1000 : 100); ++i)
            sphere(
                { type: BodyType.Dynamic },
                { ...shape, enableCustomFiltering: !acceptPressure || i === 0 },
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
    } else {
        for (const x of [0, 20])
            hull({ position: { x, y: -0.5, z: 0 } }, shape, makeBoxHull(6, 0.5, 3));
        shape.enableCustomFiltering = false;
        shape.enablePreSolveEvents = false;
        for (let i = 0; i < 5; ++i)
            sphere(
                {
                    type: BodyType.Dynamic,
                    position: { x: i >= 3 ? 20 + 4 * (i - 3) : 2 * i, y: 3, z: 0 },
                    linearVelocity: { x: 0, y: i >= 3 ? -100 : 0, z: 0 },
                    isBullet: i === 4,
                },
                shape,
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        hull({ position: { x: 500, y: -0.5, z: 0 } }, shape, makeBoxHull(400, 0.5, 3));
        for (let i = 0; i < 300; ++i)
            sphere(
                { type: BodyType.Dynamic, position: { x: 100 + 2 * i, y: 0.5, z: 0 } },
                shape,
                { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
            );
        const sensorShape = {
            ...shape,
            isSensor: true,
            enableCustomFiltering: !pre,
            enablePreSolveEvents: pre,
        };
        hull(
            { position: { x: 2, y: 3, z: 0 } },
            sensorShape,
            makeBoxHull(5, 4, 2),
        );
        hull(
            { position: { x: 22, y: 1, z: 0 } },
            sensorShape,
            makeBoxHull(4, 0.5, 2),
        );
        sensorShape.enableCustomFiltering = false;
        sensorShape.enablePreSolveEvents = false;
        for (let i = 0; i < 300; ++i)
            sphere(
                { position: { x: 100 + 2 * i, y: 1, z: 0 } },
                sensorShape,
                { center: { x: 0, y: 0, z: 0 }, radius: 0.75 },
            );
        for (let i = 0; i < 3; ++i) {
            const compound = createCompound(
                i === 0
                    ? {
                          spheres: [
                              {
                                  sphere: { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                                  material: defaultSurfaceMaterial(),
                              },
                          ],
                      }
                    : i === 1
                      ? {
                            capsules: [
                                {
                                    capsule: {
                                        center1: { x: -0.5, y: 0, z: 0 },
                                        center2: { x: 0.5, y: 0, z: 0 },
                                        radius: 0.5,
                                    },
                                    material: defaultSurfaceMaterial(),
                                },
                            ],
                        }
                      : {
                            hulls: [
                                {
                                    hull: makeBoxHull(0.5, 0.5, 0.5),
                                    transform: {
                                        p: { x: 0.25, y: 0.1, z: 0 },
                                        q: { v: { x: 0, y: 0, z: 0 }, s: 1 },
                                    },
                                    material: defaultSurfaceMaterial(),
                                },
                            ],
                        },
            );
            if (!compound) throw new Error("callback scene compound construction failed");
            const parent = body({ position: { x: 8 + 4 * i, y: -0.5, z: 0 } });
            createCompoundShape(
                world,
                parent,
                testShapeDef({ ...shape, enableCustomFiltering: !pre, enablePreSolveEvents: pre }),
                compound,
            );
            const child = body({
                type: BodyType.Dynamic,
                position: {
                    x: 8 + 4 * i + (i === 2 ? 0.25 : 0),
                    y: i === 2 ? Math.fround(Math.fround(0.49) + Math.fround(0.1)) : 0.49,
                    z: 0,
                },
            });
            if (i === 2)
                createSphereShape(world, child, testShapeDef(shape), {
                    center: { x: 0, y: 0, z: 0 },
                    radius: 0.5,
                });
            else createHullShape(world, child, testShapeDef(shape), makeBoxHull(0.5, 0.5, 0.5));
        }
    }
    for (let step = 0; step < 90; ++step) {
        callbackStep = step;
        calls = 0;
        stepTestWorld(world, 1 / 60);
        const counters = worldCounters(world);
        const begin = eventTouches(world, EventKind.ContactBegin);
        const end = eventTouches(world, EventKind.ContactEnd);
        const hits = hitEvents(world);
        const sensorBegin = sensorTouches(world, EventKind.SensorBegin);
        const sensorEnd = sensorTouches(world, EventKind.SensorEnd);
        console.log(
            `${step} 0x${hashWorldState(world).toString(16).padStart(16, "0")} ${counters.contactCount} ${begin.length} ${end.length} ${hits.length} ${sensorBegin.length} ${sensorEnd.length} ${calls}`,
        );
        for (const event of begin)
            console.log(`${step} B ${event.shapeA} ${event.shapeB} ${event.contact}`);
        for (const event of end) console.log(`${step} E ${event.shapeA} ${event.shapeB} ${event.contact}`);
        for (const event of hits)
            console.log(
                `${step} H ${event.shapeA} ${event.shapeB} ${event.contact} ${bits(event.point.x)} ${bits(event.point.y)} ${bits(event.point.z)} ${bits(event.normal.x)} ${bits(event.normal.y)} ${bits(event.normal.z)} ${bits(event.approachSpeed)} ${event.userMaterialIdA} ${event.userMaterialIdB}`,
            );
        for (const event of sensorBegin)
            console.log(`${step} SB ${event.sensor} ${event.visitor}`);
        for (const event of sensorEnd) console.log(`${step} SE ${event.sensor} ${event.visitor}`);
    }
} finally {
    destroyTestWorld(world);
    await shutdown(owner);
}

import { World } from "../../src/engine";
import { BodyType, createCompound, defaultSurfaceMaterial, hash, init, makeBoxHull, PhysicsWorld, shutdown } from "../../src/standard/physics/api";

const owner = new World();
const count = Number(process.argv[2]);
await init(owner, { threads: count === 1 ? 0 : count });
const noCallback = process.argv[3] === "no-callback-pressure";
const acceptPressure = process.argv[3] === "accept-pressure" || noCallback;
const pressure = process.argv[3] === "pressure" || acceptPressure;
const world = new PhysicsWorld({ enableSleep: false, ...(pressure ? { gravity: { x: 0, y: 0, z: 0 } } : {}) }, owner);
let calls = 0;
let callbackStep = 0;
const float = new Float32Array(1);
const integer = new Uint32Array(float.buffer);
const bits = (value: number) => { float[0] = value; return integer[0]; };
const decision = (value: boolean) => {
    if (process.env.CALLBACK_OPPOSITE === "1") value = !value;
    const c = world.getContactEvents();
    const s = world.getSensorEvents();
    return value && !(c.beginEvents.length || c.endEvents.length || c.hitEvents.length || s.beginEvents.length || s.endEvents.length || world.getBodyEvents().count || world.getJointEvents().length);
};
try {
    const pre = process.argv[3] === "pre";
    if (pre) world.setPreSolveCallback((a, b, point, normal) => {
        calls++;
        console.log(`${callbackStep} P ${a.id.index1} ${a.id.generation} ${b.id.index1} ${b.id.generation} ${bits(point.x)} ${bits(point.y)} ${bits(point.z)} ${bits(normal.x)} ${bits(normal.y)} ${bits(normal.z)}`);
        const answer = !(((point.x > 1 && point.x < 3) || (point.x > 19 && point.x < 21)) && point.y < 1 && normal.y > 0.9);
        return decision(answer);
    });
    else if (!noCallback) world.setCustomFilterCallback((a, b) => {
        calls++;
        console.log(`${callbackStep} F ${a.id.index1} ${a.id.generation} ${b.id.index1} ${b.id.generation}`);
        const answer = acceptPressure ? true : pressure ? a.id.index1 > 98 && b.id.index1 > 98 : a.id.index1 !== 4 && b.id.index1 !== 4 && a.id.index1 !== 6 && b.id.index1 !== 6;
        return decision(answer) && a.getSensorOverlaps().length === 0 && b.getSensorOverlaps().length === 0;
    });
    const shape = { enableCustomFiltering: !pre, enablePreSolveEvents: pre, enableContactEvents: true, enableHitEvents: true, enableSensorEvents: true };
    if (pressure) {
        for (let i = 0; i < (acceptPressure ? 1000 : 100); ++i) world.createBody({ type: BodyType.Dynamic })
            .createSphere({ ...shape, enableCustomFiltering: !acceptPressure || i === 0 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    } else {
    for (const x of [0, 20]) {
        world.createBody({ position: { x, y: -0.5, z: 0 } }).createHull(shape, makeBoxHull(6, 0.5, 3));
    }
    shape.enableCustomFiltering = false;
    shape.enablePreSolveEvents = false;
    for (let i = 0; i < 5; ++i) {
        world.createBody({
            type: BodyType.Dynamic,
            position: { x: i >= 3 ? 20 + 4 * (i - 3) : 2 * i, y: 3, z: 0 },
            linearVelocity: { x: 0, y: i >= 3 ? -100 : 0, z: 0 },
            isBullet: i === 4,
        }).createSphere(shape, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    }
    world.createBody({ position: { x: 500, y: -0.5, z: 0 } })
        .createHull(shape, makeBoxHull(400, 0.5, 3));
    for (let i = 0; i < 300; ++i) {
        world.createBody({ type: BodyType.Dynamic, position: { x: 100 + 2 * i, y: 0.5, z: 0 } })
            .createSphere(shape, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    }
    const sensorShape = { ...shape, isSensor: true, enableCustomFiltering: !pre, enablePreSolveEvents: pre };
    world.createBody({ position: { x: 2, y: 3, z: 0 } })
        .createHull(sensorShape, makeBoxHull(5, 4, 2));
    world.createBody({ position: { x: 22, y: 1, z: 0 } })
        .createHull(sensorShape, makeBoxHull(4, 0.5, 2));
    sensorShape.enableCustomFiltering = false;
    sensorShape.enablePreSolveEvents = false;
    for (let i = 0; i < 300; ++i) {
        world.createBody({ position: { x: 100 + 2 * i, y: 1, z: 0 } })
            .createSphere(sensorShape, { center: { x: 0, y: 0, z: 0 }, radius: 0.75 });
    }
    for (let i = 0; i < 3; ++i) {
        const compound = createCompound(i === 0 ? {
            spheres: [{ sphere: { center: { x: 0, y: 0, z: 0 }, radius: 0.5 }, material: defaultSurfaceMaterial() }],
        } : i === 1 ? {
            capsules: [{ capsule: { center1: { x: -0.5, y: 0, z: 0 }, center2: { x: 0.5, y: 0, z: 0 }, radius: 0.5 }, material: defaultSurfaceMaterial() }],
        } : {
            hulls: [{ hull: makeBoxHull(0.5, 0.5, 0.5), transform: { p: { x: 0.25, y: 0.1, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } }, material: defaultSurfaceMaterial() }],
        });
        if (!compound) throw new Error("callback scene compound construction failed");
        world.createBody({ position: { x: 8 + 4 * i, y: -0.5, z: 0 } })
            .createCompound({ ...shape, enableCustomFiltering: !pre, enablePreSolveEvents: pre }, compound);
        const body = world.createBody({ type: BodyType.Dynamic, position: { x: 8 + 4 * i + (i === 2 ? 0.25 : 0), y: i === 2 ? Math.fround(Math.fround(0.49) + Math.fround(0.1)) : 0.49, z: 0 } });
        if (i === 2) body.createSphere(shape, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        else body.createHull(shape, makeBoxHull(0.5, 0.5, 0.5));
    }
    }
    for (let step = 0; step < 90; ++step) {
        callbackStep = step;
        calls = 0;
        world.step(1 / 60);
        const c = world.getCounters();
        const e = world.getContactEvents();
        const s = world.getSensorEvents();
        console.log(`${step} 0x${hash(world).toString(16).padStart(16, "0")} ${c.contactCount} ${e.beginEvents.length} ${e.endEvents.length} ${e.hitEvents.length} ${s.beginEvents.length} ${s.endEvents.length} ${calls}`);
        for (const v of e.beginEvents) console.log(`${step} B ${v.shapeA.id.index1} ${v.shapeB.id.index1} ${v.contact.id.index1}`);
        for (const v of e.endEvents) console.log(`${step} E ${v.shapeA.id.index1} ${v.shapeB.id.index1} ${v.contact.id.index1}`);
        for (const v of e.hitEvents) console.log(`${step} H ${v.shapeA.id.index1} ${v.shapeB.id.index1} ${v.contact.id.index1} ${bits(v.point.x)} ${bits(v.point.y)} ${bits(v.point.z)} ${bits(v.normal.x)} ${bits(v.normal.y)} ${bits(v.normal.z)} ${bits(v.approachSpeed)} ${v.userMaterialIdA} ${v.userMaterialIdB}`);
        for (const v of s.beginEvents) console.log(`${step} SB ${v.sensor.id.index1} ${v.visitor.id.index1}`);
        for (const v of s.endEvents) console.log(`${step} SE ${v.sensor.id.index1} ${v.visitor.id.index1}`);
    }
} finally {
    world.destroy();
    await shutdown(owner);
}

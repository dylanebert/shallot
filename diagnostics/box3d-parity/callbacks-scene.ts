import { World } from "../../src/engine";
import { BodyType, hash, init, makeBoxHull, PhysicsWorld, shutdown } from "../../src/standard/physics/api";

const owner = new World();
const count = Number(process.argv[2]);
await init(owner, { threads: count === 1 ? 0 : count });
const world = new PhysicsWorld({ enableSleep: false }, owner);
let calls = 0;
const float = new Float32Array(1);
const integer = new Uint32Array(float.buffer);
const bits = (value: number) => { float[0] = value; return integer[0]; };
try {
    const pre = process.argv[3] === "pre";
    if (pre) world.setPreSolveCallback((_a, _b, point, normal) => {
        calls++;
        const answer = !(((point.x > 1 && point.x < 3) || (point.x > 19 && point.x < 21)) && point.y < 1 && normal.y > 0.9);
        return process.env.CALLBACK_OPPOSITE === "1" ? !answer : answer;
    });
    else world.setCustomFilterCallback((a, b) => {
        calls++;
        const answer = a.id.index1 !== 4 && b.id.index1 !== 4 && a.id.index1 !== 6 && b.id.index1 !== 6;
        return process.env.CALLBACK_OPPOSITE === "1" ? !answer : answer;
    });
    const shape = { enableCustomFiltering: !pre, enablePreSolveEvents: pre, enableContactEvents: true, enableHitEvents: true };
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
    for (let step = 0; step < 90; ++step) {
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

import { World } from "../../src/engine";
import { BodyType, hash, init, makeBoxHull, PhysicsWorld, shutdown } from "../../src/standard/physics/api";

const owner = new World();
const count = Number(process.argv[2]);
await init(owner, { threads: count === 1 ? 0 : count });
const world = new PhysicsWorld({}, owner);
try {
    world.setCustomFilterCallback((a, b) => {
        const answer = a.id.index1 !== 4 && b.id.index1 !== 4;
        return process.env.CALLBACK_OPPOSITE === "1" ? !answer : answer;
    });
    const shape = { enableCustomFiltering: true, enableContactEvents: true, enableHitEvents: true };
    for (const x of [0, 20]) {
        world.createBody({ position: { x, y: -0.5, z: 0 } }).createHull(shape, makeBoxHull(6, 0.5, 3));
    }
    for (let i = 0; i < 4; ++i) {
        world.createBody({
            type: BodyType.Dynamic,
            position: { x: i === 3 ? 20 : 2 * i, y: 3, z: 0 },
            linearVelocity: { x: 0, y: i === 3 ? -100 : 0, z: 0 },
        }).createSphere(shape, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
    }
    for (let step = 0; step < 90; ++step) {
        world.step(1 / 60);
        const c = world.getCounters();
        const e = world.getContactEvents();
        const s = world.getSensorEvents();
        console.log(`${step} 0x${hash(world).toString(16).padStart(16, "0")} ${c.contactCount} ${e.beginEvents.length} ${e.endEvents.length} ${e.hitEvents.length} ${s.beginEvents.length} ${s.endEvents.length}`);
    }
} finally {
    world.destroy();
    await shutdown(owner);
}

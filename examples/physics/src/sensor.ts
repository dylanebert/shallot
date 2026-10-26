import {
    Body,
    BodyType,
    type EntityRef,
    Hulls,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    type System,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    PhysicsEvents,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";

export const SensorSample = {
    create: () => ({
        enabled: false,
        step: 0,
        seed: 42,
        maxBeginCount: 0,
        maxEndCount: 0,
        sensors: new Map<EntityRef, { active: boolean; row: number }>(),
        reactions: new Set<EntityRef>(),
        bodyRefs: [] as EntityRef[],
        shapeRefs: [] as EntityRef[],
        slots: new Map<EntityRef, number>(),
        freeSlots: [] as number[],
        nextSlot: 1,
    }),
};
function track(world: World, eid: number): void {
    const sample = world.resource(SensorSample);
    sample.slots.set(world.ref(eid), sample.freeSlots.pop() ?? sample.nextSlot++);
    sample.bodyRefs.push(world.ref(eid));
    sample.shapeRefs.push(world.ref(eid));
}
function createRow(world: World): void {
    const sample = world.resource(SensorSample);
    const f = Math.fround;
    for (let i = 0; i < 40; i++) {
        let x = sample.seed;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        sample.seed = x >>> 0;
        const offset = f(f(2 * f((sample.seed % 32768) / 32767)) - 1);
        const eid = world.create();
        world.add(eid, Transform, { translation: [5 * i - 100, f(210 + offset), 0] });
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            gravityScale: 0,
            linearVelocity: [0, -5, 0],
        });
        world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, enableSensorEvents: 1 });
        track(world, eid);
    }
}
export function benchmarkSensor(world: World): number[] {
    const sample = world.resource(SensorSample);
    sample.enabled = true;
    world.resource(PhysicsWorldDefinition).customFilterCallback = (owner, a, b) => {
        const sensors = owner.resource(SensorSample).sensors;
        const sensor = sensors.get(a) ?? sensors.get(b);
        return !sensor || sensor.active || sensor.row !== 20;
    };
    const f = Math.fround;
    const active = world.resource(Hulls).add({
        name: "active sensor",
        halfExtents: [f(f(0.48) * 3), f(f(0.48) * 3), f(f(0.48) * 3)],
    });
    const passive = world
        .resource(Hulls)
        .add({ name: "passive sensor", halfExtents: [0.5, 0.5, 0.5] });
    const bodies: number[] = [];
    function sensor(
        translation: [number, number, number],
        geometry: number,
        row: number,
        active: boolean,
    ): void {
        const eid = world.create();
        world.add(eid, Transform, { translation });
        world.add(eid, Body);
        world.add(eid, Shape, {
            geometry,
            isSensor: 1,
            enableSensorEvents: 1,
            enableCustomFiltering: Number(!active && row === 20),
        });
        sample.sensors.set(world.ref(eid), { row, active });
        track(world, eid);
        bodies.push(eid);
    }
    for (let i = 0; i < 81; i++) sensor([-120 + i * 3, 0, 0], active, 0, true);
    for (let row = 0; row < 40; row++)
        for (let i = 0; i < 40; i++) sensor([i * 5 - 100, row * 5 + 10, 0], passive, row, false);
    return bodies;
}
const AfterSensor: System = {
    name: "sensor-after-step",
    group: "fixed",
    after: [StepPhysicsSystem],
    update(world) {
        const sample = world.resource(SensorSample);
        if (!sample.enabled) return;
        const events = world.resource(PhysicsEvents);
        const zombies = new Set<EntityRef>();
        for (const event of events.sensorBegin.records.slice(0, events.sensorBegin.count)) {
            const sensor = sample.sensors.get(event.sensorShape)!;
            if (sensor.active) zombies.add(event.visitorBody);
            else sample.reactions.add(event.visitorBody);
        }
        for (const event of events.sensorEnd.records.slice(0, events.sensorEnd.count)) {
            if (world.resolve(event.visitorShape)) sample.reactions.delete(event.visitorBody);
        }
        // Box3D's set orders zombies by body slot; destroyed slots are reused LIFO.
        for (const ref of [...zombies].sort(
            (a, b) => sample.slots.get(a)! - sample.slots.get(b)!,
        )) {
            const eid = world.resolve(ref);
            if (eid) world.destroy(eid);
            sample.freeSlots.push(sample.slots.get(ref)!);
            sample.slots.delete(ref);
            sample.reactions.delete(ref);
        }
        if (++sample.step % 32 === 0) createRow(world);
        sample.maxBeginCount = Math.max(sample.maxBeginCount, events.sensorBegin.count);
        sample.maxEndCount = Math.max(sample.maxEndCount, events.sensorEnd.count);
    },
};
export const SensorSamplePlugin: Plugin = {
    name: "SensorSample",
    dependencies: [StandardPhysicsPlugin],
    systems: [AfterSensor],
    recovery: recoverResource(SensorSample),
};

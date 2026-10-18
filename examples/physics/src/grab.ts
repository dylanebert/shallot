import {
    Body,
    BodyType,
    Camera,
    Devices,
    GlobalTransform,
    InputPlugin,
    MotorJoint,
    type Plugin,
    type System,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { Orbit, OrbitMode } from "@dylanebert/shallot/extras";
import { viewportToWorld } from "@dylanebert/shallot/rendering";
import {
    castRayClosest,
    emptyQueryHit,
    getLocalPoint,
    getMassData,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    setAwake,
    setTargetTransform,
} from "@dylanebert/shallot/standard/physics";

export const Grab = {
    create: () => ({
        mouse: 0,
        joint: 0,
        body: 0,
        fraction: 0,
        point: [0, 0, 0] as [number, number, number],
        pending: false,
        held: false,
        camera: 0,
        orbitMode: OrbitMode.Free as number,
        origin: { x: 0, y: 0, z: 0 },
        translation: { x: 0, y: 0, z: 0 },
        hit: emptyQueryHit(),
    }),
};

export function release(world: World): void {
    const grab = world.resource(Grab);
    if (grab.joint) world.destroy(grab.joint);
    if (grab.mouse) world.destroy(grab.mouse);
    grab.mouse = grab.joint = grab.body = 0;
    if (grab.camera) {
        world.storage(Orbit).mode.set(grab.camera, grab.orbitMode);
        grab.camera = 0;
    }
}

export function pick(world: World): void {
    const grab = world.resource(Grab);
    release(world);
    if (!castRayClosest(world, grab.origin, grab.translation, grab.hit)) return;
    const body = grab.hit.body;
    if (world.storage(Body).type.get(body) !== BodyType.Dynamic) return;
    const mass = getMassData(world, body);
    if (!mass) return;
    const hit = grab.hit;
    grab.point = [hit.point.x, hit.point.y, hit.point.z];
    grab.fraction = hit.fraction;
    grab.body = body;
    const anchor = getLocalPoint(world, body, hit.point)!;
    grab.mouse = world.create();
    world.add(grab.mouse, Transform, { translation: grab.point });
    world.add(grab.mouse, Body, { type: BodyType.Kinematic, enableSleep: 0 });
    const gravity = world.resource(PhysicsWorldDefinition).gravity;
    const f = Math.fround;
    const g = f(
        Math.sqrt(
            f(f(f(gravity.x * gravity.x) + f(gravity.y * gravity.y)) + f(gravity.z * gravity.z)),
        ),
    );
    const mg = f(mass.mass * g);
    const trace = f(f(mass.inertia.cx.x + mass.inertia.cy.y) + mass.inertia.cz.z);
    const lever = mass.mass > 0 ? f(Math.sqrt(f(trace / f(3 * mass.mass)))) : 0;
    grab.joint = world.create();
    world.add(grab.joint, MotorJoint, {
        a: grab.mouse,
        b: body,
        localAnchorB: [anchor.x, anchor.y, anchor.z],
        linearHertz: 7.5,
        linearDampingRatio: 1,
        maxSpringForce: f(100 * mg),
        maxVelocityTorque: f(f(0.5 * lever) * mg),
    });
    setAwake(world, body, true);
}

const SampleGrab: System = {
    name: "sample-grab",
    group: "setup",
    update(world) {
        const grab = world.resource(Grab);
        const devices = world.resource(Devices);
        grab.held = devices.enabled && devices.pointer.left;
        if (!grab.held) grab.pending = false;
        let pointerX = devices.pointer.x;
        let pointerY = devices.pointer.y;
        for (const event of devices.pointer.buttonEvents) {
            if (event.button === "left" && event.pressed && grab.held) {
                grab.pending = true;
                pointerX = event.x;
                pointerY = event.y;
            }
        }
        for (const camera of world.query([Camera])) {
            const ray = viewportToWorld(world, camera, pointerX, pointerY);
            if (!ray) continue;
            grab.origin.x = ray.origin[0];
            grab.origin.y = ray.origin[1];
            grab.origin.z = ray.origin[2];
            const q = world.storage(GlobalTransform).rotation;
            const x = q.x.get(camera),
                y = q.y.get(camera),
                z = q.z.get(camera),
                w = q.w.get(camera);
            const forwardX = -2 * (x * z + w * y);
            const forwardY = -2 * (y * z - w * x);
            const forwardZ = -(1 - 2 * (x * x + y * y));
            const cameraStorage = world.storage(Camera);
            // Keep the fraction between the near and far planes, as Camera::BuildPickRay does.
            const reach =
                (cameraStorage.far.get(camera) - cameraStorage.near.get(camera)) /
                (ray.dir[0] * forwardX + ray.dir[1] * forwardY + ray.dir[2] * forwardZ);
            grab.translation.x = ray.dir[0] * reach;
            grab.translation.y = ray.dir[1] * reach;
            grab.translation.z = ray.dir[2] * reach;
            if (!grab.held) release(world);
            if (grab.pending) {
                pick(world);
                grab.pending = false;
                if (grab.body && world.has(camera, Orbit)) {
                    grab.camera = camera;
                    grab.orbitMode = world.storage(Orbit).mode.get(camera);
                    world.storage(Orbit).mode.set(camera, OrbitMode.Locked);
                }
            }
            break;
        }
    },
};
const UpdateGrab: System = {
    name: "update-grab",
    group: "fixed",
    before: StandardPhysicsPlugin.systems?.filter((system) => system.group === "fixed"),
    update(world) {
        const grab = world.resource(Grab);
        if (!grab.held) {
            release(world);
            return;
        }
        if (grab.pending) {
            pick(world);
            grab.pending = false;
        }
        if (!grab.mouse) return;
        const o = grab.origin,
            t = grab.translation;
        grab.point[0] = o.x + grab.fraction * t.x;
        grab.point[1] = o.y + grab.fraction * t.y;
        grab.point[2] = o.z + grab.fraction * t.z;
        setTargetTransform(world, grab.mouse, grab.point, [0, 0, 0, 1]);
    },
};
export const GrabPlugin: Plugin = {
    name: "Grab",
    dependencies: [InputPlugin],
    systems: [SampleGrab, UpdateGrab],
    recovery: (world) => ({
        snapshot: () => structuredClone(world.resource(Grab)),
        restore: (saved: ReturnType<typeof Grab.create>) =>
            Object.assign(world.resource(Grab), structuredClone(saved)),
    }),
};

import {
    Body,
    BodyType,
    Camera,
    Devices,
    DistanceJoint,
    type EntityRef,
    GlobalTransform,
    Hulls,
    InputPlugin,
    MotorJoint,
    PhysicsMeshes,
    type Plugin,
    PrismaticJoint,
    RevoluteJoint,
    Shape,
    ShapeKind,
    ShapeMaterials,
    type System,
    Transform,
    WeldJoint,
    type World,
} from "@dylanebert/shallot";
import {
    applyLinearImpulseToCenter,
    applyTorque,
    computeBodyAABB,
    getLocalPoint,
    getMassData,
    PhysicsEvents,
    type PhysicsHitEvent,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAngularVelocity,
    setLinearVelocity,
    setTargetTransform,
} from "@dylanebert/shallot/standard/physics";

import type { Controls } from "./scenes";

export const EventSample = {
    create: () => ({
        scene: "",
        bodies: [] as number[],
        velocityScales: [] as number[],
        initialized: false,
        sensor: 0,
        player: 0,
        walls: 0,
        core: 0,
        target: 0,
        motor: 0,
        debrisHull: 0,
        seed: 12345,
        wait: 0.5,
        massExtent: 0,
        torque: 30000,
        debris: Array<number>(200).fill(0),
        keys: { W: 0, S: 0, A: 0, D: 0 },
        forward: [0, 0, -1] as [number, number, number],
        right: [1, 0, 0] as [number, number, number],
        speed: 0,
        time: 0,
        hits: [] as PhysicsHitEvent[],
        bodyRefs: [] as EntityRef[],
        shapeRefs: [] as EntityRef[],
        jointRefs: [] as EntityRef[],
    }),
};

const BeforeSample: System = {
    name: "sample-before-step",
    group: "fixed",
    after: [SyncPhysicsConstraintsSystem],
    before: [StepPhysicsSystem],
    update(world) {
        const sample = world.resource(EventSample);
        if (sample.scene === "contact") {
            driveContact(world);
            return;
        }
        if (sample.scene === "motor") {
            driveMotor(world);
            return;
        }
        if (sample.scene !== "hit" || sample.initialized) return;
        sample.initialized = true;
        const f = Math.fround;
        for (let i = 0; i < sample.bodies.length; i++) {
            const body = sample.bodies[i]!;
            const center = getMassData(world, body)!.center;
            const omega = f(-sample.velocityScales[i]!);
            setAngularVelocity(world, body, { x: 0, y: 0, z: omega });
            setLinearVelocity(world, body, {
                x: f(-f(omega * center.y)),
                y: f(omega * center.x),
                z: 0,
            });
        }
    },
};
const AfterSample: System = {
    name: "sample-after-step",
    group: "fixed",
    after: [StepPhysicsSystem],
    update(world) {
        const sample = world.resource(EventSample);
        const records = world.resource(PhysicsEvents);
        if (sample.scene === "sensor") {
            for (let i = 0; i < records.sensorBegin.count; i++) {
                const event = records.sensorBegin.records[i]!;
                if (world.resolve(event.sensorShape) === sample.sensor) {
                    const visitor = world.resolve(event.visitorBody);
                    if (visitor) world.destroy(visitor);
                    break;
                }
            }
        }
        if (sample.scene === "joint") {
            for (let i = 0; i < records.joints.count; i++) {
                const joint = world.resolve(records.joints.records[i]!.joint);
                if (joint) world.destroy(joint);
            }
        }
        if (sample.scene === "contact") {
            stepContact(world);
            return;
        }
        if (sample.scene !== "hit") return;
        const events = records.contactHit;
        for (let i = 0; i < events.count && sample.hits.length < 32; i++)
            sample.hits.push(structuredClone(events.records[i]!));
    },
};
export const EventSamplePlugin: Plugin = {
    name: "EventSample",
    dependencies: [StandardPhysicsPlugin],
    systems: [BeforeSample, AfterSample],
    recovery: (world) => ({
        snapshot: () => structuredClone(world.resource(EventSample)),
        restore: (state: ReturnType<typeof EventSample.create>) =>
            Object.assign(world.resource(EventSample), structuredClone(state)),
    }),
};

const SampleInputSystem: System = {
    name: "sample-input",
    group: "setup",
    update(world) {
        const sample = world.resource(EventSample);
        if (sample.scene !== "contact") return;
        const devices = world.resource(Devices);
        for (const key of ["W", "S", "A", "D"] as const)
            sample.keys[key] = Number(devices.enabled && devices.keys.held.has(`Key${key}`));
        const [camera] = world.query([Camera, GlobalTransform]);
        if (!camera) return;
        const q = world.storage(GlobalTransform).rotation;
        const x = q.x.get(camera),
            y = q.y.get(camera),
            z = q.z.get(camera),
            w = q.w.get(camera);
        const fx = -2 * (x * z + w * y),
            fz = -(1 - 2 * (x * x + y * y));
        const rx = 1 - 2 * (y * y + z * z),
            rz = 2 * (x * z - w * y);
        const forwardLength = Math.hypot(fx, fz),
            rightLength = Math.hypot(rx, rz);
        if (forwardLength) sample.forward = [fx / forwardLength, 0, fz / forwardLength];
        if (rightLength) sample.right = [rx / rightLength, 0, rz / rightLength];
    },
};
export const SampleInputPlugin: Plugin = {
    name: "SampleInput",
    dependencies: [InputPlugin, EventSamplePlugin],
    systems: [SampleInputSystem],
    recovery: "stateless",
};

export function hit(world: World): number[] {
    const sample = world.resource(EventSample);
    sample.scene = "hit";
    const ground = world.create();
    world.add(ground, Body);
    const vertices: number[] = [],
        indices: number[] = [],
        materialIndices: number[] = [];
    for (let x = 0; x <= 20; x++)
        for (let z = 0; z <= 20; z++) vertices.push(-80 + 8 * x, 0, -80 + 8 * z);
    for (let x = 0; x < 20; x++)
        for (let z = 0; z < 20; z++) {
            const a = z + 21 * x,
                b = a + 1,
                c = b + 21,
                d = c - 1;
            indices.push(a, b, c, c, d, a);
            materialIndices.push((20 * x + z) % 6, (20 * x + z) % 6);
        }
    const geometry = world.resource(PhysicsMeshes).add({
        name: "mixed-material ground",
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
        materialIndices: new Uint8Array(materialIndices),
        useMedianSplit: true,
        identifyEdges: true,
    });
    const materialSet = world.resource(ShapeMaterials).add({
        name: "ground materials",
        materials: Array.from({ length: 6 }, (_, i) => ({
            friction: 0.6,
            restitution: 0,
            rollingResistance: 0,
            tangentVelocity: [0, 0, 0],
            userMaterialId: BigInt(i + 1),
            customColor: 0,
        })),
    });
    world.add(ground, Shape, { kind: ShapeKind.Mesh, geometry, materialSet });
    const f = Math.fround;
    let r = f(0.75),
        y = r,
        offset = f(0.05),
        velocityScale = f(0.5);
    function body(): number {
        const eid = world.create();
        world.add(eid, Transform);
        world.add(eid, Body, { type: BodyType.Dynamic });
        sample.bodies.push(eid);
        sample.velocityScales.push(velocityScale);
        return eid;
    }
    let current = body();
    for (let i = 0; i < 22; i++) {
        const eid = world.create();
        // Native batches ApplyMassFromShapes after each cluster. All shapes update mass here:
        // sync completes the cluster before either velocity initialization or the solve reads it.
        world.add(eid, Shape, {
            body: current,
            kind: ShapeKind.Capsule,
            capsuleCenter1: [offset, y, 0],
            capsuleCenter2: [0, f(y + 1.5), -offset],
            radius: r,
            enableHitEvents: 1,
            rollingResistance: 0.2,
            userMaterialIdLow: 42,
        });
        if ((i + 1) % 3 === 0 || i === 21) {
            if (i < 21) {
                const previous = current;
                velocityScale = f(velocityScale * 0.75);
                current = body();
                const anchor = f(f(y + 1.5) + r);
                world.add(world.create(), WeldJoint, {
                    a: previous,
                    b: current,
                    localAnchorA: [0, anchor, 0],
                    localAnchorB: [0, anchor, 0],
                    angularHertz: 10,
                    angularDampingRatio: 2,
                });
            }
        }
        y = f(y + f(1.5 + f(2 * r)));
        r = f(f(0.95) * r);
        offset = -offset;
    }
    remember(world);
    return sample.bodies;
}

function eventGround(world: World, extent: number): void {
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -1, 0] });
    world.add(ground, Body);
    const geometry = world
        .resource(Hulls)
        .add({ name: "ground", halfExtents: [extent, 1, extent] });
    world.add(ground, Shape, { geometry });
}

export function sensor(world: World): number[] {
    const sample = world.resource(EventSample);
    sample.scene = "sensor";
    const geometry = world.resource(Hulls).add({ name: "visitor", halfExtents: [0.5, 0.5, 0.5] });
    const visitor = world.create();
    world.add(visitor, Transform, { translation: [0, 12.5, 0] });
    world.add(visitor, Body, { type: BodyType.Dynamic });
    world.add(visitor, Shape, { geometry, enableSensorEvents: 1 });
    const sensor = world.create();
    world.add(sensor, Transform, { translation: [0, 2, 0] });
    world.add(sensor, Body, { type: BodyType.Kinematic });
    const box = world.resource(Hulls).add({ name: "sensor", halfExtents: [2, 2, 2] });
    world.add(sensor, Shape, { geometry: box, isSensor: 1, enableSensorEvents: 1 });
    sample.sensor = sensor;
    remember(world);
    return [visitor];
}

export function joint(world: World): number[] {
    const sample = world.resource(EventSample);
    sample.scene = "joint";
    eventGround(world, 20);
    const ground = world.create();
    world.add(ground, Body);
    const geometry = world.resource(Hulls).add({ name: "breakable box", halfExtents: [1, 1, 0.5] });
    const bodies: number[] = [];
    // Motor and Wheel are compiled out in Box3D's Events / Joint; their columns stay empty.
    for (const [kind, x] of [
        [DistanceJoint, -12.5],
        [PrismaticJoint, -2.5],
        [RevoluteJoint, 2.5],
        [WeldJoint, 7.5],
    ] as const) {
        const body = world.create();
        world.add(body, Transform, { translation: [x, 10, 0] });
        world.add(body, Body, { type: BodyType.Dynamic, enableSleep: 0 });
        world.add(body, Shape, { geometry, density: 1 });
        const base = {
            a: ground,
            b: body,
            forceThreshold: 3000,
            torqueThreshold: 10000,
            collideConnected: 1,
        };
        const eid = world.create();
        if (kind === DistanceJoint)
            world.add(eid, DistanceJoint, {
                ...base,
                localAnchorA: [x, 13, 0],
                localAnchorB: [0, 1, 0],
                length: 2,
            });
        else if (kind === WeldJoint)
            world.add(eid, WeldJoint, {
                ...base,
                localAnchorA: [x - 1, 10, 0],
                localAnchorB: [-1, 0, 0],
                angularHertz: 2,
                angularDampingRatio: 0.5,
            });
        else if (kind === PrismaticJoint)
            world.add(eid, PrismaticJoint, {
                ...base,
                localAnchorA: [x - 1, 10, 0],
                localAnchorB: [-1, 0, 0],
            });
        else
            world.add(eid, RevoluteJoint, {
                ...base,
                localAnchorA: [x - 1, 10, 0],
                localAnchorB: [-1, 0, 0],
            });
        bodies.push(body);
    }
    remember(world);
    return bodies;
}

const f = Math.fround;
type Point = [number, number, number];
function cross(a: Point, b: Point): Point {
    return [
        f(f(a[1] * b[2]) - f(a[2] * b[1])),
        f(f(a[2] * b[0]) - f(a[0] * b[2])),
        f(f(a[0] * b[1]) - f(a[1] * b[0])),
    ];
}
function rotate(q: [number, number, number, number], v: Point, inverse = false): Point {
    const u: Point = [q[0], q[1], q[2]];
    const a = cross(u, v);
    const b = a.map((n, i) => (inverse ? f(n - f(q[3] * v[i]!)) : f(n + f(q[3] * v[i]!)))) as Point;
    const c = cross(u, b);
    return v.map((n, i) => f(n + f(2 * c[i]!))) as Point;
}
function pose(world: World, eid: number) {
    const p = world.storage(GlobalTransform).translation;
    const q = world.storage(GlobalTransform).rotation;
    return {
        p: [p.x.get(eid), p.y.get(eid), p.z.get(eid)] as Point,
        q: [q.x.get(eid), q.y.get(eid), q.z.get(eid), q.w.get(eid)] as [
            number,
            number,
            number,
            number,
        ],
    };
}
function relativeRotation(
    a: [number, number, number, number],
    b: [number, number, number, number],
): [number, number, number, number] {
    const av: Point = [a[0], a[1], a[2]],
        bv: Point = [b[0], b[1], b[2]];
    const c = cross(bv, av);
    const v = c.map((n, i) => f(f(n + f(a[3] * bv[i]!)) - f(b[3] * av[i]!)));
    return [
        v[0]!,
        v[1]!,
        v[2]!,
        f(f(a[3] * b[3]) + f(f(f(av[0] * bv[0]) + f(av[1] * bv[1])) + f(av[2] * bv[2]))),
    ];
}
function massExtent(world: World): number {
    const player = world.resource(EventSample).player;
    const bounds = computeBodyAABB(world, player)!;
    const extents = ["x", "y", "z"].map((axis) => {
        const key = axis as "x" | "y" | "z";
        return f(0.5 * f(bounds.upperBound[key] - bounds.lowerBound[key]));
    });
    return f(
        f(getMassData(world, player)!.mass / 3) * f(f(extents[0]! + extents[1]!) + extents[2]!),
    );
}
function driveContact(world: World): void {
    const sample = world.resource(EventSample);
    if (!sample.initialized) {
        sample.massExtent = massExtent(world);
        sample.initialized = true;
    }
    const throttle = sample.keys.W - sample.keys.S,
        strafe = sample.keys.D - sample.keys.A;
    const direction = sample.forward.map((n, i) =>
        f(f(throttle * n) + f(strafe * sample.right[i]!)),
    ) as Point;
    const length2 = f(
        f(f(direction[0] * direction[0]) + f(direction[1] * direction[1])) +
            f(direction[2] * direction[2]),
    );
    if (length2 <= 0) return;
    const inverseLength = f(1 / f(Math.sqrt(length2)));
    const unit = direction.map((n) => f(inverseLength * n)) as Point;
    const axis = cross([0, 1, 0], unit);
    const scale = f(massExtent(world) / sample.massExtent);
    const torque = f(scale * sample.torque);
    applyTorque(world, sample.player, {
        x: f(torque * axis[0]),
        y: f(torque * axis[1]),
        z: f(torque * axis[2]),
    });
}

export function contact(world: World): number[] {
    const sample = world.resource(EventSample);
    sample.scene = "contact";
    sample.debrisHull = world.resource(Hulls).add({ name: "debris", halfExtents: [0.4, 0.6, 0.5] });
    eventGround(world, 80);
    sample.walls = world.create();
    world.add(sample.walls, Body);
    for (const [halfExtents, position] of [
        [
            [0.5, 5, 80.5],
            [80.5, 5, 0],
        ],
        [
            [0.5, 5, 80.5],
            [-80.5, 5, 0],
        ],
        [
            [80.5, 5, 0.5],
            [0, 5, 80.5],
        ],
        [
            [80.5, 5, 0.5],
            [0, 5, -80.5],
        ],
    ] as const) {
        const geometry = world
            .resource(Hulls)
            .add({ name: "wall", halfExtents, transform: { position, rotation: [0, 0, 0, 1] } });
        world.add(world.create(), Shape, { body: sample.walls, geometry });
    }
    sample.player = world.create();
    sample.core = sample.player;
    world.add(sample.player, Transform, { translation: [0, 1, 0] });
    world.add(sample.player, Body, { type: BodyType.Dynamic, angularDamping: 0.1 });
    world.add(sample.player, Shape, {
        kind: ShapeKind.Sphere,
        radius: 1,
        enableContactEvents: 1,
        rollingResistance: 0.1,
        customColor: 0x05dda0dd,
    });
    remember(world);
    return [sample.player];
}
function random(world: World, lo: number, hi: number): number {
    const sample = world.resource(EventSample);
    let x = sample.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    sample.seed = x >>> 0;
    return f(f(f(hi - lo) * f((sample.seed % 32768) / 32767)) + lo);
}
function spawnDebris(world: World): void {
    const sample = world.resource(EventSample);
    const index = sample.debris.indexOf(0);
    if (index === -1) return;
    const position: Point = [random(world, -78, 78), random(world, 1, 6), random(world, -78, 78)];
    const u1 = random(world, 0, 1),
        u2 = random(world, 0, f(2 * f(Math.PI))),
        u3 = random(world, 0, f(2 * f(Math.PI)));
    const a = f(Math.sqrt(f(1 - u1))),
        b = f(Math.sqrt(u1));
    const cs2 = rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(2 * u2));
    const cs3 = rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(2 * u3));
    const rotation: [number, number, number, number] = [
        f(a * cs2.v.z),
        f(a * cs2.s),
        f(b * cs3.v.z),
        f(b * cs3.s),
    ];
    const linearVelocity: Point = [
        random(world, -2, 2),
        random(world, -2, 2),
        random(world, -2, 2),
    ];
    const angularVelocity: Point = [
        random(world, -1, 1),
        random(world, -1, 1),
        random(world, -1, 1),
    ];
    const eid = world.create();
    sample.debris[index] = eid;
    world.add(eid, Transform, { translation: position, rotation });
    world.add(eid, Body, { type: BodyType.Dynamic, linearVelocity, angularVelocity });
    if ((index + 1) % 3 === 0)
        world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, restitution: 0.3 });
    else if ((index + 1) % 2 === 0)
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, -0.25, 0],
            capsuleCenter2: [0, 0.25, 0],
            radius: 0.25,
            restitution: 0.3,
        });
    else {
        world.add(eid, Shape, { geometry: sample.debrisHull, restitution: 0.3 });
    }
    sample.bodyRefs.push(world.ref(eid));
    sample.shapeRefs.push(world.ref(eid));
}
function attachDebris(world: World, index: number): void {
    const sample = world.resource(EventSample);
    const debris = sample.debris[index]!;
    if (!debris) return;
    const player = pose(world, sample.player),
        other = pose(world, debris);
    const relative = getLocalPoint(world, sample.player, {
        x: other.p[0],
        y: other.p[1],
        z: other.p[2],
    })!;
    const translation: Point = [relative.x, relative.y, relative.z];
    const rotation = relativeRotation(player.q, other.q);
    const source = world.get(debris, Shape)!;
    const eid = world.create();
    const base = { body: sample.player, enableContactEvents: 1, customColor: 0xffd700 };
    function point(p: Point): Point {
        return rotate(rotation, p).map((n, i) => f(n + translation[i]!)) as Point;
    }
    // Native batches mass after all attachments/removals. Here each authored shape updates mass;
    // the whole structural sync finishes before the next drive or solve reads the result.
    if (source.kind === ShapeKind.Sphere)
        world.add(eid, Shape, {
            ...base,
            kind: ShapeKind.Sphere,
            radius: source.radius,
            sphereCenter: point(source.sphereCenter),
        });
    else if (source.kind === ShapeKind.Capsule)
        world.add(eid, Shape, {
            ...base,
            kind: ShapeKind.Capsule,
            radius: source.radius,
            capsuleCenter1: point(source.capsuleCenter1),
            capsuleCenter2: point(source.capsuleCenter2),
        });
    else {
        world.add(eid, Shape, {
            ...base,
            geometry: source.geometry,
            hullPosition: translation,
            hullRotation: rotation,
        });
    }
    sample.shapeRefs.push(world.ref(eid));
    world.destroy(debris);
    sample.debris[index] = 0;
}
function stepContact(world: World): void {
    const sample = world.resource(EventSample);
    const events = world.resource(PhysicsEvents).contactBegin;
    const attach: number[] = [],
        destroy: number[] = [];
    for (let i = 0; i < events.count; i++) {
        const event = events.records[i]!;
        const a = world.resolve(event.bodyA) === sample.player;
        const playerShape = world.resolve(a ? event.shapeA : event.shapeB);
        const other = world.resolve(a ? event.bodyB : event.bodyA);
        const index = sample.debris.indexOf(other);
        if (other && index !== -1 && attach.length < 200) attach.push(index);
        else if (
            other === sample.walls &&
            playerShape !== sample.core &&
            !destroy.includes(playerShape) &&
            destroy.length < 200
        )
            destroy.push(playerShape);
    }
    for (const index of attach) attachDebris(world, index);
    for (const shape of destroy) if (shape) world.destroy(shape);
    sample.wait = f(sample.wait - f(1 / 60));
    if (sample.wait < 0) {
        spawnDebris(world);
        sample.wait = f(sample.wait + 0.5);
    }
}

export function motor(world: World): number[] {
    const sample = world.resource(EventSample);
    sample.scene = "motor";
    eventGround(world, 20);
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -1, 0] });
    world.add(ground, Body);
    sample.target = world.create();
    world.add(sample.target, Transform, { translation: [0, 10, 0] });
    world.add(sample.target, Body, { type: BodyType.Kinematic });
    sample.player = world.create();
    const geometry = world
        .resource(Hulls)
        .add({ name: "motorized body", halfExtents: [1, 0.25, 0.25] });
    world.add(sample.player, Transform, { translation: [0, 10, 0] });
    world.add(sample.player, Body, { type: BodyType.Dynamic });
    world.add(sample.player, Shape, { geometry });
    sample.motor = world.create();
    world.add(sample.motor, MotorJoint, {
        a: sample.target,
        b: sample.player,
        linearHertz: 4,
        linearDampingRatio: 0.7,
        angularHertz: 4,
        angularDampingRatio: 0.7,
        maxSpringForce: 400000,
        maxSpringTorque: 500000,
    });
    const spring = world.create();
    const box = world.resource(Hulls).add({ name: "spring body", halfExtents: [0.5, 0.5, 0.5] });
    world.add(spring, Transform, { translation: [-2, 2, 0] });
    world.add(spring, Body, { type: BodyType.Dynamic });
    world.add(spring, Shape, { geometry: box });
    world.add(world.create(), MotorJoint, {
        a: ground,
        b: spring,
        localAnchorA: [-1.75, 3.25, 0],
        localAnchorB: [0.25, 0.25, 0],
        collideConnected: 1,
        linearHertz: 7.5,
        linearDampingRatio: 0.7,
        angularHertz: 7.5,
        angularDampingRatio: 0.7,
        maxSpringForce: 200000,
        maxSpringTorque: 10000,
    });
    remember(world);
    return [sample.player, spring];
}
function driveMotor(world: World): void {
    // The native Sample shim uses double sine rounded to float for gameplay, like Math.sin here;
    // Box3D's deterministic quaternion construction remains the physics backend's operation.
    const sample = world.resource(EventSample);
    sample.time = f(sample.time + f(sample.speed * f(1 / 60)));
    const position = {
        x: f(6 * f(Math.sin(f(2 * sample.time)))),
        y: f(10 + f(4 * f(Math.sin(sample.time)))),
        z: 0,
    };
    const rotation = rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(2 * sample.time));
    setTargetTransform(
        world,
        sample.target,
        [position.x, position.y, position.z],
        [rotation.v.x, rotation.v.y, rotation.v.z, rotation.s],
        f(1 / 60),
    );
}

export const eventScenes = {
    joint: {
        construct: joint,
        camera: { yaw: 0, pitch: 30, distance: 40, target: [0, 5, 0] },
        controls: undefined,
    },
    sensor: {
        construct: sensor,
        camera: { yaw: 0, pitch: 30, distance: 20, target: [0, 5, 0] },
        controls: undefined,
    },
    hit: {
        construct: hit,
        camera: { yaw: 0, pitch: 30, distance: 100, target: [0, 5, 0] },
        controls: undefined,
    },
    contact: {
        construct: contact,
        camera: { yaw: 0, pitch: 35, distance: 60, target: [0, 0, 0] },
        controls: {
            torque: {
                label: "Torque",
                value: 30000,
                min: 0,
                max: 60000,
                step: 1000,
                set(world, _bodies, value) {
                    world.resource(EventSample).torque = value;
                },
            },
        } satisfies Controls,
    },
    motor: {
        construct: motor,
        camera: { yaw: 0, pitch: 0, distance: 25, target: [0, 8, 0] },
        controls: {
            Speed: {
                label: "Speed",
                value: 0,
                min: -5,
                max: 5,
                step: 1,
                set(world, _bodies, value) {
                    world.resource(EventSample).speed = value;
                },
            },
            "Max Force": {
                label: "Max Force",
                value: 400000,
                min: 0,
                max: 1000000,
                step: 1000,
                set(world, _bodies, value) {
                    world
                        .storage(MotorJoint)
                        .maxSpringForce.set(world.resource(EventSample).motor, value);
                },
            },
            "Max Torque": {
                label: "Max Torque",
                value: 500000,
                min: 0,
                max: 1000000,
                step: 1000,
                set(world, _bodies, value) {
                    world
                        .storage(MotorJoint)
                        .maxSpringTorque.set(world.resource(EventSample).motor, value);
                },
            },
            "Apply Impulse": {
                label: "Apply impulse",
                value: 0,
                button: true,
                set(world) {
                    applyLinearImpulseToCenter(world, world.resource(EventSample).player, {
                        x: 100000,
                        y: 0,
                        z: 0,
                    });
                },
            },
        } satisfies Controls,
    },
} as const;

function remember(world: World): void {
    const sample = world.resource(EventSample);
    sample.bodyRefs = Array.from(world.query([Body]), (eid) => world.ref(eid));
    sample.shapeRefs = Array.from(world.query([Shape]), (eid) => world.ref(eid));
    sample.jointRefs = [DistanceJoint, MotorJoint, PrismaticJoint, RevoluteJoint, WeldJoint]
        .flatMap((kind) => Array.from(world.query([kind])))
        .sort((a, b) => a - b)
        .map((eid) => world.ref(eid));
}

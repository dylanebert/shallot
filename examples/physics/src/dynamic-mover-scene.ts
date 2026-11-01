import {
    Body,
    BodyType,
    Camera,
    Hulls,
    InputPlugin,
    type Plugin,
    PrismaticJoint,
    RevoluteJoint,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    WeldJoint,
    type World,
} from "@dylanebert/shallot";
import { Orbit } from "@dylanebert/shallot/extras";
import {
    computeCosSin,
    explode,
    getMassData,
    PhysicsEvents,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    SyncPhysicsSystem,
    setLinearVelocity,
    setTargetTransform,
    setTransform,
} from "@dylanebert/shallot/standard/physics";
import { DynamicMover } from "./dynamic-mover";
import { add, cross, dot, length, mul, pose, rotate, sub, type Vec, v } from "./mover";
import { readPlayerControls } from "./player-controls";
import type { Controls } from "./scenes";

const f = Math.fround;
// roundf ties away from zero, unlike Math.round.
const round = (x: number) => Math.sign(x) * Math.floor(Math.abs(x) + 0.5);
type V3 = [number, number, number];
type Q = [number, number, number, number];
const xyz = (p: Vec): V3 => [p.x, p.y, p.z];
const identity: Q = [0, 0, 0, 1];
const plateCenter: V3 = [0, f(0.05), 26.5];
const normal = (a: Vec) => mul(f(1 / length(a)), a);
function between(a: Vec, b: Vec): Q {
    const m = mul(0.5, add(a, b));
    const c =
        dot(m, m) > f(f(100 * 1.1920928955078125e-7) ** 2)
            ? cross(a, m)
            : Math.abs(a.x) > 0.5
              ? v(a.y, -a.x, 0)
              : v(0, a.z, -a.y);
    const s = dot(m, m) > f(f(100 * 1.1920928955078125e-7) ** 2) ? dot(a, m) : 0;
    const inv = f(1 / f(Math.sqrt(f(dot(c, c) + f(s * s)))));
    return [f(inv * c.x), f(inv * c.y), f(inv * c.z), f(inv * s)];
}
export const DynamicMoverState = {
    create: () => ({
        active: false,
        mover: new DynamicMover(),
        entity: 0,
        keys: { W: 0, A: 0, S: 0, D: 0, Space: 0, Shift: 0, Control: 0, Aim: 0, Shoot: 0 },
        forward: v(0, 0, 1),
        right: v(-1, 0, 0),
        aimDirection: v(0, 0, 1),
        jumpReleased: true,
        drawProbes: true,
        useBox: false,
        platforms: [] as number[],
        gems: [] as number[],
        balls: Array<number>(50).fill(0),
        ballShapes: Array<number>(50).fill(0),
        ballIndex: 0,
        ballSpeed: 50,
        time: 0,
        plate: 0,
        plateJoint: 0,
        plateArmed: true,
        initializePlateForce: true,
        shootRequested: false,
        shootHeld: false,
        pendingSnap: false,
        pendingSwap: false,
    }),
};
function body(
    world: World,
    position: V3,
    type: number = BodyType.Static,
    options: {
        angularDamping?: number;
        gravityScale?: number;
        isAwake?: number;
        isBullet?: number;
        angularVelocity?: V3;
        linearVelocity?: V3;
    } = {},
) {
    const eid = world.create();
    world.add(eid, Transform, { translation: position });
    world.add(eid, Body, { type, ...options });
    return eid;
}
type Material = {
    density?: number;
    friction?: number;
    restitution?: number;
    rollingResistance?: number;
    filterCategoryLow?: number;
    filterMaskLow?: number;
    filterMaskHigh?: number;
    userMaterialIdLow?: number;
    isSensor?: number;
    enableSensorEvents?: number;
    explosionScale?: number;
};
function box(
    world: World,
    owner: number,
    halfExtents: V3,
    position: V3 = [0, 0, 0],
    rotation: Q = identity,
    options: Material = {},
) {
    const shape = world.create();
    world.add(shape, Shape, {
        body: owner,
        geometry: world
            .resource(Hulls)
            .add({ name: "box", halfExtents, transform: { position, rotation } }),
        ...(options.filterCategoryLow === undefined ? {} : { filterCategoryHigh: 0 }),
        ...options,
    });
    return shape;
}
// These construction anchors use unrotated bodies' authored positions, before fixed sync.
function getLocalPoint(world: World, owner: number, point: Vec) {
    const p = world.storage(Transform).translation;
    return sub(point, v(p.x.get(owner), p.y.get(owner), p.z.get(owner)));
}
function sphere(world: World, owner: number, radius: number, options: Material = {}) {
    const shape = world.create();
    world.add(shape, Shape, {
        body: owner,
        kind: ShapeKind.Sphere,
        radius,
        ...(options.filterCategoryLow === undefined ? {} : { filterCategoryHigh: 0 }),
        ...options,
    });
    return shape;
}
function ramp(world: World, high: Vec, low: Vec, width: number) {
    const d = sub(high, low),
        n = length(d),
        q = between(v(0, 0, 1), mul(f(1 / n), d));
    const normal = rotate({ v: v(q[0], q[1], q[2]), s: q[3] }, v(0, 1, 0));
    const center = add(add(mul(0.5, low), mul(0.5, high)), mul(-f(0.1), normal));
    box(world, body(world, [0, 0, 0]), [width, f(0.1), f(0.5 * n)], xyz(center), q);
}
function platformPosition(i: number, time: number): Vec {
    return i === 0
        ? v(14, f(f(1.95) - f(f(1.95) * f(Math.cos(f(f(0.6) * time))))), 0)
        : v(f(25 - f(3.5 * f(Math.cos(f(0.5 * time))))), f(3.9), 0);
}
function construct(world: World): number[] {
    const s = world.resource(DynamicMoverState);
    Object.assign(s, DynamicMoverState.create());
    s.active = true;
    const ground = body(world, [0, -1, 0]);
    box(world, ground, [80, 1, 80]);
    s.mover.create(world);
    s.entity = s.mover.body;
    const stairs = body(world, [0, 0, 0]);
    for (let i = 0; i < 10; i++) {
        const h = f(f(0.1) * (i + 1));
        box(world, stairs, [1.5, h, f(0.3)], [-6, h, f(f(3.3) + f(f(0.6) * i))]);
    }
    const towers = [-6, 6].map((x) => {
        const eid = body(world, [x, 1, 11]);
        box(world, eid, [1.5, 1, 2]);
        return eid;
    });
    const plankLength = f(9 / 18);
    let previous = towers[0]!;
    for (let i = 0; i < 18; i++) {
        const plank = body(
            world,
            [f(-4.5 + f(plankLength * f(i + 0.5))), f(1.95), 11],
            BodyType.Dynamic,
            { angularDamping: 0.5 },
        );
        box(
            world,
            plank,
            [f(f(0.5 * plankLength) - f(0.01)), f(0.05), 0.75],
            undefined,
            undefined,
            { density: 300 },
        );
        const pivot = v(f(-4.5 + f(plankLength * i)), f(1.95), 11);
        world.add(world.create(), RevoluteJoint, {
            a: previous,
            b: plank,
            localAnchorA: xyz(getLocalPoint(world, previous, pivot)!),
            localAnchorB: xyz(getLocalPoint(world, plank, pivot)!),
            enableMotor: 1,
            maxMotorTorque: 50,
        });
        previous = plank;
    }
    const pivot = v(f(-4.5 + f(plankLength * 18)), f(1.95), 11);
    world.add(world.create(), RevoluteJoint, {
        a: previous,
        b: towers[1]!,
        localAnchorA: xyz(getLocalPoint(world, previous, pivot)!),
        localAnchorB: xyz(getLocalPoint(world, towers[1]!, pivot)!),
        enableMotor: 1,
        maxMotorTorque: 50,
    });
    ramp(world, v(6, 2, 9), v(6, 0, 3.5), 1.5);
    for (const [high, low] of [
        [v(20, 3, -40), v(20, -0.5, -40)],
        [v(35, 3, -46), v(35, -0.5, -39.5)],
        [v(50, 3, -42.5), v(50, -0.5, -39.5)],
        [v(65, 0.25, -40), v(65, -0.5, -40)],
    ] as const)
        ramp(world, high, low, 5);
    for (const x of [18, 32]) box(world, body(world, [x, 2, 0]), [2, 2, 2]);
    for (let i = 0; i < 2; i++) {
        const eid = body(world, xyz(platformPosition(i, 0)), BodyType.Kinematic);
        s.platforms.push(eid);
        box(world, eid, [1.5, f(0.1), 1.5], undefined, undefined, {
            filterCategoryLow: 16,
            userMaterialIdLow: 7,
        });
    }
    const fulcrum = body(world, [-14, f(0.4), 0]);
    box(world, fulcrum, [0.25, f(0.4), 0.75]);
    const seesaw = body(world, [-14, f(0.9), 0], BodyType.Dynamic);
    box(world, seesaw, [3.5, f(0.1), f(0.8)], undefined, undefined, {
        filterCategoryLow: 16,
        density: 200,
    });
    world.add(world.create(), RevoluteJoint, {
        a: fulcrum,
        b: seesaw,
        localAnchorA: xyz(getLocalPoint(world, fulcrum, v(-14, f(0.8), 0))!),
        localAnchorB: xyz(getLocalPoint(world, seesaw, v(-14, f(0.8), 0))!),
        enableLimit: 1,
        lowerAngle: -0.3,
        upperAngle: 0.3,
    });
    for (let i = 0; i < 100; i++)
        for (let j = i; j < 100; j++)
            box(
                world,
                body(
                    world,
                    [
                        -50,
                        f(f(2 * i + 1) * 0.25),
                        f(f(f(i + 1) * 0.25) + f(f(f(2 * f(j - i)) * 0.25)) - f(0.25 * 100)),
                    ],
                    BodyType.Dynamic,
                ),
                [0.25, 0.25, 0.25],
                undefined,
                undefined,
                { filterCategoryLow: 16, density: 50 },
            );
    const frame = body(world, [-10, f(1.6), -8]);
    const postHeight = f(0.5 * f(1.5 + f(1.6))),
        postOffset = f(0.75 + f(0.3));
    for (const side of [-1, 1])
        box(
            world,
            frame,
            [f(0.3), postHeight, f(0.2)],
            [f(side * postOffset), f(-f(1.6) + postHeight), 0],
        );
    box(world, frame, [f(postOffset + f(0.3)), f(0.3), f(0.2)], [0, f(1.5 + f(0.3)), 0]);
    const door = body(world, [-10, f(1.6), -8], BodyType.Dynamic, { gravityScale: 2 });
    box(world, door, [0.75, 1.5, f(0.1)], undefined, undefined, {
        filterCategoryLow: 16,
        density: 100,
    });
    const doorAxis = between(v(0, 0, 1), v(0, 1, 0));
    world.add(world.create(), RevoluteJoint, {
        a: frame,
        b: door,
        localAnchorA: [-0.75, 0, 0],
        localAnchorB: [-0.75, 0, 0],
        localRotationA: doorAxis,
        localRotationB: doorAxis,
        enableLimit: 1,
        lowerAngle: f(f(f(Math.PI) / 180) * -90),
        upperAngle: f(f(f(Math.PI) / 180) * 90),
        enableSpring: 1,
        hertz: 1,
        dampingRatio: 0.5,
        drawScale: 2,
    });
    for (let i = 0; i < 20; i++)
        for (let j = 0; j < 20; j++) {
            const position: V3 = [f(-9.5 + i), f(1.3), 17 + j];
            const eid = body(world, position, BodyType.Dynamic, { gravityScale: 0 });
            sphere(world, eid, f(0.35), { density: 100, friction: 0.2, filterCategoryLow: 16 });
            world.add(world.create(), WeldJoint, {
                a: ground,
                b: eid,
                collideConnected: 1,
                linearHertz: 0.5,
                linearDampingRatio: 0.2,
                localAnchorA: [position[0], f(position[1] + 1), position[2]],
            });
        }
    s.plate = body(world, plateCenter, BodyType.Dynamic);
    box(world, s.plate, [1.5, f(0.1), 1.5], undefined, undefined, {
        density: 20,
        filterCategoryLow: 16,
        filterMaskLow: 2,
        filterMaskHigh: 0,
        explosionScale: 0,
    });
    const upright = rotationFromAxisAngle(world, v(0, 0, 1), f(0.5 * f(Math.PI)));
    s.plateJoint = world.create();
    world.add(s.plateJoint, PrismaticJoint, {
        a: ground,
        b: s.plate,
        localAnchorA: [0, f(f(0.05) + 1), 26.5],
        localRotationA: upright,
        localRotationB: upright,
        enableLimit: 1,
        lowerTranslation: -0.1,
        upperTranslation: 0,
        enableMotor: 1,
        motorSpeed: 0.5,
        maxMotorForce: 0,
    });
    const points: V3[] = [];
    for (let i = 0; i < 6; i++) {
        const cs = computeCosSin(world, f(f(f(2 * f(Math.PI)) * i) / 6)),
            x = f(f(0.1) * cs[0]),
            z = f(f(0.1) * cs[1]);
        points.push([x, -f(0.05), z], [f(f(1.3) * x), 0, f(f(1.3) * z)], [x, f(0.05), z]);
    }
    const skittle = world
        .resource(Hulls)
        .add({ name: "skittle", points: new Float32Array(points.flat()) });
    for (let layer = 0; layer < 3; layer++)
        for (let i = 0; i < 32; i++)
            for (let j = 0; j < 32; j++) {
                const eid = body(
                    world,
                    [
                        f(-7.75 + f(0.5 * i)),
                        f(8 + f(0.5 * layer)),
                        f(f(f(26.5 - 7.75) + f(0.5 * j))),
                    ],
                    BodyType.Dynamic,
                    { isAwake: 0 },
                );
                world.add(world.create(), Shape, {
                    body: eid,
                    geometry: skittle,
                    filterCategoryLow: 64,
                    filterCategoryHigh: 0,
                    filterMaskLow: 113,
                    filterMaskHigh: 0,
                    density: 100,
                    explosionScale: 0.05,
                });
            }
    sphere(world, body(world, [6, 1, -6], BodyType.Dynamic), 1, {
        density: 30,
        rollingResistance: 0.05,
        filterCategoryLow: 16,
    });
    const sweeper = body(world, [0, 0, -16], BodyType.Dynamic);
    box(world, sweeper, [f(0.3), f(0.6), f(0.3)], [0, f(0.6), 0], undefined, {
        filterCategoryLow: 16,
    });
    box(world, sweeper, [5, 0.25, f(0.1)], [0, 1, 0], undefined, { filterCategoryLow: 16 });
    box(world, sweeper, [f(0.1), 0.25, 5], [0, 1, 0], undefined, { filterCategoryLow: 16 });
    const sweepAxis = rotationFromAxisAngle(world, v(1, 0, 0), f(-0.5 * f(Math.PI)));
    world.add(world.create(), RevoluteJoint, {
        a: ground,
        b: sweeper,
        localAnchorA: [0, 1, -16],
        localRotationA: sweepAxis,
        localRotationB: sweepAxis,
        enableMotor: 1,
        motorSpeed: 1,
        maxMotorTorque: 3000,
    });
    for (let i = 0; i < 6; i++) {
        const height = f(0.5 + f(f(0.6) * i));
        const eid = body(world, [-20, 0, f(-3 - 5 * i)]);
        world.add(world.create(), Shape, {
            body: eid,
            geometry: world
                .resource(Hulls)
                .add({ name: "pillar", cylinder: { height, radius: 0.75, yOffset: 0, sides: 12 } }),
        });
    }
    const positions: V3[] = [
        [0, 1.2, 5],
        [-6, 2.2, 6],
        [-6, 3.2, 11],
        [0, 3.2, 11],
        [6, 3.2, 11],
        [6, 2.2, 6],
        [18, 5.2, 0],
        [25, 5.2, 0],
        [32, 5.2, 0],
        [-17, 2.2, 0],
        [-10, 4.4, -8],
        [6, 1.2, -9.5],
        [3, 1.2, -16],
        [-3, 1.2, -16],
        [0, 1.2, -19],
        [0, 1.2, -13],
        [-20, f(f(0.5 + f(f(0.6) * 5)) + f(1.2)), -28],
    ];
    const tilt = rotationFromAxisAngle(world, normal(v(1, 0, -1)), f(0.9553));
    for (const p of positions) {
        const eid = body(world, p, BodyType.Kinematic, { angularVelocity: [0, 2, 0] });
        s.gems.push(eid);
        box(world, eid, [f(0.2), f(0.2), f(0.2)], undefined, tilt, {
            isSensor: 1,
            enableSensorEvents: 1,
            filterCategoryLow: 4,
            filterMaskLow: 2,
            filterMaskHigh: 0,
        });
    }
    return [s.mover.body];
}
function swapShape(world: World) {
    const s = world.resource(DynamicMoverState),
        m = s.mover;
    world.storage(Shape).updateBodyMass.set(m.shape, 0);
    world.destroy(m.shape);
    SyncPhysicsSystem.update!(world);
    m.shape = s.useBox
        ? box(
              world,
              m.body,
              [m.capsule.radius, f(0.5 + m.capsule.radius), m.capsule.radius],
              undefined,
              undefined,
              {
                  density: 200,
                  friction: 0,
                  restitution: 0,
                  filterCategoryLow: 2,
                  enableSensorEvents: 1,
              },
          )
        : world.create();
    if (!s.useBox)
        world.add(m.shape, Shape, {
            body: m.body,
            kind: ShapeKind.Capsule,
            radius: m.capsule.radius,
            density: 200,
            friction: 0,
            restitution: 0,
            filterCategoryLow: 2,
            filterCategoryHigh: 0,
            enableSensorEvents: 1,
        });
    SyncPhysicsSystem.update!(world);
}
function shoot(world: World) {
    const s = world.resource(DynamicMoverState),
        index = s.ballIndex;
    if (s.balls[index]) {
        world.destroy(s.ballShapes[index]!);
        world.destroy(s.balls[index]!);
        SyncPhysicsSystem.update!(world);
    }
    const direction = s.aimDirection,
        right = normal(v(s.right.x, 0, s.right.z));
    const shoulder = add(
        pose(world, s.mover.body).p,
        add(s.mover.capsule.center2, mul(f(s.mover.capsule.radius + f(0.03)), right)),
    );
    const muzzle = add(shoulder, mul(f(f(0.55) + f(0.03)), direction));
    const back = Math.min(f(s.ballSpeed * f(1 / 60)), f(f(0.55) + f(0.03)));
    const eid = body(world, xyz(sub(muzzle, mul(back, direction))), BodyType.Dynamic, {
        isBullet: 1,
        linearVelocity: xyz(mul(s.ballSpeed, direction)),
    });
    s.balls[index] = eid;
    s.ballShapes[index] = sphere(world, eid, f(0.1), {
        density: 4000,
        friction: 0.2,
        restitution: 0.5,
        rollingResistance: 0.1,
        filterCategoryLow: 32,
        filterMaskLow: 0xfffffffd,
    });
    s.ballIndex = (index + 1) % 50;
}
export const DynamicMoverPlugin: Plugin = {
    name: "DynamicMoverSample",
    dependencies: [StandardPhysicsPlugin, InputPlugin],
    recovery(world) {
        const recovery = recoverResource(DynamicMoverState)(world);
        return {
            ...recovery,
            restore(image: ReturnType<typeof DynamicMoverState.create>) {
                recovery.restore(image);
                Object.setPrototypeOf(
                    world.resource(DynamicMoverState).mover,
                    DynamicMover.prototype,
                );
            },
        };
    },
    systems: [
        {
            name: "dynamic-mover-input",
            group: "setup",
            update(world) {
                const s = world.resource(DynamicMoverState);
                if (!s.active) return;
                readPlayerControls(world, s.keys, s.forward, s.right);
                const [camera] = world.query([Camera, Orbit]);
                if (camera) {
                    const pitch = world.storage(Orbit).pitch.get(camera);
                    s.aimDirection = v(
                        f(s.forward.x * Math.cos(pitch)),
                        f(-Math.sin(pitch)),
                        f(s.forward.z * Math.cos(pitch)),
                    );
                }
            },
        },
        {
            name: "dynamic-mover-before-step",
            group: "fixed",
            after: [SyncPhysicsSystem],
            before: [SyncPhysicsConstraintsSystem, StepPhysicsSystem],
            update(world) {
                const s = world.resource(DynamicMoverState);
                if (!s.active) return;
                if (s.initializePlateForce) {
                    const plateWeight = f(10 * getMassData(world, s.plate)!.mass);
                    const moverWeight = f(
                        f(s.mover.gravityScale * 10) * getMassData(world, s.mover.body)!.mass,
                    );
                    world
                        .storage(PrismaticJoint)
                        .maxMotorForce.set(s.plateJoint, f(plateWeight + f(0.5 * moverWeight)));
                    s.initializePlateForce = false;
                }
                if (s.pendingSwap) {
                    swapShape(world);
                    s.pendingSwap = false;
                }
                if (s.pendingSnap) {
                    const p = pose(world, s.mover.body);
                    setTransform(
                        world,
                        s.mover.body,
                        [round(p.p.x), p.p.y, round(p.p.z)],
                        [p.q.v.x, p.q.v.y, p.q.v.z, p.q.s],
                    );
                    setLinearVelocity(world, s.mover.body, v());
                    s.pendingSnap = false;
                }
                s.time = f(s.time + f(1 / 60));
                for (let i = 0; i < 2; i++)
                    setTargetTransform(
                        world,
                        s.platforms[i]!,
                        xyz(platformPosition(i, s.time)),
                        identity,
                        f(1 / 60),
                        true,
                    );
                if (s.keys.Space) {
                    if (s.jumpReleased && s.mover.jump(world)) s.jumpReleased = false;
                } else s.jumpReleased = true;
                s.mover.update(
                    world,
                    f(1 / 60),
                    s.forward,
                    s.right,
                    { x: f(s.keys.W - s.keys.S), y: f(s.keys.D - s.keys.A) },
                    s.keys.Control || s.keys.Aim ? 0.25 : s.keys.Shift ? 1.5 : 1,
                );
                if (s.keys.Aim && (s.shootRequested || (s.keys.Shoot && !s.shootHeld)))
                    shoot(world);
                s.shootRequested = false;
                s.shootHeld = !!s.keys.Shoot;
                SyncPhysicsSystem.update!(world);
            },
        },
        {
            name: "dynamic-mover-after-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(DynamicMoverState);
                if (!s.active) return;
                const events = world.resource(PhysicsEvents).sensorBegin;
                for (let i = 0; i < events.count; i++) {
                    const event = events.records[i]!;
                    const shape = world.resolve(event.visitorShape);
                    if (
                        !world.has(shape, Shape) ||
                        world.storage(Shape).body.get(shape) !== s.mover.body
                    )
                        continue;
                    const sensor = world.resolve(event.sensorShape);
                    if (!world.has(sensor, Shape)) continue;
                    const gem = world.storage(Shape).body.get(sensor),
                        index = s.gems.indexOf(gem);
                    if (index >= 0) {
                        world.destroy(sensor);
                        world.destroy(gem);
                        s.gems[index] = 0;
                    }
                }
                const joint = world.storage(PrismaticJoint),
                    id = s.plateJoint;
                const a = pose(world, joint.a.get(id)),
                    b = pose(world, joint.b.get(id));
                const q = joint.localRotationA;
                const axis = rotate(
                    a.q,
                    rotate(
                        { v: v(q.x.get(id), q.y.get(id), q.z.get(id)), s: q.w.get(id) },
                        v(1, 0, 0),
                    ),
                );
                const anchorA = rotate(
                    a.q,
                    v(
                        joint.localAnchorA.x.get(id),
                        joint.localAnchorA.y.get(id),
                        joint.localAnchorA.z.get(id),
                    ),
                );
                const anchorB = rotate(
                    b.q,
                    v(
                        joint.localAnchorB.x.get(id),
                        joint.localAnchorB.y.get(id),
                        joint.localAnchorB.z.get(id),
                    ),
                );
                const translation = dot(add(sub(b.p, a.p), sub(anchorB, anchorA)), axis);
                if (s.plateArmed && translation < f(-f(0.1) + f(0.01))) {
                    explode(world, {
                        maskBits: 0xffffffffffffffffn,
                        position: v(0, f(f(0.05) - f(0.1)), 26.5),
                        radius: 10,
                        falloff: 10,
                        impulsePerArea: 3000,
                    });
                    s.mover.launch();
                    s.plateArmed = false;
                } else if (!s.plateArmed && translation > -f(0.01)) s.plateArmed = true;
            },
        },
    ],
};
const sliders = {
    "Jump Speed": ["jumpSpeed", 7, 0, 20, 0.1],
    "Max Speed": ["maxSpeed", 6, 0, 20, 1],
    Accelerate: ["accelerate", 20, 0, 100, 1],
    Friction: ["friction", 8, 0, 20, 0.1],
    "Air Steer": ["airSteer", 0.25, 0, 1, 0.01],
    "Pogo Hertz": ["pogoHertz", 3, 0, 30, 1],
    "Pogo Damping": ["pogoDampingRatio", 0.8, 0, 4, 0.1],
    "Ground Force": ["maxGroundForce", 6000, 0, 20000, 1],
} as const;
export const dynamicMoverScene = {
    construct,
    camera: { yaw: 180, pitch: 20, distance: 6, target: [0, 1.5, 0] },
    gridExtent: 80,
    gridSpacing: 1,
    controls: {
        "Skirt Pogo": {
            label: "Skirt Pogo",
            value: 1,
            set(world, _bodies, value) {
                world.resource(DynamicMoverState).mover.enableSkirtPogo = !!value;
            },
        },
        "Draw Probes": {
            label: "Draw Probes",
            value: 1,
            set(world, _bodies, value) {
                world.resource(DynamicMoverState).drawProbes = !!value;
            },
        },
        "Box Mover": {
            label: "Box Mover",
            value: 0,
            set(world, _bodies, value) {
                const s = world.resource(DynamicMoverState);
                s.useBox = !!value;
                s.pendingSwap = true;
            },
        },
        ...Object.fromEntries(
            Object.entries(sliders).map(([label, [field, value, min, max, step]]) => [
                label,
                {
                    label,
                    value,
                    min,
                    max,
                    step,
                    set(world: World, _bodies: number[], value: number) {
                        world.resource(DynamicMoverState).mover[field] = f(value);
                    },
                },
            ]),
        ),
        "Gravity Scale": {
            label: "Gravity Scale",
            value: 1.5,
            min: 0,
            max: 4,
            step: 0.1,
            set(world, _bodies, value) {
                world.resource(DynamicMoverState).mover.setGravityScale(world, value);
            },
        },
        "Ball Speed": {
            label: "Ball Speed",
            value: 50,
            min: 10,
            max: 200,
            step: 1,
            set(world, _bodies, value) {
                world.resource(DynamicMoverState).ballSpeed = f(value);
            },
        },
        "Snap Position": {
            label: "Snap Position",
            value: 0,
            button: true,
            set(world) {
                world.resource(DynamicMoverState).pendingSnap = true;
            },
        },
    } satisfies Controls,
};

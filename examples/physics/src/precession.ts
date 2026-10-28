import {
    Body,
    BodyType,
    GlobalTransform,
    Hulls,
    type Plugin,
    recoverResource,
    Shape,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    getAngularVelocity,
    getMassData,
    isAwake,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
    setAngularVelocity,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
const pi = f(Math.PI);
const dt = f(1 / 60);
export const PrecessionState = {
    create: () => ({
        bodies: [] as number[],
        angularVelocity: [0, 0, 0] as [number, number, number],
        top: 0,
        mass: 0,
        distance: 0,
        spinInertia: 0,
        transverseInertia: 0,
        initialized: false,
        groundTime: 0,
        measuring: false,
        azimuth: 0,
        actualAngle: 0,
        expectedAngle: 0,
        elapsed: 0,
    }),
};
// b3RotateVector for a vector along local Y, preserving Box3D's float operations.
function rotateY(q: readonly number[], v: number): [number, number, number] {
    const [x, y, z, w] = q as [number, number, number, number];
    const tx = f(-z * v),
        ty = f(w * v),
        tz = f(x * v);
    return [
        f(2 * f(f(y * tz) - f(z * ty))),
        f(v + f(2 * f(f(z * tx) - f(x * tz)))),
        f(2 * f(f(x * ty) - f(y * tx))),
    ];
}
// b3Atan2's deterministic minimax polynomial, used by the sample's azimuth measurement.
function atan2(y: number, x: number): number {
    if (x === 0 && y === 0) return 0;
    const ax = Math.abs(x),
        ay = Math.abs(y);
    const a = f(Math.min(ax, ay) / Math.max(ax, ay));
    const s = f(a * a),
        c = f(s * a),
        q = f(s * s);
    let r = f(f(f(0.024840285) * q) + f(0.18681418));
    const t = f(f(f(-0.094097948) * q) - f(0.33213072));
    r = f(f(r * s) + t);
    r = f(f(r * c) + a);
    if (ay > ax) r = f(f(1.57079637) - r);
    if (x < 0) r = f(f(3.14159274) - r);
    return y < 0 ? -r : r;
}
export const PrecessionPlugin: Plugin = {
    name: "Precession",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(PrecessionState),
    systems: [
        {
            name: "precession-mass",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(PrecessionState);
                if (!s.top || s.initialized) return;
                const [x, y, z] = s.angularVelocity;
                for (const eid of s.bodies) setAngularVelocity(world, eid, { x, y, z });
                const mass = getMassData(world, s.top)!;
                s.mass = mass.mass;
                s.distance = mass.center.y;
                s.spinInertia = mass.inertia.cy.y;
                s.transverseInertia = f(
                    f(0.5 * f(mass.inertia.cx.x + mass.inertia.cz.z)) +
                        f(f(s.mass * s.distance) * s.distance),
                );
                s.initialized = true;
            },
        },
        {
            name: "precession-rates",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(PrecessionState);
                if (!s.top || !isAwake(world, s.top)) return;
                const global = world.storage(GlobalTransform);
                const q = global.rotation;
                const axis = rotateY(
                    [q.x.get(s.top), q.y.get(s.top), q.z.get(s.top), q.w.get(s.top)],
                    1,
                );
                const omega = getAngularVelocity(world, s.top)!;
                const spin = f(
                    f(f(omega.x * axis[0]) + f(omega.y * axis[1])) + f(omega.z * axis[2]),
                );
                const momentum = f(s.spinInertia * spin),
                    torque = f(f(s.mass * 10) * s.distance);
                let expected = 0;
                if (momentum > 1.1920928955078125e-7) {
                    const a = f(s.transverseInertia * Math.max(-1, Math.min(1, axis[1])));
                    const discriminant = f(f(momentum * momentum) - f(f(4 * a) * torque));
                    expected =
                        a <= 1.1920928955078125e-7 || discriminant < 0
                            ? f(torque / momentum)
                            : f(f(momentum - f(Math.sqrt(discriminant))) / f(2 * a));
                }
                if (global.translation.y.get(s.top) < f(0.05)) s.groundTime = f(s.groundTime + dt);
                if (!s.measuring) {
                    if (s.groundTime > 0.5) {
                        s.measuring = true;
                        s.azimuth = atan2(-axis[2], axis[0]);
                    }
                } else {
                    const azimuth = atan2(-axis[2], axis[0]);
                    let delta = f(azimuth - s.azimuth);
                    while (delta > pi) delta = f(delta - f(2 * pi));
                    while (delta < -pi) delta = f(delta + f(2 * pi));
                    s.azimuth = azimuth;
                    s.actualAngle = f(s.actualAngle + delta);
                    s.expectedAngle = f(s.expectedAngle + f(expected * dt));
                    s.elapsed = f(s.elapsed + dt);
                }
            },
        },
    ],
};
function construct(world: World): number[] {
    const ground = world.create();
    world.add(ground, Transform, { translation: [0, -1, 0] });
    world.add(ground, Body, { type: BodyType.Static });
    world.add(ground, Shape, {
        geometry: world.resource(Hulls).add({ name: "box", halfExtents: [40, 1, 40] }),
    });
    const points: number[] = [];
    const dphi = f(f(2 * pi) / 7);
    for (let i = 0; i < 7; i++)
        points.push(f(2 * f(Math.cos(f(i * dphi)))), 2, f(2 * f(Math.sin(f(i * dphi)))));
    points.push(0, 0, 0);
    const geometry = world
        .resource(Hulls)
        .add({ name: "top", points: new Float32Array(points), maxVertexCount: 8 });
    const rotation = rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(f(15 * pi) / 180));
    const angularVelocity = rotateY(rotation, 75);
    const bodies: number[] = [];
    for (let x = 0; x < 8; x++)
        for (let z = 0; z < 8; z++) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [(x - 4) * 6, 2, (z - 4) * 6], rotation });
            world.add(eid, Body, { type: BodyType.Dynamic, allowFastRotation: 1 });
            world.add(eid, Shape, { geometry, rollingResistance: 0.1 });
            bodies.push(eid);
        }
    Object.assign(world.resource(PrecessionState), { top: bodies[0]!, bodies, angularVelocity });
    return bodies;
}
const controls: Controls = Object.fromEntries(
    [
        ["Expected average", "expectedAngle"],
        ["Actual average", "actualAngle"],
    ].map(([label, field]) => [
        label!,
        {
            label: label!,
            value: 0,
            decimals: 4,
            read(world: World) {
                const s = world.resource(PrecessionState);
                return s.elapsed > 0
                    ? f(s[field as "expectedAngle" | "actualAngle"] / s.elapsed)
                    : null;
            },
            set() {},
        },
    ]),
);
export const gyroscopicPrecession = {
    construct,
    controls,
    gridExtent: 40,
    gridSpacing: 2,
    camera: { yaw: 35, pitch: 35, distance: 80, phoneDistance: 150, target: [-3, 2, -3] },
};

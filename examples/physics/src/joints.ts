import {
    Body,
    BodyType,
    Hulls,
    MotorJoint,
    RevoluteJoint,
    Shape,
    ShapeKind,
    SphericalJoint,
    Transform,
    WeldJoint,
    type World,
} from "@dylanebert/shallot";
import { explode, setAwake } from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

type V3 = [number, number, number];
function body(world: World, position: V3, dynamic = true): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position });
    world.add(eid, Body, { type: dynamic ? BodyType.Dynamic : BodyType.Static });
    return eid;
}
function box(world: World, eid: number, halfExtents: V3, translation: V3 = [0, 0, 0]): void {
    const geometry = world.resource(Hulls).add({
        name: "box",
        halfExtents,
        transform: { position: translation, rotation: [0, 0, 0, 1] },
    });
    const shape = world.has(eid, Shape) ? world.create() : eid;
    world.add(shape, Shape, { geometry, body: eid === shape ? 0 : eid });
}
function ground(world: World): number {
    const eid = body(world, [0, -1, 0], false);
    box(world, eid, [20, 1, 20]);
    return eid;
}
function ballAndChain(world: World): number[] {
    let parent = body(world, [0, 0, 0], false);
    const bodies: number[] = [];
    for (let i = 0; i <= 32; i++) {
        const last = i === 32;
        const eid = body(world, [last ? 34 : 0.5 + i, 0, 0]);
        world.add(
            eid,
            Shape,
            last
                ? { kind: ShapeKind.Sphere, radius: 2 }
                : {
                      kind: ShapeKind.Capsule,
                      capsuleCenter1: [-0.5, 0, 0],
                      capsuleCenter2: [0.5, 0, 0],
                      radius: 0.125,
                  },
        );
        world.add(world.create(), SphericalJoint, {
            a: parent,
            b: eid,
            localAnchorA: [i ? 0.5 : 0, 0, 0],
            localAnchorB: [last ? -2 : -0.5, 0, 0],
            enableMotor: 1,
            maxMotorTorque: 10,
        });
        bodies.push(eid);
        parent = eid;
    }
    return bodies;
}
function foldOut(world: World): number[] {
    let parent = ground(world);
    const bodies: number[] = [];
    const q: [number, number, number, number] = [
        -Math.fround(Math.SQRT1_2),
        0,
        0,
        Math.fround(Math.SQRT1_2),
    ];
    for (let i = 0; i < 6; i++) {
        const eid = body(world, [0, 0.5 + i, 0]);
        box(world, eid, [2, 0.1, 0.25]);
        world.storage(Shape).density.set(eid, 2000);
        const x = i & 1 ? 2 : -2;
        world.add(world.create(), RevoluteJoint, {
            a: parent,
            b: eid,
            localAnchorA: [x, i ? 0.5 : 1, 0],
            localAnchorB: [x, -0.5, 0],
            localRotationA: q,
            localRotationB: q,
            constraintHertz: 240,
            drawScale: 0.5,
            enableMotor: 1,
            maxMotorTorque: 1000,
        });
        bodies.push(eid);
        parent = eid;
    }
    return bodies;
}
function topDownFriction(world: World): number[] {
    const ground = body(world, [0, 0, 0], false);
    box(world, ground, [10, 0.5, 4]);
    box(world, ground, [0.5, 10, 4], [-10, 10, 0]);
    box(world, ground, [0.5, 10, 4], [10, 10, 0]);
    box(world, ground, [10, 0.5, 4], [0, 20, 0]);
    const bodies: number[] = [];
    for (let i = 0; i < 10; i++)
        for (let j = 0; j < 10; j++) {
            const eid = body(world, [-5 + j, 15 - i, 0]);
            world.storage(Body).gravityScale.set(eid, 0);
            const remainder = (10 * i + j) % 4;
            if (remainder === 0)
                world.add(eid, Shape, {
                    kind: ShapeKind.Capsule,
                    capsuleCenter1: [-0.25, 0, 0],
                    capsuleCenter2: [0.25, 0, 0],
                    radius: 0.25,
                });
            else if (remainder === 1)
                world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.35 });
            else box(world, eid, [0.35, 0.35, 0.35]);
            world.storage(Shape).restitution.set(eid, 0.8);
            world.add(world.create(), MotorJoint, {
                a: ground,
                b: eid,
                collideConnected: 1,
                maxVelocityForce: 1000,
                maxVelocityTorque: 1000,
            });
            bodies.push(eid);
        }
    return bodies;
}
function weld(world: World): number[] {
    ground(world);
    const anchor = body(world, [0, -1, 0], false);
    const eid = body(world, [0, 4, 0]);
    world.storage(Body).gravityScale.set(eid, 0);
    box(world, eid, [0.5, 1.5, 0.25]);
    world.add(world.create(), WeldJoint, {
        a: anchor,
        b: eid,
        localAnchorA: [0, 6.5, 0],
        localAnchorB: [0, 1.5, 0],
        constraintHertz: 240,
        angularHertz: 2,
        angularDampingRatio: 0.7,
        drawScale: 2,
    });
    return [eid];
}
const weldControls = Object.fromEntries(
    (
        [
            ["Linear Hertz", "linearHertz", 0, 10],
            ["Linear Damping", "linearDampingRatio", 0, 2],
            ["Angular Hertz", "angularHertz", 2, 10],
            ["Angular Damping", "angularDampingRatio", 0.7, 2],
        ] as const
    ).map(([label, field, value, max]) => [
        label,
        {
            label,
            value,
            min: 0,
            max,
            step: 0.1,
            set(world, bodies, value) {
                for (const eid of world.query([WeldJoint]))
                    world.storage(WeldJoint)[field].set(eid, value);
                for (const eid of bodies) setAwake(world, eid, true);
            },
        } satisfies Controls[string],
    ]),
);
export const jointScenes = {
    ballAndChain: {
        construct: ballAndChain,
        gridExtent: 40,
        gridSpacing: 2,
        camera: { yaw: 25, pitch: 40, distance: 60, target: [8, -15, 0] },
        controls: undefined,
    },
    foldOut: {
        construct: foldOut,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 10, distance: 22, target: [0, 5, 0] },
        controls: undefined,
    },
    topDownFriction: {
        construct: topDownFriction,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 0, distance: 28, target: [0, 10, 0] },
        controls: {
            Explode: {
                label: "Explode",
                value: 0,
                button: true,
                set(world) {
                    explode(world, {
                        maskBits: 0xffffffffffffffffn,
                        position: { x: 0, y: 10, z: 0 },
                        radius: 10,
                        falloff: 5,
                        impulsePerArea: 10000,
                    });
                },
            },
        } satisfies Controls,
    },
    weld: {
        construct: weld,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 45, pitch: 30, distance: 15, target: [0, 2, 0] },
        controls: weldControls,
    },
} as const;

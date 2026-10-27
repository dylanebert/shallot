import {
    Body,
    BodyMotionLock,
    BodyType,
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
    applyLinearImpulse,
    computeCosSin,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    SyncPhysicsConstraintsSystem,
} from "@dylanebert/shallot/standard/physics";

import { SceneRestart } from "./scene-restart";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
type Q4 = [number, number, number, number];
const pi = f(Math.PI);
export const StackingImpulses = { create: () => [] as { body: number; impulse: V3; point: V3 }[] };
const BeforeStacking: System = {
    name: "stacking-before-step",
    group: "fixed",
    after: [SyncPhysicsConstraintsSystem],
    before: [StepPhysicsSystem],
    update(world) {
        const impulses = world.resource(StackingImpulses);
        for (const { body, impulse, point } of impulses)
            applyLinearImpulse(
                world,
                body,
                { x: impulse[0], y: impulse[1], z: impulse[2] },
                { x: point[0], y: point[1], z: point[2] },
            );
        impulses.length = 0;
    },
};
export const StackingSamplePlugin: Plugin = {
    name: "StackingSample",
    dependencies: [StandardPhysicsPlugin],
    systems: [BeforeStacking],
    recovery: recoverResource(StackingImpulses),
};
function box(world: World, halfExtents: V3): number {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(
    world: World,
    geometry: number,
    translation: V3,
    rotation: Q4 = [0, 0, 0, 1],
): number {
    const eid = world.create();
    world.add(eid, Transform, { translation, rotation });
    world.add(eid, Body, { type: BodyType.Dynamic });
    world.add(eid, Shape, { geometry });
    return eid;
}
function ground(world: World, extent: number): void {
    const eid = body(world, box(world, [extent, 1, extent]), [0, -1, 0]);
    world.storage(Body).type.set(eid, BodyType.Static);
}
function rotation(world: World, axis: V3, angle: number): Q4 {
    return rotationFromAxisAngle(world, { x: axis[0], y: axis[1], z: axis[2] }, angle);
}
function jengaStack(world: World, capsule = 0): number[] {
    ground(world, 20);
    const geometry = capsule ? 0 : box(world, [1, 0.1, 0.1]);
    const bodies: number[] = [];
    for (let i = 0; i < 30; i++) {
        const alpha = i & 1 ? 0 : f(0.5 * pi);
        const offset = f(1 - f(2 * f(0.1)));
        const x = i & 1 ? 0 : offset;
        const z = i & 1 ? offset : 0;
        const y = f(f(f(f(2.1) * i) + 0.5) * f(0.1));
        for (const sign of [1, -1]) {
            const eid = world.create();
            world.add(eid, Transform, {
                translation: [sign * x, y, sign * z],
                rotation: rotation(world, [0, 1, 0], alpha),
            });
            world.add(eid, Body, { type: BodyType.Dynamic });
            world.add(
                eid,
                Shape,
                capsule
                    ? {
                          kind: ShapeKind.Capsule,
                          capsuleCenter1: [-1, 0, 0],
                          capsuleCenter2: [1, 0, 0],
                          radius: 0.1,
                          rollingResistance: 0.1,
                      }
                    : { geometry, rollingResistance: 0.05 },
            );
            bodies.push(eid);
        }
    }
    return bodies;
}
const jengaControls = {
    Shape: {
        label: "Shape",
        value: 0,
        options: [
            { label: "Hull", value: 0 },
            { label: "Capsule", value: 1 },
        ],
        set(world, _bodies, value) {
            if (value !== 0 && value !== 1) throw new Error("Shape must be Hull or Capsule");
            world.resource(SceneRestart).restart((world) => jengaStack(world, value));
        },
    },
} satisfies Controls;

function cardHouse(world: World): number[] {
    ground(world, 10);
    const geometry = box(world, [0.001, 0.2, 0.1]);
    const angles = [f(f(25 * pi) / 180), f(f(-25 * pi) / 180), f(0.5 * pi)];
    const bodies: number[] = [];
    let y = f(f(0.2) - f(0.02)),
        z0 = 0;
    function card(x: number, y: number, angle: number): void {
        const eid = body(world, geometry, [x, y, 0], rotation(world, [0, 0, 1], angle));
        world.storage(Shape).rollingResistance.set(eid, 0.05);
        world.storage(Shape).friction.set(eid, 0.7);
        bodies.push(eid);
    }
    for (let n = 5; n > 0; n--) {
        let z = z0;
        for (let i = 0; i < n; i++) {
            if (i !== n - 1) card(f(z + 0.25), f(f(y + f(0.2)) - f(0.015)), angles[2]!);
            card(z, y, angles[1]!);
            z = f(z + f(0.175));
            card(z, y, angles[0]!);
            z = f(z + f(0.175));
        }
        y = f(y + f(f(f(0.2) * 2) - f(0.03)));
        z0 = f(z0 + f(0.175));
    }
    return bodies;
}
function sphereStack(world: World): number[] {
    ground(world, 15);
    const bodies: number[] = [];
    let y = 0.75;
    for (let i = 0; i < 30; i++) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, y, 0] });
        world.add(eid, Body, { type: BodyType.Dynamic });
        world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, rollingResistance: 0.1 });
        bodies.push(eid);
        y = f(y + 1.5);
    }
    return bodies;
}
function capsuleStack(world: World): number[] {
    ground(world, 40);
    const bodies: number[] = [];
    for (let i = 0; i < 20; i++) {
        const eid = world.create();
        world.add(eid, Transform, { translation: [0, 0.75 + i, 0] });
        world.add(eid, Body, {
            type: BodyType.Dynamic,
            motionLocks:
                BodyMotionLock.linearZ |
                BodyMotionLock.angularX |
                BodyMotionLock.angularY |
                BodyMotionLock.angularZ,
        });
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [-1, 0, 0],
            capsuleCenter2: [1, 0, 0],
            radius: 0.5,
        });
        bodies.push(eid);
    }
    return bodies;
}
function singleBox(world: World): number[] {
    ground(world, 20);
    return [body(world, box(world, [0.5, 0.5, 0.5]), [0, 0.5, 0])];
}
function cylinder(world: World): number[] {
    ground(world, 10);
    const geometry = world
        .resource(Hulls)
        .add({ name: "cylinder", cylinder: { height: 1, radius: 0.25, yOffset: 0, sides: 12 } });
    const eid = body(world, geometry, [0, 2, 0]);
    world.storage(Shape).rollingResistance.set(eid, 0.05);
    return [eid];
}
function cylinderStack(world: World): number[] {
    ground(world, 10);
    const geometry = world
        .resource(Hulls)
        .add({ name: "cylinder", cylinder: { height: 1, radius: 0.5, yOffset: 0, sides: 15 } });
    const scales: V3[] = [
        [1, 1, 1],
        [-0.75, 1, 1],
        [1.2, 1, -0.9],
        [0.9, 0.9, 0.9],
    ];
    const bodies: number[] = [];
    for (let i = 0; i < 10; i++) {
        const eid = body(world, geometry, [0, f(f(1.1) * i), 0]);
        world.storage(Shape).scale.set(eid, ...scales[i % 4]!);
        world.storage(Shape).enableHullTransform.set(eid, 1);
        bodies.push(eid);
    }
    return bodies;
}
function dominoes(world: World): number[] {
    ground(world, 80);
    const geometry = box(world, [0.2, 0.8, 0.05]);
    const bodies: number[] = [];
    for (let ring = 0; ring < 30; ring++) {
        const radius = f(7 + f(f(1.1) * ring));
        for (let alpha = 0; alpha <= 360; alpha += 2) {
            const angle = f(f(pi / 180) * alpha);
            const [c, s] = computeCosSin(world, angle);
            const position: V3 = [
                f(radius * c) - f(f(alpha / 630) * c),
                f(0.8),
                f(radius * s) - f(f(alpha / 630) * s),
            ];
            const eid = body(
                world,
                geometry,
                position,
                rotation(world, [0, 1, 0], f(-f(pi / 180) * alpha)),
            );
            bodies.push(eid);
            if (alpha === 0)
                world.resource(StackingImpulses).push({
                    body: eid,
                    impulse: [0, 0, 25],
                    point: [position[0], position[1] + f(0.8), position[2]],
                });
        }
    }
    return bodies;
}
function wedge(world: World): number[] {
    ground(world, 20);
    const geometry = world.resource(Hulls).add({
        name: "wedge",
        points: new Float32Array([
            -1, 1, -0.1, 1, 1, -0.1, -1, 1, 0.1, 1, 1, 0.1, -0.5, 0.5, 0, 0.5, 0.5, 0,
        ]),
        maxVertexCount: 6,
    });
    return [body(world, geometry, [0, 1, 0])];
}
function arch(world: World): number[] {
    ground(world, 40);
    const ps1 = [
        [16, 0],
        [14.93803712795643, 5.133601056842984],
        [13.79871746027416, 10.24928069555078],
        [12.56252963284711, 15.34107019122473],
        [11.20040987372525, 20.39856541571217],
        [9.66521217819836, 25.40369899225096],
        [7.87179930638133, 30.3179337000085],
        [5.635199558196225, 35.03820717801641],
        [2.405937953536585, 39.09554102558315],
    ].map((p) => p.map((v) => f(f(v) * 0.25)));
    const ps2 = [
        [24, 0],
        [22.33619528222415, 6.02299846205841],
        [20.54936888969905, 12.00964361211476],
        [18.60854610798073, 17.9470321677465],
        [16.46769273811807, 23.81367936585418],
        [14.05325025774858, 29.57079353071012],
        [11.23551045834022, 35.13775818285372],
        [7.752568160730571, 40.30450679009583],
        [3.016931552701656, 44.28891593799322],
    ].map((p) => p.map((v) => f(f(v) * 0.25)));
    const bodies: number[] = [];
    function stone(corners: number[][]): void {
        const points = [-0.5, 0.5].flatMap((z) => corners.map((p) => [p[0]!, p[1]!, z] as V3));
        const geometry = world.resource(Hulls).add({
            name: "arch stone",
            points: new Float32Array(points.flat()),
            maxVertexCount: 8,
        });
        const eid = body(world, geometry, [0, 0, 0]);
        world.storage(Shape).density.set(eid, 200);
        bodies.push(eid);
    }
    for (let i = 0; i < 8; i++) stone([ps1[i]!, ps2[i]!, ps2[i + 1]!, ps1[i + 1]!]);
    const mirror = (p: number[]): number[] => [-p[0]!, p[1]!];
    for (let i = 0; i < 8; i++)
        stone([mirror(ps2[i]!), mirror(ps1[i]!), mirror(ps1[i + 1]!), mirror(ps2[i + 1]!)]);
    stone([ps1[8]!, ps2[8]!, mirror(ps2[8]!), mirror(ps1[8]!)]);
    const geometry = box(world, [2, 0.5, 0.5]);
    for (let i = 0; i < 4; i++) {
        const eid = body(world, geometry, [0, f(f(0.5 + ps2[8]![1]!) + i), 0]);
        world.storage(Shape).density.set(eid, 200);
        bodies.push(eid);
    }
    return bodies;
}
function doubleDomino(world: World): number[] {
    ground(world, 20);
    const geometry = box(world, [0.125, 0.5, 0.25]);
    const bodies: number[] = [];
    let x = -7.5;
    for (let i = 0; i < 15; i++) {
        const eid = body(world, geometry, [x, 0.5, 0]);
        world.storage(Shape).density.set(eid, 4);
        bodies.push(eid);
        if (i === 0)
            world
                .resource(StackingImpulses)
                .push({ body: eid, impulse: [0.2, 0, 0], point: [x, 1, 0] });
        x = f(x + f(1.01));
    }
    return bodies;
}
function pyramid2D(world: World): number[] {
    ground(world, 40);
    const geometry = box(world, [1, 1, 1]);
    const bodies: number[] = [];
    for (let row = 0; row < 12; row++)
        for (let column = 0; column < 12 - row; column++) {
            const eid = body(world, geometry, [-10 + 2 * column + row, 1.5 + 2.5 * row, 0]);
            world
                .storage(Body)
                .motionLocks.set(
                    eid,
                    BodyMotionLock.linearZ | BodyMotionLock.angularX | BodyMotionLock.angularY,
                );
            bodies.push(eid);
        }
    return bodies;
}
function edgeCrossing(world: World): number[] {
    ground(world, 40);
    const box1 = box(world, [0.2, 0.02, 0.04]),
        box2 = box(world, [0.2, 0.04, 0.02]);
    const invLength = f(1 / f(Math.sqrt(f(f(f(0.1) * f(0.1)) + f(f(0.9) * f(0.9))))));
    const axis: V3 = [f(f(0.1) * invLength), f(f(0.9) * invLength), 0];
    const bodies: number[] = [];
    for (let row = 0; row < 3; row++) {
        const h = row === 1 ? f(0.04) : f(0.02);
        let x = -10;
        for (let angle = -pi; angle < f(pi + f(0.001)); angle = f(angle + f(f(0.1) * pi))) {
            bodies.push(body(world, row === 1 ? box2 : box1, [x, h, -2 + 2 * row]));
            bodies.push(
                body(
                    world,
                    row === 0 ? box1 : box2,
                    [x, f(20 * h), -2 + 2 * row],
                    rotation(world, axis, angle),
                ),
            );
            x += 1;
        }
    }
    return bodies;
}
export const stackingScenes = {
    cardHouse: {
        construct: cardHouse,
        gridExtent: 10,
        gridSpacing: 0.125,
        camera: { yaw: 40, pitch: 15, distance: 3.3, target: [0.8, 0.8, 0] },
        controls: undefined,
    },
    sphereStack: {
        construct: sphereStack,
        gridExtent: 15,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 15, distance: 43, target: [0, 21, 0] },
        controls: undefined,
    },
    capsuleStack: {
        construct: capsuleStack,
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 15, distance: 25, target: [0, 10, 0] },
        controls: undefined,
    },
    singleBox: {
        construct: singleBox,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 25, distance: 4, target: [0, 0.5, 0] },
        controls: undefined,
    },
    cylinder: {
        construct: cylinder,
        gridExtent: 10,
        gridSpacing: 0.5,
        camera: { yaw: 40, pitch: 25, distance: 5, target: [0, 1, 0] },
        controls: undefined,
    },
    jenga: {
        construct: jengaStack,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 15, distance: 12, target: [0, 3, 0] },
        controls: jengaControls,
    },
    cylinderStack: {
        construct: cylinderStack,
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 15, distance: 14, target: [0, 5, 0] },
        controls: undefined,
    },
    dominoes: {
        construct: dominoes,
        gridExtent: 80,
        gridSpacing: 5,
        camera: { yaw: 40, pitch: 45, distance: 110, target: [0, 0, 0] },
        controls: undefined,
    },
    wedge: {
        construct: wedge,
        gridExtent: 20,
        gridSpacing: 0.5,
        camera: { yaw: 40, pitch: 25, distance: 5, target: [0, 1, 0] },
        controls: undefined,
    },
    arch: {
        construct: arch,
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 15, distance: 22, target: [0, 7, 0] },
        controls: undefined,
    },
    doubleDomino: {
        construct: doubleDomino,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 25, distance: 20, target: [-0.5, 0.5, 0] },
        controls: undefined,
    },
    pyramid2D: {
        construct: pyramid2D,
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 15, distance: 34, target: [1, 14, 0] },
        controls: undefined,
    },
    edgeCrossing: {
        construct: edgeCrossing,
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 40, distance: 26, target: [0, 0, 0] },
        controls: undefined,
    },
} as const;

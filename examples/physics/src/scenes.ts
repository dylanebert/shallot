import {
    Body,
    BodyType,
    Hulls,
    Shape,
    SphericalJoint,
    Transform,
    type World,
} from "@dylanebert/shallot";

import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";

export type Scene = "boxStack" | "bridge";
export const views = {
    boxStack: { yaw: 40, pitch: 15, distance: 50, target: [0, 20, 0] },
    bridge: { yaw: 0, pitch: 20, distance: 35, target: [0, 10, 0] },
} as const;

export function createScene(world: World, scene: Scene): number[] {
    world.resource(PhysicsWorldDefinition).workerCount = 1;
    const geometry = world.resource(Hulls).register({ name: "box", halfExtents: [0.5, 0.5, 0.5] });
    const ground = world.create();
    const extent = scene === "boxStack" ? 40 : 60;
    world.add(ground, Transform, { translation: [0, -1, 0], scale: [2 * extent, 2, 2 * extent] });
    world.add(ground, Body);
    world.add(ground, Shape, { geometry });
    const bodies: number[] = [];
    if (scene === "boxStack") {
        for (let i = 0; i < 40; i++) {
            const body = world.create();
            world.add(body, Transform, { translation: [0, 0.75 + 1.25 * i, 0] });
            world.add(body, Body, { type: BodyType.Dynamic });
            world.add(body, Shape, { geometry, rollingResistance: 0.1 });
            bodies.push(body);
        }
        return bodies;
    }
    const anchor = world.create();
    world.add(anchor, Body);
    let previous = anchor;
    function joint(
        a: number,
        b: number,
        ax: number,
        ay: number,
        bx: number,
        by: number,
        z: number,
    ): void {
        const eid = world.create();
        world.add(eid, SphericalJoint, {
            a,
            b,
            localAnchorA: [ax, ay, z],
            localAnchorB: [bx, by, z],
            constraintHertz: 1000,
            enableSpring: 1,
            hertz: 2,
            dampingRatio: 1,
        });
    }
    for (let i = 0; i < 150; i++) {
        const body = world.create();
        world.add(body, Transform, {
            translation: [-20 + 0.125 * (1 + 2 * i), 20, 0],
            scale: [0.25, 0.25, 1],
        });
        world.add(body, Body, { type: BodyType.Dynamic, linearDamping: 0.1, angularDamping: 0.1 });
        world.add(body, Shape, { geometry, density: 20 });
        for (const z of [-0.5, 0.5])
            joint(previous, body, i === 0 ? -20 : 0.125, i === 0 ? 20 : 0, -0.125, 0, z);
        bodies.push(body);
        previous = body;
    }
    for (const z of [-0.5, 0.5]) joint(previous, anchor, 0.125, 0, 17.5, 20, z);
    return bodies;
}

export function setGravityScale(world: World, bodies: readonly number[], scale: number): void {
    const gravity = world.storage(Body).gravityScale;
    for (const body of bodies) gravity.set(body, scale);
}

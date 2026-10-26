import {
    Body,
    BodyType,
    Hulls,
    Shape,
    ShapeKind,
    SphericalJoint,
    Transform,
    type World,
} from "@dylanebert/shallot";
import { PhysicsWorldDefinition } from "@dylanebert/shallot/standard/physics";

const f = Math.fround;
function box(world: World, halfExtents: [number, number, number]): number {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(
    world: World,
    geometry: number,
    translation: [number, number, number],
    dynamic = true,
    density = 1000,
    enableSleep = true,
): number {
    const eid = world.create();
    world.add(eid, Transform, { translation });
    world.add(eid, Body, {
        type: dynamic ? BodyType.Dynamic : BodyType.Static,
        enableSleep: Number(enableSleep),
    });
    world.add(eid, Shape, { geometry, density });
    return eid;
}
function ground(world: World, extent: number): void {
    body(world, box(world, [extent, 1, extent]), [0, -1, 0], false);
}
function largePyramid(world: World): number[] {
    ground(world, 400);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    for (let i = 0; i < 100; i++)
        for (let j = i; j < 100; j++)
            bodies.push(
                body(
                    world,
                    geometry,
                    [(i + 1) * 0.5 + (j - i) - 50, (2 * i + 1) * 0.5, 0],
                    true,
                    100,
                ),
            );
    return bodies;
}
function widePyramid(world: World): number[] {
    ground(world, 100);
    const geometry = box(world, [f(0.975), f(0.975), f(0.975)]);
    const bodies: number[] = [];
    for (let i = 0; i < 15; i++)
        for (let j = Math.floor(i / 2); j < 15 - Math.floor((i + 1) / 2); j++)
            for (let k = Math.floor(i / 2); k < 15 - Math.floor((i + 1) / 2); k++)
                bodies.push(
                    body(world, geometry, [
                        -15 + 2 * j + (i & 1),
                        1 + 2.5 * i,
                        -15 + 2 * k + (i & 1),
                    ]),
                );
    return bodies;
}
function manyPyramids(world: World): number[] {
    ground(world, 77);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    let z = -76;
    const delta = f(152 / 13);
    for (let row = 0; row < 14; row++) {
        for (let column = 0; column < 14; column++) {
            const center = -77 + column * 11 + 1;
            for (let i = 0; i < 10; i++)
                for (let j = i; j < 10; j++)
                    bodies.push(
                        body(
                            world,
                            geometry,
                            [(i + 1) * 0.5 + (j - i) + center - 0.5, (2 * i + 1) * 0.5, z],
                            true,
                            100,
                            false,
                        ),
                    );
        }
        z = f(z + delta);
    }
    return bodies;
}
function fallingBoxes(world: World): number[] {
    ground(world, 100);
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies: number[] = [];
    for (let i = 0; i < 50; i++)
        for (let j = 0; j < 8; j++)
            for (let k = 0; k < 8; k++)
                bodies.push(body(world, geometry, [-8 + 2 * j, 2 * i + 2.5, -8 + 2 * k]));
    return bodies;
}
function jointGrid(world: World): number[] {
    world.resource(PhysicsWorldDefinition).enableSleep = false;
    const bodies: number[] = [];
    function joint(a: number, b: number, horizontal: boolean): void {
        const eid = world.create();
        world.add(eid, SphericalJoint, {
            a,
            b,
            localAnchorA: horizontal ? [0.5, 0, 0] : [0, -0.5, 0],
            localAnchorB: horizontal ? [-0.5, 0, 0] : [0, 0.5, 0],
        });
    }
    for (let k = 0; k < 100; k++) {
        for (let i = 0; i < 100; i++) {
            const eid = world.create();
            world.add(eid, Transform, { translation: [k, -i, 0] });
            world.add(eid, Body, {
                type: i === 0 ? BodyType.Static : BodyType.Dynamic,
                enableSleep: 0,
            });
            world.add(eid, Shape, {
                kind: ShapeKind.Sphere,
                radius: 0.4,
                filterCategoryLow: 2,
                filterCategoryHigh: 0,
                filterMaskLow: 0xfffffffd,
            });
            if (i > 0) joint(bodies[bodies.length - 1]!, eid, false);
            if (k > 0) joint(bodies[bodies.length - 100]!, eid, true);
            bodies.push(eid);
        }
    }
    return bodies;
}
export const benchmarkScenes = {
    largePyramid: {
        construct: largePyramid,
        gridExtent: 400,
        gridSpacing: 10,
        camera: { yaw: 40, pitch: -10, distance: 110, target: [0, 40, 0] },
        controls: undefined,
    },
    widePyramid: {
        construct: widePyramid,
        gridExtent: 100,
        gridSpacing: 5,
        camera: { yaw: 0, pitch: 5, distance: 80, target: [0, 18, 0] },
        controls: undefined,
    },
    manyPyramids: {
        construct: manyPyramids,
        gridExtent: 77,
        gridSpacing: 7,
        camera: { yaw: -10, pitch: 10, distance: 120, target: [0, 5, 0] },
        controls: undefined,
    },
    fallingBoxes: {
        construct: fallingBoxes,
        gridExtent: 100,
        gridSpacing: 5,
        camera: { yaw: 45, pitch: 10, distance: 80, target: [0, 20, 0] },
        controls: undefined,
    },
    jointGrid: {
        construct: jointGrid,
        gridExtent: 100,
        gridSpacing: 5,
        camera: { yaw: -25, pitch: 25, distance: 94, target: [30, -30, 30] },
        controls: undefined,
    },
} as const;

import {
    Body,
    BodyType,
    Hulls,
    type Plugin,
    recoverResource,
    Shape,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    getWorldCounters,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
export const RobustnessState = {
    create: () => ({
        bodies: [] as number[],
        baseCount: 4,
        overlap: 0.25,
        extent: 0.5,
        speed: 3,
        hertz: 30,
        dampingRatio: 10,
    }),
};
export const RobustnessSamplePlugin: Plugin = {
    name: "RobustnessSample",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(RobustnessState),
};
function box(world: World, halfExtents: V3): number {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(world: World, geometry: number, translation: V3, density?: number): number {
    const eid = world.create();
    world.add(eid, Transform, { translation });
    world.add(eid, Body, { type: BodyType.Dynamic });
    world.add(eid, Shape, { geometry, density });
    return eid;
}
function ground(world: World, extent: number): void {
    const eid = body(world, box(world, [extent, 1, extent]), [0, -1, 0]);
    world.storage(Body).type.set(eid, BodyType.Static);
}
function highMassRatio1(world: World): number[] {
    ground(world, 50);
    const geometry = box(world, [1, 1, 1]);
    const bodies: number[] = [];
    for (let j = 0; j < 3; j++) {
        let count = 10;
        const offset = -20 + 2 * (count + 1) * j;
        let y = 1;
        while (count > 0) {
            for (let i = 0; i < count; i++) {
                const coeff = i - 0.5 * count;
                bodies.push(
                    body(
                        world,
                        geometry,
                        [2 * coeff + offset, count === 1 ? y + 2 : y, 0],
                        count === 1 ? (j + 1) * 100 : 1,
                    ),
                );
            }
            --count;
            y += 2;
        }
    }
    return bodies;
}
function tinyPyramid(world: World): number[] {
    ground(world, 20);
    const extent = f(0.025),
        baseCount = 30;
    const geometry = box(world, [extent, extent, extent]);
    const bodies: number[] = [];
    for (let i = 0; i < baseCount; i++) {
        const y = f((2 * i + 1) * extent);
        for (let j = i; j < baseCount; j++) {
            const x = f(
                f(f((i + 1) * extent) + f(f(2 * (j - i)) * extent)) - f(baseCount * extent),
            );
            bodies.push(body(world, geometry, [x, y, 0]));
        }
    }
    return bodies;
}
function recoverOverlap(world: World): void {
    const s = world.resource(RobustnessState);
    for (const eid of s.bodies) world.destroy(eid);
    s.bodies.length = 0;
    Object.assign(world.resource(PhysicsWorldDefinition), {
        contactHertz: s.hertz,
        contactDampingRatio: s.dampingRatio,
        contactSpeed: s.speed,
    });
    const geometry = box(world, [s.extent, s.extent, s.extent]);
    const fraction = f(1 - s.overlap);
    let y = s.extent;
    for (let i = 0; i < s.baseCount; i++) {
        let x = f(f(fraction * s.extent) * (i - s.baseCount));
        for (let j = i; j < s.baseCount; j++) {
            s.bodies.push(body(world, geometry, [x, y, 0], 1));
            x = f(x + f(f(2 * fraction) * s.extent));
        }
        y = f(y + f(f(2 * fraction) * s.extent));
    }
}
function overlapRecovery(world: World): number[] {
    ground(world, 20);
    recoverOverlap(world);
    return world.resource(RobustnessState).bodies;
}
const overlapControls = {
    Extent: {
        label: "Extent",
        value: 0.5,
        min: 0.1,
        max: 1,
        step: 0.1,
        decimals: 1,
        set(world, _bodies, value) {
            world.resource(RobustnessState).extent = f(value);
            recoverOverlap(world);
        },
    },
    "Base Count": {
        label: "Base Count",
        value: 4,
        min: 1,
        max: 10,
        step: 1,
        decimals: 0,
        set(world, _bodies, value) {
            world.resource(RobustnessState).baseCount = value;
            recoverOverlap(world);
        },
    },
    Overlap: {
        label: "Overlap",
        value: 0.25,
        min: 0,
        max: 1,
        step: 0.01,
        decimals: 2,
        set(world, _bodies, value) {
            world.resource(RobustnessState).overlap = f(value);
            recoverOverlap(world);
        },
    },
    Speed: {
        label: "Speed",
        value: 3,
        min: 0,
        max: 10,
        step: 0.1,
        decimals: 1,
        set(world, _bodies, value) {
            world.resource(RobustnessState).speed = f(value);
            recoverOverlap(world);
        },
    },
    Hertz: {
        label: "Hertz",
        value: 30,
        min: 0,
        max: 240,
        step: 1,
        decimals: 0,
        set(world, _bodies, value) {
            world.resource(RobustnessState).hertz = f(value);
            recoverOverlap(world);
        },
    },
    "Damping Ratio": {
        label: "Damping Ratio",
        value: 10,
        min: 0,
        max: 20,
        step: 0.1,
        decimals: 1,
        set(world, _bodies, value) {
            world.resource(RobustnessState).dampingRatio = f(value);
            recoverOverlap(world);
        },
    },
    "Reset Scene": {
        label: "Reset Scene",
        value: 0,
        button: true,
        set(world) {
            recoverOverlap(world);
        },
    },
} satisfies Controls;
function overflowColorPile(world: World): number[] {
    ground(world, 20);
    const bodies = [body(world, box(world, [0.5, 2.5, 0.5]), [0, 2.5, 0], 50)];
    const neighborHalf = f(0.2);
    const radius = f(f(0.5 + neighborHalf) - f(0.03));
    const geometry = box(world, [neighborHalf, neighborHalf, neighborHalf]);
    const baseY = f(neighborHalf + f(0.05));
    const pi = f(Math.PI);
    for (let ring = 0; ring < 5; ring++) {
        const y = f(baseY + 0.5 * ring);
        const offset = ring & 1 ? f(pi / 5) : 0;
        for (let slot = 0; slot < 5; slot++) {
            const theta = f(offset + f(f(f(2 * pi) * slot) / 5));
            bodies.push(
                body(world, geometry, [
                    f(radius * f(Math.cos(theta))),
                    y,
                    f(radius * f(Math.sin(theta))),
                ]),
            );
        }
    }
    return bodies;
}
const overflowReadouts = {
    Neighbors: {
        label: "Neighbors",
        value: 0,
        decimals: 0,
        read() {
            return 25;
        },
        set() {},
    },
    "Overflow contacts": {
        label: "Overflow contacts",
        value: 0,
        decimals: 0,
        read(world) {
            return getWorldCounters(world)?.colorCounts[23] ?? 0;
        },
        set() {},
    },
    "Total contacts": {
        label: "Total contacts",
        value: 0,
        decimals: 0,
        read(world) {
            return getWorldCounters(world)?.contactCount ?? 0;
        },
        set() {},
    },
} satisfies Controls;
export const robustnessScenes = {
    highMassRatio1: {
        construct: highMassRatio1,
        gridExtent: 50,
        gridSpacing: 2,
        camera: { yaw: 15, pitch: 20, distance: 45, phoneDistance: 135, target: [1, 10, 0] },
        controls: undefined,
    },
    tinyPyramid: {
        construct: tinyPyramid,
        gridExtent: 20,
        gridSpacing: 0.25,
        camera: { yaw: 25, pitch: 20, distance: 3, phoneDistance: 4.5, target: [0, 0.6, 0] },
        controls: undefined,
    },
    overlapRecovery: {
        construct: overlapRecovery,
        gridExtent: 20,
        gridSpacing: 0.5,
        camera: {
            yaw: 25,
            pitch: 20,
            distance: 7,
            phoneDistance: 12,
            target: [-0.4, 1, 0],
            phoneTarget: [-0.4, 5.5, 0],
        },
        controls: overlapControls,
    },
    overflowColorPile: {
        construct: overflowColorPile,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 20, distance: 8, target: [0, 2, 0], phoneTarget: [0, 3.6, 0] },
        controls: overflowReadouts,
    },
} as const;

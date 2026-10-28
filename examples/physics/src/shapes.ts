import {
    Body,
    BodyType,
    FilterJoint,
    Hulls,
    PhysicsMeshes,
    type Plugin,
    RevoluteJoint,
    recoverResource,
    Shape,
    ShapeKind,
    ShapeMaterials,
    SphericalJoint,
    type System,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    applyWind,
    computeCosSin,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import conveyor from "./conveyor.json" with { type: "json" };
import type { Controls } from "./scenes";

const f = Math.fround;
const pi = f(Math.PI);
const radians = f(pi / 180);
type V3 = [number, number, number];
type Q4 = [number, number, number, number];
export const WindState = {
    create: () => ({ shapes: [] as number[], joints: [] as number[], lift: 0, time: 0 }),
};
const AfterShapes: System = {
    name: "shapes-after-step",
    group: "fixed",
    after: [StepPhysicsSystem],
    update(world) {
        const sample = world.resource(ShapeControlsState);
        if (sample.scene === "staticInvoke" && ++sample.step === 20) createStatic(world);
        if (sample.scene === "wind") {
            const speed = f(Math.sqrt(f(sample.wind * sample.wind)));
            const direction = speed ? f(f(1 / speed) * sample.wind) : 0;
            const wind = {
                x: f(speed * f(direction + sample.noise[0]!)),
                y: f(speed * sample.noise[1]!),
                z: f(speed * sample.noise[2]!),
            };
            for (const eid of sample.bodies)
                applyWind(world, eid, wind, sample.drag, sample.lift, 10, true);
            for (let i = 0; i < 3; i++) {
                let x = sample.seed;
                x ^= x << 13;
                x ^= x >>> 17;
                x ^= x << 5;
                sample.seed = x >>> 0;
                const rand = f(f(f(f(0.3) - f(-0.3)) * f((sample.seed % 32768) / 32767)) + f(-0.3));
                sample.noise[i] = f(f(f(1 - f(0.05)) * sample.noise[i]!) + f(f(0.05) * rand));
            }
        }
        const state = world.resource(WindState);
        for (const shape of state.shapes)
            applyWind(world, shape, { x: 0, y: 0, z: 0 }, 1, state.lift, 10, !state.joints.length);
        if (state.joints.length) {
            const angle = computeCosSin(world, f(10 * state.time))[1];
            for (const [index, joint] of state.joints.entries())
                world.storage(RevoluteJoint).targetAngle.set(joint, index ? -angle : angle);
            state.time = f(state.time + f(1 / 60));
        }
    },
};
function rotation(world: World, axis: V3, angle: number): Q4 {
    return rotationFromAxisAngle(world, { x: axis[0], y: axis[1], z: axis[2] }, angle);
}
function box(world: World, halfExtents: V3): number {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(world: World, position: V3, dynamic = true, rotation: Q4 = [0, 0, 0, 1]): number {
    const eid = world.create();
    world.add(eid, Transform, { translation: position, rotation });
    world.add(eid, Body, { type: dynamic ? BodyType.Dynamic : BodyType.Static });
    return eid;
}
function ground(world: World, extent: number): void {
    const eid = body(world, [0, -1, 0], false);
    world.add(eid, Shape, { geometry: box(world, [extent, 1, extent]) });
}
function inclinedPlane(world: World): number[] {
    ground(world, 50);
    const q = rotation(world, [1, 0, 0], f(40 * radians));
    const plane = body(world, [0, 7.5, -5], false, q);
    world.add(plane, Shape, { geometry: box(world, [16, 0.5, 10]), friction: 1 });
    const geometry = box(world, [1, 1, 1]);
    const bodies = [];
    for (let i = 0; i < 5; i++) {
        const eid = body(world, [-10 + 5 * i, 14.25, -10.6], true, q);
        world.add(eid, Shape, { geometry, friction: f((i + 1) * (i + 1) * f(0.04)) });
        bodies.push(eid);
    }
    return bodies;
}
function rollingResistance(world: World): number[] {
    ground(world, 80);
    const q = rotation(world, [1, 0, 0], f(10 * radians));
    const plane = body(world, [0, 2, -20], false, q);
    world.add(plane, Shape, { geometry: box(world, [32, 0.5, 15]) });
    const bodies = [];
    for (const capsule of [false, true])
        for (let i = 0; i < 5; i++) {
            const eid = body(world, [(capsule ? 2 : -25) + 5 * i, 8, -24], true, q);
            world.add(
                eid,
                Shape,
                capsule
                    ? {
                          kind: ShapeKind.Capsule,
                          capsuleCenter1: [-1, 0, 0],
                          capsuleCenter2: [1, 0, 0],
                          radius: 0.5,
                          rollingResistance: f(f(0.05) * i),
                      }
                    : { kind: ShapeKind.Sphere, radius: 1, rollingResistance: f(f(0.05) * i) },
            );
            bodies.push(eid);
        }
    return bodies;
}
function highResistance(world: World): number[] {
    ground(world, 50);
    const q = rotation(world, [0, 0, 1], f(radians * 30));
    const bodies = [];
    for (let i = 0; i < 10; i++) {
        const eid = body(world, [-22 + 5 * i, 1.5, 0], true, q);
        world.add(eid, Shape, {
            kind: ShapeKind.Capsule,
            capsuleCenter1: [0, -1, 0],
            capsuleCenter2: [0, 1, 0],
            radius: 0.5,
            rollingResistance: f(f(0.2) * i),
        });
        bodies.push(eid);
    }
    return bodies;
}
function isotropicFriction(world: World): number[] {
    ground(world, 100);
    const geometry = box(world, [1, 1, 1]);
    const bodies = [];
    for (let i = 0; i < 32; i++) {
        const alpha = f(f(pi / 16) * i);
        const [c, s] = computeCosSin(world, alpha);
        const eid = body(
            world,
            [f(15 * c), 1, f(15 * s)],
            true,
            rotation(world, [0, 1, 0], -alpha),
        );
        world.storage(Body).linearVelocity.set(eid, f(25 * c), 0, f(25 * s));
        world.add(eid, Shape, { geometry, friction: 0.6 });
        bodies.push(eid);
    }
    return bodies;
}
function slideTwist(world: World): number[] {
    ground(world, 50);
    const q = rotation(world, [1, 0, 0], f(20 * radians));
    const plane = body(world, [0, 4, 0], false, q);
    world.add(plane, Shape, { geometry: box(world, [10, 0.5, 10]), friction: 0.6 });
    const eid = body(world, [0, 5, 0], true, q);
    // b3RotateVector(q, axisY).
    world
        .storage(Body)
        .angularVelocity.set(
            eid,
            0,
            f(25 * f(1 + f(2 * f(-q[0] * q[0])))),
            f(25 * f(2 * f(q[0] * q[3]))),
        );
    world.add(eid, Shape, { geometry: box(world, [1, 0.5, 1]), friction: 0.3 });
    return [eid];
}
function conveyorBelt(world: World): number[] {
    ground(world, 20);
    const plane = body(world, [-5, 5, 0], false, rotation(world, [0, 1, 0], 0.2));
    world.add(plane, Shape, {
        geometry: box(world, [10, 0.25, 2]),
        friction: 0.8,
        tangentVelocity: [2, 0, 0],
    });
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const bodies = [];
    for (let i = 0; i < 5; i++) {
        const eid = body(world, [-10 + 2 * i, 7, 0]);
        world.add(eid, Shape, { geometry });
        bodies.push(eid);
    }
    return bodies;
}
function conveyorMesh(world: World): number[] {
    ground(world, 20);
    // Box3D data/meshes/conveyor.obj; material indices map the sample's built triangles to source faces.
    const geometry = world.resource(PhysicsMeshes).add({
        name: "conveyor",
        vertices: new Float32Array(conveyor.vertices),
        indices: new Uint32Array(conveyor.indices),
        materialIndices: new Uint8Array(conveyor.materialIndices),
        useMedianSplit: true,
        identifyEdges: true,
        weldVertices: true,
        weldTolerance: 0.002,
    });
    const velocities: V3[] = [
        [0, 0, 0],
        [0.7, 0, -0.2],
        [0.6, 0, 0.4],
        [0, 0, 1.3],
        [-0.6, 0, 0.4],
        [-0.75, 0, -0.4],
        [0, 0, -1.3],
    ];
    const colors = [0x008000, 0xadff2f, 0xf0fff0, 0xff69b4, 0xcd5c5c, 0x4b0082, 0xfffff0];
    const materialSet = world.resource(ShapeMaterials).add({
        name: "conveyor",
        materials: velocities.map((v, i) => ({
            friction: 0.8,
            restitution: 0,
            rollingResistance: 0,
            tangentVelocity: v.map((x) => f(2 * f(x))) as V3,
            userMaterialId: 0n,
            customColor: colors[i]!,
        })),
    });
    const mesh = body(world, [0, 0.5, 6], false, rotation(world, [0, 1, 0], f(0.5 * pi)));
    world.add(mesh, Shape, { kind: ShapeKind.Mesh, geometry, materialSet });
    const cylinder = world
        .resource(Hulls)
        .add({ name: "cylinder", cylinder: { height: 0.3, radius: 0.15, yOffset: 0, sides: 32 } });
    const bodies = [];
    for (let i = 0; i < 20; i++) {
        const eid = body(world, [f(f(-8.5) + f(f(0.9) * i)), 1.5, -5.5]);
        world.add(eid, Shape, { geometry: cylinder });
        bodies.push(eid);
    }
    return bodies;
}
function windDrop(world: World): number[] {
    ground(world, 15);
    const radius = f(0.1);
    const eid = body(world, [0, 10, 0], true, rotation(world, [1, 0, 0], 0.25));
    world.storage(Body).gravityScale.set(eid, 0.5);
    world.add(eid, Shape, {
        geometry: box(world, [f(4 * radius), f(f(0.1) * radius), f(4 * radius)]),
        density: 2,
    });
    Object.assign(world.resource(WindState), { shapes: [eid], joints: [], lift: 4, time: 0 });
    return [eid];
}
function windFlap(world: World): number[] {
    ground(world, 50);
    const a = f(0.4);
    const geometry = world.resource(Hulls).add({
        name: "wing",
        halfExtents: [f(2 * a), 0.01, a],
        transform: { position: [0, 0, 0], rotation: rotation(world, [1, 0, 0], 0.1) },
    });
    const wing1 = body(world, [f(-2 * a), 20, 0]);
    world.add(wing1, Shape, { geometry, density: 5 });
    const wing2 = body(world, [f(2 * a), 20, 0]);
    world.add(wing2, Shape, { geometry, density: 5 });
    const torso = body(world, [0, 20, 0]);
    world.add(torso, Shape, {
        kind: ShapeKind.Capsule,
        capsuleCenter1: [0, 0, -a],
        capsuleCenter2: [0, 0, a],
        radius: f(0.25 * a),
        density: 10,
    });
    const joints = [];
    for (const [index, wing] of [wing1, wing2].entries()) {
        const joint = world.create();
        world.add(joint, RevoluteJoint, {
            a: torso,
            b: wing,
            localAnchorB: [f((index ? -2 : 2) * a), 0, 0],
            drawScale: 0.1,
            enableSpring: 1,
            hertz: 6,
            dampingRatio: 0.5,
            enableLimit: 1,
            lowerAngle: f(f(-30 * pi) / 180),
            upperAngle: f(f(30 * pi) / 180),
        });
        joints.push(joint);
    }
    world.add(world.create(), FilterJoint, { a: wing1, b: wing2 });
    Object.assign(world.resource(WindState), { shapes: [wing1, wing2], joints, lift: 2, time: 0 });
    return [wing1, wing2, torso];
}
export const ShapeControlsState = {
    create: () => ({
        scene: "",
        bodies: [] as number[],
        joints: [] as number[],
        staticBody: -1,
        ground: -1,
        step: 0,
        invoke: 0,
        shape: 2,
        count: 10,
        wind: 6,
        drag: 1,
        lift: 0.75,
        noise: [0, 0, 0],
        seed: 12345,
    }),
};
const ShapeControlsPlugin: Plugin = {
    name: "ShapeControls",
    recovery: recoverResource(ShapeControlsState),
};
export const ShapesSamplePlugin: Plugin = {
    name: "ShapesSample",
    dependencies: [StandardPhysicsPlugin, ShapeControlsPlugin],
    systems: [AfterShapes],
    recovery: recoverResource(WindState),
};
function restitutionBodies(world: World): number[] {
    const state = world.resource(ShapeControlsState);
    for (const eid of state.bodies) world.destroy(eid);
    state.bodies = [];
    const geometry = box(world, [0.5, 0.5, 0.5]);
    const dr = f(1 / 39);
    let restitution = 0;
    for (let i = 0; i < 40; i++) {
        const eid = body(world, [-39 + 2 * i, 40, 0]);
        world.add(
            eid,
            Shape,
            state.shape === 0
                ? { kind: ShapeKind.Sphere, radius: 0.5, restitution }
                : { geometry, restitution },
        );
        state.bodies.push(eid);
        restitution = f(restitution + dr);
    }
    return state.bodies;
}
function restitution(world: World): number[] {
    ground(world, 50);
    Object.assign(world.resource(ShapeControlsState), { scene: "restitution", shape: 0 });
    return restitutionBodies(world);
}
function createStatic(world: World): void {
    const state = world.resource(ShapeControlsState);
    if (state.staticBody !== -1) world.destroy(state.staticBody);
    state.staticBody = body(world, [0, 0.5, 0], false);
    world.add(state.staticBody, Shape, {
        kind: ShapeKind.Sphere,
        radius: 0.5,
        invokeContactCreation: state.invoke,
    });
}
function staticInvoke(world: World): number[] {
    ground(world, 20);
    const eid = body(world, [0.25, 1, 0]);
    world.add(eid, Shape, { kind: ShapeKind.Sphere, radius: 0.5, rollingResistance: 0.2 });
    world.resource(ShapeControlsState).scene = "staticInvoke";
    return [eid];
}
function windBodies(world: World): number[] {
    const state = world.resource(ShapeControlsState);
    for (const eid of state.joints) world.destroy(eid);
    for (const eid of state.bodies) world.destroy(eid);
    state.bodies = [];
    state.joints = [];
    const r = f(0.1);
    const geometry = box(world, [f(1.25 * r), f(0.75 * r), f(0.125 * r)]);
    let a = state.ground;
    for (let i = 0; i < state.count; i++) {
        const eid = body(world, [f((2 * i + 1) * r), 2, 0]);
        world.storage(Body).gravityScale.set(eid, 0.5);
        world.storage(Body).enableSleep.set(eid, 0);
        world.add(
            eid,
            Shape,
            state.shape === 0
                ? { kind: ShapeKind.Sphere, radius: r, density: 20 }
                : state.shape === 1
                  ? {
                        kind: ShapeKind.Capsule,
                        capsuleCenter1: [-r, 0, 0],
                        capsuleCenter2: [r, 0, 0],
                        radius: f(0.5 * r),
                        density: 20,
                    }
                  : { geometry, density: 20 },
        );
        const joint = world.create();
        world.add(joint, SphericalJoint, {
            a,
            b: eid,
            localAnchorA: i ? [r, 0, 0] : [0, 2, 0],
            localAnchorB: [-r, 0, 0],
            drawScale: 0.1,
        });
        state.bodies.push(eid);
        state.joints.push(joint);
        a = eid;
    }
    return state.bodies;
}
function wind(world: World): number[] {
    ground(world, 20);
    Object.assign(world.resource(ShapeControlsState), {
        scene: "wind",
        ground: body(world, [0, 0, 0], false),
    });
    return windBodies(world);
}
const restitutionControls = {
    Shape: {
        label: "Shape",
        value: 0,
        options: [
            { label: "Sphere", value: 0 },
            { label: "Box", value: 1 },
        ],
        set(world, _bodies, value) {
            world.resource(ShapeControlsState).shape = value;
            restitutionBodies(world);
        },
    },
} satisfies Controls;
const staticControls = {
    Mode: {
        label: "Mode",
        value: 0,
        select: true,
        options: [
            { label: "Invoke", value: 1 },
            { label: "Passive", value: 0 },
        ],
        set(world, _bodies, value) {
            world.resource(ShapeControlsState).invoke = value;
        },
    },
    Create: {
        label: "Create",
        value: 0,
        button: true,
        enabled: (world) => world.resource(ShapeControlsState).staticBody === -1,
        set(world) {
            createStatic(world);
        },
    },
    Destroy: {
        label: "Destroy",
        value: 0,
        button: true,
        enabled: (world) => world.resource(ShapeControlsState).staticBody !== -1,
        set(world) {
            const state = world.resource(ShapeControlsState);
            world.destroy(state.staticBody);
            state.staticBody = -1;
        },
    },
} satisfies Controls;
const windControls = {
    Shape: {
        label: "Shape",
        value: 2,
        select: true,
        options: [
            { label: "Circle", value: 0 },
            { label: "Capsule", value: 1 },
            { label: "Box", value: 2 },
        ],
        set(world, _bodies, value) {
            world.resource(ShapeControlsState).shape = value;
            windBodies(world);
        },
    },
    Wind: {
        label: "Wind",
        value: 6,
        min: -50,
        max: 50,
        step: 0.1,
        decimals: 1,
        set(world, _bodies, value) {
            world.resource(ShapeControlsState).wind = f(value);
        },
    },
    Drag: {
        label: "Drag",
        value: 1,
        min: 0,
        max: 1,
        step: 0.01,
        decimals: 2,
        set(world, _bodies, value) {
            world.resource(ShapeControlsState).drag = f(value);
        },
    },
    Lift: {
        label: "Lift",
        value: 0.75,
        min: 0,
        max: 4,
        step: 0.01,
        decimals: 2,
        set(world, _bodies, value) {
            world.resource(ShapeControlsState).lift = f(value);
        },
    },
    Count: {
        label: "Count",
        value: 10,
        min: 1,
        max: 60,
        step: 1,
        set(world, _bodies, value) {
            world.resource(ShapeControlsState).count = value;
            windBodies(world);
        },
    },
} satisfies Controls;
export const shapeScenes = {
    restitution: {
        construct: restitution,
        gridExtent: 50,
        gridSpacing: 5,
        camera: { yaw: 5, pitch: 20, distance: 160, target: [0, 20, 0] },
        controls: restitutionControls,
    },
    staticInvoke: {
        construct: staticInvoke,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 30, pitch: 20, distance: 6, target: [0, 1, 0] },
        controls: staticControls,
    },
    wind: {
        construct: wind,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 15, pitch: 15, distance: 6, target: [1, 2.2, 0] },
        controls: windControls,
    },
    inclinedPlane: {
        construct: inclinedPlane,
        gridExtent: 50,
        gridSpacing: 2,
        camera: { yaw: 25, pitch: 25, distance: 43, target: [0, 7, -5] },
        controls: undefined,
    },
    rollingResistance: {
        construct: rollingResistance,
        gridExtent: 80,
        gridSpacing: 5,
        camera: { yaw: 20, pitch: 30, distance: 75, target: [0, 3, -20] },
        controls: undefined,
    },
    highResistance: {
        construct: highResistance,
        gridExtent: 50,
        gridSpacing: 2,
        camera: { yaw: 10, pitch: 25, distance: 55, target: [0.5, 1, 0] },
        controls: undefined,
    },
    isotropicFriction: {
        construct: isotropicFriction,
        gridExtent: 100,
        gridSpacing: 5,
        camera: { yaw: 40, pitch: 55, distance: 120, target: [0, 0, 0] },
        controls: undefined,
    },
    slideTwist: {
        construct: slideTwist,
        gridExtent: 50,
        gridSpacing: 1,
        camera: { yaw: 30, pitch: 30, distance: 28, target: [0, 4, 0] },
        controls: undefined,
    },
    conveyorBelt: {
        construct: conveyorBelt,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 25, distance: 29, target: [-3, 4, 0] },
        controls: undefined,
    },
    conveyorMesh: {
        construct: conveyorMesh,
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 25, pitch: 50, distance: 30, target: [0, 1, 0] },
        controls: undefined,
    },
    windDrop: {
        construct: windDrop,
        gridExtent: 15,
        gridSpacing: 1,
        camera: { yaw: 35, pitch: 15, distance: 16, target: [0, 5, 0] },
        controls: undefined,
    },
    windFlap: {
        construct: windFlap,
        gridExtent: 50,
        gridSpacing: 2,
        camera: { yaw: 35, pitch: 20, distance: 30, target: [0, 10, 0] },
        controls: undefined,
    },
} as const;

import {
    Body,
    BodyType,
    Hulls,
    ParallelJoint,
    PhysicsMeshes,
    type Plugin,
    RevoluteJoint,
    recoverResource,
    Shape,
    ShapeKind,
    SphericalJoint,
    type System,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    SyncPhysicsConstraintsSystem,
} from "@dylanebert/shallot/standard/physics";
import { human } from "./human";
import bones from "./human.json" with { type: "json" };
import type { Controls } from "./scenes";

const f = Math.fround;
type V3 = [number, number, number];
export const RagdollState = {
    create: () => ({
        human: { bodies: [] as number[], joints: [] as number[], shapes: [] as number[] },
        anchorJoints: [] as number[],
        mesh: false,
        incline: false,
        time: 0,
        motorized: true,
    }),
};
function tune(
    world: World,
    field: "maxMotorTorque" | "hertz" | "dampingRatio",
    value: number,
): void {
    for (const [i, eid] of world
        .resource(RagdollState)
        .human.joints.slice(0, bones.length - 1)
        .entries()) {
        const kind = world.has(eid, RevoluteJoint) ? RevoluteJoint : SphericalJoint;
        world
            .storage(kind)
            [field].set(
                eid,
                field === "maxMotorTorque" ? f(f(bones[i + 1]!.friction) * f(value)) : value,
            );
    }
}
const BeforeRagdoll: System = {
    name: "ragdoll-before-step",
    group: "fixed",
    before: [SyncPhysicsConstraintsSystem],
    update(world) {
        const state = world.resource(RagdollState);
        if (!state.incline) return;
        if (state.time > 2 && state.motorized) {
            tune(world, "maxMotorTorque", 0.5);
            tune(world, "hertz", 0.5);
            state.motorized = false;
        }
        state.time = f(state.time + f(1 / 60));
    },
};
export const RagdollPlugin: Plugin = {
    name: "RagdollSample",
    dependencies: [StandardPhysicsPlugin],
    systems: [BeforeRagdoll],
    recovery: recoverResource(RagdollState),
};
function ground(
    world: World,
    geometry: number,
    translation: V3,
    mesh = false,
    scale: V3 = [1, 1, 1],
    rotation: [number, number, number, number] = [0, 0, 0, 1],
): number {
    const eid = world.create();
    world.add(eid, Transform, { translation, rotation, scale });
    world.add(eid, Body);
    world.add(eid, Shape, { geometry, kind: mesh ? ShapeKind.Mesh : ShapeKind.Hull });
    return eid;
}
function grid(world: World, count: number, width: number, materials: number): number {
    return world.resource(PhysicsMeshes).add({
        name: "ragdoll ground",
        grid: {
            xCount: count,
            zCount: count,
            cellWidth: width,
            materialCount: materials,
            identifyEdges: true,
        },
    });
}
function spawn(world: World, values: Record<string, number>): number[] {
    const state = world.resource(RagdollState);
    state.human = human(
        world,
        [0, state.mesh ? 1 : 2, 0],
        1,
        values["Joint Friction"]!,
        values.Hertz!,
        values.Damping!,
    );
    if (state.mesh) {
        const pose = world.storage(Transform);
        for (const body of state.human.bodies) {
            const rotation = [
                pose.rotation.x.get(body),
                pose.rotation.y.get(body),
                pose.rotation.z.get(body),
                pose.rotation.w.get(body),
            ] as [number, number, number, number];
            const anchor = world.create();
            world.add(anchor, Transform, {
                translation: [
                    pose.translation.x.get(body),
                    pose.translation.y.get(body),
                    pose.translation.z.get(body),
                ],
                rotation,
            });
            world.add(anchor, Body, { type: BodyType.Kinematic });
            // b3InvMulQuat(bodyTransform.q, b3ComputeQuatBetweenUnitVectors(axisZ, axisY)).
            const [x, y, z, w] = rotation,
                s = f(Math.sqrt(0.5));
            const frame = [
                f(f(-w * s) - f(x * s)),
                f(f(z * s) - f(y * s)),
                f(f(-y * s) - f(z * s)),
                f(f(w * s) - f(x * s)),
            ] as [number, number, number, number];
            const joint = world.create();
            world.add(joint, ParallelJoint, {
                a: anchor,
                b: body,
                localRotationA: frame,
                localRotationB: frame,
                hertz: 8,
                dampingRatio: 1,
                maxTorque: 800,
            });
            state.anchorJoints.push(joint);
        }
    }
    return state.human.bodies;
}
function controls(hertz: number): Controls {
    return {
        "Joint Friction": {
            label: "Joint Friction",
            value: 5,
            min: 0,
            max: 20,
            step: 1,
            decimals: 0,
            set: (world, _bodies, value) => tune(world, "maxMotorTorque", value),
        },
        Hertz: {
            label: "Hertz",
            value: hertz,
            min: 0,
            max: 20,
            step: 0.1,
            decimals: 1,
            set: (world, _bodies, value) => tune(world, "hertz", value),
        },
        Damping: {
            label: "Damping",
            value: 0.7,
            min: 0,
            max: 4,
            step: 0.1,
            decimals: 1,
            set: (world, _bodies, value) => tune(world, "dampingRatio", value),
        },
        Respawn: {
            label: "Respawn",
            value: 0,
            button: true,
            set(world, _bodies, _value, values) {
                const state = world.resource(RagdollState);
                for (const joint of [
                    ...state.human.joints.slice(bones.length - 1),
                    ...state.human.joints.slice(0, bones.length - 1),
                ])
                    world.destroy(joint);
                // DestroyHuman leaves anchor bodies alive; their joints disappear with the bones.
                for (const joint of state.anchorJoints) world.destroy(joint);
                state.anchorJoints.length = 0;
                for (const [index, body] of state.human.bodies.entries()) {
                    // Body removal destroys its attached shapes in Box3D's linked-list order.
                    world.remove(body, Body);
                    world.destroy(body);
                    if (state.human.shapes[index]) world.destroy(state.human.shapes[index]!);
                }
                spawn(world, values);
            },
        },
    };
}
function construct(world: World, mesh: boolean): number[] {
    const state = world.resource(RagdollState);
    state.mesh = mesh;
    if (!mesh)
        ground(
            world,
            world.resource(Hulls).add({ name: "ground", halfExtents: [20, 1, 20] }),
            [0, -1, 0],
        );
    else {
        const base = ground(world, grid(world, 20, 2, 2), [0, 0, 0], true);
        for (const [halfExtents, translation] of [
            [
                [20, 5, 0.1],
                [0, 5, -20],
            ],
            [
                [20, 5, 0.1],
                [0, 5, 20],
            ],
            [
                [0.1, 5, 20],
                [-20, 5, 0],
            ],
            [
                [0.1, 5, 20],
                [20, 5, 0],
            ],
        ] as [V3, V3][]) {
            const shape = world.create();
            world.add(shape, Shape, {
                body: base,
                geometry: world.resource(Hulls).add({
                    name: "wall",
                    halfExtents,
                    transform: { position: translation, rotation: [0, 0, 0, 1] },
                }),
            });
        }
    }
    return spawn(world, { "Joint Friction": 5, Hertz: mesh ? 2 : 1, Damping: 0.7 });
}
function pile(world: World): number[] {
    ground(world, grid(world, 20, 1, 1), [0, -1, 0], true);
    let seed = 42;
    function random(): number {
        let x = seed;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        seed = x >>> 0;
        return f((seed % 32768) / 32767);
    }
    const bodies: number[] = [];
    for (let i = 0; i < 20; i++) {
        const x = f(f(4 * random()) - 2);
        random();
        const z = f(f(4 * random()) - 2);
        bodies.push(...human(world, [x, 2, z], i, 10, 0.5, 0.7).bodies);
    }
    return bodies;
}
function incline(world: World): number[] {
    const mesh = grid(world, 4, 2, 1);
    ground(
        world,
        mesh,
        [-10, 2, 0],
        true,
        [1, 1, 1],
        rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(f(-0.2) * f(Math.PI))),
    );
    ground(world, mesh, [0, 0, 0], true, [4, 4, 4]);
    const state = world.resource(RagdollState);
    state.incline = true;
    state.human = human(world, [-12, 6, 0], 1, 10, 2, 0.7);
    return state.human.bodies;
}
export const ragdollScenes = {
    ragdollBox: {
        construct: (world: World) => construct(world, false),
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 40, pitch: 25, distance: 7, target: [0, 1, 0] },
        controls: controls(1),
    },
    ragdollMesh: {
        construct: (world: World) => construct(world, true),
        gridExtent: 20,
        gridSpacing: 2,
        camera: { yaw: 40, pitch: 25, distance: 7, target: [0, 1, 0] },
        controls: controls(2),
    },
    ragdollPile: {
        construct: pile,
        gridExtent: 10,
        gridSpacing: 1,
        gridHeight: -1,
        camera: { yaw: 40, pitch: 30, distance: 13, phoneDistance: 20, target: [0, 1, 0] },
        controls: undefined,
    },
    ragdollIncline: {
        construct: incline,
        gridExtent: 16,
        gridSpacing: 2,
        camera: { yaw: 20, pitch: 25, distance: 29, phoneDistance: 46, target: [-5, 2, 0] },
        controls: undefined,
    },
};

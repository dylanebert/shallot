import {
    Body,
    BodyType,
    GlobalTransform,
    HeightFields,
    Hulls,
    InputPlugin,
    PhysicsMeshes,
    type Plugin,
    RevoluteJoint,
    recoverResource,
    Shape,
    ShapeKind,
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
    setAwake,
    setLinearVelocity,
    setTransform,
    timeOfImpactBodyMover,
} from "@dylanebert/shallot/standard/physics";
import { decodeMesh } from "./mesh-data";
import {
    add,
    cross,
    dot,
    type MoverShape,
    move,
    moverState,
    mul,
    pose,
    rotate,
    sub,
    v,
} from "./mover";
import { readPlayerControls } from "./player-controls";
import type { Controls } from "./scenes";
import stairsData from "./stairs.json" with { type: "json" };
import testMap from "./test_map01.json" with { type: "json" };

const f = Math.fround;
const TREE_SPAWN: [number, number, number] = [22, 1.5, 2];
const treeSpin = () => mul(0.5, v(f(Math.SQRT1_2), 0, -f(Math.SQRT1_2)));
const impact = () => ({ point: v(), normal: v(), speed: 0, valid: false });
export const GeometricMoverState = {
    create: () => ({
        active: false,
        mover: moverState(),
        entity: 0,
        clipVelocity: true,
        keys: { W: 0, A: 0, S: 0, D: 0, Space: 0, Shift: 0 },
        forward: v(0, 0, -1),
        right: v(1, 0, 0),
        policies: new Map<number, MoverShape>(),
        ignore: 0,
        tree: 0,
        treeShapes: [] as number[],
        treeGeometry: [] as number[],
        fallingBox: 0,
        treeTimer: 0,
        respawnTimer: 0,
        treeNeedsVelocity: true,
        respawnBox: false,
        boxStart: { p: v(), q: { v: v(), s: 1 } },
        treeStart: { p: v(), q: { v: v(), s: 1 } },
        moverStart: v(),
        boxImpact: impact(),
        treeImpact: impact(),
        impactTransform: { p: v(), q: { v: v(), s: 1 } },
    }),
};
function body(world: World, position: [number, number, number], type: number = BodyType.Static) {
    const eid = world.create();
    world.add(eid, Transform, { translation: position });
    world.add(eid, Body, { type });
    return eid;
}
function box(
    world: World,
    owner: number,
    halfExtents: [number, number, number],
    position: [number, number, number] = [0, 0, 0],
    rotation: [number, number, number, number] = [0, 0, 0, 1],
    density = 1000,
) {
    const shape = world.create();
    const geometry = world
        .resource(Hulls)
        .add({ name: "box", halfExtents, transform: { position, rotation } });
    world.add(shape, Shape, { body: owner, geometry, density });
    return shape;
}
function createTree(world: World): number {
    const s = world.resource(GeometricMoverState);
    const tree = body(world, TREE_SPAWN, BodyType.Dynamic);
    world.storage(Body).sleepThreshold.set(tree, 0.2);
    s.tree = tree;
    s.treeShapes = [];
    let y = 1,
        r = f(0.75);
    const l = f(1.5);
    for (let i = 0; i < 22; i++) {
        const shape = world.create();
        s.treeShapes.push(shape);
        const geometry = (s.treeGeometry[i] ??= world.resource(Hulls).add({
            name: "tree segment",
            cylinder: { height: f(l + f(2 * r)), radius: r, yOffset: f(y - r), sides: 6 },
        }));
        world.add(shape, Shape, {
            body: tree,
            geometry,
            friction: 0.9,
            rollingResistance: 0.05,
            updateBodyMass: i === 21 ? 1 : 0,
        });
        s.policies.set(shape, { maxPush: 10, clipVelocity: true, canMoverPush: false });
        y = f(y + f(l + f(2 * r)));
        r = f(f(0.95) * r);
    }
    // The center is read after fixed sync, once all segments have supplied their mass.
    return tree;
}
function initializeTreeVelocity(world: World) {
    const s = world.resource(GeometricMoverState);
    const mass = getMassData(world, s.tree)!;
    const placement = pose(world, s.tree);
    const center = add(placement.p, rotate(placement.q, mass.center));
    setAngularVelocity(world, s.tree, treeSpin());
    setLinearVelocity(world, s.tree, cross(treeSpin(), sub(center, v(...TREE_SPAWN))));
}
function construct(world: World): number[] {
    const s = world.resource(GeometricMoverState);
    Object.assign(s, GeometricMoverState.create());
    s.active = true;
    const level = body(world, [0, 0, 0]);
    world.add(world.create(), Shape, {
        body: level,
        kind: ShapeKind.Mesh,
        geometry: world.resource(PhysicsMeshes).add({
            name: "test map",
            ...decodeMesh(testMap),
            identifyEdges: true,
            weldVertices: true,
            weldTolerance: 0.002,
        }),
    });
    box(world, level, [1, 1, 1], [4, 1, 14]);
    box(world, level, [1, 1, 1], [4, 1, f(13.95)]);
    box(
        world,
        level,
        [1, 1, 1],
        [f(5.8), 1, f(13.7)],
        rotationFromAxisAngle(world, v(0, 1, 0), f(f(0.1) * f(Math.PI))),
    );
    const stairs = body(world, [-10, 0, 0]);
    world.add(world.create(), Shape, {
        body: stairs,
        kind: ShapeKind.Mesh,
        scale: [0.75, 0.75, -1.5],
        geometry: world.resource(PhysicsMeshes).add({
            name: "stairs",
            ...decodeMesh(stairsData),
            identifyEdges: true,
            weldVertices: true,
            weldTolerance: 0.002,
        }),
    });
    const torus = body(world, [-10, 1, -8]);
    world
        .storage(Transform)
        .rotation.set(torus, ...rotationFromAxisAngle(world, v(0, 1, 0), f(0.5 * f(Math.PI))));
    world.add(world.create(), Shape, {
        body: torus,
        kind: ShapeKind.Mesh,
        scale: [-0.75, 1.5, 0.5],
        geometry: world.resource(PhysicsMeshes).add({
            name: "torus",
            torus: { radialResolution: 10, tubularResolution: 12, radius: 2, thickness: 1 },
        }),
    });
    const terrain = body(world, [20, 0, 0]);
    world.add(world.create(), Shape, {
        body: terrain,
        kind: ShapeKind.HeightField,
        geometry: world.resource(HeightFields).add({
            name: "wave",
            wave: {
                rowCount: 50,
                columnCount: 50,
                scale: [1, 1, 1],
                rowFrequency: f(0.02),
                columnFrequency: f(0.04),
                makeHoles: true,
            },
        }),
    });
    createTree(world);
    for (const friendly of [false, true]) {
        const capsule = body(world, [0, f(1.4), friendly ? 5 : 6]);
        const shape = world.create();
        world.add(shape, Shape, {
            body: capsule,
            kind: ShapeKind.Capsule,
            radius: 0.3,
            filterCategoryLow: friendly ? 2 : 1,
        });
        s.policies.set(shape, {
            maxPush: friendly ? f(0.01) : 1,
            clipVelocity: !friendly,
            canMoverPush: false,
        });
    }
    const sphere = body(world, [7, 5, 0], BodyType.Dynamic);
    world.add(world.create(), Shape, { body: sphere, kind: ShapeKind.Sphere, radius: 0.5 });
    s.fallingBox = body(world, [10.5, 30, 0], BodyType.Dynamic);
    box(world, s.fallingBox, [1, f(0.1), 1]);
    const ignored = body(world, [7, 2, -3]);
    s.ignore = box(world, ignored, [0.5, 0.25, 0.5]);
    const ground = body(world, [0, 0, 0]);
    const door = body(world, [-2, f(1.6), 0], BodyType.Dynamic);
    world.storage(Body).gravityScale.set(door, 2);
    box(world, door, [0.75, 1.5, f(0.1)]);
    const q: [number, number, number, number] = [-f(Math.SQRT1_2), 0, 0, f(Math.SQRT1_2)];
    world.add(world.create(), RevoluteJoint, {
        a: ground,
        b: door,
        localAnchorA: [-2.75, f(1.6), 0],
        localAnchorB: [-0.75, 0, 0],
        localRotationA: q,
        localRotationB: q,
        enableLimit: 1,
        lowerAngle: f(f(f(Math.PI) / 180) * -90),
        upperAngle: f(f(f(Math.PI) / 180) * 90),
        enableSpring: 1,
        hertz: 1,
        dampingRatio: 0.5,
        maxMotorTorque: 100,
        drawScale: 2,
    });
    s.entity = world.create();
    world.add(s.entity, Transform, { translation: [7.5, 0.75, 9] });
    return [sphere, s.fallingBox, door, s.tree];
}
function settled(world: World, owner: number, field: "treeTimer" | "respawnTimer") {
    const s = world.resource(GeometricMoverState);
    if (isAwake(world, owner)) {
        s[field] = 0;
        return false;
    }
    s[field] = f(s[field] + f(1 / 60));
    if (s[field] < 1) return false;
    s[field] = 0;
    return true;
}

export const GeometricMoverPlugin: Plugin = {
    name: "GeometricMoverSample",
    dependencies: [StandardPhysicsPlugin, InputPlugin],
    recovery: recoverResource(GeometricMoverState),
    systems: [
        {
            name: "geometric-mover-input",
            group: "setup",
            update(world) {
                const s = world.resource(GeometricMoverState);
                if (!s.active) return;
                readPlayerControls(world, s.keys, s.forward, s.right);
            },
        },
        {
            name: "geometric-mover-before-step",
            group: "fixed",
            after: [SyncPhysicsConstraintsSystem],
            before: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(GeometricMoverState);
                if (!s.active) return;
                if (s.treeNeedsVelocity) {
                    initializeTreeVelocity(world);
                    s.treeNeedsVelocity = false;
                }
                if (s.respawnBox) {
                    setTransform(world, s.fallingBox, [10.5, 30, 0], [0, 0, 0, 1]);
                    setLinearVelocity(world, s.fallingBox, v());
                    setAngularVelocity(world, s.fallingBox, v());
                    setAwake(world, s.fallingBox, true);
                    s.respawnBox = false;
                }
                s.moverStart = { ...s.mover.position };
                s.boxStart = pose(world, s.fallingBox);
                s.treeStart = pose(world, s.tree);
                move(
                    world,
                    s.mover,
                    s.forward,
                    s.right,
                    s.keys,
                    s.clipVelocity,
                    s.ignore,
                    s.policies,
                );
                world
                    .storage(Transform)
                    .translation.set(
                        s.entity,
                        s.mover.position.x,
                        s.mover.position.y,
                        s.mover.position.z,
                    );
            },
        },
        {
            name: "geometric-mover-after-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(GeometricMoverState);
                if (!s.active) return;
                for (const tree of [false, true]) {
                    const owner = tree ? s.tree : s.fallingBox,
                        start = tree ? s.treeStart : s.boxStart;
                    const end = pose(world, owner);
                    const toi = timeOfImpactBodyMover(
                        world,
                        owner,
                        s.moverStart,
                        s.mover.capsule,
                        sub(s.mover.position, s.moverStart),
                        start,
                        end,
                    );
                    if (toi.fraction > 0) {
                        const mass = getMassData(world, owner)!;
                        const center = add(end.p, rotate(end.q, mass.center));
                        const g = world.storage(GlobalTransform).linearVelocity;
                        const velocity = add(
                            v(g.x.get(owner), g.y.get(owner), g.z.get(owner)),
                            cross(getAngularVelocity(world, owner)!, sub(toi.point, center)),
                        );
                        const result = tree ? s.treeImpact : s.boxImpact;
                        Object.assign(result, {
                            point: { ...toi.point },
                            normal: { ...toi.normal },
                            speed: dot(toi.normal, sub(velocity, s.mover.velocity)),
                            valid: true,
                        });
                        if (!tree) {
                            s.impactTransform.p = add(
                                start.p,
                                mul(toi.fraction, sub(end.p, start.p)),
                            );
                            const q1 =
                                f(dot(start.q.v, end.q.v) + f(start.q.s * end.q.s)) < 0
                                    ? { v: mul(-1, start.q.v), s: -start.q.s }
                                    : start.q;
                            let q = {
                                v: add(mul(f(1 - toi.fraction), q1.v), mul(toi.fraction, end.q.v)),
                                s: f(f(f(1 - toi.fraction) * q1.s) + f(toi.fraction * end.q.s)),
                            };
                            const inv = f(1 / f(Math.sqrt(f(dot(q.v, q.v) + f(q.s * q.s)))));
                            q = { v: mul(inv, q.v), s: f(inv * q.s) };
                            s.impactTransform.q = q;
                        }
                    }
                    if (settled(world, owner, tree ? "treeTimer" : "respawnTimer")) {
                        if (tree) {
                            for (const shape of s.treeShapes) {
                                s.policies.delete(shape);
                                world.destroy(shape);
                            }
                            world.destroy(s.tree);
                            createTree(world);
                            s.treeNeedsVelocity = true;
                        } else {
                            s.respawnBox = true;
                        }
                    }
                }
            },
        },
    ],
};
export const geometricMoverScene = {
    construct,
    camera: { yaw: 0, pitch: 25, distance: 6, target: [7.5, 0.75, 9] },
    gridExtent: 80,
    gridSpacing: 1,
    controls: {
        "Clip Velocity": {
            label: "Clip Velocity",
            value: 1,
            set(world, _bodies, value) {
                world.resource(GeometricMoverState).clipVelocity = !!value;
            },
        },
    } satisfies Controls,
};

import {
    Body,
    BodyType,
    GlobalTransform,
    Hulls,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    getContactData,
    getMassData,
    PhysicsEvents,
    PhysicsWorldDefinition,
    rotationFromAxisAngle,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
type Point = [number, number, number];
export const RestitutionState = {
    create: () => ({
        scene: "",
        bodies: [] as number[],
        ground: -1,
        balls: [] as number[],
        apex: [0, 0, 0],
        rebounding: [false, false, false],
        done: [false, false, false],
        restitution: f(0.9),
        spin: 0,
        friction: 0,
        safetyFactor: 0.5,
        stack: 2,
        gravity: false,
        height: 20,
        mass: 0,
        approachSpeed: 0,
        totalImpulse: 0,
        expectedImpulse: 0,
        contactSteps: 0,
        hit: false,
        bouncing: false,
    }),
};
export const RestitutionPlugin: Plugin = {
    name: "RestitutionSample",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(RestitutionState),
    systems: [
        {
            name: "restitution-after-step",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(RestitutionState);
                if (s.scene === "restitutionPropagation") {
                    const pose = world.storage(GlobalTransform);
                    for (let i = 0; i < 3; i++) {
                        const ball = s.balls[i]!;
                        const vy = pose.linearVelocity.y.get(ball);
                        const height = f(f(pose.translation.y.get(ball)) - f(0.5 + i));
                        if (!s.rebounding[i] && vy > 0) s.rebounding[i] = true;
                        if (s.rebounding[i] && !s.done[i]) {
                            s.apex[i] = Math.max(s.apex[i]!, height);
                            if (vy < 0) s.done[i] = true;
                        }
                    }
                }
                if (s.scene !== "bounceImpulse") return;
                const ball = s.bodies[0]!;
                if (!s.mass) s.mass = getMassData(world, ball)!.mass;
                const events = world.resource(PhysicsEvents);
                if (!s.hit && events.contactHit.count) {
                    s.hit = s.bouncing = true;
                    s.approachSpeed = events.contactHit.records[0]!.approachSpeed;
                    s.expectedImpulse = f(f(f(1 + s.restitution) * s.mass) * s.approachSpeed);
                }
                if (s.bouncing) {
                    const data = getContactData(world, ball, s.ground);
                    if (data) {
                        for (let m = 0; m < data.manifoldCount; m++) {
                            const manifold = data.manifolds[m]!;
                            for (let p = 0; p < manifold.pointCount; p++)
                                s.totalImpulse = f(
                                    s.totalImpulse + manifold.points[p]!.totalNormalImpulse,
                                );
                        }
                        s.contactSteps++;
                    } else s.bouncing = false;
                }
            },
        },
    ],
};
function box(world: World, halfExtents: Point = [0.5, 0.5, 0.5]): number {
    return world.resource(Hulls).add({ name: "box", halfExtents });
}
function body(world: World, position: Point): number {
    const e = world.create();
    world.add(e, Transform, { translation: position });
    world.add(e, Body, { type: BodyType.Dynamic });
    world.resource(RestitutionState).bodies.push(e);
    return e;
}
function begin(world: World, scene: string, extent: number, slick = false) {
    const s = world.resource(RestitutionState);
    Object.assign(s, RestitutionState.create(), { scene });
    const e = world.create();
    world.add(e, Transform, { translation: [0, -1, 0] });
    world.add(e, Body, { type: BodyType.Static });
    world.add(e, Shape, {
        geometry: box(world, [extent, 1, extent]),
        ...(slick ? { friction: 0, restitution: 0 } : {}),
    });
    s.ground = e;
    if (scene === "boxRestitution")
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: 0, z: 0 };
    if (scene === "sphereStackRestitution") s.restitution = 1;
    if (scene === "bounceImpulse") s.restitution = 0.5;
    return s;
}
function rebuild(world: World): void {
    const s = world.resource(RestitutionState);
    for (const e of s.bodies) world.destroy(e);
    s.bodies = [];
    s.balls = [];
    s.apex.fill(0);
    s.rebounding.fill(false);
    s.done.fill(false);
    s.mass = s.approachSpeed = s.totalImpulse = s.expectedImpulse = s.contactSteps = 0;
    s.hit = s.bouncing = false;
    if (s.scene === "boxRestitution") {
        const e = body(world, [0, f(0.25 + f(2.5 * f(1 / 60))), 0]);
        world.storage(Body).linearVelocity.set(e, 0, -5, 0);
        world.storage(Body).angularVelocity.set(e, 0, 0, s.spin);
        world.storage(Body).enableSleep.set(e, 0);
        world.add(e, Shape, {
            geometry: box(world, [1, 0.25, 1]),
            density: 1,
            friction: 0,
            restitution: s.restitution,
            enableHitEvents: 1,
        });
    } else if (s.scene === "bounceImpulse") {
        const e = body(world, [0, f(0.5 + s.height), 0]);
        world.storage(Body).enableSleep.set(e, 0);
        world.add(e, Shape, {
            kind: ShapeKind.Sphere,
            radius: 0.5,
            friction: 0,
            restitution: s.restitution,
            enableHitEvents: 1,
        });
    } else if (s.scene === "rotatedBoxRestitution") {
        const dq = f(f(0.5 * f(Math.PI)) / 12);
        const geometry = box(world);
        for (let i = 0; i < 12; i++) {
            const e = body(world, [-11 + 2 * i, 10, 0]);
            world.storage(Body).safetyFactor.set(e, s.safetyFactor);
            world
                .storage(Transform)
                .rotation.set(e, ...rotationFromAxisAngle(world, { x: 0, y: 0, z: 1 }, f(i * dq)));
            world.add(e, Shape, {
                geometry,
                friction: s.friction,
                restitution: s.restitution,
                filterGroupIndex: -1,
            });
        }
    } else if (s.scene === "sphereStackRestitution") {
        world.resource(PhysicsWorldDefinition).gravity = { x: 0, y: s.gravity ? -10 : 0, z: 0 };
        for (let i = 0; i <= s.stack; i++) {
            const impactor = i === s.stack;
            const y = f(f(0.5 + i) + (impactor ? f(2.5 * f(1 / 60)) : 0));
            const e = body(world, [0, y, 0]);
            world.storage(Body).enableSleep.set(e, 0);
            world.storage(Body).linearVelocity.set(e, 0, impactor ? -5 : 0, 0);
            world.add(e, Shape, {
                kind: ShapeKind.Sphere,
                radius: 0.5,
                friction: 0,
                restitution: impactor ? s.restitution : 0,
                enableHitEvents: Number(impactor),
            });
        }
    } else if (s.scene === "restitutionPropagation") {
        const geometry = box(world);
        for (let column = 0; column < 3; column++) {
            const x = 4 * (column - 1);
            for (let k = 0; k < column; k++) {
                const e = body(world, [x, 0.5 + k, 0]);
                world.storage(Body).enableSleep.set(e, 0);
                world.add(e, Shape, { geometry });
            }
            const e = body(world, [x, 5.5 + column, 0]);
            world.storage(Body).enableSleep.set(e, 0);
            world.add(e, Shape, {
                kind: ShapeKind.Sphere,
                radius: 0.5,
                restitution: s.restitution,
            });
            s.balls.push(e);
        }
    }
}
function slider(
    label: string,
    property: "restitution" | "spin" | "friction" | "safetyFactor" | "stack" | "height",
    value: number,
    min: number,
    max: number,
    step: number,
) {
    return {
        label,
        value,
        min,
        max,
        step,
        set(world: World, _b: readonly number[], value: number) {
            world.resource(RestitutionState)[property] = f(value);
            rebuild(world);
        },
    };
}
const reset = {
    label: "Reset",
    value: 0,
    button: true,
    set(world: World) {
        rebuild(world);
    },
};
const coefficient = (value: number) => slider("Restitution", "restitution", value, 0, 1, 0.01);
const impulseControls = {
    Restitution: coefficient(0.5),
    Height: slider("Height", "height", 20, 0.5, 50, 0.1),
    Reset: reset,
    Approach: {
        label: "Approach (m/s)",
        value: 0,
        set() {},
        read(world) {
            const s = world.resource(RestitutionState);
            return s.hit ? s.approachSpeed : null;
        },
    },
    Measured: {
        label: "Measured (N s)",
        value: 0,
        set() {},
        read(world) {
            const s = world.resource(RestitutionState);
            return s.hit ? s.totalImpulse : null;
        },
    },
    Expected: {
        label: "Expected (N s)",
        value: 0,
        set() {},
        read(world) {
            const s = world.resource(RestitutionState);
            return s.hit ? s.expectedImpulse : null;
        },
    },
    Steps: {
        label: "Contact steps",
        value: 0,
        set() {},
        read(world) {
            const s = world.resource(RestitutionState);
            return s.hit ? s.contactSteps : null;
        },
    },
} satisfies Controls;
export const restitutionScenes = {
    singleBoxRestitution: {
        construct(world: World) {
            const s = begin(world, "singleBoxRestitution", 4);
            const e = body(world, [0, 5, 0]);
            world.storage(Body).safetyFactor.set(e, 0.01);
            world.add(e, Shape, {
                geometry: box(world),
                density: 1,
                friction: 0,
                restitution: 1,
                enableHitEvents: 1,
            });
            return s.bodies;
        },
        gridExtent: 4,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 25, distance: 16, phoneDistance: 20, target: [0, 3, 0] },
        controls: undefined,
    },
    singleSphereRestitution: {
        construct(world: World) {
            const s = begin(world, "singleSphereRestitution", 4);
            const e = body(world, [0, 10, 0]);
            world.storage(Body).safetyFactor.set(e, 0.01);
            world.add(e, Shape, {
                kind: ShapeKind.Sphere,
                radius: 0.5,
                density: 1,
                friction: 0,
                restitution: 1,
                enableHitEvents: 1,
            });
            return s.bodies;
        },
        gridExtent: 4,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 25, distance: 24, phoneDistance: 30, target: [0, 5, 0] },
        controls: undefined,
    },
    twoBoxRestitution: {
        construct(world: World) {
            const s = begin(world, "twoBoxRestitution", 10);
            const geometry = box(world);
            for (const position of [
                [0, 0.5, 0],
                [f(0.1), 6, 0],
            ] as Point[]) {
                const e = body(world, position);
                world.add(e, Shape, { geometry, restitution: 0.9 });
            }
            return s.bodies;
        },
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 25, distance: 16, phoneDistance: 22, target: [0, 3, 0] },
        controls: undefined,
    },
    sphereRestitution: {
        construct(world: World) {
            const s = begin(world, "sphereRestitution", 20);
            let restitution = 0;
            for (let i = 0; i < 10; i++) {
                for (const y of [1, 4]) {
                    const e = body(world, [-9 + 2 * i, y, 0]);
                    world.add(e, Shape, {
                        kind: ShapeKind.Sphere,
                        radius: 0.5,
                        density: 1,
                        restitution,
                    });
                }
                restitution = f(restitution + f(1 / 9));
            }
            return s.bodies;
        },
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 25, distance: 30, phoneDistance: 50, target: [0, 3, 0] },
        controls: undefined,
    },
    boxRestitution: {
        construct(world: World) {
            const s = begin(world, "boxRestitution", 40, true);
            rebuild(world);
            return s.bodies;
        },
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 25, distance: 14, phoneDistance: 18, target: [0, 3, 0] },
        controls: {
            Restitution: coefficient(0.9),
            Spin: slider("Spin", "spin", 0, 0, 2, 0.01),
            Reset: reset,
        } satisfies Controls,
    },
    rotatedBoxRestitution: {
        construct(world: World) {
            const s = begin(world, "rotatedBoxRestitution", 25);
            rebuild(world);
            return s.bodies;
        },
        gridExtent: 25,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 25, distance: 36, phoneDistance: 60, target: [0, 5, 0] },
        controls: {
            Restitution: coefficient(0.9),
            Friction: slider("Friction", "friction", 0, 0, 1, 0.1),
            Safety: slider("Safety Factor", "safetyFactor", 0.5, 0.01, 0.5, 0.01),
            Reset: reset,
        } satisfies Controls,
    },
    sphereStackRestitution: {
        construct(world: World) {
            const s = begin(world, "sphereStackRestitution", 40, true);
            rebuild(world);
            return s.bodies;
        },
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 25, distance: 12, phoneDistance: 16, target: [0, 2.5, 0] },
        controls: {
            Restitution: coefficient(1),
            Stack: slider("Stack", "stack", 2, 0, 3, 1),
            Gravity: {
                label: "Gravity",
                value: 0,
                set(world, _b, value) {
                    world.resource(RestitutionState).gravity = Boolean(value);
                    rebuild(world);
                },
            },
            Reset: reset,
        } satisfies Controls,
    },
    restitutionPropagation: {
        construct(world: World) {
            const s = begin(world, "restitutionPropagation", 40);
            rebuild(world);
            return s.bodies;
        },
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 0, pitch: 25, distance: 24, phoneDistance: 34, target: [0, 4, 0] },
        controls: { Restitution: coefficient(0.9), Reset: reset } satisfies Controls,
    },
    bounceImpulse: {
        construct(world: World) {
            const s = begin(world, "bounceImpulse", 40, true);
            rebuild(world);
            return s.bodies;
        },
        gridExtent: 40,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 25, distance: 36, phoneDistance: 44, target: [0, 8, 0] },
        controls: impulseControls,
    },
} as const;

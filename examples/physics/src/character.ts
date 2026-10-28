import {
    Body,
    Hulls,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    type CollisionPlane,
    collideMover,
    type PhysicsPlanes,
    PhysicsWorldDefinition,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
    solvePlanes,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

export const CharacterSample = {
    create: () => ({
        scene: "" as "" | "capsulePlane" | "moverOverlap",
        position: { x: 0, y: 0, z: 0 },
        capsule: {
            center1: { x: 0, y: -0.5, z: 0 },
            center2: { x: 0, y: 0.5, z: 0 },
            radius: 0.25,
        },
        planes: { records: [], count: 0 } as PhysicsPlanes,
        solverPlanes: [] as CollisionPlane[],
        pushed: { x: 0, y: 0, z: 0 },
        zeroNormalCount: 0,
        tracking: false,
        base: { x: 0, y: 0, z: 0 },
        origin: { x: 0, y: 0, z: 0 },
    }),
};
type Vec = { x: number; y: number; z: number };
export function characterDrag(
    world: World,
    origin: Vec,
    translation: Vec,
    press: boolean,
    held: boolean,
): boolean {
    const s = world.resource(CharacterSample);
    const f = Math.fround;
    const length = f(
        Math.sqrt(
            f(
                f(f(translation.x * translation.x) + f(translation.y * translation.y)) +
                    f(translation.z * translation.z),
            ),
        ),
    );
    const p = {
        x: f(origin.x + f(10 * f(translation.x / length))),
        y: f(origin.y + f(10 * f(translation.y / length))),
        z: f(origin.z + f(10 * f(translation.z / length))),
    };
    if (press && held) {
        s.tracking = hitsUprightCapsule(s.position, s.capsule, origin, translation);
        if (s.tracking) {
            Object.assign(s.origin, p);
            Object.assign(s.base, s.position);
        }
    }
    if (s.tracking) {
        s.position.x = f(s.base.x + f(p.x - s.origin.x));
        s.position.y = f(s.base.y + f(p.y - s.origin.y));
        s.position.z = f(s.base.z + f(p.z - s.origin.z));
    }
    if (!held) s.tracking = false;
    return s.tracking;
}
function hitsUprightCapsule(
    position: Vec,
    capsule: { center1: Vec; center2: Vec; radius: number },
    origin: Vec,
    direction: Vec,
): boolean {
    const { center1, center2, radius } = capsule;
    const lower = Math.min(center1.y, center2.y),
        upper = Math.max(center1.y, center2.y);
    const x = origin.x - position.x - center1.x,
        y = origin.y - position.y,
        z = origin.z - position.z - center1.z;
    const endDistance = y - Math.max(lower, Math.min(upper, y));
    if (x * x + z * z + endDistance * endDistance <= radius * radius) return true;
    const a = direction.x * direction.x + direction.z * direction.z;
    const b = x * direction.x + z * direction.z;
    const c = x * x + z * z - radius * radius;
    const disc = b * b - a * c;
    if (a > 0 && disc >= 0) {
        const t = (-b - Math.sqrt(disc)) / a;
        if (t >= 0 && t <= 1 && y + t * direction.y >= lower && y + t * direction.y <= upper)
            return true;
    }
    for (const end of [lower, upper]) {
        const yy = y - end;
        const aa = a + direction.y * direction.y;
        const bb = b + yy * direction.y;
        const cc = c + yy * yy;
        const d = bb * bb - aa * cc;
        if (d < 0) continue;
        const t = (-bb - Math.sqrt(d)) / aa;
        if (t >= 0 && t <= 1) return true;
    }
    return false;
}
export function solveCapsule(world: World): void {
    const s = world.resource(CharacterSample);
    const result = solvePlanes({ x: 0, y: 0, z: 0 }, s.solverPlanes, s.planes.count);
    const f = Math.fround;
    s.position.x = f(s.position.x + result.delta.x);
    s.position.y = f(s.position.y + result.delta.y);
    s.position.z = f(s.position.z + result.delta.z);
}
export const CharacterSamplePlugin: Plugin = {
    name: "CharacterSample",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(CharacterSample),
    systems: [
        {
            name: "character-sample-queries",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(CharacterSample);
                if (!s.scene) return;
                s.planes = collideMover(
                    world,
                    s.position,
                    s.capsule,
                    s.scene === "capsulePlane" ? 3 : 32,
                );
                s.zeroNormalCount = 0;
                for (let i = 0; i < s.planes.count; i++) {
                    const p = s.planes.records[i]!.plane.plane;
                    const solver = (s.solverPlanes[i] ??= {
                        plane: { normal: { x: 0, y: 0, z: 0 }, offset: 0 },
                        pushLimit: 3.4028234663852886e38,
                        push: 0,
                        clipVelocity: true,
                    });
                    Object.assign(solver.plane.normal, p.normal);
                    solver.plane.offset = p.offset;
                    solver.push = 0;
                    if (!validNormal(p.normal)) s.zeroNormalCount++;
                }
                if (s.scene !== "moverOverlap") return;
                const solved = solvePlanes({ x: 0, y: 0, z: 0 }, s.solverPlanes, s.planes.count);
                const f = Math.fround;
                s.pushed.x = f(s.position.x + solved.delta.x);
                s.pushed.y = f(s.position.y + solved.delta.y);
                s.pushed.z = f(s.position.z + solved.delta.z);
            },
        },
    ],
};
export function validNormal(n: { x: number; y: number; z: number }): boolean {
    const f = Math.fround;
    return (
        Math.abs(f(f(f(n.x * n.x) + f(n.y * n.y)) + f(n.z * n.z)) - 1) < 100 * 1.1920928955078125e-7
    );
}
function construct(world: World, scene: "capsulePlane" | "moverOverlap"): number[] {
    // These samples query without stepping their world.
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    const s = world.resource(CharacterSample);
    s.scene = scene;
    s.position =
        scene === "capsulePlane" ? { x: 0, y: 1, z: Math.fround(0.4) } : { x: 0, y: 3.5, z: 0 };
    Object.assign(s.pushed, s.position);
    s.capsule.radius = Math.fround(scene === "capsulePlane" ? 0.25 : 0.35);
    const bodies: number[] = [];
    const box = (x: number, y: number, z: number, size: number) => {
        const body = world.create();
        world.add(body, Transform, { translation: [x, y, z] });
        world.add(body, Body);
        world.add(body, Shape, {
            geometry: world.resource(Hulls).add({ name: "box", halfExtents: [size, size, size] }),
        });
        bodies.push(body);
    };
    if (scene === "capsulePlane") box(0, 1, 1, 0.5);
    else {
        for (const [x, kind] of [
            [-3, ShapeKind.Sphere],
            [0, ShapeKind.Capsule],
        ] as const) {
            const body = world.create();
            world.add(body, Transform, { translation: [x, 1, 0] });
            world.add(body, Body);
            world.add(body, Shape, {
                kind,
                radius: x === -3 ? 0.6 : 0.4,
                capsuleCenter1: [0, 0, -0.7],
                capsuleCenter2: [0, 0, 0.7],
            });
            bodies.push(body);
        }
        box(3, 1, 0, 0.6);
    }
    return bodies;
}
export const characterScenes = {
    capsulePlane: {
        construct: (world: World) => construct(world, "capsulePlane"),
        gridExtent: 10,
        gridSpacing: 1,
        camera: { yaw: 30, pitch: 25, distance: 5, phoneDistance: 7, target: [0, 1, 0.5] },
        controls: {
            Solve: { label: "Solve", value: 0, button: true, set: solveCapsule },
        } satisfies Controls,
    },
    moverOverlap: {
        construct: (world: World) => construct(world, "moverOverlap"),
        gridExtent: 12,
        gridSpacing: 1,
        camera: { yaw: 20, pitch: 25, distance: 11, phoneDistance: 22, target: [0, 1.75, 0] },
        controls: {
            Planes: {
                label: "Planes",
                value: 0,
                decimals: 0,
                read: (world: World) => world.resource(CharacterSample).planes.count,
                set() {},
            },
            Degenerate: {
                label: "Degenerate normals",
                value: 0,
                decimals: 0,
                read: (world: World) => world.resource(CharacterSample).zeroNormalCount,
                set() {},
            },
        } satisfies Controls,
    },
};

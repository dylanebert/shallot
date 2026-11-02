import { type Plugin, recoverResource, type World } from "@dylanebert/shallot";
import {
    hull2D,
    PhysicsWorldDefinition,
    type Point2D,
    StandardPhysicsPlugin,
    shapeCast,
    simplifyHull2D,
} from "@dylanebert/shallot/standard/physics";
import { box, identity, vec } from "./collision-debug";
import type { Controls } from "./scenes";

const f = Math.fround;
export const CollisionGeometry = {
    create: () => ({
        scene: "",
        seed: 12345,
        source: 0,
        pointCount: 20,
        targetCount: 6,
        drawScale: 1,
        drawLabels: true,
        input: [] as Point2D[],
        hull: [] as Point2D[],
        simplified: [] as Point2D[],
        gap: 4,
        angle: 0.5,
        distance: 2,
        canEncroach: true,
        transform: identity(),
        translation: vec(),
        cast: null as ReturnType<typeof shapeCast> | null,
    }),
};
export const CollisionGeometryPlugin: Plugin = {
    name: "CollisionGeometry",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(CollisionGeometry),
};
function random(s: ReturnType<typeof CollisionGeometry.create>): number {
    let x = s.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    s.seed = x >>> 0;
    return f(f(10 * f((s.seed % 32768) / 32767)) - 5);
}
function generate(world: World): void {
    const s = world.resource(CollisionGeometry);
    s.input = [];
    s.drawScale = s.source === 2 ? 50 : 1;
    const point = (x: number, y: number) =>
        s.input.push({ p: { x: f(x), y: f(y) }, separation: 0, originalIndex: s.input.length });
    if (s.source === 0) for (let i = 0; i < s.pointCount; i++) point(random(s), random(s));
    else if (s.source === 1) {
        const delta = f(f(2 * f(Math.PI)) / s.pointCount);
        for (let i = 0; i < s.pointCount; i++) {
            const angle = f(delta * i);
            point(f(5 * f(Math.cos(angle))), f(5 * f(Math.sin(angle))));
        }
    } else {
        for (const [x, y] of [
            [0, 0],
            [-0.00305632968, 0.0000554900616],
            [-0.00305632968, 0.0246563386],
            [-0.00305632968, 0.0246562902],
            [-0.00305632968, 0.0800221488],
        ])
            point(x!, y!);
    }
    s.targetCount = Math.max(3, Math.min(s.targetCount, s.input.length));
    s.hull = hull2D(world, s.input);
    s.simplified = simplifyHull2D(world, s.hull, s.targetCount);
}
function cast(world: World): void {
    const s = world.resource(CollisionGeometry);
    const radians = f(s.angle * f(f(Math.PI) / 180));
    s.transform = { p: vec(0, f(0.5 + f(f(0.001) * s.gap)), 0), q: identity().q };
    s.translation = vec(
        f(s.distance * f(Math.cos(radians))),
        f(s.distance * f(Math.sin(radians))),
        0,
    );
    s.cast = shapeCast(world, {
        proxyA: { points: box(3, 0.25, 3, -0.25), count: 8, radius: 0 },
        proxyB: { points: box(0.5, 0.5, 0.5), count: 8, radius: 0 },
        transform: s.transform,
        translationB: s.translation,
        maxFraction: 1,
        canEncroach: s.canEncroach,
    });
}
function construct(world: World, scene: string): number[] {
    Object.assign(world.resource(CollisionGeometry), CollisionGeometry.create(), { scene });
    world.resource(PhysicsWorldDefinition).enableAutomaticStep = false;
    if (scene === "hull2D") generate(world);
    else cast(world);
    return [];
}
const hullControls = {
    Source: {
        label: "Source",
        value: 0,
        select: true,
        options: ["Random", "Circle", "Sliver"].map((label, value) => ({ label, value })),
        set(world, _b, value, values) {
            world.resource(CollisionGeometry).source = value;
            generate(world);
            values.Target = world.resource(CollisionGeometry).targetCount;
        },
    },
    Points: {
        label: "Point Count",
        value: 20,
        min: 4,
        max: 64,
        step: 1,
        enabled: (world) => world.resource(CollisionGeometry).source !== 2,
        set(world, _b, value, values) {
            world.resource(CollisionGeometry).pointCount = value;
            generate(world);
            values.Target = world.resource(CollisionGeometry).targetCount;
        },
    },
    Regenerate: {
        label: "Regenerate",
        value: 0,
        button: true,
        enabled: (world) => world.resource(CollisionGeometry).source === 0,
        set(world) {
            generate(world);
        },
    },
    Target: {
        label: "Target Count",
        value: 6,
        min: 3,
        max: 20,
        step: 1,
        bounds: (_v, world) => [3, world?.resource(CollisionGeometry).input.length ?? 20],
        set(world, _b, value) {
            const s = world.resource(CollisionGeometry);
            s.targetCount = value;
            s.simplified = simplifyHull2D(world, s.hull, value);
        },
    },
    Labels: {
        label: "Draw Labels",
        value: 1,
        set(world, _b, value) {
            world.resource(CollisionGeometry).drawLabels = Boolean(value);
        },
    },
} satisfies Controls;
function castSlider(
    label: string,
    property: "gap" | "angle" | "distance",
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
            world.resource(CollisionGeometry)[property] = f(value);
            cast(world);
        },
    };
}
export const collisionGeometryScenes = {
    encroachCast: {
        construct: (world: World) => construct(world, "encroachCast"),
        gridExtent: 20,
        gridSpacing: 1,
        camera: { yaw: 120, pitch: 25, distance: 10, phoneDistance: 18, target: [0, 0.5, 0] },
        controls: {
            Encroach: {
                label: "Can Encroach",
                value: 1,
                set(world, _b, value) {
                    world.resource(CollisionGeometry).canEncroach = Boolean(value);
                    cast(world);
                },
            },
            Gap: castSlider("Gap (mm)", "gap", 4, 0, 20, 0.1),
            Angle: castSlider("Angle", "angle", 0.5, -5, 5, 0.1),
            Distance: castSlider("Distance", "distance", 2, 0, 4, 0.1),
        } satisfies Controls,
    },
    hull2D: {
        construct: (world: World) => construct(world, "hull2D"),
        gridExtent: 10,
        gridSpacing: 1,
        camera: {
            yaw: 0,
            pitch: 25,
            distance: 20,
            phoneDistance: 35,
            target: [0, 0, 0],
            phoneTarget: [0, 7, 0],
        },
        controls: hullControls,
    },
} as const;

import {
    Body,
    HeightFields,
    Hulls,
    type Plugin,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    castRayClosest,
    castShape,
    cloneAndTransformHull,
    createHull,
    emptyQueryHit,
    type HullData,
    type PhysicsQueryHit,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";

const f = Math.fround;
const identity = { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } };
const scale = { x: -1, y: 1, z: 1 };
export const QuerySample = {
    create: () => ({
        scene: "",
        radius: f(0.1),
        points: [] as { x: number; y: number; z: number }[],
        hull: null as HullData | null,
        clone: null as HullData | null,
    }),
};
export function heightField(world: World): number[] {
    const sample = world.resource(QuerySample);
    sample.scene = "heightField";
    sample.radius = f(0.1);
    const geometry = world.resource(HeightFields).add({
        name: "Wave",
        wave: {
            rowCount: 50,
            columnCount: 50,
            scale: [1, 1, 1],
            rowFrequency: f(0.02),
            columnFrequency: f(0.04),
            makeHoles: true,
        },
    });
    const eid = world.create();
    world.add(eid, Transform, { translation: [-25, 0, -25] });
    world.add(eid, Body);
    world.add(eid, Shape, { kind: ShapeKind.HeightField, geometry });
    return [eid];
}
export function hull(world: World): number[] {
    const sample = world.resource(QuerySample);
    sample.scene = "hull";
    let seed = 42;
    const random = () => {
        seed ^= seed << 13;
        seed ^= seed >>> 17;
        seed ^= seed << 5;
        seed >>>= 0;
        return f(f(2 * f((seed % 32768) / 32767)) - 1);
    };
    sample.points = Array.from({ length: 100 }, () => ({ x: random(), y: random(), z: random() }));
    sample.hull = createHull(sample.points, 16, world)!;
    sample.clone = cloneAndTransformHull(sample.hull, identity, scale, world)!;
    const geometry = world.resource(Hulls).add({
        name: "Hull",
        points: new Float32Array(sample.points.flatMap((p) => [p.x, p.y, p.z])),
        maxVertexCount: 16,
    });
    return [-2, 2].map((x) => {
        const eid = world.create();
        world.add(eid, Transform, { translation: [x, 0, 0] });
        world.add(eid, Body);
        world.add(eid, Shape, { geometry, scale: [x < 0 ? 1 : -1, 1, 1] });
        return eid;
    });
}
/** One release Render's grid, including misses; the observer owns any retained records. */
export function castHeightField(
    world: World,
    observe?: (hit: PhysicsQueryHit, found: boolean) => void,
): void {
    const sample = world.resource(QuerySample);
    const translation = { x: 80000, y: -80000, z: 8 };
    const proxy = { points: [{ x: 0, y: 0, z: 0 }], count: 1, radius: sample.radius };
    const result = emptyQueryHit();
    const empty = emptyQueryHit();
    const origin = { x: 0, y: 2, z: 0 };
    let hit = false;
    const closest = (value: PhysicsQueryHit) => {
        hit = true;
        Object.assign(result, value);
        return value.fraction;
    };
    for (let x = -f(f(0.94) * 25); x <= f(f(0.94) * 25); x = f(x + f(0.4)))
        for (let z = -f(f(0.96) * 25); z <= f(f(0.96) * 25); z = f(z + f(0.4))) {
            Object.assign(result, empty);
            origin.x = x;
            origin.z = z;
            hit = false;
            if (sample.radius === 0) hit = castRayClosest(world, origin, translation, result);
            else castShape(world, origin, proxy, translation, closest);
            observe?.(result, hit);
        }
}
export function buildHulls(world: World, observe?: (hull: HullData) => void): void {
    createHulls(world, observe);
    cloneHulls(world, observe);
}
function createHulls(world: World, observe?: (hull: HullData) => void): void {
    const sample = world.resource(QuerySample);
    for (let i = 0; i < 2000; i++) {
        const built = createHull(sample.points, 16, world)!;
        observe?.(built);
    }
}
function cloneHulls(world: World, observe?: (hull: HullData) => void): void {
    const sample = world.resource(QuerySample);
    for (let i = 0; i < 2000; i++) {
        const built = cloneAndTransformHull(sample.hull!, identity, scale, world)!;
        observe?.(built);
    }
}

export const QuerySamplePlugin: Plugin = {
    name: "QuerySample",
    dependencies: [StandardPhysicsPlugin],
    systems: [
        {
            name: "hull-create-trials",
            group: "fixed",
            update(world) {
                if (world.resource(QuerySample).scene === "hull") createHulls(world);
            },
        },
        {
            name: "hull-clone-trials",
            group: "fixed",
            update(world) {
                if (world.resource(QuerySample).scene === "hull") cloneHulls(world);
            },
        },
    ],
    recovery: (world) => ({
        snapshot: () => structuredClone(world.resource(QuerySample)),
        restore: (state: ReturnType<typeof QuerySample.create>) =>
            Object.assign(world.resource(QuerySample), structuredClone(state)),
    }),
};
export const queryScenes = {
    heightField: {
        construct: heightField,
        gridExtent: 30,
        gridSpacing: 5,
        camera: {
            yaw: 0,
            pitch: (Math.atan2(20, 50) * 180) / Math.PI,
            distance: Math.hypot(20, 50),
            target: [0, 0, 0],
        },
        controls: undefined,
    },
    hull: {
        construct: hull,
        gridExtent: 5,
        gridSpacing: 1,
        camera: {
            yaw: 0,
            pitch: (Math.atan2(15, 5) * 180) / Math.PI,
            distance: Math.hypot(15, 5),
            target: [0, 0, 0],
        },
        controls: undefined,
    },
};

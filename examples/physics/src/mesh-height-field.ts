import {
    Body,
    HeightFields,
    type Plugin,
    recoverResource,
    Shape,
    ShapeKind,
    Transform,
    type World,
} from "@dylanebert/shallot";
import {
    castRayClosest,
    castShapeClosest,
    emptyQueryHit,
    StandardPhysicsPlugin,
    StepPhysicsSystem,
} from "@dylanebert/shallot/standard/physics";
import type { Controls } from "./scenes";

const f = Math.fround;
const initial = {
    columns: 10,
    rows: 10,
    amplitude: 0.75,
    holes: 0,
    "ray x": 5.5,
    "ray z": 1.01,
    "delta x": 0,
    "delta z": 0,
    radius: 0.2,
};
export const HeightFieldState = {
    create: () => ({
        active: false,
        ground: 0,
        geometry: 0,
        values: { ...initial },
        hit: false,
        result: emptyQueryHit(),
    }),
};
export const HeightFieldSamplePlugin: Plugin = {
    name: "HeightFieldSample",
    dependencies: [StandardPhysicsPlugin],
    recovery: recoverResource(HeightFieldState),
    systems: [
        {
            name: "height-field-cast",
            group: "fixed",
            after: [StepPhysicsSystem],
            update(world) {
                const s = world.resource(HeightFieldState);
                if (!s.active) return;
                const v = s.values;
                const origin = { x: v["ray x"], y: 4, z: v["ray z"] };
                const translation = { x: v["delta x"], y: -8, z: v["delta z"] };
                s.hit =
                    v.radius === 0
                        ? castRayClosest(world, origin, translation, s.result)
                        : castShapeClosest(
                              world,
                              { x: 0, y: 0, z: 0 },
                              { points: [origin], count: 1, radius: v.radius },
                              translation,
                              s.result,
                          );
            },
        },
    ],
};
function ground(world: World): void {
    const s = world.resource(HeightFieldState),
        v = s.values;
    if (s.ground) {
        world.destroy(s.ground);
        world.resource(HeightFields).remove(s.geometry);
    }
    const scale: [number, number, number] = [2, f(2 * v.amplitude), 2];
    s.geometry = world.resource(HeightFields).add(
        v.amplitude === 0
            ? {
                  name: "grid",
                  countX: v.columns,
                  countZ: v.rows,
                  scale,
                  heights: new Float32Array(v.rows * v.columns),
                  materialIndices: Uint8Array.from(
                      { length: (v.rows - 1) * (v.columns - 1) },
                      (_, k) => (v.holes && k > 0 && k % 16 === 0 ? 255 : 0),
                  ),
                  globalMinimumHeight: -256,
                  globalMaximumHeight: 256,
              }
            : {
                  name: "wave",
                  wave: {
                      rowCount: v.rows,
                      columnCount: v.columns,
                      scale,
                      rowFrequency: 0.1,
                      columnFrequency: 0.03333,
                      makeHoles: !!v.holes,
                  },
              },
    );
    s.ground = world.create();
    world.add(s.ground, Transform, { translation: [-(v.columns - 1), 0, -(v.rows - 1)] });
    world.add(s.ground, Body);
    world.add(s.ground, Shape, { kind: ShapeKind.HeightField, geometry: s.geometry });
}
const controls: Controls = Object.fromEntries(
    Object.entries(initial).map(([label, value]) => {
        const dimension = label === "columns" || label === "rows";
        const ray = label.startsWith("ray "),
            delta = label.startsWith("delta ");
        const axis = label.endsWith("x") ? "columns" : "rows";
        const bounds = (values: Record<string, number>): [number, number] => {
            const extent = values[axis]! - 1;
            const limit = ray ? 2 * values.radius! + extent : 4 * extent;
            return [-limit, limit];
        };
        const range = dimension
            ? [2, 500]
            : label === "amplitude"
              ? [0, 2]
              : label === "radius"
                ? [0, 1]
                : ray || delta
                  ? bounds(initial)
                  : [];
        return [
            label,
            {
                label,
                value,
                min: range[0],
                max: range[1],
                step: dimension || delta ? 1 : ray ? 0.1 : 0.001,
                decimals: dimension || delta ? 0 : ray ? 1 : 3,
                ...(ray || delta ? { bounds } : {}),
                set(world: World, _bodies: readonly number[], value: number) {
                    world.resource(HeightFieldState).values[label as keyof typeof initial] =
                        f(value);
                    if (dimension || label === "amplitude" || label === "holes") ground(world);
                },
            },
        ];
    }),
);
export const heightFieldScene = {
    construct(world: World): number[] {
        const s = world.resource(HeightFieldState);
        Object.assign(s, HeightFieldState.create(), { active: true });
        ground(world);
        return [];
    },
    controls,
    gridExtent: 9,
    gridSpacing: 2,
    camera: { yaw: 35, pitch: 35, distance: 32, phoneDistance: 53, target: [0, 0, 0] },
};

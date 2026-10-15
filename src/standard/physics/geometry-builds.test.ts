import { expect, test } from "bun:test";
import { type Registry, type Resource, Time, World } from "@dylanebert/shallot";
import {
    Body,
    type HeightField,
    HeightFields,
    type PhysicsMesh,
    PhysicsMeshes,
    PhysicsPlugin,
    Shape,
    ShapeKind,
} from "@dylanebert/shallot/physics";
import { hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { expectWarning } from "../../../scripts/test-console";
import { heightFieldBuild, meshBuild } from "./geometry-builds";
import { kernel } from "./kernel/kernel";
import { physicsState } from "./runtime";
import { createHeightField } from "./shapes/heightfield";
import { createMesh } from "./shapes/mesh";
import { getShapeGeometryReference } from "./shapes/shape";
import { worldCounters } from "./world/world";

async function withPhysics(run: (world: World) => void): Promise<void> {
    const world = new World();
    for (const component of PhysicsPlugin.components!)
        world.registry.register(component, PhysicsPlugin.name);
    for (const plugin of [PhysicsPlugin, StandardPhysicsPlugin]) {
        const recovery = plugin.recovery;
        if (typeof recovery !== "function") throw new Error(`${plugin.name} must recover`);
        world.registerRecovery(plugin.name, recovery(world));
    }
    await StandardPhysicsPlugin.initialize!(world);
    await StandardPhysicsPlugin.warm!(world);
    for (const system of StandardPhysicsPlugin.systems!)
        world.addSystem(system, StandardPhysicsPlugin.name);
    try {
        run(world);
    } finally {
        await StandardPhysicsPlugin.dispose!(world);
        world.dispose();
    }
}

function state(world: World) {
    const physics = physicsState(world);
    if (!physics) throw new Error("physics state is unavailable");
    return physics;
}

function databaseCount(world: World): number {
    return kernel(world).geometryDatabaseCount(state(world).worldId);
}

function geometryShape(world: World, kind: ShapeKind, geometry: number): number {
    const eid = world.create();
    world.add(eid, Body);
    world.add(eid, Shape, { kind, geometry, scale: [1, 1, 1] });
    return eid;
}

function geometryReferences(world: World): number[] {
    const physics = state(world);
    const count = worldCounters(physics).shapeCount;
    return Array.from({ length: count }, (_, i) => getShapeGeometryReference(physics, i));
}

const quad = (name: string, size = 1): PhysicsMesh => ({
    name,
    vertices: new Float32Array([0, 0, 0, size, 0, 0, size, 0, size, 0, 0, size]),
    indices: new Uint32Array([0, 2, 1, 0, 3, 2]),
});

const field = (name: string, size = 1): HeightField => ({
    name,
    heights: new Float32Array([0, 0.25, 0.5, 1]),
    scale: [size, 1, 1],
    countX: 2,
    countZ: 2,
    globalMinimumHeight: 0,
    globalMaximumHeight: 1,
});

interface Geometry<Def extends { name: string }> {
    noun: string;
    plural: string;
    registry: Resource<Registry<Def>>;
    registryName: string;
    kind: ShapeKind;
    /** A definition `size` meters along x. */
    define(name: string, size?: number): Def;
    lookup(world: World, id: number): object | undefined;
    /** Its build's upper x bound. */
    upperX(world: World, id: number): number;
}

const mesh: Geometry<PhysicsMesh> = {
    noun: "mesh",
    plural: "meshes",
    registry: PhysicsMeshes,
    registryName: "PhysicsMeshes",
    kind: ShapeKind.Mesh,
    define: quad,
    lookup: meshBuild,
    upperX: (world, id) => meshBuild(world, id)!.bounds.upperBound.x,
};

const heightField: Geometry<HeightField> = {
    noun: "height field",
    plural: "height fields",
    registry: HeightFields,
    registryName: "HeightFields",
    kind: ShapeKind.HeightField,
    define: field,
    lookup: heightFieldBuild,
    upperX: (world, id) => heightFieldBuild(world, id)!.aabb.upperBound.x,
};

// A bumpy n-by-n grid of quads, two triangles each, so every build option changes its MeshData.
function grid(n: number): { vertices: Float32Array; indices: Uint32Array } {
    const vertices = new Float32Array(3 * (n + 1) * (n + 1));
    for (let x = 0; x <= n; x++)
        for (let z = 0; z <= n; z++)
            vertices.set(
                [x + 0.3 * (z % 2), 0.25 * ((x * 7 + z * 3) % 5), z * 1.5],
                3 * (x * (n + 1) + z),
            );
    const indices = new Uint32Array(6 * n * n);
    for (let x = 0, o = 0; x < n; x++)
        for (let z = 0; z < n; z++, o += 6) {
            const a = x * (n + 1) + z;
            indices.set([a, a + 1, a + n + 2, a + n + 2, a + n + 1, a], o);
        }
    return { vertices, indices };
}

function buildTests<Def extends { name: string }>(geometry: Geometry<Def>): void {
    const { noun, plural, registryName, kind } = geometry;
    const register = (world: World, name: string, size?: number) =>
        world.resource(geometry.registry).register(geometry.define(name, size));

    test(`Shapes naming one ${registryName} id share one build and one geometry database entry`, async () => {
        await withPhysics((world) => {
            const id = register(world, "shared");
            const before = databaseCount(world);
            geometryShape(world, kind, id);
            geometryShape(world, kind, id);
            world.tick();
            expect(worldCounters(state(world)).shapeCount).toBe(2);
            expect(databaseCount(world) - before).toBe(1);
            const [first, second] = geometryReferences(world);
            expect(second).toBe(first);
        });
    });

    test(`a ${noun} registered again under its name builds anew for the next Shape, and earlier Shapes keep theirs`, async () => {
        await withPhysics((world) => {
            const id = register(world, "replaced", 1);
            geometryShape(world, kind, id);
            // A marked Shape compares against the registry at each sync; the second frame's sync is its last.
            world.step(Time.FIXED_DT);
            world.step(Time.FIXED_DT);
            expect(register(world, "replaced", 2)).toBe(id);
            geometryShape(world, kind, id);
            world.tick();
            const [first, second] = geometryReferences(world);
            expect(second).not.toBe(first);
            expect(geometry.upperX(world, id)).toBe(2);
        });
    });

    test(`a restored World keeps its ${noun} builds, so a Shape spawned after restore adds no geometry and hashes alike`, async () => {
        await withPhysics((world) => {
            const id = register(world, "restored");
            geometryShape(world, kind, id);
            // Two frames settle the spawn, so the restored Shape is not compared against the registry again.
            world.step(Time.FIXED_DT);
            world.step(Time.FIXED_DT);
            const saved = world.snapshot();
            geometryShape(world, kind, id);
            world.tick();
            const count = databaseCount(world);
            const hash = hashPhysics(world);
            world.restore(saved);
            geometryShape(world, kind, id);
            world.tick();
            expect(worldCounters(state(world)).shapeCount).toBe(2);
            expect(databaseCount(world)).toBe(count);
            expect(hashPhysics(world)).toBe(hash);
        });
    });

    test(`deleted ${plural} are collectable with their builds, with no later lookup or snapshot`, async () => {
        await withPhysics((world) => {
            const registry = world.resource(geometry.registry);
            // Cycles run in their own frame, so the conservative stack scan cannot pin their values.
            const cycle = (i: number) => {
                const def = geometry.define(`cycle-${i}`);
                const build = geometry.lookup(world, registry.register(def))!;
                registry.delete(def.name);
                return { def: new WeakRef(def), build: new WeakRef(build) };
            };
            const cycles = Array.from({ length: 100 }, (_, i) => cycle(i));
            Bun.gc(true);
            const alive = (key: "def" | "build") =>
                cycles.flatMap((refs, i) => (refs[key].deref() ? [i] : []));
            // A build lives exactly as long as its definition, and the cache keeps no definition alive;
            // outside the cache, the engine may still pin an odd deleted definition until a later GC.
            expect(alive("build")).toEqual(alive("def"));
            expect(alive("def").length).toBeLessThan(5);
        });
    });
}

buildTests(mesh);
buildTests(heightField);

test("each PhysicsMesh build option reaches Box3D's mesh build", async () => {
    await withPhysics((world) => {
        const { vertices, indices } = grid(4);
        const points = Array.from({ length: vertices.length / 3 }, (_, i) => ({
            x: vertices[3 * i],
            y: vertices[3 * i + 1],
            z: vertices[3 * i + 2],
        }));
        const meshes = world.resource(PhysicsMeshes);
        const built = (def: Omit<PhysicsMesh, "name">) =>
            meshBuild(world, meshes.register({ name: JSON.stringify(def), ...def }))!;
        const plain = built({ vertices, indices });
        expect(plain).toEqual(createMesh({ vertices: points, indices })!);

        const interleaved = new Float32Array(2 * vertices.length);
        for (let i = 0; i < points.length; i++)
            interleaved.set([...vertices.subarray(3 * i, 3 * i + 3), 9, 9, 9], 6 * i);
        expect(built({ vertices: interleaved, stride: 6, indices })).toEqual(plain);

        const reversed = indices.slice();
        for (let i = 0; i < reversed.length; i += 3)
            [reversed[i + 1], reversed[i + 2]] = [reversed[i + 2], reversed[i + 1]];
        const welded = new Float32Array([...vertices, ...vertices.subarray(0, 3)]);
        welded[welded.length - 1] += 0.001;
        const weldedIndices = indices.slice();
        weldedIndices[0] = points.length;
        const materialIndices = Uint8Array.from({ length: indices.length / 3 }, (_, i) => i % 3);
        const options: [string, Omit<PhysicsMesh, "name">, Parameters<typeof createMesh>[0]][] = [
            [
                "clockwiseWinding",
                { vertices, indices: reversed, clockwiseWinding: true },
                { vertices: points, indices: reversed, clockwiseWinding: true },
            ],
            [
                "identifyEdges",
                { vertices, indices, identifyEdges: true },
                { vertices: points, indices, identifyEdges: true },
            ],
            [
                "useMedianSplit",
                { vertices, indices, useMedianSplit: true },
                { vertices: points, indices, useMedianSplit: true },
            ],
            [
                "materialIndices",
                { vertices, indices, materialIndices },
                { vertices: points, indices, materialIndices },
            ],
            [
                "weldVertices",
                {
                    vertices: welded,
                    indices: weldedIndices,
                    weldVertices: true,
                    weldTolerance: 0.01,
                },
                {
                    vertices: [...points, { ...points[0], z: points[0].z + 0.001 }],
                    indices: weldedIndices,
                    weldVertices: true,
                    weldTolerance: 0.01,
                },
            ],
        ];
        for (const [option, def, expected] of options) {
            const mesh = built(def);
            expect({ option, mesh }).toEqual({ option, mesh: createMesh(expected)! });
            expect({
                option,
                differs: !Bun.deepEquals(mesh, built({ ...def, [option]: undefined })),
            }).toEqual({ option, differs: true });
        }
    });
});

test("a PhysicsMesh Box3D cannot build leaves its Shape unattached with a warning naming it", async () => {
    const triangle = new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 1]);
    const cases: [Omit<PhysicsMesh, "name">, RegExp][] = [
        [
            { vertices: triangle, indices: new Uint32Array([0, 2, 1, 0]) },
            /has 4 indices, not a multiple of 3/,
        ],
        [
            { vertices: triangle, indices: new Uint32Array([0, 2, 3]) },
            /index 2 is 3, past its 3 vertices/,
        ],
        [
            {
                vertices: triangle,
                indices: new Uint32Array([0, 2, 1]),
                materialIndices: new Uint8Array(2),
            },
            /has 2 material indices for 1 triangles/,
        ],
        [
            { vertices: triangle.subarray(0, 8), indices: new Uint32Array([0, 2, 1]) },
            /has 8 vertex floats, not whole vertices at stride 3/,
        ],
        [
            { vertices: triangle, stride: 2, indices: new Uint32Array([0, 2, 1]) },
            /has stride 2, not 0 or an integer of at least 3/,
        ],
        [
            { vertices: new Float32Array(9), indices: new Uint32Array([0, 2, 1]) },
            /could not be built/,
        ],
    ];
    for (const [i, [, message]] of cases.entries())
        expectWarning(
            new RegExp(`Shape ${i + 2} was not attached:.*mesh "bad-${i}" ${message.source}`),
        );
    await withPhysics((world) => {
        const meshes = world.resource(PhysicsMeshes);
        const body = world.create();
        world.add(body, Body);
        for (const [i, [def]] of cases.entries()) {
            const eid = world.create();
            world.add(eid, Shape, {
                body,
                kind: ShapeKind.Mesh,
                geometry: meshes.register({ name: `bad-${i}`, ...def }),
            });
        }
        world.tick();
        expect(worldCounters(state(world)).bodyCount).toBe(1);
        expect(worldCounters(state(world)).shapeCount).toBe(0);
    });
});

test("each HeightField field reaches Box3D's height-field build", async () => {
    await withPhysics((world) => {
        const fields = world.resource(HeightFields);
        const built = (def: Omit<HeightField, "name">) =>
            heightFieldBuild(world, fields.register({ name: JSON.stringify(def), ...def }))!;
        // Four points along x by three along z, so swapping the counts reorders the grid.
        const heights = new Float32Array([0, 0.5, 0.2, 0.8, 0.7, 1, 0.3, 0.6, 0.1, 0.4, 0.9, 0.35]);
        const base: Omit<HeightField, "name"> = {
            heights,
            scale: [1, 2, 3],
            countX: 4,
            countZ: 3,
            globalMinimumHeight: 0,
            globalMaximumHeight: 1,
        };
        const expected: Parameters<typeof createHeightField>[0] = {
            heights,
            materialIndices: null,
            scale: { x: 1, y: 2, z: 3 },
            countX: 4,
            countZ: 3,
            globalMinimumHeight: 0,
            globalMaximumHeight: 1,
            clockwiseWinding: false,
        };
        const plain = built(base);
        expect(plain).toEqual(createHeightField(expected));

        const materialIndices = new Uint8Array([0, 1, 255, 2, 0, 1]);
        const other = heights.map((h) => 1 - h);
        const variants: [string, Partial<HeightField>, Partial<typeof expected>][] = [
            ["heights", { heights: other }, { heights: other }],
            ["materialIndices", { materialIndices }, { materialIndices }],
            ["scale", { scale: [2, 1, 0.5] }, { scale: { x: 2, y: 1, z: 0.5 } }],
            ["counts", { countX: 3, countZ: 4 }, { countX: 3, countZ: 4 }],
            ["globalMinimumHeight", { globalMinimumHeight: -1 }, { globalMinimumHeight: -1 }],
            ["globalMaximumHeight", { globalMaximumHeight: 2 }, { globalMaximumHeight: 2 }],
            ["clockwiseWinding", { clockwiseWinding: true }, { clockwiseWinding: true }],
        ];
        for (const [option, change, expectedChange] of variants) {
            const field = built({ ...base, ...change });
            expect({ option, field }).toEqual({
                option,
                field: createHeightField({ ...expected, ...expectedChange }),
            });
            expect({ option, differs: !Bun.deepEquals(field, plain) }).toEqual({
                option,
                differs: true,
            });
        }
    });
});

test("a HeightField Box3D cannot build leaves its Shape unattached with a warning naming it", async () => {
    const square = {
        heights: new Float32Array(4),
        scale: [1, 1, 1] as [number, number, number],
        countX: 2,
        countZ: 2,
        globalMinimumHeight: 0,
        globalMaximumHeight: 1,
    };
    const cases: [Omit<HeightField, "name">, RegExp][] = [
        [
            { ...square, heights: new Float32Array(2), countX: 1 },
            /is 1 by 2 points, not integers of at least 2/,
        ],
        [
            { ...square, heights: new Float32Array(5), countX: 2.5 },
            /is 2.5 by 2 points, not integers of at least 2/,
        ],
        [{ ...square, heights: new Float32Array(3) }, /has 3 heights for 4 points/],
        [{ ...square, materialIndices: new Uint8Array(2) }, /has 2 material indices for 1 cells/],
        [{ ...square, scale: [1, 0, 1] }, /has scale \[1, 0, 1\], not positive/],
        [
            { ...square, globalMinimumHeight: 1, globalMaximumHeight: 0 },
            /has globalMinimumHeight 1 above globalMaximumHeight 0/,
        ],
    ];
    for (const [i, [, message]] of cases.entries())
        expectWarning(
            new RegExp(
                `Shape ${i + 2} was not attached:.*height field "bad-${i}" ${message.source}`,
            ),
        );
    await withPhysics((world) => {
        const fields = world.resource(HeightFields);
        const body = world.create();
        world.add(body, Body);
        for (const [i, [def]] of cases.entries()) {
            const eid = world.create();
            world.add(eid, Shape, {
                body,
                kind: ShapeKind.HeightField,
                geometry: fields.register({ name: `bad-${i}`, ...def }),
            });
        }
        world.tick();
        expect(worldCounters(state(world)).bodyCount).toBe(1);
        expect(worldCounters(state(world)).shapeCount).toBe(0);
    });
});

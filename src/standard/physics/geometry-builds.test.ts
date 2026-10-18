import { expect, test } from "bun:test";
import { type Resource, Time, Transform, TransformPlugin, World } from "@dylanebert/shallot";
import {
    Body,
    type Compound,
    Compounds,
    type HeightField,
    HeightFields,
    Hulls,
    type PhysicsAssets,
    type PhysicsMesh,
    PhysicsMeshes,
    PhysicsPlugin,
    Shape,
    ShapeKind,
    type SurfaceMaterial,
    UNIT_CUBE_ID,
} from "@dylanebert/shallot/physics";
import {
    castRayClosest,
    emptyQueryHit,
    hashPhysics,
    StandardPhysicsPlugin,
} from "@dylanebert/shallot/standard/physics";
import { expectWarning } from "../../../scripts/test-console";
import { defaultSurfaceMaterial } from "./common/types";
import { compoundBuild, heightFieldBuild, meshBuild } from "./geometry-builds";
import { kernel } from "./kernel/kernel";
import { physicsState } from "./runtime";
import { createHull } from "./shapes/build-hull";
import { type CompoundData, type CompoundDef, createCompound } from "./shapes/compound";
import { createHeightField } from "./shapes/heightfield";
import { createMesh } from "./shapes/mesh";
import { getShapeGeometryReference } from "./shapes/shape";
import { worldCounters } from "./world/world";

async function withPhysics(run: (world: World) => void): Promise<void> {
    const world = new World();
    for (const component of [...TransformPlugin.components!, ...PhysicsPlugin.components!])
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
    registry: Resource<PhysicsAssets<Def>>;
    registryName: string;
    kind: ShapeKind;
    /** A definition `size` meters along x, registering any geometry it names in `world`. */
    define(world: World, name: string, size?: number): Def;
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
    define: (_, name, size) => quad(name, size),
    lookup: meshBuild,
    upperX: (world, id) => meshBuild(world, id)!.bounds.upperBound.x,
};

// A sphere reaching `size` along x, beside the unit cube and a quad mesh registered under its name,
// so a compound's build depends on both child registries.
const compound: Geometry<Compound> = {
    noun: "compound",
    plural: "compounds",
    registry: Compounds,
    registryName: "Compounds",
    kind: ShapeKind.Compound,
    define: (world, name, size = 1) => ({
        name,
        hulls: [
            { hull: UNIT_CUBE_ID, transform: { position: [-2, 0, 0], rotation: [0, 0, 0, 1] } },
        ],
        meshes: [
            {
                mesh: world.resource(PhysicsMeshes).add(quad(`${name}-child`)),
                transform: { position: [-5, 0, 0], rotation: [0, 0, 0, 1] },
                scale: [1, 1, 1],
            },
        ],
        spheres: [{ center: [size - 0.5, 0, 0], radius: 0.5 }],
    }),
    lookup: compoundBuild,
    upperX: (world, id) => {
        const { sphere } = compoundBuild(world, id)!.spheres[0];
        return sphere.center.x + sphere.radius;
    },
};

const heightField: Geometry<HeightField> = {
    noun: "height field",
    plural: "height fields",
    registry: HeightFields,
    registryName: "HeightFields",
    kind: ShapeKind.HeightField,
    define: (_, name, size) => field(name, size),
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
    const add = (world: World, name: string, size?: number) =>
        world.resource(geometry.registry).add(geometry.define(world, name, size));

    test(`Shapes naming one ${registryName} handle share one build and one geometry database entry`, async () => {
        await withPhysics((world) => {
            const id = add(world, "shared");
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

    test(`two ${plural} with one name have distinct handles and each Shape keeps its own geometry`, async () => {
        await withPhysics((world) => {
            const id = add(world, "replaced", 1);
            geometryShape(world, kind, id);
            // A marked Shape compares against the registry at each sync; the second frame's sync is its last.
            world.step(Time.FIXED_DT);
            world.step(Time.FIXED_DT);
            const secondId = add(world, "replaced", 2);
            expect(secondId).not.toBe(id);
            geometryShape(world, kind, secondId);
            world.tick();
            const [first, second] = geometryReferences(world);
            expect(second).not.toBe(first);
            expect(geometry.upperX(world, id)).toBe(1);
            expect(geometry.upperX(world, secondId)).toBe(2);
            const saved = world.snapshot();
            const count = databaseCount(world);
            world.resource(geometry.registry).clear();
            world.restore(saved);
            expect(geometry.upperX(world, id)).toBe(1);
            expect(geometry.upperX(world, secondId)).toBe(2);
            geometryShape(world, kind, id);
            geometryShape(world, kind, secondId);
            world.tick();
            const references = geometryReferences(world);
            expect(references[2]).toBe(references[0]);
            expect(references[3]).toBe(references[1]);
            expect(databaseCount(world)).toBe(count);
        });
    });

    test(`a restored World keeps its ${noun} builds, so a Shape spawned after restore adds no geometry and hashes alike`, async () => {
        await withPhysics((world) => {
            const id = add(world, "restored");
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
            // Cycles run in their own frame, so the conservative stack scan cannot pin their values; a
            // last, unmeasured cycle overwrites the stack slots the final measured one left behind.
            const cycle = (i: number) => {
                const def = geometry.define(world, `cycle-${i}`);
                const handle = registry.add(def);
                const build = geometry.lookup(world, handle)!;
                registry.remove(handle);
                return { def: new WeakRef(def), build: new WeakRef(build) };
            };
            const cycles = Array.from({ length: 101 }, (_, i) => cycle(i)).slice(0, 100);
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
buildTests(compound);

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
            meshBuild(world, meshes.add({ name: JSON.stringify(def), ...def }))!;
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
                geometry: meshes.add({ name: `bad-${i}`, ...def }),
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
            heightFieldBuild(world, fields.add({ name: JSON.stringify(def), ...def }))!;
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
                geometry: fields.add({ name: `bad-${i}`, ...def }),
            });
        }
        world.tick();
        expect(worldCounters(state(world)).bodyCount).toBe(1);
        expect(worldCounters(state(world)).shapeCount).toBe(0);
    });
});

// A tetrahedron `size` meters along each axis.
const tetrahedron = (size: number) =>
    new Float32Array([0, 0, 0, size, 0, 0, 0, size, 0, 0, 0, size]);

test("a compound keeps its child handles when same-name hulls and meshes are added", async () => {
    await withPhysics((world) => {
        const hulls = world.resource(Hulls);
        const meshes = world.resource(PhysicsMeshes);
        const hull = hulls.add({ name: "child-hull", points: tetrahedron(1) });
        const mesh = meshes.add(quad("child-mesh"));
        const id = world.resource(Compounds).add({
            name: "parent",
            hulls: [{ hull, transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1] } }],
            meshes: [
                {
                    mesh,
                    transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1] },
                    scale: [1, 1, 1],
                },
            ],
        });
        const children = () => {
            const built = compoundBuild(world, id)!;
            return [built.hulls[0].hull, built.meshes[0].meshData] as const;
        };
        const spawn = () => {
            geometryShape(world, ShapeKind.Compound, id);
            // A marked Shape compares against the registry at each sync; the second frame's sync is its last.
            world.step(Time.FIXED_DT);
            world.step(Time.FIXED_DT);
        };
        spawn();
        const [firstHull, firstMesh] = children();
        expect(firstHull.aabb.upperBound.x).toBe(1);

        hulls.add({ name: "child-hull", points: tetrahedron(2) });
        spawn();
        const [secondHull, secondMesh] = children();
        expect(secondHull).toBe(firstHull);
        expect(secondMesh).toBe(firstMesh);

        meshes.add(quad("child-mesh", 2));
        spawn();
        const [thirdHull, thirdMesh] = children();
        expect(thirdHull).toBe(secondHull);
        expect(thirdMesh).toBe(firstMesh);

        const [first, second, third] = geometryReferences(world);
        expect(new Set([first, second, third]).size).toBe(1);
    });
});

test("each Compound field reaches Box3D's compound build", async () => {
    await withPhysics((world) => {
        const points = tetrahedron(1);
        const hull = world.resource(Hulls).add({ name: "fields-hull", points });
        // Two material indices, so the mesh child takes two materials.
        const twoMaterials = { ...quad("fields-mesh"), materialIndices: new Uint8Array([0, 1]) };
        const mesh = world.resource(PhysicsMeshes).add(twoMaterials);
        const compounds = world.resource(Compounds);
        let count = 0;
        // The tree's spare node floats are NaN, which typed-array equality never matches; arrays compare them alike.
        const comparable = (data: CompoundData) => ({
            ...data,
            tree: { ...data.tree, nf: Array.from(data.tree.nf) },
        });
        const built = (def: Omit<Compound, "name">) =>
            comparable(compoundBuild(world, compounds.add({ name: `fields-${count++}`, ...def }))!);
        const materialOf = (userMaterialId: bigint): SurfaceMaterial => ({
            ...defaultSurfaceMaterial(),
            userMaterialId,
        });
        const [a, b, c, d, e] = [1n, 2n, 3n, 4n, 5n].map(materialOf);
        const base: Omit<Compound, "name"> = {
            capsules: [{ center1: [0, -1, 0], center2: [0, 1, 0], radius: 0.25, material: a }],
            hulls: [
                {
                    hull,
                    transform: { position: [1, 2, 3], rotation: [0, 0, 0.6, 0.8] },
                    material: b,
                },
            ],
            meshes: [
                {
                    mesh,
                    transform: { position: [-1, 0, 0], rotation: [0.6, 0, 0, 0.8] },
                    scale: [1, 2, -1],
                    materials: [c, d],
                },
            ],
            spheres: [{ center: [0, 0, 2], radius: 0.5, material: e }],
        };
        const hullData = createHull(
            Array.from({ length: 4 }, (_, i) => ({
                x: points[3 * i],
                y: points[3 * i + 1],
                z: points[3 * i + 2],
            })),
            128,
        )!;
        const meshData = createMesh({
            vertices: [
                { x: 0, y: 0, z: 0 },
                { x: 1, y: 0, z: 0 },
                { x: 1, y: 0, z: 1 },
                { x: 0, y: 0, z: 1 },
            ],
            indices: twoMaterials.indices,
            materialIndices: twoMaterials.materialIndices,
        })!;
        const expected: CompoundDef = {
            capsules: [
                {
                    capsule: {
                        center1: { x: 0, y: -1, z: 0 },
                        center2: { x: 0, y: 1, z: 0 },
                        radius: 0.25,
                    },
                    material: a,
                },
            ],
            hulls: [
                {
                    hull: hullData,
                    transform: {
                        p: { x: 1, y: 2, z: 3 },
                        q: { v: { x: 0, y: 0, z: 0.6 }, s: 0.8 },
                    },
                    material: b,
                },
            ],
            meshes: [
                {
                    meshData,
                    transform: {
                        p: { x: -1, y: 0, z: 0 },
                        q: { v: { x: 0.6, y: 0, z: 0 }, s: 0.8 },
                    },
                    scale: { x: 1, y: 2, z: -1 },
                    materials: [c, d],
                    materialCount: 2,
                },
            ],
            spheres: [{ sphere: { center: { x: 0, y: 0, z: 2 }, radius: 0.5 }, material: e }],
        };
        const plain = built(base);
        expect(plain).toEqual(comparable(createCompound(expected)!));

        const other = materialOf(9n);
        const identity = { position: [0, 0, 0], rotation: [0, 0, 0, 1] } as const;
        const child = <K extends "capsules" | "hulls" | "meshes" | "spheres">(
            key: K,
            change: object,
        ) => ({ [key]: [{ ...base[key]![0], ...change }] }) as Partial<Compound>;
        const variants: [string, Partial<Compound>][] = [
            ["capsule center1", child("capsules", { center1: [0, -2, 0] })],
            ["capsule center2", child("capsules", { center2: [0, 2, 0] })],
            ["capsule radius", child("capsules", { radius: 0.5 })],
            ["capsule material", child("capsules", { material: other })],
            ["capsule default material", child("capsules", { material: undefined })],
            [
                "hull position",
                child("hulls", { transform: { ...identity, rotation: [0, 0, 0.6, 0.8] } }),
            ],
            ["hull rotation", child("hulls", { transform: { ...identity, position: [1, 2, 3] } })],
            ["hull material", child("hulls", { material: other })],
            [
                "mesh position",
                child("meshes", { transform: { ...identity, rotation: [0.6, 0, 0, 0.8] } }),
            ],
            [
                "mesh rotation",
                child("meshes", { transform: { ...identity, position: [-1, 0, 0] } }),
            ],
            ["mesh scale", child("meshes", { scale: [1, 1, 1] })],
            ["mesh materials", child("meshes", { materials: [d, c] })],
            ["sphere center", child("spheres", { center: [0, 0, 3] })],
            ["sphere radius", child("spheres", { radius: 0.75 })],
            ["sphere material", child("spheres", { material: other })],
        ];
        for (const [option, change] of variants)
            expect({
                option,
                differs: !Bun.deepEquals(built({ ...base, ...change }), plain),
            }).toEqual({
                option,
                differs: true,
            });
        const defaults = built({ ...base, ...child("capsules", { material: undefined }) });
        expect(defaults.materials[defaults.capsules[0].materialIndex]).toEqual(
            defaultSurfaceMaterial(),
        );
    });
});

test("a Compound Box3D cannot build leaves its Shape unattached with a warning naming it", async () => {
    const at = {
        position: [0, 0, 0] as [number, number, number],
        rotation: [0, 0, 0, 1] as [number, number, number, number],
    };
    await withPhysics((world) => {
        const ragged = world.resource(Hulls).add({ name: "ragged", points: new Float32Array(7) });
        const mesh = world.resource(PhysicsMeshes).add(quad("one-material"));
        const six = world.resource(PhysicsMeshes).add({
            ...quad("six-materials"),
            materialIndices: new Uint8Array([0, 5]),
        });
        const material = defaultSurfaceMaterial();
        const meshChild = (id: number, count: number) => ({
            mesh: id,
            transform: at,
            scale: [1, 1, 1] as [number, number, number],
            materials: Array.from({ length: count }, () => material),
        });
        const cases: [Omit<Compound, "name">, RegExp][] = [
            [
                { hulls: [{ hull: 99, transform: at }] },
                /compound "bad-0" hull 0 names missing hull 99/,
            ],
            [{ meshes: [meshChild(99, 1)] }, /compound "bad-1" mesh 0 names missing mesh 99/],
            [
                { meshes: [meshChild(mesh, 2)] },
                /compound "bad-2" mesh 0 has 2 materials for its mesh's 1, at most 4/,
            ],
            [
                { meshes: [meshChild(mesh, 0)] },
                /compound "bad-3" mesh 0 has 0 materials for its mesh's 1, at most 4/,
            ],
            [
                { meshes: [meshChild(six, 6)] },
                /compound "bad-4" mesh 0 has 6 materials for its mesh's 6, at most 4/,
            ],
            [
                { hulls: [{ hull: ragged, transform: at }] },
                /hull "ragged" has 7 point coordinates, not a multiple of 3/,
            ],
        ];
        for (const [i, [, message]] of cases.entries())
            expectWarning(new RegExp(`Shape ${i + 2} was not attached:.*${message.source}`));
        const compounds = world.resource(Compounds);
        const body = world.create();
        world.add(body, Body);
        for (const [i, [def]] of cases.entries()) {
            const eid = world.create();
            world.add(eid, Shape, {
                body,
                kind: ShapeKind.Compound,
                geometry: compounds.add({ name: `bad-${i}`, ...def }),
            });
        }
        world.tick();
        expect(worldCounters(state(world)).bodyCount).toBe(1);
        expect(worldCounters(state(world)).shapeCount).toBe(0);
    });
});

test("a compound mesh child flattened below B3_MIN_SCALE by body Transform.scale meets rays at its scaled height", async () => {
    await withPhysics((world) => {
        const box = world.resource(PhysicsMeshes).add({
            name: "flattened-box",
            vertices: new Float32Array([
                -1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1,
            ]),
            indices: new Uint32Array([
                3, 7, 6, 3, 6, 2, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3, 4, 5, 6, 4,
                6, 7, 0, 3, 2, 0, 2, 1,
            ]),
        });
        const id = world.resource(Compounds).add({
            name: "flattened",
            meshes: [
                {
                    mesh: box,
                    transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1] },
                    scale: [1, 1, 1],
                },
            ],
        });
        const eid = world.create();
        world.add(eid, Transform, { scale: [1, 0.005, 1] });
        world.add(eid, Body);
        world.add(eid, Shape, { kind: ShapeKind.Compound, geometry: id });
        world.step(Time.FIXED_DT);
        const hit = emptyQueryHit();
        expect(castRayClosest(world, { x: 0.3, y: 5, z: 0.2 }, { x: 0, y: -10, z: 0 }, hit)).toBe(
            true,
        );
        // Box3D keeps the child's 0.005; b3SafeScale's 0.01 floor is for mesh shapes only.
        expect(hit.point.y).toBeCloseTo(0.005, 4);
    });
});

import { expect, test } from "bun:test";
import { Time, World } from "@dylanebert/shallot";
import {
    Body,
    type PhysicsMesh,
    PhysicsMeshes,
    PhysicsPlugin,
    Shape,
    ShapeKind,
} from "@dylanebert/shallot/physics";
import { hashPhysics, StandardPhysicsPlugin } from "@dylanebert/shallot/standard/physics";
import { expectWarning } from "../../../scripts/test-console";
import { kernel } from "./kernel/kernel";
import { meshBuild } from "./mesh-builds";
import { physicsState } from "./runtime";
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

function meshShape(world: World, geometry: number): number {
    const eid = world.create();
    world.add(eid, Body);
    world.add(eid, Shape, { kind: ShapeKind.Mesh, geometry, scale: [1, 1, 1] });
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

test("Shapes naming one PhysicsMeshes id share one build and one geometry database entry", async () => {
    await withPhysics((world) => {
        const geometry = world.resource(PhysicsMeshes).register(quad("shared"));
        const before = databaseCount(world);
        meshShape(world, geometry);
        meshShape(world, geometry);
        world.tick();
        expect(worldCounters(state(world)).shapeCount).toBe(2);
        expect(databaseCount(world) - before).toBe(1);
        const [first, second] = geometryReferences(world);
        expect(second).toBe(first);
    });
});

test("a mesh registered again under its name builds anew for the next Shape, and earlier Shapes keep theirs", async () => {
    await withPhysics((world) => {
        const meshes = world.resource(PhysicsMeshes);
        const geometry = meshes.register(quad("replaced", 1));
        meshShape(world, geometry);
        // A marked Shape compares against the registry at each sync; the second frame's sync is its last.
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        expect(meshes.register(quad("replaced", 2))).toBe(geometry);
        meshShape(world, geometry);
        world.tick();
        const [first, second] = geometryReferences(world);
        expect(second).not.toBe(first);
        expect(meshBuild(world, geometry)!.bounds.upperBound.x).toBe(2);
    });
});

test("a restored World keeps its mesh builds, so a Shape spawned after restore adds no geometry and hashes alike", async () => {
    await withPhysics((world) => {
        const geometry = world.resource(PhysicsMeshes).register(quad("restored"));
        meshShape(world, geometry);
        // Two frames settle the spawn, so the restored Shape is not compared against the registry again.
        world.step(Time.FIXED_DT);
        world.step(Time.FIXED_DT);
        const saved = world.snapshot();
        meshShape(world, geometry);
        world.tick();
        const count = databaseCount(world);
        const hash = hashPhysics(world);
        world.restore(saved);
        meshShape(world, geometry);
        world.tick();
        expect(worldCounters(state(world)).shapeCount).toBe(2);
        expect(databaseCount(world)).toBe(count);
        expect(hashPhysics(world)).toBe(hash);
    });
});

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

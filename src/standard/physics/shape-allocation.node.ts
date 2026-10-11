import { beforeAll, expect, setDefaultTimeout, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import {
    allocationFailure,
    sampleAllocation,
} from "../../../diagnostics/first-person-allocation/allocation";
import { CEILING } from "../../../scripts/test-tiers";
import { BodyType, defaultSurfaceMaterial, ShapeType } from "./common/types";
import { BodyField, setBodyField } from "./kernel/bodyrecords";
import { hullDatabaseIndex } from "./kernel/geocolumns";
import { clockImport, type Kernel, kernelState } from "./kernel/kernel";
import { S_GEO_REFERENCE, SHAPE_STRIDE } from "./kernel/shapecolumns";
import { shapeAllocationSubject } from "./shape-allocation.fixture";
import { createCompound } from "./shapes/compound";
import { createGrid } from "./shapes/heightfield";
import { makeBoxHull } from "./shapes/hull";
import { createGridMesh } from "./shapes/mesh";
import {
    createCompoundShape,
    createHeightFieldShape,
    createHullShape,
    createMeshShape,
    createSphereShape,
    destroyShape,
} from "./shapes/shape";
import { createTestBody, createTestWorld, destroyTestWorld, testShapeDef } from "./testing";
import { bodyIsAwake, bodySetAwake, destroyBody } from "./world/body";
import type { WorldState } from "./world/world";

setDefaultTimeout(CEILING.node);
const entry = resolve(import.meta.dir, "fixtures/shape-allocation.entry.ts");
let countedKernelBytes: ArrayBuffer | undefined;

beforeAll(async () => {
    const dir = mkdtempSync(join(tmpdir(), "shallot-shape-alloc-"));
    try {
        const build = Bun.spawnSync(
            [
                "cargo",
                "build",
                "-p",
                "shallot-physics",
                "--release",
                "--target",
                "wasm32-unknown-unknown",
                "--features",
                "count-allocations",
                "--target-dir",
                dir,
            ],
            { cwd: resolve(import.meta.dir, "../../..") },
        );
        if (build.exitCode !== 0) throw new Error(build.stderr.toString());
        countedKernelBytes = await Bun.file(
            join(dir, "wasm32-unknown-unknown/release/shallot_physics.wasm"),
        ).arrayBuffer();
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}, CEILING.startup);

test("warm recycled shape create/destroy, filters, inline materials, mass walks, static enable/disable, sleeping wake with shape query sync and pair finding allocate no WASM heap; the allocating control fails", async () => {
    const bytes = countedKernelBytes;
    if (!bytes) throw new Error("counting kernel was not built");
    const runtime = kernelState(undefined);
    const previous = runtime.instance;
    let world: WorldState | undefined;
    try {
        const built = await WebAssembly.instantiate(bytes, {
            env: {
                now: clockImport(() => k.memory),
                queryCallback() {
                    throw new Error("unexpected shape allocation callback");
                },
                materialCallback() {
                    throw new Error("unexpected shape allocation material callback");
                },
                collisionCallback() {
                    throw new Error("unexpected shape allocation collision callback");
                },
            },
        });
        const k = built.instance.exports as unknown as Kernel & {
            allocationCount(): number;
            allocationControl(): void;
        };
        runtime.instance = k;
        world = createTestWorld();
        const state = world;
        for (const type of [BodyType.Static, BodyType.Kinematic]) {
            const id = createTestBody(world, { type });
            // A high shape count must not reserve temporary mass storage for non-dynamic bodies.
            setBodyField(world, id, BodyField.shapeCount, 1024);
            const massBefore = k.allocationCount();
            k.bodyUpdateMass(world.worldId, id);
            expect(k.allocationCount() - massBefore).toBe(0);
            setBodyField(world, id, BodyField.shapeCount, 0);
            destroyBody(world, id);
        }
        const step = shapeAllocationSubject(world, false);
        for (let i = 0; i < 1200; ++i) step();
        const before = k.allocationCount();
        for (let i = 0; i < 600; ++i) step();
        expect(k.allocationCount() - before).toBe(0);
        const ground = createTestBody(world, {
            type: BodyType.Static,
            position: { x: 100, y: 0, z: 0 },
        });
        const compound = createCompound({
            spheres: [
                {
                    sphere: { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                    material: defaultSurfaceMaterial(),
                },
            ],
        })!;
        createCompoundShape(world, ground, testShapeDef(), compound);
        for (let i = 0; i < 32; ++i) {
            const warm = createCompoundShape(world, ground, testShapeDef(), compound)!;
            destroyShape(world, warm, true);
        }
        for (let i = 0; i < 32; ++i) {
            const createBefore = k.allocationCount();
            const shape = createCompoundShape(world, ground, testShapeDef(), compound)!;
            // Native compounds own even their single material; this path is not a zero span.
            expect(k.allocationCount() - createBefore).toBe(1);
            const destroyBefore = k.allocationCount();
            destroyShape(world, shape, true);
            expect(k.allocationCount() - destroyBefore).toBe(0);
        }
        const sleeper = createTestBody(world, { type: BodyType.Dynamic });
        createSphereShape(world, sleeper, testShapeDef(), {
            center: { x: 0, y: 0, z: 0 },
            radius: 0.5,
        });
        // Warm the awake set's capacity for the newly added body before measuring recycled wake.
        bodySetAwake(world, sleeper, false);
        bodySetAwake(world, sleeper, true);
        for (let i = 0; i < 32; ++i) {
            // Sleeping rebuilds Box3D's four owned solver-set buffers; only wake is allocation-free.
            bodySetAwake(world, sleeper, false);
            expect(bodyIsAwake(world, sleeper)).toBe(false);
            const wakeBefore = k.allocationCount();
            bodySetAwake(world, sleeper, true);
            expect(bodyIsAwake(world, sleeper)).toBe(true);
            expect(k.allocationCount() - wakeBefore).toBe(0);
        }
        const missingHull = makeBoxHull(0.625, 0.375, 0.875);
        for (let i = 0; i < 32; ++i) {
            const shape = createHullShape(world, ground, testShapeDef(), missingHull)!;
            destroyShape(world, shape, true);
        }
        for (let i = 0; i < 32; ++i) {
            const count = k.hullDatabaseCount(world.worldId);
            const missBefore = k.allocationCount();
            const shape = createHullShape(world, ground, testShapeDef(), missingHull)!;
            const handle = hullDatabaseIndex(world, missingHull);
            expect(k.hullDatabaseCount(world.worldId)).toBe(count + 1);
            expect(k.allocationCount() - missBefore).toBe(1);
            const hitBefore = k.allocationCount();
            const duplicate = createHullShape(world, ground, testShapeDef(), missingHull)!;
            expect(k.hullDatabaseRefs(world.worldId, handle)).toBe(2);
            expect(k.allocationCount() - hitBefore).toBe(0);
            const destroyBefore = k.allocationCount();
            destroyShape(world, shape, true);
            expect(k.hullDatabaseRefs(world.worldId, handle)).toBe(1);
            destroyShape(world, duplicate, true);
            expect(k.hullDatabaseCount(world.worldId)).toBe(count);
            // b3AddHullToDatabase clones on a miss; a hit and releasing either reference allocate nothing.
            expect(k.allocationCount() - destroyBefore).toBe(0);
        }
        const meshData = createGridMesh(3, 3, 1, 0, true);
        const heightData = createGrid(3, 3, { x: 1, y: 1, z: 1 }, false);
        const compoundData = createCompound({
            spheres: [
                {
                    sphere: { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
                    material: defaultSurfaceMaterial(),
                },
            ],
        })!;
        const geometryCases = [
            {
                kind: ShapeType.Mesh,
                create: () =>
                    createMeshShape(state, ground, testShapeDef(), meshData, { x: 1, y: 1, z: 1 })!,
            },
            {
                kind: ShapeType.HeightField,
                create: () => createHeightFieldShape(state, ground, testShapeDef(), heightData)!,
            },
            {
                kind: ShapeType.Compound,
                create: () => createCompoundShape(state, ground, testShapeDef(), compoundData)!,
            },
        ];
        for (const { kind, create } of geometryCases) {
            for (let i = 0; i < 12; ++i) destroyShape(world, create(), true);
            for (let i = 0; i < 12; ++i) {
                const missBefore = k.allocationCount();
                const first = create();
                const pointer = world.shapeStore.shapeU[first * SHAPE_STRIDE + S_GEO_REFERENCE];
                expect(k.allocationCount() - missBefore).toBeGreaterThan(0);
                expect(k.geometryDatabaseRefs(world.worldId, kind, pointer)).toBe(1);
                const hitBefore = k.allocationCount();
                const duplicate = create();
                // Box3D creates an owned material for every compound; database hits add nothing else.
                expect(k.allocationCount() - hitBefore).toBe(kind === ShapeType.Compound ? 1 : 0);
                expect(k.geometryDatabaseRefs(world.worldId, kind, pointer)).toBe(2);
                const releaseBefore = k.allocationCount();
                destroyShape(world, first, true);
                expect(k.geometryDatabaseRefs(world.worldId, kind, pointer)).toBe(1);
                destroyShape(world, duplicate, true);
                expect(k.geometryDatabaseRefs(world.worldId, kind, pointer)).toBe(0);
                expect(k.allocationCount() - releaseBefore).toBe(0);
            }
        }
        const controlBefore = k.allocationCount();
        k.allocationControl();
        const allocated = k.allocationCount() - controlBefore;
        expect(allocated).toBeGreaterThan(0);
        expect(() => expect(allocated).toBe(0)).toThrow();
    } finally {
        if (world) destroyTestWorld(world);
        runtime.instance = previous;
    }
});

test("the same warm recycled shape lifecycle allocates no steady JavaScript heap", async () => {
    const sample = await sampleAllocation(entry, { warm: 1200, frames: 600 });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});

test("warmed geometry uploads on the hull database miss path allocate no transient JavaScript heap", async () => {
    const sample = await sampleAllocation(entry, { warm: 1200, frames: 600, input: "upload" });
    expect(sample.control.length).toBeGreaterThan(0);
    const failure = allocationFailure(sample);
    if (failure !== undefined) throw new Error(failure);
});

test("the shape allocation assertion rejects deliberate JavaScript allocation", async () => {
    const sample = await sampleAllocation(entry, { warm: 1200, frames: 600, input: "allocating" });
    expect(sample.control.length).toBeGreaterThan(0);
    expect(allocationFailure(sample)).toContain("steady play allocated JavaScript heap");
});

import { expect, test } from "bun:test";
import { BodyType, PhysicsWorld, makeBoxHull, createMesh, createCompound, defaultSurfaceMaterial, createHeightField } from "../../src/standard/physics/api";
import { kernel } from "../../src/standard/physics/kernel/kernel";
import { assertPublicOracleKernel } from "./oracle-kernel";
function count(world: PhysicsWorld, lane: number): number {
    return (kernel(world.state.ecsState) as unknown as { box3dCallbackWork(world: number, lane: number): number }).box3dCallbackWork(world.state.worldId, lane);
}
test("fitted filtered collection adds only the counted candidate bytes to capacity-sized scratch", async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        for (let i = 0; i < 2; ++i) world.createBody({ type: BodyType.Dynamic }).createSphere(
            { enableCustomFiltering: i === 0 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
        );
        let calls = 0;
        world.setCustomFilterCallback(() => { ++calls; return true; });
        world.step(1 / 60);
        const k = kernel(world.state.ecsState) as unknown as { box3dPairScratchBytes(world: number): number };
        // 32 survivor slots + two heads/counts/offsets, aligned to 16; one 12-byte candidate, aligned.
        expect(k.box3dPairScratchBytes(world.state.worldId)).toBe(544 + 16);
        expect(world.getCounters().contactCount).toBe(1);
        expect(calls).toBe(1);
    } finally { world.destroy(); }
});
test("a saturated serial walk keeps native filter-before-capacity invocation semantics", async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        for (let i = 0; i < 100; ++i) world.createBody({ type: BodyType.Dynamic }).createSphere(
            { enableCustomFiltering: true }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
        );
        let calls = 0;
        world.setCustomFilterCallback(() => { ++calls; return true; });
        world.step(1 / 60);
        expect(world.getCounters().contactCount).toBe(1600);
        expect(calls).toBe(4950);
    } finally { world.destroy(); }
});
for (const filtered of [false, true]) test(`${filtered ? "filtered" : "no-callback"} 1000-sphere overflow reserves at most 32 bytes per survivor slot`, async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        for (let i = 0; i < 1000; ++i) world.createBody({ type: BodyType.Dynamic }).createSphere(
            { enableCustomFiltering: i === 0 }, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 },
        );
        if (filtered) world.setCustomFilterCallback(() => true);
        world.step(1 / 60);
        expect(world.getCounters().contactCount).toBe(16000);
        const k = kernel(world.state.ecsState) as unknown as { box3dPairScratchBytes(world: number): number };
        expect(k.box3dPairScratchBytes(world.state.worldId)).toBeLessThanOrEqual(32 * 16000);
    } finally { world.destroy(); }
});
test("unflagged pair candidates remain in tasks with a custom filter installed", async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        world.createBody().createHull({}, makeBoxHull(1000, 0.5, 2));
        for (let i = 0; i < 300; ++i) world.createBody({ type: BodyType.Dynamic, position: { x: i * 3, y: 0.99, z: 0 } }).createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        world.createBody({ position: { x: 10000, y: 0, z: 0 } }).createSphere({ enableCustomFiltering: true }, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
        let calls = 0;
        world.setCustomFilterCallback(() => { ++calls; return true; });
        world.step(1 / 60);
        expect(calls).toBe(0);
        expect(count(world, 0)).toBe(300);
        expect(count(world, 1)).toBe(0);
    } finally { world.destroy(); }
});
test("an unflagged fast body remains in its task when the only flagged target is distant", async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableSleep: false });
    try {
        world.createBody().createHull({}, makeBoxHull(10, 0.1, 2));
        world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 5, z: 0 }, linearVelocity: { x: 0, y: -400, z: 0 } }).createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.2 });
        world.createBody({ position: { x: 10000, y: 0, z: 0 } }).createSphere({ enablePreSolveEvents: true }, { center: { x: 0, y: 0, z: 0 }, radius: 1 });
        let calls = 0;
        world.setPreSolveCallback(() => { ++calls; return true; });
        world.step(1 / 60);
        expect(calls).toBe(0);
        expect(count(world, 4)).toBe(1);
        expect(count(world, 5)).toBe(0);
    } finally { world.destroy(); }
});
test("recycled pre-solve contacts finish in the task without a second deferral", async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 } });
    try {
        world.createBody().createHull({ enablePreSolveEvents: true }, makeBoxHull(2, 0.5, 2));
        world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 0.99, z: 0 } }).createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        let calls = 0;
        world.setPreSolveCallback(() => { ++calls; return false; });
        world.step(1 / 60);
        expect(calls).toBe(1);
        expect(count(world, 3)).toBe(1);
        world.step(1 / 60);
        expect(calls).toBe(1);
        expect(count(world, 3)).toBe(0);
    } finally { world.destroy(); }
});
for (const kind of ["mesh", "compound mesh", "height"] as const) test(`${kind} contacts never defer for pre-solve`, async () => {
    await assertPublicOracleKernel();
    const world = new PhysicsWorld({ gravity: { x: 0, y: 0, z: 0 }, enableContinuous: false });
    try {
        const ground = world.createBody();
        const def = { enablePreSolveEvents: true, enableContactEvents: true };
        if (kind === "height") ground.createHeightField(def, createHeightField({ heights: [0, 0, 0, 0], materialIndices: null, countX: 2, countZ: 2, scale: { x: 4, y: 1, z: 4 }, globalMinimumHeight: -1, globalMaximumHeight: 1, clockwiseWinding: false }));
        else {
            const mesh = createMesh({ vertices: [{ x: -2, y: 0, z: -2 }, { x: 2, y: 0, z: -2 }, { x: 2, y: 0, z: 2 }, { x: -2, y: 0, z: 2 }], indices: [0, 2, 1, 0, 3, 2], identifyEdges: true });
            if (!mesh) throw new Error("mesh construction failed");
            if (kind === "mesh") ground.createMesh(def, mesh);
            else {
                const compound = createCompound({ meshes: [{ meshData: mesh, scale: { x: 1, y: 1, z: 1 }, transform: { p: { x: 0, y: 0, z: 0 }, q: { v: { x: 0, y: 0, z: 0 }, s: 1 } }, materials: [defaultSurfaceMaterial()], materialCount: 1 }] });
                if (!compound) throw new Error("compound construction failed");
                ground.createCompound(def, compound);
            }
        }
        world.createBody({ type: BodyType.Dynamic, position: { x: 0, y: 0.4, z: 0 } }).createSphere({}, { center: { x: 0, y: 0, z: 0 }, radius: 0.5 });
        let calls = 0;
        world.setPreSolveCallback(() => { ++calls; return true; });
        world.step(1 / 60);
        expect(calls).toBe(0);
        expect(count(world, 2)).toBe(1);
        expect(count(world, 3)).toBe(0);
    } finally { world.destroy(); }
});

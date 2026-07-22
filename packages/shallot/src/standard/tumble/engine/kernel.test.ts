import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { createBodyStore } from "./bodycolumns";
import {
    D_CONTACT,
    D_GEOM_A,
    D_GEOM_B,
    D_TYPE_A,
    D_TYPE_B,
    D_XF_A,
    D_XF_B,
    DISPATCH_STRIDE,
    reserveColumns,
    STATE_LIVE,
} from "./columns";
import { type UploadHull, uploadGeometry } from "./geocolumns";
import { AUTO_THREADS, announce, COOP_COEP_HINT, type Host, init, kernel, resolve } from "./kernel";
import { SHARED_STACK_SIZE } from "./kernel.shared.wasm";
import { createManifoldStore, DIR_STRIDE, MANIFOLD_STRIDE } from "./manifoldstore";
import { maxWorkers } from "./pool";

// `init({ threads: 0 })` throughout: `init()` now multithreads by default (bun/node have SAB), which
// would spawn a pool and flip the whole process to the shared artifact — bun runs one process for every
// test file. These tests exercise the single-thread kernel's exports, so they force it, same as
// pool.test.ts drives the shared artifact in isolation. The resolver's own decision table is unit-tested
// below over its pure seam.

// The load-bearing stage-3 toolchain gate: wasm-simd128 must build, instantiate, and execute
// correctly in the JS host. `smoke_scale` runs a real `f32x4_mul` over shared linear memory; if
// simd128 hit a JSC cliff (bad codegen, unsupported op) this is where it surfaces.
test("wasm-simd128 kernel loads and runs f32x4 over shared memory", async () => {
    await init({ threads: 0 });
    const k = kernel();
    const view = new Float32Array(k.memory.buffer, k.scratchPtr(), 8);
    view.set([1, 2, 3, 4, 5, 6, 7, 8]);
    k.smokeScale(8, 2.5);
    expect(Array.from(view)).toEqual([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]);
});

// End-to-end plumbing gate: drive the finalize phase over the shared columns exactly as `step()` will
// (reserve → write inputs through the views → run the export shim → read outputs back), and assert
// bit-for-bit against the committed C gold. This exercises the arena layout, the layoutPtr header, the
// view derivation, and the finalize shim — everything between TS and the gold-verified Rust arithmetic.
// Body `state` is resident (4a.2), so it's driven through the body store over the persistent region
// (which `reserveBodies` establishes and `arena::reserve` aliases `LAYOUT[STATE]` at), not `cols`.
const _b = new ArrayBuffer(4);
const _f = new Float32Array(_b);
const _u = new Uint32Array(_b);
function hexBits(h: string): number {
    return Number.parseInt(h, 16) >>> 0;
}
function f32FromHex(h: string): number {
    _u[0] = hexBits(h);
    return _f[0];
}
function bitsOf(x: number): number {
    _f[0] = x;
    return _u[0];
}

test("finalize export shim matches the C gold over shared columns", async () => {
    await init({ threads: 0 });
    const k = kernel();
    const gold = JSON.parse(
        readFileSync(new URL("./finalize.gold.json", import.meta.url), "utf8"),
    ) as { cases: Record<string, string | string[]>[] };

    for (const [c, cse] of gold.cases.entries()) {
        const simIn = (cse.sim as string[]).map(f32FromHex);
        const stateIn = (cse.state as string[]).map(f32FromHex);
        const finIn = (cse.fin as string[]).map(f32FromHex);
        const h = f32FromHex(cse.h as string);
        const invDt = f32FromHex(cse.invDt as string);

        // The sim/fin/state columns are all resident (arena::reserve points LAYOUT[SIM]/[FIN]/[STATE] at
        // the body region), so drive the inputs through the body store's views (record 0). Establish the
        // region first, then reserve the per-step columns; refresh the store after both grows.
        k.reserveBodies(1);
        const cols = reserveColumns(1, 0, 0, 0, 0, 0);
        const store = createBodyStore();
        store.refreshViews();
        store.simF.set(simIn);
        store.stateF.subarray(0, STATE_LIVE).set(stateIn);
        store.finF.set(finIn);

        k.finalize(h, invDt, 1);

        const check = (got: Float32Array, want: string[], label: string) => {
            for (let i = 0; i < want.length; ++i) {
                if (bitsOf(got[i]) !== hexBits(want[i])) {
                    throw new Error(
                        `case ${c} ${label}[${i}]: got 0x${bitsOf(got[i]).toString(16)}, want ${want[i]}`,
                    );
                }
            }
        };
        check(store.simF, cse.outSim as string[], "sim");
        check(store.stateF.subarray(0, STATE_LIVE), cse.outState as string[], "state");
        check(store.finF, cse.outFin as string[], "fin");
        check(cols.finOut, cse.out as string[], "out");
    }
});

// Geometry-columns gate (3c.2b): upload the convex hulls from the committed manifold gold into the
// kernel's static geometry columns, then run the hull-hull narrowphase over those column-backed hulls
// in wasm (`collideHullsGeo`) and assert the manifold bit-for-bit against the same gold the native
// Vec-backed path verifies. This exercises the whole geometry read path — the reserveGeometry layout,
// the TS upload (geocolumns.ts), the wasm reinterpret into the HullData view, and every pool the
// narrowphase touches (points, vertices, edges, faces, planes, center) — the seed of 3c.3's dispatch.
type GoldVec = [string, string, string];
type GoldHull = {
    center: GoldVec;
    vertexCount: number;
    edgeCount: number;
    faceCount: number;
    points: GoldVec[];
    vertices: number[];
    edges: [number, number, number, number][];
    faces: number[];
    planes: { normal: GoldVec; offset: string }[];
};

function uploadHull(h: GoldHull): UploadHull {
    return {
        center: {
            x: f32FromHex(h.center[0]),
            y: f32FromHex(h.center[1]),
            z: f32FromHex(h.center[2]),
        },
        vertexCount: h.vertexCount,
        edgeCount: h.edgeCount,
        faceCount: h.faceCount,
        points: h.points.map((p) => ({
            x: f32FromHex(p[0]),
            y: f32FromHex(p[1]),
            z: f32FromHex(p[2]),
        })),
        vertices: h.vertices.map((edge) => ({ edge })),
        edges: h.edges.map(([next, twin, origin, face]) => ({ next, twin, origin, face })),
        faces: h.faces.map((edge) => ({ edge })),
        planes: h.planes.map((pl) => ({
            normal: {
                x: f32FromHex(pl.normal[0]),
                y: f32FromHex(pl.normal[1]),
                z: f32FromHex(pl.normal[2]),
            },
            offset: f32FromHex(pl.offset),
        })),
        geoIndex: -1,
    };
}

test("collideHullsGeo matches the manifold gold over the static geometry columns", async () => {
    await init({ threads: 0 });
    const k = kernel();
    const gold = JSON.parse(
        readFileSync(new URL("./manifold.gold.json", import.meta.url), "utf8"),
    ) as {
        hulls: {
            name: string;
            hullA: GoldHull;
            hullB: GoldHull;
            xf: { p: GoldVec; q: [string, string, string, string] };
            manifolds: {
                normal: GoldVec;
                pointCount: number;
                points: { point: GoldVec; separation: string; featureId: number }[];
            }[];
        }[];
    };

    for (const scene of gold.hulls) {
        const a = uploadHull(scene.hullA);
        const b = uploadHull(scene.hullB);
        // Fresh upload per scene so geoIndex assignment is deterministic (a→0, b→1).
        uploadGeometry([a, b]);

        const p = scene.xf.p;
        const q = scene.xf.q;
        const count = k.collideHullsGeo(
            a.geoIndex,
            b.geoIndex,
            f32FromHex(p[0]),
            f32FromHex(p[1]),
            f32FromHex(p[2]),
            f32FromHex(q[0]),
            f32FromHex(q[1]),
            f32FromHex(q[2]),
            f32FromHex(q[3]),
        );

        // Read the output buffer through both a float and a u32 view (the feature id is a raw u32).
        const outF = new Float32Array(k.memory.buffer, k.geoOutPtr(), 4 + 8 * 5);
        const outU = new Uint32Array(k.memory.buffer, k.geoOutPtr(), 4 + 8 * 5);

        // collideHullsGeo uses a fresh SAT cache, matching the gold's first (cold) call.
        const want = scene.manifolds[0];
        expect(count).toBe(want.pointCount);
        if (want.pointCount > 0) {
            expect(bitsOf(outF[1])).toBe(hexBits(want.normal[0]));
            expect(bitsOf(outF[2])).toBe(hexBits(want.normal[1]));
            expect(bitsOf(outF[3])).toBe(hexBits(want.normal[2]));
        }
        for (let i = 0; i < want.pointCount; ++i) {
            const wp = want.points[i];
            const o = 4 + i * 5;
            const label = `${scene.name}.points[${i}]`;
            expect(bitsOf(outF[o]), `${label}.x`).toBe(hexBits(wp.point[0]));
            expect(bitsOf(outF[o + 1]), `${label}.y`).toBe(hexBits(wp.point[1]));
            expect(bitsOf(outF[o + 2]), `${label}.z`).toBe(hexBits(wp.point[2]));
            expect(bitsOf(outF[o + 3]), `${label}.sep`).toBe(hexBits(wp.separation));
            expect(outU[o + 4], `${label}.featureId`).toBe(wp.featureId);
        }
    }
});

// Convex dispatch gate (3c.3.a): drive `dispatchConvex` over the full column stack the live collide will
// use — the static geometry pools (hulls), the persistent manifold pool (warm-start in/out), the folded
// GJK/SAT cache — for every convex pair type in the convex-manifold gold, and assert the mapped
// persistent manifold bit-for-bit against the same gold the native `compute_convex_manifold` path
// verifies. Mirrors the native `check_scene`: two runs per scene (cold: no old points; warm: old points
// seeded so impulses carry by feature id), each from a cold cache (fresh store per run). This isolates the
// dispatch column marshaling + pool↔manifold round-trip + cache read/write from the live restructure
// (3c.3.b). Inert until then (nothing calls `dispatchConvex` in the live path).
type GoldSphere = { center: GoldVec; radius: string };
type GoldCapsule = { center1: GoldVec; center2: GoldVec; radius: string };
type GoldXf = { p: GoldVec; q: [string, string, string, string] };
type GoldRun = {
    old: { featureId: number; normalImpulse: string }[];
    out: {
        normal: GoldVec;
        pointCount: number;
        points: {
            anchorA: GoldVec;
            anchorB: GoldVec;
            separation: string;
            normalImpulse: string;
            featureId: number;
            persisted: boolean;
        }[];
    };
};

test("dispatchConvex maps the convex-manifold gold through the geometry + manifold columns", async () => {
    await init({ threads: 0 });
    const k = kernel();
    const gold = JSON.parse(
        readFileSync(new URL("./convex_manifold.gold.json", import.meta.url), "utf8"),
    ) as Record<string, Record<string, unknown>[]>;

    const TyCapsule = 0;
    const TyHull = 3;
    const TySphere = 5;
    // Per section: the two shape types + which gold key holds each shape (hull shapes are uploaded).
    const sections = [
        { key: "spheres", ta: TySphere, tb: TySphere, aKey: "a", bKey: "b" },
        { key: "capsuleSphere", ta: TyCapsule, tb: TySphere, aKey: "a", bKey: "b" },
        { key: "capsules", ta: TyCapsule, tb: TyCapsule, aKey: "a", bKey: "b" },
        { key: "hullSphere", ta: TyHull, tb: TySphere, aKey: "hull", bKey: "b" },
        { key: "hullCapsule", ta: TyHull, tb: TyCapsule, aKey: "hull", bKey: "b" },
        { key: "hulls", ta: TyHull, tb: TyHull, aKey: "hullA", bKey: "hullB" },
    ];

    const writeXf = (view: Float32Array, o: number, x: GoldXf) => {
        view[o] = f32FromHex(x.p[0]);
        view[o + 1] = f32FromHex(x.p[1]);
        view[o + 2] = f32FromHex(x.p[2]);
        view[o + 3] = f32FromHex(x.q[0]);
        view[o + 4] = f32FromHex(x.q[1]);
        view[o + 5] = f32FromHex(x.q[2]);
        view[o + 6] = f32FromHex(x.q[3]);
    };
    // Write a shape's geom slots; a hull encodes its geoIndex (u32), sphere/capsule their params (f32).
    const writeGeom = (
        f: Float32Array,
        u: Uint32Array,
        o: number,
        ty: number,
        shape: unknown,
        geoIndex: number,
    ) => {
        if (ty === TyHull) {
            u[o] = geoIndex;
        } else if (ty === TySphere) {
            const s = shape as GoldSphere;
            f[o] = f32FromHex(s.center[0]);
            f[o + 1] = f32FromHex(s.center[1]);
            f[o + 2] = f32FromHex(s.center[2]);
            f[o + 3] = f32FromHex(s.radius);
        } else {
            const c = shape as GoldCapsule;
            f[o] = f32FromHex(c.center1[0]);
            f[o + 1] = f32FromHex(c.center1[1]);
            f[o + 2] = f32FromHex(c.center1[2]);
            f[o + 3] = f32FromHex(c.center2[0]);
            f[o + 4] = f32FromHex(c.center2[1]);
            f[o + 5] = f32FromHex(c.center2[2]);
            f[o + 6] = f32FromHex(c.radius);
        }
    };

    for (const sec of sections) {
        for (const scene of gold[sec.key]) {
            const name = scene.name as string;
            const hulls: UploadHull[] = [];
            const aGeo =
                sec.ta === TyHull ? hulls.push(uploadHull(scene[sec.aKey] as GoldHull)) - 1 : -1;
            const bGeo =
                sec.tb === TyHull ? hulls.push(uploadHull(scene[sec.bKey] as GoldHull)) - 1 : -1;

            for (const run of scene.runs as GoldRun[]) {
                // Fresh store per run = cold GJK/SAT cache (a fresh directory is zeroed), matching the
                // native gold; the manifold region is reserved before geo so geo lands past it.
                const store = createManifoldStore();
                store.ensureSlot(0);
                store.flush();
                const [m] = store.alloc(0, 1);
                // Re-upload geometry every run so GEO_END tracks this run's manifold region (empty upload
                // for pairs with no hull keeps the region layout consistent).
                uploadGeometry(hulls);
                k.reserveDispatch(1);
                store.refreshViews();

                // Zero contact 0's folded GJK/SAT cache so this run collides cold, matching the native
                // gold's fresh-cache-per-run. A fresh store reuses heap_base memory, so the directory
                // isn't auto-zeroed; the live path guarantees cold-at-birth via freeSlot (manifoldstore.ts).
                // Cache slots are DIR_CACHE(12)..+10 within contact 0's directory record (DIR_STRIDE).
                for (let w = 0; w < 10; ++w) store.dirU[12 + w] = 0;

                // Seed the resident manifold's old points (warm-start source): feature id + impulse only,
                // the fields the bridge reads from the old manifold.
                m.pointCount = run.old.length;
                for (let j = 0; j < run.old.length; ++j) {
                    m.points[j].featureId = run.old[j].featureId;
                    m.points[j].normalImpulse = f32FromHex(run.old[j].normalImpulse);
                }

                const dispF = new Float32Array(k.memory.buffer, k.dispatchPtr(), DISPATCH_STRIDE);
                const dispU = new Uint32Array(k.memory.buffer, k.dispatchPtr(), DISPATCH_STRIDE);
                dispU[D_CONTACT] = 0;
                dispU[D_TYPE_A] = sec.ta;
                dispU[D_TYPE_B] = sec.tb;
                writeXf(dispF, D_XF_A, scene.xfA as GoldXf);
                writeXf(dispF, D_XF_B, scene.xfB as GoldXf);
                writeGeom(dispF, dispU, D_GEOM_A, sec.ta, scene[sec.aKey], aGeo);
                writeGeom(dispF, dispU, D_GEOM_B, sec.tb, scene[sec.bKey], bGeo);

                k.dispatchConvex(1);

                const out = new Uint32Array(k.memory.buffer, k.dispatchOutPtr(), 1);
                const want = run.out;
                expect(out[0], `${sec.key}/${name}.touching`).toBe(want.pointCount > 0 ? 1 : 0);
                expect(m.pointCount, `${sec.key}/${name}.pointCount`).toBe(want.pointCount);
                if (want.pointCount === 0) continue;
                expect(bitsOf(m.normal.x), `${sec.key}/${name}.normal.x`).toBe(
                    hexBits(want.normal[0]),
                );
                expect(bitsOf(m.normal.y), `${sec.key}/${name}.normal.y`).toBe(
                    hexBits(want.normal[1]),
                );
                expect(bitsOf(m.normal.z), `${sec.key}/${name}.normal.z`).toBe(
                    hexBits(want.normal[2]),
                );
                for (let i = 0; i < want.pointCount; ++i) {
                    const wp = want.points[i];
                    const p = m.points[i];
                    const label = `${sec.key}/${name}.points[${i}]`;
                    expect(bitsOf(p.anchorA.x), `${label}.anchorA.x`).toBe(hexBits(wp.anchorA[0]));
                    expect(bitsOf(p.anchorA.y), `${label}.anchorA.y`).toBe(hexBits(wp.anchorA[1]));
                    expect(bitsOf(p.anchorA.z), `${label}.anchorA.z`).toBe(hexBits(wp.anchorA[2]));
                    expect(bitsOf(p.anchorB.x), `${label}.anchorB.x`).toBe(hexBits(wp.anchorB[0]));
                    expect(bitsOf(p.anchorB.y), `${label}.anchorB.y`).toBe(hexBits(wp.anchorB[1]));
                    expect(bitsOf(p.anchorB.z), `${label}.anchorB.z`).toBe(hexBits(wp.anchorB[2]));
                    expect(bitsOf(p.separation), `${label}.sep`).toBe(hexBits(wp.separation));
                    expect(bitsOf(p.normalImpulse), `${label}.ni`).toBe(hexBits(wp.normalImpulse));
                    expect(p.featureId, `${label}.featureId`).toBe(wp.featureId);
                    expect(p.persisted, `${label}.persisted`).toBe(wp.persisted);
                }
            }
        }
    }
});

// Persistent manifold columns gate (3c.2c.ii): the store's allocator + wasm region + view protocol.
// Proves the ABI end-to-end — reserve, block allocation, the mixed f32/u32 aliasing on both columns,
// free-list recycling, and (the load-bearing bit) that a directory grow which shifts the pool base
// memmoves live pool data in place so every block keeps its element offset across the `memory.grow`
// detach. Inert in the live solve until 3c.2c.iii, so the fixture gate is unaffected; this test is the
// standing proof the storage round-trips.
test("persistent manifold store round-trips and preserves data across a directory grow", async () => {
    await init({ threads: 0 });
    const store = createManifoldStore();

    // Two contacts: blocks of 1 and 2 manifold records.
    store.ensureSlot(0);
    store.ensureSlot(1);
    const base0 = store.allocBlock(0, 1);
    const base1 = store.allocBlock(1, 2);
    expect(base0).toBe(0);
    expect(base1).toBe(1); // bumped past the 1-record block
    expect(store.flush()).toBe(true); // first reserve grows from empty
    store.refreshViews();

    // Write a directory record (material row f32 + block-descriptor u32) and a manifold record. Values
    // are exactly f32-representable so the read-back compares by ===.
    const writeDir = (cid: number, base: number, count: number) => {
        const o = cid * DIR_STRIDE;
        store.dirF[o] = 0.25; // friction
        store.dirF[o + 1] = 0.5; // restitution
        store.dirF[o + 2] = 0.125; // rollingResistance
        store.dirF[o + 3] = 1.5; // tangentVelocity.x
        store.dirU[o + 6] = 0xdead0000; // flags
        store.dirU[o + 7] = count; // manifoldCount
        store.dirU[o + 8] = base; // manifoldBase
    };
    const writeManifold = (base: number, marker: number) => {
        const o = base * MANIFOLD_STRIDE;
        store.poolF[o] = marker; // normal.x
        store.poolF[o + 6] = marker + 0.5; // twistImpulse
        store.poolU[o + 10] = 4; // pointCount (u32 alias)
        store.poolU[o + 11 + 11] = 0xfeed0000 + base; // point[0].featureId
    };
    writeDir(0, base0, 1);
    writeDir(1, base1, 2);
    writeManifold(base0, 32);
    writeManifold(base1, 64);
    writeManifold(base1 + 1, 128);

    const readManifold = (base: number, marker: number, label: string) => {
        const o = base * MANIFOLD_STRIDE;
        expect(store.poolF[o], `${label}.normal.x`).toBe(marker);
        expect(store.poolF[o + 6], `${label}.twist`).toBe(marker + 0.5);
        expect(store.poolU[o + 10], `${label}.pointCount`).toBe(4);
        expect(store.poolU[o + 11 + 11], `${label}.featureId`).toBe(0xfeed0000 + base);
    };

    // Force a directory grow: a high contactId pushes the pool base up. The pool run must be memmoved,
    // so the block contents survive.
    store.ensureSlot(64);
    expect(store.flush()).toBe(true);
    store.refreshViews();

    readManifold(base0, 32, "block0 after dir grow");
    readManifold(base1, 64, "block1[0] after dir grow");
    readManifold(base1 + 1, 128, "block1[1] after dir grow");

    // Directory records survived (base fixed at heap_base — in place, never moved).
    expect(store.dirF[0]).toBe(0.25);
    expect(store.dirU[0 * DIR_STRIDE + 8]).toBe(base0);
    expect(store.dirF[1 * DIR_STRIDE]).toBe(0.25);
    expect(store.dirU[1 * DIR_STRIDE + 7]).toBe(2);

    // Free-list recycling: a freed block of the same size class is handed back before the bump grows.
    store.freeBlock(0);
    expect(store.allocBlock(2, 1)).toBe(base0);

    // Pool grow (base unchanged) also preserves the earlier blocks.
    const big = store.allocBlock(3, 40);
    expect(store.flush()).toBe(true);
    store.refreshViews();
    expect(big).toBeGreaterThan(base1 + 1);
    readManifold(base1, 64, "block1[0] after pool grow");
    readManifold(base1 + 1, 128, "block1[1] after pool grow");
});

// Pool-backed manifold views gate (3c.2c.iii): the `Manifold`/`ManifoldPoint` accessors the narrowphase
// writes and the solver/events read. Proves every header + point field round-trips through the pool
// (incl. the signed `triangleIndex` i32 alias and the `persisted` bool), and that views re-fetched over
// a contact's block after a pool grow read the same data. All values are exactly f32-representable.
test("pool-backed manifold views round-trip every field across a grow", async () => {
    await init({ threads: 0 });
    const store = createManifoldStore();

    store.ensureSlot(0);
    expect(store.flush()).toBe(true); // sizes the directory + pool from empty

    const [m] = store.alloc(0, 1);
    m.normal = { x: 0.5, y: -0.25, z: 0.125 };
    m.frictionImpulse = { x: 1.5, y: -2.5, z: 0.75 };
    m.twistImpulse = -0.5;
    m.rollingImpulse = { x: 0.25, y: 0.5, z: -1.25 };
    m.pointCount = 2;
    m.points[0].anchorA = { x: 1.0, y: 2.0, z: 3.0 };
    m.points[0].anchorB = { x: -1.0, y: -2.0, z: -3.0 };
    m.points[0].separation = -0.0625;
    m.points[0].baseSeparation = 0.125;
    m.points[0].normalImpulse = 4.5;
    m.points[0].totalNormalImpulse = 4.75;
    m.points[0].normalVelocity = -3.25;
    m.points[0].featureId = 0xfeed0001;
    m.points[0].triangleIndex = -1; // NULL_INDEX through the i32 alias
    m.points[0].persisted = true;
    m.points[1].anchorA = { x: 5.0, y: 6.0, z: 7.0 };
    m.points[1].featureId = 0x1234;
    m.points[1].triangleIndex = 42;
    m.points[1].persisted = false;

    const check = (label: string) => {
        const [v] = store.views(0, 1);
        expect(v.normal, `${label}.normal`).toEqual({ x: 0.5, y: -0.25, z: 0.125 });
        expect(v.frictionImpulse, `${label}.friction`).toEqual({ x: 1.5, y: -2.5, z: 0.75 });
        expect(v.twistImpulse, `${label}.twist`).toBe(-0.5);
        expect(v.rollingImpulse, `${label}.rolling`).toEqual({ x: 0.25, y: 0.5, z: -1.25 });
        expect(v.pointCount, `${label}.pointCount`).toBe(2);
        const p0 = v.points[0];
        expect(p0.anchorA, `${label}.p0.anchorA`).toEqual({ x: 1.0, y: 2.0, z: 3.0 });
        expect(p0.anchorB, `${label}.p0.anchorB`).toEqual({ x: -1.0, y: -2.0, z: -3.0 });
        expect(p0.separation, `${label}.p0.sep`).toBe(-0.0625);
        expect(p0.baseSeparation, `${label}.p0.baseSep`).toBe(0.125);
        expect(p0.normalImpulse, `${label}.p0.ni`).toBe(4.5);
        expect(p0.totalNormalImpulse, `${label}.p0.tni`).toBe(4.75);
        expect(p0.normalVelocity, `${label}.p0.nv`).toBe(-3.25);
        expect(p0.featureId, `${label}.p0.feat`).toBe(0xfeed0001);
        expect(p0.triangleIndex, `${label}.p0.tri`).toBe(-1);
        expect(p0.persisted, `${label}.p0.persisted`).toBe(true);
        expect(v.points[1].triangleIndex, `${label}.p1.tri`).toBe(42);
        expect(v.points[1].persisted, `${label}.p1.persisted`).toBe(false);
    };
    check("before grow");

    // A big allocation grows the pool (re-deriving every view); the contact-0 block keeps its offset,
    // so views re-fetched over it read the same data.
    store.alloc(1, 40);
    expect(store.grew).toBe(true);
    check("after pool grow");
});

// Persistent body region relocation gate (4a.1): the body region sits first in linear memory, so its
// own growth shifts the manifold + geometry regions above it. Prove the kernel relocates them in place
// — one overlapping memmove of the whole span + a static rebase — so a live contact's warm-start
// manifold and an interned hull's geometry survive a body-region grow bit-for-bit. This is the only
// native-uncovered path (the arena/manifold/geo regions are wasm-only), on top of the spawn fixtures
// that exercise mid-scene body growth over live manifolds. Runs last so its reserveBodies calls (which
// establish the singleton body region) never perturb the earlier tests, which assume no body region.
test("a body-region grow relocates the manifold + geometry regions in place", async () => {
    await init({ threads: 0 });
    const k = kernel();
    const buf = () => k.memory.buffer;
    const nManifold = 2; // [DIR, POOL]
    const nGeo = 6; // [REC, POINTS, VERTICES, EDGES, FACES, PLANES]

    // Establish a small body region, then reserve the manifold + geometry regions above it.
    k.reserveBodies(64);
    k.reserveManifolds(8, 8);
    k.reserveGeometry(2, 4, 4, 4);

    // The body region owns live columns too (4a.2): STATE stays at the region base across a grow, but
    // FLAGS shifts up as capacity grows, so its contents must be copied. Mark record 5 in both and
    // assert they survive at the (re-derived) offsets — the DYNAMIC-flag-loss regression that made the
    // solver stop writing velocity back once a scene crossed the 16-body cap.
    const nBody = 6; // [STATE, SIM, FIN, FIN_OUT, FLAGS, SIM2]
    const bl0 = new Uint32Array(buf(), k.bodyLayoutPtr(), nBody);
    const stateBase0 = bl0[0];
    const flagsBase0 = bl0[4];
    const markState = 0x57a7_0006;
    const markFlags = 0xf1a6_0007;
    new Uint32Array(buf(), stateBase0, 6 * 16)[5 * 16] = markState; // record 5, slot 0
    new Uint32Array(buf(), flagsBase0, 6)[5] = markFlags; // flags[5]

    const ml0 = new Uint32Array(buf(), k.manifoldLayoutPtr(), nManifold);
    const gl0 = new Uint32Array(buf(), k.geoLayoutPtr(), nGeo);
    const dirOff0 = ml0[0];
    const poolOff0 = ml0[1];
    const recOff0 = gl0[0];
    const pointsOff0 = gl0[1];
    const planesOff0 = gl0[5];
    // The manifold region anchors at the body region's end; geometry sits above it.
    expect(dirOff0).toBeLessThan(poolOff0);
    expect(poolOff0).toBeLessThan(recOff0);

    // Write recognizable markers into a live directory record, a live manifold record, and every
    // geometry pool the relocation must preserve (u32 markers dodge f32 NaN canonicalization).
    const markDir = 0xd1c0_0001;
    const markPool = 0x900f_0002;
    const markRec = 0x9ec0_0003;
    const markPoints = 0xba11_0004;
    const markPlanes = 0x71a5_0005;
    new Uint32Array(buf(), dirOff0, DIR_STRIDE * 8)[3] = markDir;
    new Uint32Array(buf(), poolOff0, MANIFOLD_STRIDE * 8)[5] = markPool;
    new Uint32Array(buf(), recOff0, 2 * 12)[7] = markRec;
    new Uint32Array(buf(), pointsOff0, 4 * 3)[2] = markPoints;
    new Uint32Array(buf(), planesOff0, 4 * 4)[9] = markPlanes;

    // Grow the body region past its current capacity, forcing the relocation.
    k.reserveBodies(8192);

    const ml1 = new Uint32Array(buf(), k.manifoldLayoutPtr(), nManifold);
    const gl1 = new Uint32Array(buf(), k.geoLayoutPtr(), nGeo);
    const delta = ml1[0] - dirOff0;
    // The body region grew, so every region above it shifted up by the same nonzero delta.
    expect(delta).toBeGreaterThan(0);
    expect(ml1[1]).toBe(poolOff0 + delta);
    expect(gl1[0]).toBe(recOff0 + delta);
    expect(gl1[1]).toBe(pointsOff0 + delta);
    expect(gl1[5]).toBe(planesOff0 + delta);

    // Every marker survived the memmove, readable at the rebased offset.
    expect(new Uint32Array(buf(), ml1[0], DIR_STRIDE * 8)[3]).toBe(markDir);
    expect(new Uint32Array(buf(), ml1[1], MANIFOLD_STRIDE * 8)[5]).toBe(markPool);
    expect(new Uint32Array(buf(), gl1[0], 2 * 12)[7]).toBe(markRec);
    expect(new Uint32Array(buf(), gl1[1], 4 * 3)[2]).toBe(markPoints);
    expect(new Uint32Array(buf(), gl1[5], 4 * 4)[9]).toBe(markPlanes);

    // The body region's own columns survived too: STATE at the same base, FLAGS at its shifted offset.
    const bl1 = new Uint32Array(buf(), k.bodyLayoutPtr(), nBody);
    expect(bl1[0]).toBe(stateBase0); // STATE base is capacity-independent
    expect(bl1[4]).toBeGreaterThan(flagsBase0); // FLAGS shifted up with the larger capacity
    expect(new Uint32Array(buf(), bl1[0], 6 * 16)[5 * 16]).toBe(markState);
    expect(new Uint32Array(buf(), bl1[4], 6)[5]).toBe(markFlags);
});

// Persistent fat-AABB region relocation gate (4b.1): a second low persistent region, above the body
// region and below the manifold region. Prove both directions — its own grow relocates the manifold +
// geometry regions above it in place (its own single base-anchored column staying put), and a body
// region grow below shifts the whole fat-AABB region up with everything above it. Wasm-only, like the
// body-region gate above; runs after it (the singleton body region is already established).
test("a fat-AABB grow relocates the regions above it, and a body grow relocates the fat-AABB region", async () => {
    await init({ threads: 0 });
    const k = kernel();
    const buf = () => k.memory.buffer;
    const nManifold = 2;
    const nGeo = 6;
    const fatStride = 6; // AABB_STRIDE (lower.xyz + upper.xyz)

    // Establish the fat-AABB region above the body region, then the manifold + geometry regions above it.
    k.reserveFatAabb(16);
    k.reserveManifolds(8, 8);
    k.reserveGeometry(2, 4, 4, 4);

    const fatBase0 = new Uint32Array(buf(), k.fatAabbLayoutPtr(), 1)[0];
    const dirOff0 = new Uint32Array(buf(), k.manifoldLayoutPtr(), nManifold)[0];
    const recOff0 = new Uint32Array(buf(), k.geoLayoutPtr(), nGeo)[0];
    // The manifold region anchors above the fat-AABB region, geometry above that.
    expect(fatBase0).toBeLessThan(dirOff0);
    expect(dirOff0).toBeLessThan(recOff0);

    // Markers in the fat-AABB column (shape 5), a live manifold directory record, and a geometry pool.
    const markFat = 0xfa7a_0001;
    const markDir = 0xd1c0_0002;
    const markRec = 0x9ec0_0003;
    new Uint32Array(buf(), fatBase0, 16 * fatStride)[5 * fatStride] = markFat;
    new Uint32Array(buf(), dirOff0, DIR_STRIDE * 8)[3] = markDir;
    new Uint32Array(buf(), recOff0, 2 * 12)[7] = markRec;

    // Grow the fat-AABB region past its capacity, forcing it to relocate the regions above it.
    k.reserveFatAabb(8192);

    const fatBase1 = new Uint32Array(buf(), k.fatAabbLayoutPtr(), 1)[0];
    const dirOff1 = new Uint32Array(buf(), k.manifoldLayoutPtr(), nManifold)[0];
    const recOff1 = new Uint32Array(buf(), k.geoLayoutPtr(), nGeo)[0];
    const delta = dirOff1 - dirOff0;
    expect(delta).toBeGreaterThan(0);
    // The fat-AABB column is base-anchored, so its own bytes never move on its own grow.
    expect(fatBase1).toBe(fatBase0);
    expect(recOff1).toBe(recOff0 + delta);
    expect(new Uint32Array(buf(), fatBase1, 8192 * fatStride)[5 * fatStride]).toBe(markFat);
    expect(new Uint32Array(buf(), dirOff1, DIR_STRIDE * 8)[3]).toBe(markDir);
    expect(new Uint32Array(buf(), recOff1, 2 * 12)[7]).toBe(markRec);

    // Grow the body region below: the fat-AABB region shifts up with everything above it, and its marker
    // survives at the rebased offset (the body-grow memmove covers the fat-AABB span too).
    k.reserveBodies(16384);
    const fatBase2 = new Uint32Array(buf(), k.fatAabbLayoutPtr(), 1)[0];
    const dirOff2 = new Uint32Array(buf(), k.manifoldLayoutPtr(), nManifold)[0];
    const bodyDelta = fatBase2 - fatBase1;
    expect(bodyDelta).toBeGreaterThan(0);
    expect(dirOff2).toBe(dirOff1 + bodyDelta);
    expect(new Uint32Array(buf(), fatBase2, 8192 * fatStride)[5 * fatStride]).toBe(markFat);
    expect(new Uint32Array(buf(), dirOff2, DIR_STRIDE * 8)[3]).toBe(markDir);
});

// Broad-region anchor regression (3d): reserveBroad must anchor the broad region at the *raw* manifold
// top, not an align16'd copy of it. The manifold top is only 4-aligned and its residue mod 16 shifts as
// the manifold caps grow (DIR_STRIDE*4 ≡ 4 mod 16), so an align16'd base desyncs the preserve-copy — a
// manifold grow relocates the broad region by a 4-but-not-16-aligned delta (base + memmoved bytes both
// shift by that delta), then the next reserveBroad recomputes an align16'd base that no longer matches
// the relocated static-tree pool (TREE_S), which the copy loop skips as "anchor-fixed", silently
// orphaning it. Drive that exact sequence and assert the static pool's bytes survive. Wasm-only, like the
// relocation gates above; runs among them (its reserve grows perturb the singleton region).
test("reserveBroad keeps the static tree pool anchored across a manifold grow then broad grow", async () => {
    await init({ threads: 0 });
    const k = kernel();
    const buf = () => k.memory.buffer;
    const treeStride = 12; // u32 slots per tree node (broad.rs TREE_STRIDE)
    const nBroad = 6; // [TREE_S, TREE_K, TREE_D, KEY_HI, KEY_LO, HASHES]
    const treeS = () => new Uint32Array(buf(), k.broadLayoutPtr(), nBroad)[0];
    // The kernel singleton is shared across test files (bun runs them in one process), so read the
    // resident caps and grow above them rather than assuming absolute sizes. A large manifold cap keeps
    // the manifold grow below monotone (a shrink under the resident broad region is a different hazard).
    const capOf = (i: number) => k.broadTreeCap(i);

    // Manifold region large, broad region above it (growing every column past the current high-water so
    // it actually reserves), a geometry region above that so the relocation memmoves real data.
    k.reserveManifolds(8192, 8192);
    const capS = Math.max(capOf(0), 8);
    k.reserveBroad(
        capS,
        Math.max(capOf(1), 8),
        Math.max(capOf(2), 8),
        Math.max(k.broadSetCap(), 64),
    );
    k.reserveGeometry(4, 8, 8, 8);

    // Sentinels spanning a static-tree node's slots (u32 markers dodge f32 NaN canonicalization).
    const slots = [0, 5, 11];
    const marks = [0x57a7_0000, 0x57a7_0005, 0x57a7_000b];
    const treeS0 = treeS();
    {
        const pool = new Uint32Array(buf(), treeS0, capS * treeStride);
        for (let i = 0; i < slots.length; ++i) pool[slots[i]] = marks[i];
    }

    // Grow the manifold region by one contact (DIR_STRIDE*4 = 148 bytes ≡ 4 mod 16), forcing a
    // non-16-aligned relocation of the broad region above it.
    k.reserveManifolds(8193, 8192);

    // The relocate memmove alone must preserve the sentinels at the rebased TREE_S.
    const treeS1 = treeS();
    expect(treeS1).toBeGreaterThan(treeS0);
    {
        const pool = new Uint32Array(buf(), treeS1, capS * treeStride);
        for (let i = 0; i < slots.length; ++i) expect(pool[slots[i]]).toBe(marks[i]);
    }

    // Grow a broad tree pool (dynamic, above its current cap so it always grows). Pre-fix this recomputes
    // an align16'd base != treeS1 and the preserve-copy skips TREE_S, orphaning the static pool; post-fix
    // the raw base == treeS1, so TREE_S is genuinely anchor-fixed and the sentinels survive.
    expect(k.reserveBroad(capOf(0), capOf(1), capOf(2) + 8, k.broadSetCap())).not.toBe(0);
    const treeS2 = treeS();
    {
        const pool = new Uint32Array(buf(), treeS2, capS * treeStride);
        for (let i = 0; i < slots.length; ++i) expect(pool[slots[i]]).toBe(marks[i]);
    }
});

// The threading resolver's decision table. Pure over its host seam, so the
// browser branches are tested by faking the predicate rather than a real cross-origin-isolated page.
const STANDALONE: Host = { browser: false, shared: true };
const STANDALONE_NO_SAB: Host = { browser: false, shared: false };
const BROWSER_ISOLATED: Host = { browser: true, shared: true };
const BROWSER_BLOCKED: Host = { browser: true, shared: false };

function captureLog(fn: () => void): string[] {
    const logs: string[] = [];
    const orig = console.log;
    console.log = (...a: unknown[]) => {
        logs.push(a.map(String).join(" "));
    };
    try {
        fn();
    } finally {
        console.log = orig;
    }
    return logs;
}

test("resolver: standalone always multithreads at the auto count", () => {
    expect(resolve(undefined, STANDALONE)).toEqual({ want: AUTO_THREADS, warn: false });
    expect(AUTO_THREADS).toBe(4);
});

test("resolver: a cross-origin-isolated browser multithreads at the auto count", () => {
    expect(resolve(undefined, BROWSER_ISOLATED)).toEqual({ want: AUTO_THREADS, warn: false });
});

test("resolver: a non-isolated browser runs single-thread and warns", () => {
    expect(resolve(undefined, BROWSER_BLOCKED)).toEqual({ want: 0, warn: true });
});

test("resolver: standalone without SharedArrayBuffer runs single-thread, no warning", () => {
    // Not a browser, so no COOP/COEP log — that hint is only actionable on a page.
    expect(resolve(undefined, STANDALONE_NO_SAB)).toEqual({ want: 0, warn: false });
});

test("resolver escape: threads 0 forces single-thread with no warning", () => {
    expect(resolve(0, STANDALONE)).toEqual({ want: 0, warn: false });
    expect(resolve(0, BROWSER_ISOLATED)).toEqual({ want: 0, warn: false });
    // even where a warning would otherwise fire — the caller opted out deliberately.
    expect(resolve(0, BROWSER_BLOCKED)).toEqual({ want: 0, warn: false });
});

test("resolver escape: threads n overrides the auto count", () => {
    expect(resolve(8, STANDALONE)).toEqual({ want: 8, warn: false });
    expect(resolve(2, BROWSER_ISOLATED)).toEqual({ want: 2, warn: false });
    // an explicit count still can't beat a blocked browser — single-thread, and it still warns.
    expect(resolve(2, BROWSER_BLOCKED)).toEqual({ want: 0, warn: true });
});

test("resolver escape: an over-large count clamps to the link bound at the pool", () => {
    // The resolver passes the request through; the shadow stack caps it (pool.ts). 3 MiB ⇒ 7 workers ⇒ 8.
    const want = resolve(100, STANDALONE).want;
    expect(want).toBe(100);
    expect(Math.min(want, 1 + maxWorkers(SHARED_STACK_SIZE))).toBe(8);
});

test("announce logs the COOP/COEP hint exactly once, only when a browser blocked threads", () => {
    const blocked = captureLog(() => announce(resolve(undefined, BROWSER_BLOCKED)));
    expect(blocked.length).toBe(1);
    expect(blocked[0]).toContain("COOP");
    expect(blocked[0]).toContain("COEP");
    expect(blocked[0]).toBe(COOP_COEP_HINT);

    // no log on any path that got what it asked for, and none on the explicit opt-out.
    expect(captureLog(() => announce(resolve(undefined, STANDALONE)))).toEqual([]);
    expect(captureLog(() => announce(resolve(undefined, BROWSER_ISOLATED)))).toEqual([]);
    expect(captureLog(() => announce(resolve(0, BROWSER_BLOCKED)))).toEqual([]);
});

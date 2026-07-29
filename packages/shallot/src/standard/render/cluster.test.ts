import { describe, expect, test } from "bun:test";
import { body, flat, integerDiscipline } from "../../../tests/wgsl";
import { State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { Camera, CameraMode } from "./camera";
import {
    CLUSTER_COUNT,
    CLUSTER_X,
    CLUSTER_Y,
    CLUSTER_Z,
    type ClusterView,
    clusterAabb,
    clusterCell,
    clusterCoord,
    clusterIndex,
    clusterView,
    gridWgsl,
    lightClusters,
    lightCullWgsl,
    sliceDepth,
    zSlice,
} from "./cluster";

const persp: ClusterView = { perspective: true, halfW: 0.7, halfH: 0.5, near: 0.5, far: 200 };
const ortho: ClusterView = { perspective: false, halfW: 8, halfH: 5, near: 0.1, far: 50 };

describe("clusterView", () => {
    test("maps Camera fields per mode — fov° → tan half-extents, size absolute", () => {
        clear();
        register("Camera", Camera);
        const state = new State();
        const eid = state.create();
        state.add(eid, Camera);
        Camera.fov.set(eid, 90);
        Camera.near.set(eid, 0.5);
        Camera.far.set(eid, 100);
        Camera.size.set(eid, 5);

        const p = clusterView(eid, 2);
        expect(p.perspective).toBe(true);
        expect(p.halfH).toBeCloseTo(1, 6); // tan(45°)
        expect(p.halfW).toBeCloseTo(2, 6); // aspect-widened
        expect(p.near).toBe(0.5);
        expect(p.far).toBe(100);

        Camera.mode.set(eid, CameraMode.Orthographic);
        const o = clusterView(eid, 2);
        expect(o.perspective).toBe(false);
        expect(o.halfH).toBe(5); // size, depth-independent
        expect(o.halfW).toBe(10);
    });
});

describe("cluster index", () => {
    test("linearization roundtrips for every cluster", () => {
        for (let i = 0; i < CLUSTER_COUNT; i++) {
            const { x, y, z } = clusterCoord(i);
            expect(clusterIndex(x, y, z)).toBe(i);
            expect(x).toBeGreaterThanOrEqual(0);
            expect(x).toBeLessThan(CLUSTER_X);
            expect(y).toBeGreaterThanOrEqual(0);
            expect(y).toBeLessThan(CLUSTER_Y);
            expect(z).toBeGreaterThanOrEqual(0);
            expect(z).toBeLessThan(CLUSTER_Z);
        }
    });
});

describe("z slicing", () => {
    test("slice boundaries land on near and far", () => {
        expect(sliceDepth(persp, 0)).toBeCloseTo(persp.near, 6);
        expect(sliceDepth(persp, CLUSTER_Z)).toBeCloseTo(persp.far, 4);
    });

    test("viewZ → slice → depth range contains viewZ", () => {
        // geometric sampling covers every slice; linear would crowd the far end
        for (let i = 0; i <= 200; i++) {
            const viewZ = persp.near * (persp.far / persp.near) ** (i / 200) * 0.9999 + 1e-6;
            if (viewZ < persp.near || viewZ >= persp.far) continue;
            const s = zSlice(persp, viewZ);
            expect(sliceDepth(persp, s)).toBeLessThanOrEqual(viewZ * (1 + 1e-12));
            expect(sliceDepth(persp, s + 1)).toBeGreaterThan(viewZ);
        }
    });

    test("clamps outside the depth range", () => {
        expect(zSlice(persp, persp.near / 2)).toBe(0);
        expect(zSlice(persp, persp.far * 2)).toBe(CLUSTER_Z - 1);
    });
});

describe("cluster AABBs", () => {
    test("orthographic neighbors share faces — no gaps, no overlap", () => {
        for (let y = 0; y < CLUSTER_Y; y++) {
            for (let x = 0; x < CLUSTER_X; x++) {
                for (let z = 0; z < CLUSTER_Z; z++) {
                    const a = clusterAabb(ortho, x, y, z);
                    if (x + 1 < CLUSTER_X) {
                        expect(clusterAabb(ortho, x + 1, y, z).min[0]).toBeCloseTo(a.max[0], 10);
                    }
                    if (y + 1 < CLUSTER_Y) {
                        expect(clusterAabb(ortho, x, y + 1, z).min[1]).toBeCloseTo(a.max[1], 10);
                    }
                    // deeper slice is more negative in view space
                    if (z + 1 < CLUSTER_Z) {
                        expect(clusterAabb(ortho, x, y, z + 1).max[2]).toBeCloseTo(a.min[2], 10);
                    }
                }
            }
        }
    });

    // a perspective cluster's AABB is the conservative box around a slanted frustum
    // cell, so adjacent boxes overlap in x/y by construction; the partition property
    // is that the shared frustum edge — `edgeNdc · halfW · depth` at both slice
    // boundary depths — lies inside both boxes (no gap), and z faces stay exact
    test("perspective neighbors cover the shared frustum face — no gaps", () => {
        for (let y = 0; y < CLUSTER_Y; y++) {
            for (let x = 0; x < CLUSTER_X; x++) {
                for (let z = 0; z < CLUSTER_Z; z++) {
                    const a = clusterAabb(persp, x, y, z);
                    const depths = [sliceDepth(persp, z), sliceDepth(persp, z + 1)];
                    if (x + 1 < CLUSTER_X) {
                        const b = clusterAabb(persp, x + 1, y, z);
                        const edge = (-1 + (2 * (x + 1)) / CLUSTER_X) * persp.halfW;
                        for (const d of depths) {
                            expect(edge * d).toBeLessThanOrEqual(a.max[0] + 1e-12);
                            expect(edge * d).toBeGreaterThanOrEqual(b.min[0] - 1e-12);
                        }
                    }
                    if (y + 1 < CLUSTER_Y) {
                        const b = clusterAabb(persp, x, y + 1, z);
                        const edge = (-1 + (2 * (y + 1)) / CLUSTER_Y) * persp.halfH;
                        for (const d of depths) {
                            expect(edge * d).toBeLessThanOrEqual(a.max[1] + 1e-12);
                            expect(edge * d).toBeGreaterThanOrEqual(b.min[1] - 1e-12);
                        }
                    }
                    if (z + 1 < CLUSTER_Z) {
                        expect(clusterAabb(persp, x, y, z + 1).max[2]).toBeCloseTo(a.min[2], 10);
                    }
                }
            }
        }
    });

    test("grid spans the frustum exactly", () => {
        const first = clusterAabb(persp, 0, 0, 0);
        const last = clusterAabb(persp, CLUSTER_X - 1, CLUSTER_Y - 1, CLUSTER_Z - 1);
        expect(first.max[2]).toBeCloseTo(-persp.near, 6);
        expect(last.min[2]).toBeCloseTo(-persp.far, 4);
        // widest extent is at the far boundary of the deepest slice
        expect(clusterAabb(persp, 0, 0, CLUSTER_Z - 1).min[0]).toBeCloseTo(
            -persp.halfW * persp.far,
            4,
        );
        expect(last.max[1]).toBeCloseTo(persp.halfH * persp.far, 4);
    });

    test("perspective AABB widens with depth; orthographic doesn't", () => {
        const shallow = clusterAabb(persp, 0, 0, 0);
        const deep = clusterAabb(persp, 0, 0, CLUSTER_Z - 1);
        expect(deep.min[0]).toBeLessThan(shallow.min[0]);

        const oShallow = clusterAabb(ortho, 0, 0, 0);
        const oDeep = clusterAabb(ortho, 0, 0, CLUSTER_Z - 1);
        expect(oDeep.min[0]).toBe(oShallow.min[0]);
        expect(oDeep.max[1]).toBe(oShallow.max[1]);
    });
});

describe("lightClusters", () => {
    test("a sphere inside one cluster hits that cluster", () => {
        // center of cluster (8, 4, 12)'s AABB, radius far smaller than the box
        const { min, max } = clusterAabb(persp, 8, 4, 12);
        const c: [number, number, number] = [
            (min[0] + max[0]) / 2,
            (min[1] + max[1]) / 2,
            (min[2] + max[2]) / 2,
        ];
        const r = Math.min(max[0] - min[0], max[1] - min[1], max[2] - min[2]) / 4;
        const hit = lightClusters(persp, c, r);
        expect(hit).toContain(clusterIndex(8, 4, 12));
        // every hit cluster genuinely touches the sphere: re-verify by distance
        for (const idx of hit) {
            const { x, y, z } = clusterCoord(idx);
            const a = clusterAabb(persp, x, y, z);
            let d = 0;
            for (let i = 0; i < 3; i++) {
                const p = Math.min(Math.max(c[i], a.min[i]), a.max[i]);
                d += (p - c[i]) ** 2;
            }
            expect(d).toBeLessThanOrEqual(r * r);
        }
    });

    test("a sphere covering the whole frustum hits every cluster", () => {
        const hit = lightClusters(ortho, [0, 0, -25], 1e4);
        expect(hit.length).toBe(CLUSTER_COUNT);
    });

    test("a sphere behind the camera or past far hits nothing", () => {
        expect(lightClusters(persp, [0, 0, 10], 5).length).toBe(0);
        expect(lightClusters(persp, [0, 0, -500], 5).length).toBe(0);
    });

    test("growing the range only adds clusters — monotone", () => {
        const c: [number, number, number] = [0.2, -0.1, -3];
        let prev = new Set<number>();
        for (const r of [0.5, 1, 2, 4, 8]) {
            const hit = new Set(lightClusters(persp, c, r));
            for (const idx of prev) expect(hit.has(idx)).toBe(true);
            prev = hit;
        }
    });
});

// The GPU twins' device-free structural seam. The oracles above are the numeric truth and the gym
// `render` scenario pins the twins to them on a real device; what needs asserting without a device is
// the shape the TGSL port can get silently wrong — the grid's two integer divisions (TGSL's `/` emits
// *float* division, which would leave `x`/`y` fractional and every AABB in the wrong tile), the u32
// discipline, and the cull's uniformity-analysis scaffolding.
describe("emitted WGSL", () => {
    const light = { base: 0, mask: 8 };
    const spot = { base: 0, mask: 16 };
    const vol = { base: 65536, mask: 1 };

    test("the grid's cluster→(x, y, z) decompose divides as integers, never as floats", () => {
        const src = gridWgsl();
        expect(src).toContain(`let xy = idiv(cluster, ${CLUSTER_Z}u);`);
        expect(src).toContain(`let y = idiv(xy, ${CLUSTER_X}u);`);
        expect(src).toContain(`let z = (cluster % ${CLUSTER_Z}u);`);
        expect(src).toContain(`let x = (xy % ${CLUSTER_X}u);`);
        // the log-Z depths are the only real division, and both operands are f32 there
        expect(src).toContain("pow((far / near)");
    });

    test("the grid folds the whole grid geometry to literals — no uniform but the per-view params", () => {
        const src = gridWgsl();
        expect(src).toContain(`cluster >= ${CLUSTER_COUNT}u`);
        expect(src).toContain(`(slot * ${CLUSTER_COUNT}u)`);
        expect(src).not.toContain("var<uniform>");
        integerDiscipline(src);
    });

    test("the compact pass reuses the shared sRGB and transform codecs, not a local copy", () => {
        const { compact } = lightCullWgsl(light, spot, vol);
        expect(compact).toContain("fn srgbToLinear1(");
        expect(compact).not.toContain("fn srgb1(");
        expect(compact).toContain("fn xformQuat(");
        expect(compact).toContain("fn octEncodeNormal(");
        // the count word is atomic on the writer's side only
        expect(compact).toContain("count: array<atomic<u32>, 4>");
    });

    test("the compact pass folds all three membership gates to literals", () => {
        const { compact } = lightCullWgsl(light, spot, vol);
        expect(compact).toContain(`membership[(${light.base}u + eid)] & ${light.mask}u`);
        expect(compact).toContain(`membership[(${spot.base}u + eid)] & ${spot.mask}u`);
        expect(compact).toContain(`membership[(${vol.base}u + eid)] & ${vol.mask}u`);
        integerDiscipline(compact);
    });

    test("the cull's batch loop keeps its uniformity scaffolding", () => {
        const { cull } = lightCullWgsl(light, spot, vol);
        // the loop bound comes through workgroupUniformLoad, which is what makes the in-loop
        // barriers legal; out-of-range threads mask on `live`, they never return early
        expect(cull).toContain("workgroupUniformLoad");
        expect(flat(cull)).toContain("uniformLoad((&wgCount))");
        expect(cull).toContain("workgroupBarrier();");
        expect(body(cull, "@compute")).not.toContain("return;");
        expect(flat(cull)).toMatch(/var<workgroup> batch: array<vec4f, ?64>/);
        integerDiscipline(cull);
    });

    test("the cull reads the list through the non-atomic schema", () => {
        const { cull } = lightCullWgsl(light, spot, vol);
        expect(flat(cull)).toContain("var<storage, read> lights: PointLights;");
        expect(cull).toContain("count: vec4u");
        expect(cull).not.toContain("atomic<u32>, 4");
    });
});

// `clusterCell` is the froxel index the shader reads the light grid at — one function on both sides now,
// so this pins it against the CPU grid oracles rather than against a second WGSL spelling of the same math
describe("clusterCell", () => {
    const view: ClusterView = { perspective: true, halfW: 1.6, halfH: 0.9, near: 0.1, far: 200 };

    test("the froxel index agrees with the tile + zSlice oracles", () => {
        for (const fx of [0, 0.03, 0.5, 0.97, 0.999]) {
            for (const fy of [0, 0.03, 0.5, 0.97, 0.999]) {
                for (const viewZ of [0.1, 0.4, 3, 40, 199]) {
                    for (const slot of [0, 1, 3]) {
                        const tx = Math.min(Math.floor(fx * CLUSTER_X), CLUSTER_X - 1);
                        // tile (0,0) is NDC (-1,-1) — bottom-left — so the y tile flips screen y
                        const ty =
                            CLUSTER_Y - 1 - Math.min(Math.floor(fy * CLUSTER_Y), CLUSTER_Y - 1);
                        const want =
                            slot * CLUSTER_COUNT + clusterIndex(tx, ty, zSlice(view, viewZ));
                        expect(clusterCell(fx, fy, viewZ, view.near, view.far, slot)).toBe(want);
                    }
                }
            }
        }
    });

    test("the index stays inside the grid past both depth bounds", () => {
        for (const viewZ of [1e-6, 0.01, 1e6]) {
            const cell = clusterCell(0.5, 0.5, viewZ, view.near, view.far, 0);
            expect(cell).toBeGreaterThanOrEqual(0);
            expect(cell).toBeLessThan(CLUSTER_COUNT);
        }
    });
});

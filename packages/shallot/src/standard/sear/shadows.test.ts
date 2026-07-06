import { beforeEach, describe, expect, spyOn, test } from "bun:test";
import { lookAt, multiply, perspective, State } from "../../engine";
import { clear, register } from "../../engine/ecs/core";
import { Camera, CameraMode, DirectionalLight, PointLight } from "../render";
import { computeViewProj, FRUSTUM_FLOATS, frustumPlanes, Views } from "../render/core";
import { Slab } from "../slab";
import { Transform, TransformsPlugin } from "../transforms";
import {
    cascadeAtlasSize,
    cascadeComboEids,
    cascadeCount,
    cascadeFars,
    cascadeFit,
    cascadeRecvVP,
    cascadeSplits,
    cascadeTileRect,
    createPacker,
    MAX_CASCADES,
    orthoFootprintFit,
    POINT_FACES,
    type PointShadowFrame,
    packCasters,
    pointCasters,
    pointComboCount,
    pointComboEids,
    pointFace,
    pointFaceVP,
    pointFov,
    pointReceiver,
    pointTanHalf,
    pointTileRects,
    resetCascades,
    resetPointShadows,
    Shadow,
    SunShadows,
    spotBasis,
    sunCascades,
    tileTransform,
    updateCascades,
    updatePointShadows,
} from "./shadows";

const PERSP = CameraMode.Perspective;
const ORTHO = CameraMode.Orthographic;

// a perspective camera world matrix needs only its position (col 3) and look direction; the fit reads the
// Z column (local +Z) as -forward. A camera looking down -Z has identity rotation.
function cameraAt(px: number, py: number, pz: number): Float32Array {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, px, py, pz, 1]);
}

// an ortho camera at height `h` looking straight down -Y: right = +X, up = -Z, so col2 (+Z) = +Y
function topDownCam(h: number): Float32Array {
    return new Float32Array([1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, h, 0, 1]);
}

// The CSM split + per-cascade fit are pure (no GPU): the practical/PSSM split is checked by its defining
// invariants (λ=0 uniform world depth, λ=1 uniform depth ratio, the last bound = far), and the fit by the
// property that defines it — every corner of the camera's frustum slice lands inside the cascade's ortho
// box. The receiver's cascade-select + the GPU atlas render live in later sub-stages (`bun bench`).
describe("cascadeSplits", () => {
    const near = 1;
    const far = 100;
    const n = 4;

    test("the last bound is the camera far, and the bounds strictly increase within (near, far]", () => {
        const s = cascadeSplits(near, far, n, 0.5);
        expect(s.length).toBe(n);
        expect(s[n - 1]).toBeCloseTo(far, 10); // the last cascade reaches exactly the far plane
        for (let i = 0; i < n; i++) {
            expect(s[i]).toBeGreaterThan(i === 0 ? near : s[i - 1]); // monotone, first past near
            expect(s[i]).toBeLessThanOrEqual(far + 1e-9);
        }
    });

    test("λ=0 is a uniform split: equal world depth per cascade", () => {
        const s = cascadeSplits(near, far, n, 0);
        const step = (far - near) / n;
        let prev = near;
        for (const bound of s) {
            expect(bound - prev).toBeCloseTo(step, 10); // constant depth per cascade
            prev = bound;
        }
    });

    test("λ=1 is a logarithmic split: equal depth *ratio* per cascade", () => {
        const s = cascadeSplits(near, far, n, 1);
        const ratio = (far / near) ** (1 / n);
        let prev = near;
        for (const bound of s) {
            expect(bound / prev).toBeCloseTo(ratio, 10); // constant ratio per cascade
            prev = bound;
        }
    });

    test("the practical split lies between uniform and logarithmic at every bound", () => {
        const uni = cascadeSplits(near, far, n, 0);
        const log = cascadeSplits(near, far, n, 1);
        const prac = cascadeSplits(near, far, n, 0.5);
        for (let i = 0; i < n - 1; i++) {
            // log front-loads (tighter near cascades), so log[i] < practical[i] < uniform[i] for i < last
            const lo = Math.min(uni[i], log[i]);
            const hi = Math.max(uni[i], log[i]);
            expect(prac[i]).toBeGreaterThanOrEqual(lo - 1e-9);
            expect(prac[i]).toBeLessThanOrEqual(hi + 1e-9);
            expect(prac[i]).toBeCloseTo((uni[i] + log[i]) / 2, 10); // λ=0.5 is the midpoint
        }
    });
});

describe("sunCascades", () => {
    test("clamps the configured count to [1, MAX_CASCADES] and rounds", () => {
        const saved = SunShadows.cascades;
        try {
            SunShadows.cascades = 0;
            expect(sunCascades()).toBe(1); // never zero cascades
            SunShadows.cascades = MAX_CASCADES + 3;
            expect(sunCascades()).toBe(MAX_CASCADES); // capped at the reserved slot budget
            SunShadows.cascades = 2.7;
            expect(sunCascades()).toBe(3); // rounded to a whole cascade count
        } finally {
            SunShadows.cascades = saved;
        }
    });
});

describe("cascadeFit", () => {
    const res = 2048;
    const dir: [number, number, number] = [-0.3, -0.8, -0.55];

    // the camera's frustum-slice corners in world space, derived independently from the camera geometry
    // (the half-height at depth d — `d·tan(fov/2)` for perspective, the constant `size` for ortho), so the
    // test's expectation doesn't reimplement the fit it checks
    function sliceCorners(
        cam: Float32Array,
        half: (d: number) => number,
        aspect: number,
        nearD: number,
        farD: number,
    ): number[][] {
        const corners: number[][] = [];
        for (const d of [nearD, farD]) {
            const hh = half(d);
            const hw = hh * aspect;
            for (const sh of [-1, 1]) {
                for (const sv of [-1, 1]) {
                    corners.push([
                        cam[12] - cam[8] * d + cam[0] * sh * hw + cam[4] * sv * hh,
                        cam[13] - cam[9] * d + cam[1] * sh * hw + cam[5] * sv * hh,
                        cam[14] - cam[10] * d + cam[2] * sh * hw + cam[6] * sv * hh,
                    ]);
                }
            }
        }
        return corners;
    }

    // each slice corner lands inside the cascade's ortho box — the box is the slice's bounding sphere
    // (radius `cover`), so projected into the light view every corner is within ±cover and in front
    function expectContains(fit: ReturnType<typeof cascadeFit>, corners: number[][]): void {
        const view = lookAt(
            fit.eye[0],
            fit.eye[1],
            fit.eye[2],
            fit.focus[0],
            fit.focus[1],
            fit.focus[2],
            fit.up[0],
            fit.up[1],
            fit.up[2],
        );
        const tol = fit.cover * 1e-4 + 1e-3;
        for (const c of corners) {
            // corner → light view space (column-major mat4 × point)
            const lx = view[0] * c[0] + view[4] * c[1] + view[8] * c[2] + view[12];
            const ly = view[1] * c[0] + view[5] * c[1] + view[9] * c[2] + view[13];
            const lz = view[2] * c[0] + view[6] * c[1] + view[10] * c[2] + view[14];
            expect(Math.abs(lx)).toBeLessThanOrEqual(fit.cover + tol);
            expect(Math.abs(ly)).toBeLessThanOrEqual(fit.cover + tol);
            expect(lz).toBeLessThan(tol); // in front of the light (looks down -Z)
        }
    }

    test("a perspective camera's box contains each frustum slice", () => {
        const cam = cameraAt(0, 0, 0); // perspective, looking down -Z
        const fov = 60;
        const aspect = 1.6;
        const half = (d: number) => d * Math.tan((fov * Math.PI) / 360);
        let nearD = 0.1;
        for (const farD of cascadeSplits(0.1, 60, 4, 0.5)) {
            const fit = cascadeFit(cam, PERSP, fov, 0, aspect, dir, nearD, farD, res, 60);
            expectContains(fit, sliceCorners(cam, half, aspect, nearD, farD));
            nearD = farD;
        }
    });

    test("an orthographic camera's box contains each slice (constant half-extents)", () => {
        const cam = topDownCam(50); // ortho, looking straight down -Y
        const size = 30;
        const aspect = 1.6;
        const half = () => size; // ortho cross-section is constant with depth
        let nearD = 1;
        for (const farD of cascadeSplits(1, 50, 4, 0.5)) {
            const fit = cascadeFit(cam, ORTHO, 0, size, aspect, dir, nearD, farD, res, 50);
            expectContains(fit, sliceCorners(cam, half, aspect, nearD, farD));
            nearD = farD;
        }
    });

    test("farther cascades get larger boxes (the CSM resolution gradient)", () => {
        const cam = cameraAt(0, 0, 0);
        const splits = cascadeSplits(0.1, 80, 4, 0.5);
        let nearD = 0.1;
        let prev = 0;
        for (const farD of splits) {
            const { cover } = cascadeFit(cam, PERSP, 60, 0, 1.6, dir, nearD, farD, res, 80);
            expect(cover).toBeGreaterThan(prev); // each cascade covers more world than the one before
            prev = cover;
            nearD = farD;
        }
    });

    test("a sub-texel camera move leaves a cascade's eye fixed (per-cascade texel snap)", () => {
        // a straight-down sun, so a camera move along x is entirely in the snap plane (⊥ the sun) — the
        // plane where shadow-map crawl happens, the one the texel snap quantizes (an angled sun would let
        // part of the move ride the depth axis the snap doesn't touch, which isn't crawl)
        const down: [number, number, number] = [0, -1, 0];
        const fit = (px: number) =>
            cascadeFit(cameraAt(px, 0, 0), PERSP, 60, 0, 1.6, down, 5, 20, res, 20);
        const a = fit(0);
        const texel = (2 * a.cover) / res;
        const b = fit(texel * 0.3); // a fraction of one shadow texel
        expect(b.eye[0]).toBeCloseTo(a.eye[0], 6);
        expect(b.eye[1]).toBeCloseTo(a.eye[1], 6);
        expect(b.eye[2]).toBeCloseTo(a.eye[2], 6);
    });
});

// The ortho path (single footprint box) + the near-plane margin every box extends toward the light. Both are
// pure (no GPU): the footprint box is checked by the property that defines it — every corner of the camera's
// visible y=0 footprint lands inside the box — and the margin by the depth/eye it produces.
describe("orthoFootprintFit + near extension", () => {
    const res = 2048;
    const dir: [number, number, number] = [-0.3, -0.8, -0.55];
    const norm = (v: number[]): number[] => {
        const l = Math.hypot(v[0], v[1], v[2]) || 1;
        return [v[0] / l, v[1] / l, v[2] / l];
    };
    const cross = (a: number[], b: number[]): number[] => [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];

    // an ortho camera world matrix (col0 right, col1 up, col2 +Z = −forward) from a position + look direction,
    // the basis `aim`/`lookAt` produce — so the fit reads a real angled-down ortho pose
    function orthoCam(pos: number[], fwd: number[]): Float32Array {
        const f = norm(fwd);
        const up0 = Math.abs(f[1]) > 0.99 ? [0, 0, 1] : [0, 1, 0];
        const r = norm(cross(f, up0));
        const u = cross(r, f);
        return new Float32Array([
            r[0],
            r[1],
            r[2],
            0,
            u[0],
            u[1],
            u[2],
            0,
            -f[0],
            -f[1],
            -f[2],
            0,
            pos[0],
            pos[1],
            pos[2],
            1,
        ]);
    }

    // the 4 corners of the camera's visible ground (y=0) footprint: an ortho camera's rays are parallel along
    // forward, so each screen-corner origin (pos ± right·hw ± up·hh) slides along forward to y=0. Derived from
    // the camera geometry independently of the fit it checks.
    function groundCorners(cam: Float32Array, size: number, aspect: number): number[][] {
        const r = [cam[0], cam[1], cam[2]];
        const u = [cam[4], cam[5], cam[6]];
        const f = [-cam[8], -cam[9], -cam[10]];
        const p = [cam[12], cam[13], cam[14]];
        const hw = size * aspect;
        const hh = size;
        const out: number[][] = [];
        for (const sh of [-1, 1]) {
            for (const sv of [-1, 1]) {
                const ox = p[0] + r[0] * sh * hw + u[0] * sv * hh;
                const oy = p[1] + r[1] * sh * hw + u[1] * sv * hh;
                const oz = p[2] + r[2] * sh * hw + u[2] * sv * hh;
                const t = -oy / f[1];
                out.push([ox + f[0] * t, oy + f[1] * t, oz + f[2] * t]);
            }
        }
        return out;
    }

    function expectInBox(fit: ReturnType<typeof orthoFootprintFit>, corners: number[][]): void {
        const view = lookAt(
            fit.eye[0],
            fit.eye[1],
            fit.eye[2],
            fit.focus[0],
            fit.focus[1],
            fit.focus[2],
            fit.up[0],
            fit.up[1],
            fit.up[2],
        );
        const tol = fit.cover * 1e-4 + 1e-3;
        for (const c of corners) {
            const lx = view[0] * c[0] + view[4] * c[1] + view[8] * c[2] + view[12];
            const ly = view[1] * c[0] + view[5] * c[1] + view[9] * c[2] + view[13];
            const lz = view[2] * c[0] + view[6] * c[1] + view[10] * c[2] + view[14];
            expect(Math.abs(lx)).toBeLessThanOrEqual(fit.cover + tol);
            expect(Math.abs(ly)).toBeLessThanOrEqual(fit.cover + tol);
            expect(lz).toBeLessThan(tol); // in front of the light
        }
    }

    test("an angled-down ortho camera's box contains its visible ground footprint", () => {
        const cam = orthoCam([0, 40, 40], [0, -1, -1]); // 45° down, posed far from the scene
        const size = 20;
        const aspect = 1.6;
        const fit = orthoFootprintFit(cam, size, aspect, dir, 50, res, 50);
        expectInBox(fit, groundCorners(cam, size, aspect));
    });

    test("a non-down ortho camera falls back to a forward-distance box", () => {
        const cam = orthoCam([0, 5, 0], [0, 0, -1]); // horizontal — no convergent ground footprint
        const fit = orthoFootprintFit(cam, 20, 1.6, dir, 50, res, 0);
        expect(fit.cover).toBeCloseTo(50, 4); // cover = distance
    });

    test("the near margin extends the box depth and pushes the eye toward the light, snap unchanged", () => {
        const d = norm([-0.3, -0.8, -0.55]) as [number, number, number];
        const cam = cameraAt(0, 0, 0);
        const a = cascadeFit(cam, PERSP, 60, 0, 1.6, d, 5, 20, res, 0);
        const b = cascadeFit(cam, PERSP, 60, 0, 1.6, d, 5, 20, res, 10);
        expect(b.cover).toBeCloseTo(a.cover, 6); // XY extent is unchanged by the depth margin
        expect(a.depth).toBeCloseTo(2 * a.cover, 5);
        expect(b.depth).toBeCloseTo(2 * b.cover + 10, 5);
        // the eye moves exactly the margin delta along −dir (toward the light); the texel snap is margin-invariant
        expect(b.eye[0]).toBeCloseTo(a.eye[0] - d[0] * 10, 4);
        expect(b.eye[1]).toBeCloseTo(a.eye[1] - d[1] * 10, 4);
        expect(b.eye[2]).toBeCloseTo(a.eye[2] - d[2] * 10, 4);
    });
});

// The cascade atlas tiling is a pure fixed grid (cascades are equal-resolution — no importance packer), so
// the per-cascade tile rect + the atlas side are checked directly.
describe("cascade atlas tiling", () => {
    test("n=1 fills the whole atlas", () => {
        expect(cascadeTileRect(0, 1)).toEqual([0, 0, 1, 1]);
    });

    test("the tiles partition the atlas without overlap", () => {
        for (const n of [2, 3, 4]) {
            const rects = Array.from({ length: n }, (_, k) => cascadeTileRect(k, n));
            // every tile is inside the atlas
            for (const [u0, v0, du, dv] of rects) {
                expect(u0).toBeGreaterThanOrEqual(0);
                expect(v0).toBeGreaterThanOrEqual(0);
                expect(u0 + du).toBeLessThanOrEqual(1 + 1e-9);
                expect(v0 + dv).toBeLessThanOrEqual(1 + 1e-9);
            }
            // no two tiles overlap (axis-separated, since equal-size grid cells)
            for (let i = 0; i < n; i++)
                for (let j = i + 1; j < n; j++) {
                    const [ax, ay, aw, ah] = rects[i];
                    const [bx, by, bw, bh] = rects[j];
                    const disjoint =
                        ax + aw <= bx + 1e-9 ||
                        bx + bw <= ax + 1e-9 ||
                        ay + ah <= by + 1e-9 ||
                        by + bh <= ay + 1e-9;
                    expect(disjoint).toBe(true);
                }
        }
    });

    test("the atlas side is ceil(√n)·resolution, pow2-clamped", () => {
        expect(cascadeAtlasSize(2048, 1)).toBe(2048);
        expect(cascadeAtlasSize(2048, 2)).toBe(4096);
        expect(cascadeAtlasSize(2048, 4)).toBe(4096); // 2·2048, clamped at the 4096 ceiling
        expect(cascadeAtlasSize(1024, 4)).toBe(2048);
        // pow2 snap + the 256 floor
        expect(cascadeAtlasSize(100, 1)).toBe(256);
    });
});

// updateCascades poses one pooled depth-only ortho camera per cascade (the per-cascade cull) and fills the
// dense per-cascade viewProjs the atlas + receiver read. The integration twin of the pure cascadeFit/Splits
// tests: it exercises the real pool + pose + fold, and pins each cascade camera's `computeViewProj` (the
// frustum the Part pack culls into) to the unfolded receiver viewProj — the cull-vs-render consistency the
// gym `render` scenario then pins end-to-end against the GPU survivor counts.
describe("updateCascades", () => {
    let state: State;
    let main: number;

    beforeEach(() => {
        clear();
        resetCascades();
        Views.clear(); // cascade cameras attachView'd by a prior test would collide on recycled eids
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Camera", Camera);
        register("DirectionalLight", DirectionalLight);
        register("Shadow", Shadow);
        Slab.collect();
        state = new State();
        main = state.create();
        state.add(main, Transform);
        state.add(main, Camera);
        Camera.mode.set(main, CameraMode.Perspective);
        Camera.fov.set(main, 60);
        Camera.near.set(main, 0.1);
        Transform.pos.set(main, 0, 8, 12, 1); // an off-origin, tilted view so the cascades fit a real frustum
    });

    function sun(dx: number, dy: number, dz: number, distance = 50): number {
        const eid = state.create();
        state.add(eid, DirectionalLight);
        state.add(eid, Shadow);
        DirectionalLight.direction.set(eid, dx, dy, dz, 0);
        Shadow.distance.set(eid, distance);
        return eid;
    }

    test("no directional Shadow → no cascades", () => {
        // a directional light WITHOUT Shadow casts nothing
        const eid = state.create();
        state.add(eid, DirectionalLight);
        updateCascades(state, main);
        expect(cascadeCount()).toBe(0);
        expect(cascadeComboEids().length).toBe(0);
    });

    test("a casting sun poses one cull camera per cascade", () => {
        sun(-0.4, -0.8, -0.45);
        updateCascades(state, main);
        const n = sunCascades();
        expect(cascadeCount()).toBe(n);
        expect(cascadeComboEids().length).toBe(n);
    });

    test("an orthographic main camera collapses to a single shadow box", () => {
        // uniform texel density → one box, not N depth slices (the ortho regression: the slice fit never
        // reaches an ortho camera's visible ground). Pitched 45° down so the footprint branch runs.
        Camera.mode.set(main, CameraMode.Orthographic);
        Camera.size.set(main, 20);
        Transform.pos.set(main, 0, 30, 30, 1);
        Transform.rot.set(main, -0.38268, 0, 0, 0.92388); // −45° pitch about X → forward (0, −0.707, −0.707)
        sun(-0.4, -0.8, -0.45);
        updateCascades(state, main);
        expect(cascadeCount()).toBe(1);
        expect(cascadeComboEids().length).toBe(1);
        // its far-bound is a sentinel beyond any visible fragment, so the receiver always selects it
        expect(cascadeFars()[0]).toBeGreaterThan(1e8);
        // cull-vs-render consistency still holds for the single ortho box
        const got = new Float32Array(16);
        computeViewProj(cascadeComboEids()[0], 1, got);
        const recv = cascadeRecvVP();
        for (let k = 0; k < 16; k++) expect(got[k]).toBeCloseTo(recv[k], 3);
    });

    test("the far-bounds are monotone, last = Shadow.distance", () => {
        sun(-0.3, -0.9, -0.3, 60);
        updateCascades(state, main);
        const n = sunCascades();
        const fars = cascadeFars();
        for (let i = 1; i < n; i++) expect(fars[i]).toBeGreaterThan(fars[i - 1]);
        expect(fars[n - 1]).toBeCloseTo(60, 4);
    });

    // the cull-vs-render consistency: render's `computeViewProj` of each posed cascade camera (the frustum the
    // pack culls casters into) must reproduce the unfolded receiver viewProj (`aim → compose → invert` equals
    // the direct ortho × lookAt the fold builds on)
    test("each cascade camera's computeViewProj matches its receiver viewProj", () => {
        sun(-0.4, -0.8, -0.45);
        updateCascades(state, main);
        const n = sunCascades();
        const recv = cascadeRecvVP();
        const got = new Float32Array(16);
        for (let i = 0; i < n; i++) {
            computeViewProj(cascadeComboEids()[i], 1, got);
            for (let k = 0; k < 16; k++) expect(got[k]).toBeCloseTo(recv[i * 16 + k], 3); // f32 aim→compose→invert vs direct lookAt
        }
    });
});

// The point-shadow atlas math is pure: the face selection (the WGSL twin is generated from the same
// POINT_FACES table), the per-tile face projection (pinned to the engine's real `perspective()` at the
// tile's widened FOV so the FS's analytic receiver depth compares against exactly what the atlas render
// wrote), and the importance allocator (the buddy packer + the per-(caster, face) rects). GPU compilation
// + the atlas render live in `bun bench`; the look is the render gym scenario (the pointShadow / spotShadow modes).
describe("point shadows", () => {
    // a deterministic golden-angle sphere sprinkle — covers every face, grazes the boundaries
    const dirs: [number, number, number][] = [];
    for (let i = 0; i < 200; i++) {
        const a = i * 0.61803 * Math.PI * 2;
        const b = Math.acos(1 - (2 * (i + 0.5)) / 200);
        dirs.push([Math.sin(b) * Math.cos(a), Math.cos(b), Math.sin(b) * Math.sin(a)]);
    }

    test("face coordinates reconstruct the direction (roundtrip)", () => {
        for (const d of dirs) {
            const { face, s, t, z } = pointFace(d);
            const f = POINT_FACES[face];
            for (let i = 0; i < 3; i++) {
                const r = f.right[i] * s + f.up[i] * t + f.fwd[i] * z;
                expect(r).toBeCloseTo(d[i], 10); // orthonormal axis basis — f64-exact decomposition
            }
            // the dominant axis is forward: z bounds |s|, |t| (a face never sees past its 45° edge)
            expect(z).toBeGreaterThan(0);
            expect(Math.abs(s)).toBeLessThanOrEqual(z + 1e-12);
            expect(Math.abs(t)).toBeLessThanOrEqual(z + 1e-12);
        }
    });

    test("every face is selected by its axis direction", () => {
        const axes: [number, number, number][] = [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ];
        for (const [i, d] of axes.entries()) expect(pointFace(d).face).toBe(i);
    });

    // the FS's receiver depth + ndc must equal what a tile's face camera wrote into the atlas — pin both
    // formulas to the engine's real perspective() at that tile's widened pointFov(tilePx). The tanHalf
    // scales with the tile's pixel size, so the seam margin is a constant number of texels for every tile.
    test("the analytic face projection matches perspective(pointFov(tilePx))", () => {
        const near = 0.006;
        const far = 6;
        for (const tilePx of [64, 256, 512, 2048]) {
            const tanHalf = pointTanHalf(tilePx);
            const proj = perspective(pointFov(tilePx), 1, near, far);
            for (const zv of [near, 0.01, 0.5, 2.3, far]) {
                for (const sv of [0, -0.3 * zv, 0.9 * zv]) {
                    // a view-space point (s, t, -z): camera looks down -Z
                    const cx = proj[0] * sv;
                    const cz = proj[10] * -zv + proj[14];
                    const cw = zv; // proj[11] = -1
                    expect(sv / (zv * tanHalf)).toBeCloseTo(cx / cw, 6);
                    expect((near * (far - zv)) / (zv * (far - near))).toBeCloseTo(cz / cw, 6);
                }
            }
        }
    });

    // the clip-space fold: tileVP = tileTransform(rect) · faceVP must land a face-space point at the exact
    // atlas ndc + depth pointShadowOf samples (rect.xy + ndc remap, y-flipped). This is what lets the VS emit
    // `tileVP · world` with NO manual divide — the load-bearing oracle that catches a y-flip-on-scale or a
    // dropped term in `D` before the GPU. Each face owns an independent atlas-UV rect now (no slot/cell math).
    test("tileTransform folds the tile remap into clip space (matches the receiver's atlas uv)", () => {
        const near = 0.02;
        const far = 20;
        const tilePx = 512;
        const tanHalf = pointTanHalf(tilePx);
        const faceProj = perspective(pointFov(tilePx), 1, near, far);
        const light = [10, 5, -3];
        // an arbitrary off-origin tile (a non-zero u0/v0 exercises both offset terms in `D`)
        const rect: [number, number, number, number] = [0.25, 0.5, tilePx / 2048, tilePx / 2048];
        for (let f = 0; f < 6; f++) {
            const { fwd, up, right } = POINT_FACES[f];
            const faceView = lookAt(
                light[0],
                light[1],
                light[2],
                light[0] + fwd[0],
                light[1] + fwd[1],
                light[2] + fwd[2],
                up[0],
                up[1],
                up[2],
            );
            const tileVP = multiply(tileTransform(rect), multiply(faceProj, faceView));
            const [u0, v0, du, dv] = rect;
            for (const z of [near + 0.01, 1.7, far - 0.5]) {
                for (const [s, t] of [
                    [0, 0],
                    [0.6 * z, -0.4 * z],
                ]) {
                    const w = [
                        light[0] + fwd[0] * z + right[0] * s + up[0] * t,
                        light[1] + fwd[1] * z + right[1] * s + up[1] * t,
                        light[2] + fwd[2] * z + right[2] * s + up[2] * t,
                    ];
                    const cx4 = tileVP[0] * w[0] + tileVP[4] * w[1] + tileVP[8] * w[2] + tileVP[12];
                    const cy4 = tileVP[1] * w[0] + tileVP[5] * w[1] + tileVP[9] * w[2] + tileVP[13];
                    const cz4 =
                        tileVP[2] * w[0] + tileVP[6] * w[1] + tileVP[10] * w[2] + tileVP[14];
                    const cw4 =
                        tileVP[3] * w[0] + tileVP[7] * w[1] + tileVP[11] * w[2] + tileVP[15];
                    // the atlas uv the receiver reconstructs, → atlas ndc (y flips twice: uv then ndc)
                    const fnx = s / (z * tanHalf);
                    const fny = t / (z * tanHalf);
                    const uvx = u0 + (fnx * 0.5 + 0.5) * du;
                    const uvy = v0 + (0.5 - fny * 0.5) * dv;
                    // digit 4 (5e-5): two chained f32 mat4 multiplies (D · proj · view) accumulate more
                    // than the single proj·view the bare-projection test pins at digit 5
                    expect(cx4 / cw4).toBeCloseTo(uvx * 2 - 1, 4);
                    expect(cy4 / cw4).toBeCloseTo(1 - uvy * 2, 4);
                    expect(cz4 / cw4).toBeCloseTo((near * (far - z)) / (z * (far - near)), 4);
                }
            }
        }
    });

    // pointReceiver's base mapping (zero bias) is the unbiased perspective depth the atlas render
    // wrote — pinned to the engine's real perspective(), the bias-free twin of the projection test
    test("pointReceiver at zero bias is the unbiased perspective depth", () => {
        const near = 0.006;
        const far = 6;
        const proj = perspective(pointFov(512), 1, near, far);
        for (const z of [near, 0.01, 0.5, 2.3, far]) {
            const cz = proj[10] * -z + proj[14];
            const cw = z; // proj[11] = -1
            expect(pointReceiver(z, near, far, 0)).toBeCloseTo(cz / cw, 6);
        }
    });

    // the peter-panning fix: depthBias is applied in LINEAR depth, so the world-space lift toward the
    // light is constant across distance from the light. The pre-fix form subtracted depthBias straight
    // from the hyperbolic NDC depth, where a fixed offset grows with z² (the ortho sun never sees it —
    // its depth is linear). Recover the implied lift by inverting the unbiased remap; assert z-invariance.
    test("the depth bias is a z-invariant world-space lift (no perspective peter-panning)", () => {
        const near = 0.05;
        const far = 50;
        const bias = 0.0005;
        const ndc = (z: number) => (near * (far - z)) / (z * (far - near)); // what the render wrote (reverse-Z)
        const invNdc = (d: number) => (near * far) / (d * (far - near) + near); // its inverse
        // round-trip self-check: the inverse actually inverts the remap (else the lift below is noise)
        for (const z of [2, 10, 40]) expect(invNdc(ndc(z))).toBeCloseTo(z, 6);
        const lift = (z: number) => z - invNdc(pointReceiver(z, near, far, bias));
        const lifts = [2, 5, 10, 20, 40].map(lift);
        for (const l of lifts) expect(l).toBeCloseTo(lifts[0], 6); // constant across z — the fix
        expect(lifts[0]).toBeCloseTo(bias * (far - near), 6); // == the sun's depthBias·depth-range lift
        expect(lifts[0]).toBeGreaterThan(0); // toward the light (reduces shadow), never away
    });

    // the widened FOV's purpose: a face-boundary direction (|s| = z, the 45° edge) lands a constant number
    // of texels inside the tile — EDGE_TEXELS regardless of tile size — so the 3×3 PCF footprint (~1.5
    // texels) never crosses into the neighbour. The margin must clear 1.5 at every tile size we allocate.
    test("a face-boundary direction keeps the PCF footprint inside the tile (every tile size)", () => {
        for (const tilePx of [64, 128, 256, 512, 1024, 2048]) {
            const ndc = 1 / pointTanHalf(tilePx);
            const marginTexels = ((1 - ndc) / 2) * tilePx;
            expect(marginTexels).toBeGreaterThan(1.5);
        }
    });
});

// The buddy quadtree packer: power-of-two square tiles into a power-of-two square atlas, no overlap, in
// bounds, freed (reset) space reused, null when full. The standard primitive the importance sizing builds on.
describe("createPacker", () => {
    // decode + bounds + overlap check over a placed set; the packer returns pixel origins
    function noOverlap(tiles: { x: number; y: number; size: number }[], side: number): boolean {
        for (const t of tiles) {
            if (t.x < 0 || t.y < 0 || t.x + t.size > side || t.y + t.size > side) return false;
        }
        for (let i = 0; i < tiles.length; i++) {
            for (let j = i + 1; j < tiles.length; j++) {
                const a = tiles[i];
                const b = tiles[j];
                const sep =
                    a.x + a.size <= b.x ||
                    b.x + b.size <= a.x ||
                    a.y + a.size <= b.y ||
                    b.y + b.size <= a.y;
                if (!sep) return false;
            }
        }
        return true;
    }

    test("packs a mixed power-of-two set with no overlap, in bounds", () => {
        const side = 1024;
        const packer = createPacker(side);
        const sizes = [512, 256, 256, 128, 128, 128, 64, 64, 64, 64];
        const tiles: { x: number; y: number; size: number }[] = [];
        for (const size of sizes) {
            const o = packer.alloc(size);
            expect(o).not.toBeNull();
            tiles.push({ x: o![0], y: o![1], size });
        }
        expect(noOverlap(tiles, side)).toBe(true);
        // origins are size-aligned (the buddy invariant)
        for (const t of tiles) {
            expect(t.x % t.size).toBe(0);
            expect(t.y % t.size).toBe(0);
        }
    });

    test("alloc returns null once the atlas is full", () => {
        const packer = createPacker(256);
        // four 128 tiles fill a 256 atlas exactly
        for (let i = 0; i < 4; i++) expect(packer.alloc(128)).not.toBeNull();
        expect(packer.alloc(128)).toBeNull();
        expect(packer.alloc(64)).toBeNull(); // no room even for a smaller tile
        expect(packer.alloc(512)).toBeNull(); // larger than the atlas
    });

    test("reset reclaims the atlas — a dropped frame's space is reused next frame", () => {
        const packer = createPacker(256);
        for (let i = 0; i < 4; i++) packer.alloc(128); // fill
        expect(packer.alloc(128)).toBeNull();
        packer.reset();
        // after reset the whole atlas is free again — the per-frame reuse path
        const o = packer.alloc(256);
        expect(o).toEqual([0, 0]);
    });
});

// The importance allocator: a point requests 6 same-size face tiles, a spot 1; tile AREA tracks the score
// (hero light large, distant small); the whole set buddy-packs into the square atlas, or null on overflow.
describe("packCasters", () => {
    const atlas = 2048;
    function frame(slot: number, score: number, spot: boolean): PointShadowFrame {
        return {
            light: slot,
            slot,
            score,
            tilePx: 0,
            pos: [0, 0, 0],
            near: 0.01,
            far: 10,
            depthBias: 0,
            normalBias: 0,
            spot,
            fwd: [0, 0, -1],
            right: [1, 0, 0],
            up: [0, 1, 0],
            coneTanHalf: spot ? 1 : 0,
            coneFov: 0,
        };
    }
    // collect every placed face rect (atlas-uv) into pixel tiles for the overlap/bounds check
    function tiles(rects: number[][][]): { x: number; y: number; size: number }[] {
        const out: { x: number; y: number; size: number }[] = [];
        for (const faces of rects) {
            for (const r of faces) {
                if (!r) continue;
                out.push({ x: r[0] * atlas, y: r[1] * atlas, size: r[2] * atlas });
            }
        }
        return out;
    }
    function noOverlap(ts: { x: number; y: number; size: number }[]): boolean {
        for (const t of ts) {
            if (t.x < 0 || t.y < 0 || t.x + t.size > atlas || t.y + t.size > atlas) return false;
        }
        for (let i = 0; i < ts.length; i++) {
            for (let j = i + 1; j < ts.length; j++) {
                const a = ts[i];
                const b = ts[j];
                const sep =
                    a.x + a.size <= b.x ||
                    b.x + b.size <= a.x ||
                    a.y + a.size <= b.y ||
                    b.y + b.size <= a.y;
                if (!sep) return false;
            }
        }
        return true;
    }

    test("a spot claims one tile, a point six", () => {
        const rects = packCasters([frame(0, 1, false), frame(1, 1, true)], atlas);
        expect(rects).not.toBeNull();
        expect(rects![0].length).toBe(6); // point: six face tiles
        expect(rects![1].length).toBe(1); // spot: one tile
    });

    test("every tile is square, in bounds, and non-overlapping across all casters", () => {
        const frames = [
            frame(0, 100, false),
            frame(1, 10, false),
            frame(2, 1, true),
            frame(3, 0.5, false),
        ];
        const rects = packCasters(frames, atlas);
        expect(rects).not.toBeNull();
        for (const faces of rects!) {
            for (const r of faces) {
                expect(r[2]).toBeCloseTo(r[3], 12); // square
            }
        }
        expect(noOverlap(tiles(rects!))).toBe(true);
    });

    test("a point's six faces share one tile size", () => {
        const rects = packCasters([frame(0, 100, false), frame(1, 1, false)], atlas);
        expect(rects).not.toBeNull();
        for (const faces of rects!) {
            for (const r of faces) expect(r[2]).toBeCloseTo(faces[0][2], 12);
        }
    });

    test("importance sizes tiles: the hero light's tile is at least the distant light's", () => {
        // a bright/near hero (score 1000) + a dim/far light (score 1) — area ∝ score, so the hero's tile
        // side is ≥ the distant's (often strictly larger; never smaller)
        const rects = packCasters([frame(0, 1000, false), frame(1, 1, false)], atlas);
        expect(rects).not.toBeNull();
        const hero = rects![0][0][2];
        const distant = rects![1][0][2];
        expect(hero).toBeGreaterThanOrEqual(distant);
        expect(hero).toBeGreaterThan(distant); // a 1000× score gap is several power-of-two levels
    });

    test("equal scores tile uniformly", () => {
        const frames = [frame(0, 5, false), frame(1, 5, false), frame(2, 5, false)];
        const rects = packCasters(frames, atlas);
        expect(rects).not.toBeNull();
        const size = rects![0][0][2];
        for (const faces of rects!) for (const r of faces) expect(r[2]).toBeCloseTo(size, 12);
    });

    test("overflow returns null (a tiny atlas can't hold many casters)", () => {
        // 8 point casters = 48 tiles; at MIN_TILE 64 they need 48·64² = 196k px², a 256×256 atlas holds 64k
        const frames = Array.from({ length: 8 }, (_, i) => frame(i, 1, false));
        expect(packCasters(frames, 256)).toBeNull();
    });
});

// The spot caster's cone basis: a quaternion → the forward axis + an orthonormal lookAt basis, and the
// FS's analytic receiver reconstruction pinned to the engine's real perspective × lookAt. The cone tangent
// + FOV widen by the matched tile's PCF margin (tilePx), so the basis tests pass a representative size.
describe("spotBasis", () => {
    const rad = (deg: number) => (deg * Math.PI) / 180;
    // a unit quaternion for `deg` about axis (ax, ay, az), normalized so it's a proper rotation
    const quat = (
        ax: number,
        ay: number,
        az: number,
        deg: number,
    ): [number, number, number, number] => {
        const l = Math.hypot(ax, ay, az) || 1;
        const s = Math.sin(rad(deg) / 2);
        return [(ax / l) * s, (ay / l) * s, (az / l) * s, Math.cos(rad(deg) / 2)];
    };
    const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    const Tile = 512;

    test("forward is the entity's local -Z under its rotation", () => {
        // identity → -Z; the scenes' straight-down case + orientations a sign error would flip
        const cases: [[number, number, number, number], [number, number, number]][] = [
            [
                [0, 0, 0, 1],
                [0, 0, -1],
            ],
            [quat(1, 0, 0, -90), [0, -1, 0]], // pitched straight down
            [quat(1, 0, 0, 90), [0, 1, 0]], // straight up
            [quat(0, 1, 0, 90), [-1, 0, 0]], // yawed
            [quat(1, 0, 0, -45), [0, -Math.SQRT1_2, -Math.SQRT1_2]], // pitched 45° below the -Z forward
        ];
        for (const [[qx, qy, qz, qw], want] of cases) {
            const { fwd } = spotBasis(qx, qy, qz, qw, 30, Tile);
            for (let i = 0; i < 3; i++) expect(fwd[i]).toBeCloseTo(want[i], 6);
        }
    });

    test("right/up/fwd is an orthonormal basis", () => {
        for (const deg of [0, 17, -33, 80]) {
            const { fwd, right, up } = spotBasis(...quat(0.3, 0.7, 0.2, deg), 28, Tile);
            expect(dot(fwd, fwd)).toBeCloseTo(1, 9);
            expect(dot(right, right)).toBeCloseTo(1, 9);
            expect(dot(up, up)).toBeCloseTo(1, 9);
            expect(dot(fwd, right)).toBeCloseTo(0, 9);
            expect(dot(fwd, up)).toBeCloseTo(0, 9);
            expect(dot(right, up)).toBeCloseTo(0, 9);
        }
    });

    // the FS reconstructs the spot receiver analytically — z = dot(d, fwd); ndc = (dot(d,right),
    // dot(d,up)) / (z·coneTanHalf); depth = far·(z−near)/(z·(far−near)) — and it must equal what the
    // rendered perspective(coneFov) × lookAt wrote into the tile, off-axis included (where a basis sign
    // error shows). The cube-face twin is "the analytic face projection matches perspective(pointFov())"
    test("the analytic spot projection matches perspective(coneFov) × lookAt", () => {
        const q = quat(0.4, -0.2, 0.55, 37); // an arbitrary off-axis orientation
        const near = 0.02;
        const far = 20;
        const b = spotBasis(...q, 32, Tile);
        const view = lookAt(0, 0, 0, b.fwd[0], b.fwd[1], b.fwd[2], b.up[0], b.up[1], b.up[2]);
        const vp = multiply(perspective(b.coneFov, 1, near, far), view);
        // world points in front of the spot (at origin), spread across the cone
        for (const t of [1, 4, 12]) {
            for (const off of [0, 0.2, -0.3]) {
                const d = [0, 1, 2].map(
                    (i) => b.fwd[i] * t + b.right[i] * off * t + b.up[i] * off * 0.5 * t,
                );
                // rendered clip (column-major mat4 × (d, 1))
                const cx = vp[0] * d[0] + vp[4] * d[1] + vp[8] * d[2] + vp[12];
                const cy = vp[1] * d[0] + vp[5] * d[1] + vp[9] * d[2] + vp[13];
                const cz = vp[2] * d[0] + vp[6] * d[1] + vp[10] * d[2] + vp[14];
                const cw = vp[3] * d[0] + vp[7] * d[1] + vp[11] * d[2] + vp[15];
                const z = dot(d, b.fwd);
                expect(dot(d, b.right) / (z * b.coneTanHalf)).toBeCloseTo(cx / cw, 5); // ndc.x
                expect(dot(d, b.up) / (z * b.coneTanHalf)).toBeCloseTo(cy / cw, 5); // ndc.y
                expect((near * (far - z)) / (z * (far - near))).toBeCloseTo(cz / cw, 5); // receiver depth
            }
        }
    });
});

// Caster selection + the importance allocation through the real ECS path: the highest-importance lights
// win the atlas (apparent contribution at the main camera), never query order; the overflow warns once;
// and each combo viewProj projects a face-space point to the FS's atlas ndc + depth via its allocated rect.
describe("updatePointShadows caster selection", () => {
    let state: State;
    let main: number;

    beforeEach(() => {
        clear();
        resetPointShadows();
        Views.clear(); // combo + union cameras attachView'd by a prior test would collide on recycled eids
        // with the traits, `state.add(eid, Transform)` applies scale [1,1,1,1] — the combo cameras' world
        // matrix is then invertible, so `computeViewProj` reproduces their face projection (as in production)
        register("Transform", Transform, TransformsPlugin.traits?.Transform);
        register("Camera", Camera);
        register("PointLight", PointLight);
        register("Shadow", Shadow);
        Slab.collect();
        state = new State();
        main = state.create();
        state.add(main, Transform);
        state.add(main, Camera);
        Transform.pos.set(main, 0, 0, 0, 0);
    });

    function caster(x: number, intensity: number, range = 10): number {
        const eid = state.create();
        state.add(eid, PointLight);
        state.add(eid, Shadow);
        state.add(eid, Transform);
        Transform.pos.set(eid, x, 0, 0, 0);
        PointLight.intensity.set(eid, intensity);
        PointLight.range.set(eid, range);
        return eid;
    }

    test("the nearest/brightest lights win over query order", () => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        // pointCasters() + 1 dim far lights created FIRST (query order would pick them)...
        const far: number[] = [];
        for (let i = 0; i <= pointCasters(); i++) far.push(caster(100 + i, 0.5));
        // ...then one bright near light — importance must keep it
        const hero = caster(2, 5);
        const frames = updatePointShadows(state, main);
        expect(frames.length).toBe(pointCasters());
        expect(frames.map((f) => f.light)).toContain(hero);
        expect(frames[0].light).toBe(hero); // highest score takes slot 0
        expect(warn).toHaveBeenCalledTimes(1); // over-cap warns, once
        updatePointShadows(state, main);
        expect(warn).toHaveBeenCalledTimes(1); // latched until the episode ends
        warn.mockRestore();
    });

    // the view-angle disappearance: with more shadowed lights than the cap, ranking by distance to the main
    // camera re-selects the winners as the camera moves, so a light's shadow pops out (while its light stays
    // lit) on a tiny move near a score crossover. Hysteresis must keep a casting light its slot until a
    // challenger beats it by a margin — so a sub-margin camera nudge never flips the set.
    test("a marginal camera nudge doesn't churn the caster set (hysteresis)", () => {
        // pointCasters()-1 bright anchors always hold the top slots; two equal lights A (x=0) and B (x=13)
        // trade the LAST slot at the crossover (camera x≈6.5). One over the cap warns once — let it print
        for (let i = 0; i < pointCasters() - 1; i++) caster(5 + i * 0.5, 1000);
        const a = caster(0, 10);
        const b = caster(13, 10);
        // establish A as the marginal caster with the camera at x=6.4 (A marginally closer than B)
        Transform.pos.set(main, 6.4, 0, 0, 0);
        const first = updatePointShadows(state, main).map((f) => f.light);
        expect(first.length).toBe(pointCasters());
        expect(first).toContain(a);
        expect(first).not.toContain(b);
        // nudge the camera 0.2 across the crossover to x=6.6: B is now marginally closer than A, but within
        // the hysteresis margin — incumbent A keeps the slot, B stays out. Without hysteresis A↔B flip
        Transform.pos.set(main, 6.6, 0, 0, 0);
        const second = updatePointShadows(state, main).map((f) => f.light);
        expect(second).toContain(a); // sticky: the marginal challenger B doesn't evict incumbent A
        expect(second).not.toContain(b);
    });

    test("under the cap every caster gets a slot, no warn", () => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        const a = caster(5, 1);
        const b = caster(8, 1);
        const frames = updatePointShadows(state, main);
        expect(frames.map((f) => f.light).sort()).toEqual([a, b].sort());
        expect(warn).not.toHaveBeenCalled();
        warn.mockRestore();
    });

    test("a spot caster claims one combo, a point caster six", () => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        const point = caster(0, 1);
        const frames = updatePointShadows(state, main);
        expect(frames.find((f) => f.light === point)?.spot).toBe(false);
        // a point caster fills six face rects (the first six of its slot's rect block are all sized)
        const rects = pointTileRects();
        for (let f = 0; f < 6; f++) expect(rects[f * 4 + 2]).toBeGreaterThan(0);
        warn.mockRestore();
    });

    // the production path: updatePointShadows sizes + packs the caster's tiles, then folds each tile's
    // allocated rect into its combo viewProj, so pointFaceVP() · world lands a face-space point at the exact
    // ATLAS ndc + depth `pointShadowOf` samples (its rect read from pointTileRects). The integration twin of
    // the pure "tileTransform folds…" oracle — it exercises the real allocation, lookAt, near/far, and fold.
    test("each combo viewProj projects a face-space point to the FS's atlas ndc + depth", () => {
        const range = 20;
        const c = caster(0, 1, range);
        Transform.pos.set(c, 10, 5, -3, 1); // a non-origin light, so the lookAt translation is exercised
        const frames = updatePointShadows(state, main);
        const vp = pointFaceVP();
        const rects = pointTileRects();
        const frame = frames.find((f) => f.light === c)!;
        const slot = frame.slot;
        const near = range / 1000;
        const far = range;
        const tanHalf = pointTanHalf(frame.tilePx);
        const pos = [10, 5, -3];
        const zf = 4;
        const sr = 1.2;
        const tu = -0.7; // a point inside the face frustum (|sr|, |tu| < zf)
        for (let f = 0; f < 6; f++) {
            const { fwd, up, right } = POINT_FACES[f];
            const w = [
                pos[0] + fwd[0] * zf + right[0] * sr + up[0] * tu,
                pos[1] + fwd[1] * zf + right[1] * sr + up[1] * tu,
                pos[2] + fwd[2] * zf + right[2] * sr + up[2] * tu,
            ];
            // a single caster fills slot 0, so its dense combo index is the face index
            const m = vp.subarray((slot * 6 + f) * 16, (slot * 6 + f) * 16 + 16);
            const cx = m[0] * w[0] + m[4] * w[1] + m[8] * w[2] + m[12];
            const cy = m[1] * w[0] + m[5] * w[1] + m[9] * w[2] + m[13];
            const cz = m[2] * w[0] + m[6] * w[1] + m[10] * w[2] + m[14];
            const cw = m[3] * w[0] + m[7] * w[1] + m[11] * w[2] + m[15];
            // this face's allocated atlas-UV rect (sparse, slot·6 + face)
            const ri = (slot * 6 + f) * 4;
            const [u0, v0, du, dv] = [rects[ri], rects[ri + 1], rects[ri + 2], rects[ri + 3]];
            const uvx = u0 + (sr / (zf * tanHalf)) * 0.5 * du + 0.5 * du;
            const uvy = v0 + 0.5 * dv - (tu / (zf * tanHalf)) * 0.5 * dv;
            // digit 4 (5e-5): the fold is two chained f32 mat4 multiplies (D · proj · view)
            expect(cx / cw).toBeCloseTo(uvx * 2 - 1, 4);
            expect(cy / cw).toBeCloseTo(1 - uvy * 2, 4);
            expect(cz / cw).toBeCloseTo((near * (far - zf)) / (zf * (far - near)), 4);
        }
    });

    // Sub-stage 0 — the per-combo cull slots. updatePointShadows poses one pooled depth-only camera per
    // active combo (a point caster's six cube faces), and render's `computeViewProj` of that camera must
    // reproduce the combo's face projection — the frustum the Part pack culls casters into. Pin its six
    // planes to the pre-fold `perspective(pointFov) × lookAt` the atlas VS folds the tile onto (the
    // consistency invariant: `aim → compose → invert` equals the direct `lookAt`). This is the cull the
    // gym `render` scenario then pins end-to-end against the GPU survivor counts.
    test("each active combo poses a frustum-cull camera matching its face projection", () => {
        const range = 12;
        const c = caster(0, 1, range);
        Transform.pos.set(c, 4, 2, -1, 1); // off-origin, so the lookAt translation is exercised
        const frames = updatePointShadows(state, main);
        const frame = frames.find((f) => f.light === c)!;
        const combos = pointComboEids();
        // one point caster → six combo cameras (its cube faces), combo-major from its slot
        expect(combos.length).toBe(6);
        expect(combos.length).toBe(pointComboCount());
        const near = range / 1000;
        const far = range;
        const fov = pointFov(frame.tilePx);
        const pos = [4, 2, -1];
        const want = new Float32Array(FRUSTUM_FLOATS);
        const gotVP = new Float32Array(16);
        const got = new Float32Array(FRUSTUM_FLOATS);
        for (let f = 0; f < 6; f++) {
            const { fwd, up } = POINT_FACES[f];
            // the pre-fold face projection (what the atlas VS folds the tile onto) — the cull frustum's source
            const view = lookAt(
                pos[0],
                pos[1],
                pos[2],
                pos[0] + fwd[0],
                pos[1] + fwd[1],
                pos[2] + fwd[2],
                up[0],
                up[1],
                up[2],
            );
            frustumPlanes(multiply(perspective(fov, 1, near, far), view), want);
            // the combo camera's frustum the way render packs it: computeViewProj → frustumPlanes
            computeViewProj(combos[f], 1, gotVP);
            frustumPlanes(gotVP, got);
            // f32 aim→compose→invert vs the direct lookAt: digit 4 covers the accumulated rounding
            for (let i = 0; i < FRUSTUM_FLOATS; i++) expect(got[i]).toBeCloseTo(want[i], 4);
        }
    });

    // one point caster spawns six combo cameras (its cube faces) the pack culls into; removing all casters
    // tears the pool down (freeing the slots) — the per-combo cull's lifecycle
    test("a caster spawns its combo cameras; removing it tears the pool down", () => {
        const warn = spyOn(console, "warn").mockImplementation(() => {});
        const c = caster(5, 1);
        updatePointShadows(state, main);
        const eids = [...pointComboEids()];
        expect(eids.length).toBe(6);
        for (const e of eids) expect(Views.has(e)).toBe(true);

        state.destroy(c);
        updatePointShadows(state, main);
        expect(pointComboEids().length).toBe(0);
        for (const e of eids) expect(Views.has(e)).toBe(false);
        warn.mockRestore();
    });
});

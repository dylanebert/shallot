import { beforeEach, describe, expect, test } from "bun:test";
import { compose, invert, multiply } from "../../engine";
import { bakeVat, type SkinInput } from "../gltf/vat";
import {
    blockVec4,
    HEADER_VEC4,
    jwVec4,
    LiveSkin,
    PALETTE_STRIDE,
    paletteEntry,
    SKIN_SHEAR_EPSILON,
    skinMatrix,
    skinNormal,
    skinPoint,
    writeHeader,
} from "./live";

// Spec tests for the live joint-palette substrate (live.ts) — the runtime twin of the VAT bake. The
// load-bearing case is the EQUIVALENCE GATE: the Xform-decomposed palette path (decompose each skin matrix
// → blend the Xforms per vertex) must reproduce `bakeVat`'s matrix linear-blend skinning on the same hand
// rigs, byte-for-byte to f32. That equivalence is what licenses the surface `vs` to reuse the spliced
// `xformPoint` (a decomposed-Xform blend) as its skinning math. `bakeVat` is the glTF importer's
// clip baker — the substrate's reference lives across the module boundary on purpose: it's the independently
// derived truth, not a second copy of this file's math. The rest pins the block layout arithmetic (both grow
// paths, hole reuse, base stability) + the header fold + the shear residual guard. The `skin-live` surface
// codegen that splices this substrate is gated in `extras/gltf/live.test.ts` (the material path owns it).

const IDENT = compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
const Z90 = [0, 0, Math.SQRT1_2, Math.SQRT1_2] as const;

beforeEach(() => LiveSkin.reset());

describe("block layout arithmetic", () => {
    test("blockVec4 = header + 3 vec4 per joint; jwVec4 = 2 verts per vec4", () => {
        expect(blockVec4(0)).toBe(HEADER_VEC4);
        expect(blockVec4(2)).toBe(HEADER_VEC4 + 2 * PALETTE_STRIDE);
        expect(jwVec4(0)).toBe(0);
        expect(jwVec4(1)).toBe(1);
        expect(jwVec4(2)).toBe(1);
        expect(jwVec4(3)).toBe(2);
    });

    test("the header packs color + jointCount + flags into the block's leading vec4", () => {
        const u32 = new Uint32Array(8);
        writeHeader(u32, 1, 0xdeadbeef, 7, 0b101);
        // lane order is the GPU contract: LIVE_TINT_WGSL reads the packed color as `skinData[base].x`
        expect(Array.from(u32.subarray(4, 8))).toEqual([0xdeadbeef, 7, 0b101, 0]);
    });

    test("blocks pack contiguously; bases stay stable as region A fills", () => {
        const bases = [0, 1, 2, 3].map((eid) => LiveSkin.alloc(eid, 4, 1));
        const step = blockVec4(4);
        expect(bases).toEqual([0, step, 2 * step, 3 * step]);
        expect(LiveSkin.paletteEnd).toBe(4 * step);
        // alloc is idempotent for an unchanged jointCount — same base, no growth
        expect(LiveSkin.alloc(1, 4, 1)).toBe(step);
        expect(LiveSkin.paletteEnd).toBe(4 * step);
    });

    test("free returns an exact-size hole that the next same-size alloc reuses; a tail free shrinks", () => {
        const step = blockVec4(4);
        for (const eid of [0, 1, 2]) LiveSkin.alloc(eid, 4, 1);
        LiveSkin.free(1); // middle → hole at base `step`
        expect(LiveSkin.paletteEnd).toBe(3 * step); // unchanged (not the tail)
        expect(LiveSkin.alloc(9, 4, 1)).toBe(step); // reuses the hole
        LiveSkin.free(2); // tail → shrinks
        expect(LiveSkin.paletteEnd).toBe(2 * step);
    });

    test("palette grow doubles the capacity, keeps bases stable, and flags the layout dirty", () => {
        const step = blockVec4(4); // 13 vec4; the 64-vec4 initial cap holds 4 blocks (52), the 5th overflows
        LiveSkin.layoutDirty = false;
        const first = [0, 1, 2, 3].map((eid) => LiveSkin.alloc(eid, 4, 1));
        expect(LiveSkin.layoutDirty).toBe(false);
        expect(LiveSkin.paletteCap).toBe(64);
        const fifth = LiveSkin.alloc(4, 4, 1);
        expect(fifth).toBe(4 * step); // base unchanged by the grow
        expect(first).toEqual([0, step, 2 * step, 3 * step]);
        expect(LiveSkin.paletteCap).toBe(128);
        expect(LiveSkin.layoutDirty).toBe(true);
        // the f32 + u32 shadows must stay views of the one backing buffer flush uploads — a grow that
        // reallocated only one of them would drop every subsequent header write
        expect(LiveSkin.palette.buffer).toBe(LiveSkin.paletteAB);
        expect(LiveSkin.paletteU32.buffer).toBe(LiveSkin.paletteAB);
    });

    // A same-update destroy+create realias (ecs.md "An eid is a borrow"): the eid stays a live instance
    // the whole time, so alloc — idempotent on the block, not on membership — would return the destroyed
    // instance's block, serving its last pose to the new one. The create-stamp is the only realias signal.
    test("a realias (bumped stamp, same jointCount) reseeds the bind pose, not the prior pose", () => {
        const base = LiveSkin.alloc(0, 2, 1); // instance A, seeded to the bind pose
        const posX = (base + HEADER_VEC4) * 4; // joint 0's Xform pos.x, 0 at the bind pose
        LiveSkin.palette[posX] = 999; // pose A away from bind

        // idempotent for the SAME stamp — the discriminator is the stamp, not an unconditional realloc
        expect(LiveSkin.alloc(0, 2, 1)).toBe(base);
        expect(LiveSkin.palette[posX]).toBe(999);

        // recycle the eid to instance B (bumped stamp): free + realloc + reseed the bind pose
        expect(LiveSkin.alloc(0, 2, 2)).toBe(base);
        expect(LiveSkin.palette[posX]).toBe(0); // B renders the bind pose, not A's 999 — red without the stamp
    });
});

describe("mesh joints/weights region", () => {
    test("jwBase = paletteCap + local; a JW-region grow doubles jwCap", () => {
        const a = LiveSkin.registerMesh(0, new Uint32Array(128), new Uint32Array(128)); // 64 vec4 → fills the cap
        expect(a).toBe(LiveSkin.paletteCap + 0);
        expect(LiveSkin.jwCap).toBe(64);
        const b = LiveSkin.registerMesh(1, new Uint32Array(2), new Uint32Array(2)); // overflows → grow
        expect(b).toBe(LiveSkin.paletteCap + 64);
        expect(LiveSkin.jwCap).toBe(128);
        expect(LiveSkin.jwBaseOf(1)).toBe(LiveSkin.paletteCap + 64);
    });

    test("jwBase shifts when region A's capacity grows (region B moves right)", () => {
        LiveSkin.registerMesh(0, new Uint32Array(4), new Uint32Array(4));
        expect(LiveSkin.jwBaseOf(0)).toBe(64); // paletteCap 64 + local 0
        for (let eid = 0; eid < 5; eid++) LiveSkin.alloc(eid, 4, 1); // forces paletteCap 64 → 128
        expect(LiveSkin.paletteCap).toBe(128);
        expect(LiveSkin.jwBaseOf(0)).toBe(128);
    });

    test("2 vertices pack per vec4 (lower pair .xy, upper pair .zw)", () => {
        LiveSkin.registerMesh(
            0,
            new Uint32Array([0xaa, 0xbb, 0xcc]),
            new Uint32Array([0x11, 0x22, 0x33]),
        );
        // vert 0 → element 0 .xy, vert 1 → element 0 .zw, vert 2 → element 1 .xy
        expect(Array.from(LiveSkin.jw.subarray(0, 6))).toEqual([
            0xaa, 0x11, 0xbb, 0x22, 0xcc, 0x33,
        ]);
    });
});

describe("equivalence gate — palette LBS reproduces bakeVat", () => {
    // build the skin matrices a producer would (jointGlobal · inverseBind), decompose them into the Xform
    // palette via writePalette, blend with skinPoint, and assert the result matches bakeVat's matrix LBS.
    // Tolerance: bakeVat stores its positions as f32 (~1e-6 at unit scale), so compare to 5 decimals — the
    // algebraic identity is exact, this only absorbs bakeVat's f32 storage rounding.

    test("2-bone rotate: a child-bound vertex carried by the parent's 90°Z", () => {
        const input: SkinInput = {
            nodes: [
                { t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], children: [1] },
                { t: [1, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], children: [] },
            ],
            roots: [0],
            channels: [
                {
                    node: 0,
                    path: "rotation",
                    times: new Float32Array([0, 1]),
                    values: new Float32Array([0, 0, 0, 1, ...Z90]),
                    step: false,
                },
            ],
            joints: [0, 1],
            inverseBind: new Float32Array([...IDENT, ...compose(-1, 0, 0, 0, 0, 0, 1, 1, 1, 1)]),
            jointIndex: new Uint16Array([1, 0, 0, 0]),
            weights: new Float32Array([1, 0, 0, 0]),
            restPos: new Float32Array([1, 0, 0]),
            restNormal: new Float32Array([1, 0, 0]),
            duration: 1,
        };
        const vat = bakeVat(input, { fps: 1 });

        // the pose at t=1: node0 rotated 90°Z, node1 its child translated +1x. skin = jointGlobal · IBM.
        const g0 = compose(0, 0, 0, Z90[0], Z90[1], Z90[2], Z90[3], 1, 1, 1);
        const g1 = multiply(g0, compose(1, 0, 0, 0, 0, 0, 1, 1, 1, 1));
        const skin0 = multiply(g0, IDENT);
        const skin1 = multiply(g1, compose(-1, 0, 0, 0, 0, 0, 1, 1, 1, 1));
        const matrices = new Float32Array(32);
        matrices.set(skin0, 0);
        matrices.set(skin1, 16);

        const base = LiveSkin.alloc(0, 2, 1);
        LiveSkin.writePalette(0, matrices);
        const p = skinPoint(LiveSkin.palette, base, [1, 0, 0, 0], [1, 0, 0, 0], 1, 0, 0);

        // frame 1 of the VAT (positions[3..5]) is the same deformed vertex
        expect(p[0]).toBeCloseTo(vat.positions[3], 5);
        expect(p[1]).toBeCloseTo(vat.positions[4], 5);
        expect(p[2]).toBeCloseTo(vat.positions[5], 5);
        expect(p[1]).toBeCloseTo(1, 5); // sanity: (1,0,0) carried to (0,1,0)

        // the normal path blends + renormalizes the same way
        const n = skinNormal(LiveSkin.palette, base, [1, 0, 0, 0], [1, 0, 0, 0], 1, 0, 0);
        expect(n[0]).toBeCloseTo(vat.normals[3], 5);
        expect(n[1]).toBeCloseTo(vat.normals[4], 5);
    });

    test("50/50 blend: a vertex split between two joints lands at the weighted average", () => {
        const input: SkinInput = {
            nodes: [
                { t: [0, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], children: [] },
                { t: [2, 0, 0], r: [0, 0, 0, 1], s: [1, 1, 1], children: [] },
            ],
            roots: [0, 1],
            channels: [],
            joints: [0, 1],
            inverseBind: new Float32Array([...IDENT, ...IDENT]),
            jointIndex: new Uint16Array([0, 1, 0, 0]),
            weights: new Float32Array([0.5, 0.5, 0, 0]),
            restPos: new Float32Array([0, 0, 0]),
            restNormal: new Float32Array([0, 1, 0]),
            duration: 0,
        };
        const vat = bakeVat(input);

        const matrices = new Float32Array(32);
        matrices.set(IDENT, 0); // skin0 = global0 · I = I
        matrices.set(compose(2, 0, 0, 0, 0, 0, 1, 1, 1, 1), 16); // skin1 = translate(2,0,0)

        const base = LiveSkin.alloc(0, 2, 1);
        LiveSkin.writePalette(0, matrices);
        const p = skinPoint(LiveSkin.palette, base, [0, 1, 0, 0], [0.5, 0.5, 0, 0], 0, 0, 0);

        expect(p[0]).toBeCloseTo(vat.positions[0], 5);
        expect(p[1]).toBeCloseTo(vat.positions[1], 5);
        expect(p[2]).toBeCloseTo(vat.positions[2], 5);
        expect(p[0]).toBeCloseTo(1, 5); // sanity: midpoint of 0 and 2
    });

    test("a rest-seeded (unposed) block renders the bind pose — identity palette", () => {
        const base = LiveSkin.alloc(0, 3, 1);
        const p = skinPoint(LiveSkin.palette, base, [2, 0, 0, 0], [1, 0, 0, 0], 4, 5, 6);
        expect(p).toEqual([4, 5, 6]);
    });
});

describe("decompose residual — the shear guard", () => {
    test("a similarity skin matrix decomposes cleanly (residual under the epsilon)", () => {
        const out = new Float32Array(12);
        const q = Z90;
        const m = compose(1, 2, 3, q[0], q[1], q[2], q[3], 2, 2, 2);
        expect(paletteEntry(m, out, 0)).toBeLessThan(SKIN_SHEAR_EPSILON);
    });

    test("a sheared skin matrix exceeds the epsilon (TRS can't represent it)", () => {
        const out = new Float32Array(12);
        const m = compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        m[4] = 0.5; // col1 gains an x component → non-orthogonal basis (shear)
        expect(paletteEntry(m, out, 0)).toBeGreaterThan(SKIN_SHEAR_EPSILON);
    });
});

// The pose-composition helper a live-skin producer (the ragdoll) feeds writePalette: skinMatrix(pos, quat,
// invBind) = compose(pos, quat) · invBind. Geometric checks (transform a point by the resulting matrix), not
// a recompute of the same product. The anchor is that the bind pose (pos/quat == the bone's bind) returns
// identity, so an unposed instance renders undeformed.
describe("skinMatrix — pose composition", () => {
    // apply a column-major mat4 to a point
    const apply = (m: Float32Array, x: number, y: number, z: number): [number, number, number] => [
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    ];

    test("bind pose (now == bind) is the identity — renders undeformed", () => {
        // bone bind at (1,0,0), 90° about Z; invBind = its inverse
        const bind = compose(1, 0, 0, Z90[0], Z90[1], Z90[2], Z90[3], 1, 1, 1);
        const invBind = invert(bind);
        const skin = skinMatrix([1, 0, 0], Z90, invBind);
        for (let i = 0; i < 16; i++) expect(skin[i]).toBeCloseTo(i % 5 === 0 ? 1 : 0, 5);
    });

    test("a pure translation delta moves a vertex by the delta", () => {
        // bind at origin, identity; the bone now sits at (0,2,0)
        const invBind = invert(compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1));
        const skin = skinMatrix([0, 2, 0], [0, 0, 0, 1], invBind);
        const p = apply(skin, 1, 0, 0); // any vertex shifts up by 2
        expect(p[0]).toBeCloseTo(1, 5);
        expect(p[1]).toBeCloseTo(2, 5);
        expect(p[2]).toBeCloseTo(0, 5);
    });

    test("a rotation about the bone origin swings a vertex around it", () => {
        // bone bind + now both at (1,0,0); now rotated 90° about Z. A vertex one unit +X of the bone
        // origin (2,0,0) swings to +Y of it: (1,1,0).
        const invBind = invert(compose(1, 0, 0, 0, 0, 0, 1, 1, 1, 1));
        const skin = skinMatrix([1, 0, 0], Z90, invBind);
        const p = apply(skin, 2, 0, 0);
        expect(p[0]).toBeCloseTo(1, 5);
        expect(p[1]).toBeCloseTo(1, 5);
        expect(p[2]).toBeCloseTo(0, 5);
    });

    test("writes into a caller subarray without aliasing invBind", () => {
        const invBind = invert(compose(0, 0, 0, 0, 0, 0, 1, 1, 1, 1));
        const buf = new Float32Array(32);
        const ret = skinMatrix([3, 0, 0], [0, 0, 0, 1], invBind, buf.subarray(16));
        expect(ret[12]).toBeCloseTo(3, 5); // wrote translation into the second slot
        expect(buf[16 + 12]).toBeCloseTo(3, 5);
        expect(invBind[12]).toBeCloseTo(0, 5); // invBind untouched
    });
});

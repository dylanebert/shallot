// The billboard corner math is the executable spec the surface's vs calls directly (billboard.ts) — a
// TGSL fn is callable as plain JS/GPU-dual code outside a trace (the `lineQuad`/`sdfToSignedDistance`
// precedent), so these expectations pin the real production kernels, not a hand-derived mirror. The
// surface + packing structure below is checked device-free (`tgpu.resolve`, the text `typedTextSurface`
// precedent); the draw itself, the shadow cast, and the billboard orientation are `bun bench` concerns.

import { beforeEach, describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { build, type State } from "../../engine";
import { RenderPlugin } from "../../standard/render";
import { SlabPlugin } from "../../standard/slab";
import { Transform, TransformsPlugin } from "../../standard/transforms";
import { image, SpritePlugin } from ".";
import { screenCorner, worldCorner, yLockedCorner } from "./billboard";
import { packSprites, Sprite, SpriteBillboard, SpriteBlend, SpriteFill, signature } from "./pack";
import { SpriteData, spriteFs, spriteSurface } from "./surface";

type Vec3 = [number, number, number];

// column-major TRS without rotation — enough to exercise translation + per-axis scale
function trs(t: Vec3, s: Vec3 = [1, 1, 1]): d.m4x4f {
    // biome-ignore format: column-major, one row per matrix column reads clearer
    return d.mat4x4f(
        s[0], 0,    0,    0,
        0,    s[1], 0,    0,
        0,    0,    s[2], 0,
        t[0], t[1], t[2], 1,
    );
}

// a 90° yaw about Y (local +X → world -Z), translation zero — the one rotated case the corner tests need
function yaw90(t: Vec3 = [0, 0, 0]): d.m4x4f {
    // biome-ignore format: column-major, one row per matrix column reads clearer
    return d.mat4x4f(
        0, 0, -1, 0,
        0, 1, 0,  0,
        1, 0, 0,  0,
        t[0], t[1], t[2], 1,
    );
}

function expectVec(actual: d.v3f, expected: Vec3) {
    expect(actual.x).toBeCloseTo(expected[0], 5);
    expect(actual.y).toBeCloseTo(expected[1], 5);
    expect(actual.z).toBeCloseTo(expected[2], 5);
}

describe("billboard corners", () => {
    const X = d.vec3f(1, 0, 0);
    const Y = d.vec3f(0, 1, 0);

    test("world: plain transform applies", () => {
        expectVec(worldCorner(trs([0, 0, 0]), 0.5, 1), [0.5, 1, 0]);
        expectVec(worldCorner(trs([2, 3, 4]), 0.5, 1), [2.5, 4, 4]);
        // rotated model: local x maps along the rotated column (yaw 90°)
        expectVec(worldCorner(yaw90(), 1, 0), [0, 0, -1]);
    });

    test("screen: identity camera basis reproduces the local quad, translated + scaled", () => {
        expectVec(screenCorner(trs([2, 3, 4], [2, 2, 2]), X, Y, 1, 1), [4, 5, 4]);
        expectVec(screenCorner(trs([0, 0, 0]), X, Y, -0.5, 0.5), [-0.5, 0.5, 0]);
    });

    test("screen: camera basis replaces the model rotation", () => {
        // camera rolled 90°: right = +Y, up = -X; model rotation (yaw90) is ignored entirely
        expectVec(screenCorner(yaw90(), d.vec3f(0, 1, 0), d.vec3f(-1, 0, 0), 1, 0), [0, 1, 0]);
    });

    test("screen: per-axis scale comes from the model columns", () => {
        expectVec(screenCorner(trs([0, 0, 0], [3, 5, 1]), X, Y, 1, 1), [3, 5, 0]);
    });

    test("yLocked: camera down -Z faces the quad at +Z with world up", () => {
        // backward = cross(right, up) = (0,0,1); quad right = cross(up, facing) = +X
        expectVec(yLockedCorner(trs([0, 0, 0]), X, Y, 1, 1), [1, 1, 0]);
    });

    test("yLocked: camera down -X yaws the quad, y stays locked", () => {
        // camera looking along -X: right = (0,0,-1), up = (0,1,0), backward = (1,0,0)
        const right = d.vec3f(0, 0, -1);
        expectVec(yLockedCorner(trs([0, 0, 0]), right, Y, 1, 0), [0, 0, -1]);
        expectVec(yLockedCorner(trs([0, 0, 0]), right, Y, 0, 1), [0, 1, 0]);
    });

    test("yLocked: a pitched camera keeps the quad vertical (xz projection)", () => {
        // camera pitched 45° down, looking -Z: backward = (0, 0.7071, 0.7071) → facing (0,0,1)
        const s = Math.SQRT1_2;
        expectVec(yLockedCorner(trs([2, 0, 0]), X, d.vec3f(0, s, -s), 1, 1), [3, 1, 0]);
    });

    test("yLocked: straight-down camera falls back to the up xz projection, stays finite", () => {
        // looking straight down -Y with up = (0,0,-1): backward = (0,1,0), xz collapses
        const up = d.vec3f(0, 0, -1);
        const corner = yLockedCorner(trs([0, 0, 0]), X, up, 1, 1);
        // facing = up.xz = (0,-1) → quad right = cross((0,1,0),(0,0,-1)) = (-1,0,0)
        expectVec(corner, [-1, 1, 0]);
    });

    test("yLocked: scale re-applies from the model columns", () => {
        expectVec(yLockedCorner(trs([0, 0, 0], [2, 4, 1]), X, Y, 1, 1), [2, 4, 0]);
    });
});

describe("image registry", () => {
    test("image() registers by name and dedupes", () => {
        const id = image("/icons/a.png", "a");
        expect(image("/icons/a.png", "a")).toBe(id);
        expect(image("/icons/b.png", "b")).not.toBe(id);
    });

    test("a url source defaults its name to the url", () => {
        const id = image("/icons/url-keyed.png");
        expect(image("/icons/url-keyed.png")).toBe(id);
    });
});

// `SpriteData`'s layout the CPU packer writes and the vs/fs read — the byte offsets are pinned against
// pack.ts's literal word indices (never re-derived from the schema itself, testing.md's "a check that
// re-derives the rule it checks discriminates almost nothing").
describe("SpriteData layout", () => {
    const at = (field: (s: d.Infer<typeof SpriteData>) => unknown) =>
        d.memoryLayoutOf(SpriteData, field).offset / 4;

    test("32 bytes / two vec4 reads, matching pack.ts's word indices", () => {
        expect(d.sizeOf(SpriteData)).toBe(32);
        expect(at((s) => s.offset)).toBe(0);
        expect(at((s) => s.size)).toBe(2);
        expect(at((s) => s.eid)).toBe(4);
        expect(at((s) => s.layer)).toBe(5);
        expect(at((s) => s.color)).toBe(6);
        expect(at((s) => s.fill)).toBe(7);
    });
});

describe("surface variants", () => {
    test("six variants: (screen|y|world) × (clip|alpha), clip default per pair, one shared layout", () => {
        const names = Array.from({ length: 6 }, (_, b) => spriteSurface(b).name);
        expect(names).toEqual([
            "sprite-screen",
            "sprite-screen-alpha",
            "sprite-y",
            "sprite-y-alpha",
            "sprite-world",
            "sprite-world-alpha",
        ]);
        for (let b = 0; b < 6; b++) {
            expect(spriteSurface(b).blend).toBe(b & 1 ? "alpha" : "clip");
        }
        // every bucket shares one layout object (the gltf-trio / skin shape) — not six copies
        const layouts = new Set(Array.from({ length: 6 }, (_, b) => spriteSurface(b).layout));
        expect(layouts.size).toBe(1);
    });

    test("declares the eids+transforms instancing convention (the typed twin of typedInstanced)", () => {
        const { layout } = spriteSurface(0);
        expect("eids" in layout.entries).toBe(true);
        expect("transforms" in layout.entries).toBe(true);
    });

    test("resolving each bucket's vs/fs emits the billboard kernel it calls, no shared code duplicated", () => {
        for (let b = 0; b < 6; b++) {
            const { vs, fs } = spriteSurface(b);
            const wgsl = tgpu.resolve([vs, fs], { names: "strict" });
            expect(wgsl).toContain("fn spriteFillMask(");
            expect(wgsl).toContain("fn spriteSrgbToLinear(");
            const variant = ["screenCorner", "yLockedCorner", "worldCorner"][[0, 0, 1, 1, 2, 2][b]];
            expect(wgsl).toContain(`fn ${variant}(`);
        }
    });

    test("clip discards below the cutout and alpha never does", () => {
        for (let b = 0; b < 6; b++) {
            const wgsl = tgpu.resolve([spriteSurface(b).fs], { names: "strict" });
            if (b & 1) {
                expect(wgsl).not.toContain("discard");
            } else {
                expect(wgsl).toContain("discard");
            }
        }
    });

    test("declares no custom varying — the fs reads the packed instance by the built-in eid", () => {
        expect("varyings" in spriteSurface(0)).toBe(false);
        const wgsl = tgpu.resolve([spriteFs(false)], { names: "strict" });
        expect(wgsl).toContain("spriteData[");
        expect(wgsl).toContain("ctx.eid");
        expect(wgsl).not.toContain(".slot");
    });
});

describe("packing", () => {
    let state: State;

    beforeEach(async () => {
        ({ state } = await build({
            plugins: [SlabPlugin, TransformsPlugin, RenderPlugin, SpritePlugin],
            defaults: false,
        }));
    });

    function spawn(fields: Record<string, unknown> = {}): number {
        const eid = state.create();
        state.add(eid, Transform);
        state.add(eid, Sprite);
        for (const [key, value] of Object.entries(fields)) {
            // @ts-expect-error keyed field access on the component record
            Sprite[key].set(eid, value);
        }
        return eid;
    }

    test("packs a sprite's instance words: anchor offset, size, eid, layer, color", () => {
        const eid = spawn();
        Sprite.size.set(eid, 2, 4);
        Sprite.anchor.set(eid, 0.5, 0);
        Sprite.image.set(eid, 3);
        Sprite.color.set(eid, 0xff8040);
        const { ranges, count, f32, u32, eids } = packSprites(state);
        const o = eid * 8;

        expect(count).toBe(1);
        expect(f32[o]).toBeCloseTo(-1); // -size.x * anchor.x
        expect(f32[o + 1]).toBeCloseTo(0);
        expect(f32[o + 2]).toBe(2);
        expect(f32[o + 3]).toBe(4);
        expect(u32[o + 4]).toBe(eid);
        expect(u32[o + 5]).toBe(3);
        expect((u32[o + 6] >>> 0) & 0xff).toBe(0xff); // r in byte 0 (packColor)
        // the eids array stays slot-major (bucket-contiguous ranges), one u32 per slot
        expect(eids[0]).toBe(eid);
        // default bucket = Screen + Clip = 0
        expect(ranges[0]).toEqual({ start: 0, count: 1 });
    });

    test("instance data is eid-indexed, not slot-indexed — offset is eid * SPRITE_FLOATS", () => {
        // burn a few eids on entities with no Sprite component so the sprite's eid diverges from
        // its packing slot (slot 0) — a slot-indexed regression reads/writes offset 0 here instead
        for (let i = 0; i < 3; i++) state.create();
        const eid = spawn();
        expect(eid).toBeGreaterThan(0);
        Sprite.image.set(eid, 7);
        const { u32 } = packSprites(state);
        expect(u32[eid * 8 + 5]).toBe(7);
        expect(u32[5]).not.toBe(7); // slot-0 offset must NOT hold this instance's data
    });

    test("buckets by (billboard, blend) into contiguous ranges, bucket-ordered", () => {
        const world = spawn({ billboard: SpriteBillboard.World });
        const screenAlpha = spawn({ blend: SpriteBlend.Alpha });
        const screenA = spawn();
        const screenB = spawn();
        const { ranges, count, u32, eids } = packSprites(state);

        expect(count).toBe(4);
        expect(ranges[0]).toEqual({ start: 0, count: 2 }); // screen+clip
        expect(ranges[1]).toEqual({ start: 2, count: 1 }); // screen+alpha
        expect(ranges[4]).toEqual({ start: 3, count: 1 }); // world+clip
        expect([ranges[2].count, ranges[3].count, ranges[5].count]).toEqual([0, 0, 0]);
        // instance words land at their own eid's offset, not their slot's
        expect(u32[screenA * 8 + 4]).toBe(screenA);
        expect(u32[screenB * 8 + 4]).toBe(screenB);
        expect(u32[screenAlpha * 8 + 4]).toBe(screenAlpha);
        expect(u32[world * 8 + 4]).toBe(world);
        // the eids array is slot-major, one word per slot, in bucket order
        expect([eids[0], eids[1]].sort()).toEqual([screenA, screenB].sort());
        expect(eids[2]).toBe(screenAlpha);
        expect(eids[3]).toBe(world);
    });

    test("invisible sprites are skipped", () => {
        spawn({ visible: 0 });
        spawn();
        expect(packSprites(state).count).toBe(1);
    });

    test("packs fill as unorm16 amount | mode << 16, default whole image", () => {
        const eid = spawn();
        const o = eid * 8;
        expect(packSprites(state).u32[o + 7]).toBe(0xffff); // fill 1, mode None
        Sprite.fill.set(eid, 0.5);
        Sprite.fillMode.set(eid, SpriteFill.Radial);
        const word = packSprites(state).u32[o + 7];
        expect(word >>> 16).toBe(SpriteFill.Radial);
        expect(word & 0xffff).toBe(Math.round(0.5 * 0xffff));
        Sprite.fill.set(eid, -1); // clamps
        expect(packSprites(state).u32[o + 7] & 0xffff).toBe(0);
    });

    test("signature ignores transform, tracks layout + bucket fields", () => {
        const eid = spawn();
        const base = signature(state);
        Transform.pos.set(eid, 5, 6, 7, 0);
        expect(signature(state)).toBe(base);
        Sprite.billboard.set(eid, SpriteBillboard.YLocked);
        expect(signature(state)).not.toBe(base);
        Sprite.billboard.set(eid, SpriteBillboard.Screen);
        Sprite.size.set(eid, 9, 9);
        expect(signature(state)).not.toBe(base);
        // fill included, so a gauge write rebuilds the instance buffer
        Sprite.size.set(eid, 1, 1);
        Sprite.fill.set(eid, 0.25);
        expect(signature(state)).not.toBe(base);
        Sprite.fill.set(eid, 1);
        Sprite.fillMode.set(eid, SpriteFill.Vertical);
        expect(signature(state)).not.toBe(base);
    });
});

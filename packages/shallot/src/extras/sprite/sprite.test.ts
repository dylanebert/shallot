// the billboard corner math is the executable spec the surface WGSL is ported from — these
// expectations are hand-derived from the reference formulas (Godot material.cpp), so a WGSL port
// that disagrees with them is a port bug.

import { beforeEach, describe, expect, test } from "bun:test";
import { build, type State } from "../../engine";
import { RenderPlugin } from "../../standard/render";
import { surfaceCode } from "../../standard/sear/forward";
import { SlabPlugin } from "../../standard/slab";
import { Transform, TransformsPlugin } from "../../standard/transforms";
import { image, SpritePlugin } from ".";
import { screenCorner, worldCorner, yLockedCorner } from "./billboard";
import { packSprites, Sprite, SpriteBillboard, SpriteBlend, SpriteFill, signature } from "./pack";
import { spriteSurface } from "./surface";

type Vec3 = [number, number, number];

// column-major TRS without rotation — enough to exercise translation + per-axis scale
function trs(t: Vec3, s: Vec3 = [1, 1, 1]): Float32Array {
    const m = new Float32Array(16);
    m[0] = s[0];
    m[5] = s[1];
    m[10] = s[2];
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    m[15] = 1;
    return m;
}

function expectVec(actual: Vec3, expected: Vec3) {
    expect(actual[0]).toBeCloseTo(expected[0], 5);
    expect(actual[1]).toBeCloseTo(expected[1], 5);
    expect(actual[2]).toBeCloseTo(expected[2], 5);
}

describe("billboard corners", () => {
    const X: Vec3 = [1, 0, 0];
    const Y: Vec3 = [0, 1, 0];

    test("world: plain transform applies", () => {
        expectVec(worldCorner(trs([0, 0, 0]), 0.5, 1), [0.5, 1, 0]);
        expectVec(worldCorner(trs([2, 3, 4]), 0.5, 1), [2.5, 4, 4]);
        // rotated model: local x maps along the rotated column
        const rot = trs([0, 0, 0]);
        rot[0] = 0;
        rot[2] = -1; // local +X → world -Z (yaw 90°)
        expectVec(worldCorner(rot, 1, 0), [0, 0, -1]);
    });

    test("screen: identity camera basis reproduces the local quad, translated + scaled", () => {
        expectVec(screenCorner(trs([2, 3, 4], [2, 2, 2]), X, Y, 1, 1), [4, 5, 4]);
        expectVec(screenCorner(trs([0, 0, 0]), X, Y, -0.5, 0.5), [-0.5, 0.5, 0]);
    });

    test("screen: camera basis replaces the model rotation", () => {
        // camera rolled 90°: right = +Y, up = -X; model rotation is ignored entirely
        const rolled = trs([0, 0, 0]);
        rolled[0] = 0;
        rolled[2] = -1;
        expectVec(screenCorner(rolled, [0, 1, 0], [-1, 0, 0], 1, 0), [0, 1, 0]);
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
        expectVec(yLockedCorner(trs([0, 0, 0]), [0, 0, -1], Y, 1, 0), [0, 0, -1]);
        expectVec(yLockedCorner(trs([0, 0, 0]), [0, 0, -1], Y, 0, 1), [0, 1, 0]);
    });

    test("yLocked: a pitched camera keeps the quad vertical (xz projection)", () => {
        // camera pitched 45° down, looking -Z: backward = (0, 0.7071, 0.7071) → facing (0,0,1)
        const s = Math.SQRT1_2;
        expectVec(yLockedCorner(trs([2, 0, 0]), X, [0, s, -s], 1, 1), [3, 1, 0]);
    });

    test("yLocked: straight-down camera falls back to the up xz projection, stays finite", () => {
        // looking straight down -Y with up = (0,0,-1): backward = (0,1,0), xz collapses
        const corner = yLockedCorner(trs([0, 0, 0]), X, [0, 0, -1], 1, 1);
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

describe("surface variants", () => {
    test("six variants: (screen|y|world) × (clip|alpha), clip default per pair", () => {
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
    });

    test("sear codegen compiles each variant's shape", () => {
        for (let b = 0; b < 6; b++) {
            const code = surfaceCode(spriteSurface(b));
            expect(code).toContain("spriteData[iid]");
            expect(code).toContain("transforms[s.eid]");
            if (b & 1) {
                expect(code).not.toContain("discard");
            } else {
                // clip: cutout discard + the authored tag so sprites are pick targets
                expect(code).toContain("discard");
                expect(code).toContain("tag = eid;");
            }
        }
        expect(surfaceCode(spriteSurface(0))).toContain("view.right.xyz");
        expect(surfaceCode(spriteSurface(2))).toContain("cross(view.right.xyz, view.up.xyz)");
        expect(surfaceCode(spriteSurface(4))).toContain("world = t * vec4<f32>(lp, 0.0, 1.0);");
    });

    test("every variant masks alpha by the per-instance fill", () => {
        for (let b = 0; b < 6; b++) {
            const code = surfaceCode(spriteSurface(b));
            expect(code).toContain("fn spriteFillMask");
            expect(code).toContain("spriteFillMask(sfill, uv)");
            expect(code).toContain("sfill = s.fill;");
        }
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
        const { ranges, count, f32, u32 } = packSprites(state);

        expect(count).toBe(1);
        expect(f32[0]).toBeCloseTo(-1); // -size.x * anchor.x
        expect(f32[1]).toBeCloseTo(0);
        expect(f32[2]).toBe(2);
        expect(f32[3]).toBe(4);
        expect(u32[4]).toBe(eid);
        expect(u32[5]).toBe(3);
        expect((u32[6] >>> 0) & 0xff).toBe(0xff); // r in byte 0 (packColor)
        // default bucket = Screen + Clip = 0
        expect(ranges[0]).toEqual({ start: 0, count: 1 });
    });

    test("buckets by (billboard, blend) into contiguous ranges, bucket-ordered", () => {
        const world = spawn({ billboard: SpriteBillboard.World });
        const screenAlpha = spawn({ blend: SpriteBlend.Alpha });
        const screenA = spawn();
        const screenB = spawn();
        const { ranges, count, u32 } = packSprites(state);

        expect(count).toBe(4);
        expect(ranges[0]).toEqual({ start: 0, count: 2 }); // screen+clip
        expect(ranges[1]).toEqual({ start: 2, count: 1 }); // screen+alpha
        expect(ranges[4]).toEqual({ start: 3, count: 1 }); // world+clip
        expect([ranges[2].count, ranges[3].count, ranges[5].count]).toEqual([0, 0, 0]);
        expect([u32[4], u32[8 + 4]].sort()).toEqual([screenA, screenB].sort());
        expect(u32[2 * 8 + 4]).toBe(screenAlpha);
        expect(u32[3 * 8 + 4]).toBe(world);
    });

    test("invisible sprites are skipped", () => {
        spawn({ visible: 0 });
        spawn();
        expect(packSprites(state).count).toBe(1);
    });

    test("packs fill as unorm16 amount | mode << 16, default whole image", () => {
        const eid = spawn();
        expect(packSprites(state).u32[7]).toBe(0xffff); // fill 1, mode None
        Sprite.fill.set(eid, 0.5);
        Sprite.fillMode.set(eid, SpriteFill.Radial);
        const word = packSprites(state).u32[7];
        expect(word >>> 16).toBe(SpriteFill.Radial);
        expect(word & 0xffff).toBe(Math.round(0.5 * 0xffff));
        Sprite.fill.set(eid, -1); // clamps
        expect(packSprites(state).u32[7] & 0xffff).toBe(0);
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

// the Sprite component + the CPU half of the producer: bucket every visible sprite by
// (billboard, blend), pack each instance into a shared eid-indexed staging buffer (the shadow atlas
// re-gather preserves only `eid`, so instance data can't be slot-major once a surface casts — see
// surface.ts), a slot-major `eids` array parallel to the bucket-contiguous ranges (each variant's
// draw indexes its range via firstInstance), and the FNV signature that gates the rebuild. Pure
// over State — no GPU — so the packing contract is what sprite.test.ts exercises directly.

import * as d from "typegpu/data";
import { f32, GlobalTransform, type World, u32, vec2 } from "../../engine";
import { packColor } from "../../engine/utils";
import { SpriteData } from "./surface";

/** how a sprite quad orients toward the camera */
export const SpriteBillboard = {
    /** camera-plane aligned (the default for icons) */
    Screen: 0,
    /** upright, yawing toward the viewer (foliage, standees) */
    YLocked: 1,
    /** plain transform: the quad lives in the entity's local xy plane (decals, ground markers) */
    World: 2,
} as const;

/** how a sprite composites against the scene */
export const SpriteBlend = {
    /** alpha-tested cutout at 0.5: depth-written, unsorted, holed shadows (the default) */
    Clip: 0,
    /** translucent: blended over the opaque scene, casts nothing */
    Alpha: 1,
} as const;

/** which portion of a sprite's image shows, for progress rings and gauges */
export const SpriteFill = {
    /** the whole image (the default) */
    None: 0,
    /** clockwise wedge from 12 o'clock: progress rings */
    Radial: 1,
    /** bottom-up: tanks, vertical gauges */
    Vertical: 2,
    /** left-to-right: bars */
    Horizontal: 3,
} as const;

/**
 * a textured world-space quad (icon, marker) anchored to an entity's {@link Transform}. `image` is
 * a registered image id ({@link image}), `size` the world-space quad size, `anchor` the 0..1 pivot
 * (0.5 0.5 = centered), `color` a hex sRGB tint, `billboard` a {@link SpriteBillboard} mode,
 * `blend` a {@link SpriteBlend} mode. `opacity` multiplies the texture alpha; under the default
 * `clip` blend that shrinks the cutout (the sprite vanishes below 0.5);
 * a smooth fade needs `blend: alpha`. The quad scales by the transform's scale on top of `size`.
 * `fill` shows only the leading 0..1 fraction of the image along a {@link SpriteFill} `fillMode`:
 * a radial fill over a ring icon is a progress ring, a vertical fill over a bar icon a gauge
 *
 * @example
 * ```
 * <a sprite="image: house; size: 2 2; anchor: 0.5 0" transform="pos: 4 0 4" />
 * ```
 */
export const Sprite = {
    /** registered image id (see {@link image}); a scene's `image:` resolves the registered name */
    image: u32,
    /** quad size in world units, before the transform's scale */
    size: vec2,
    /** 0..1 pivot within the quad; 0.5 0.5 centers, 0.5 0 pins the bottom edge to the entity */
    anchor: vec2,
    /** hex sRGB tint multiplied into the texture */
    color: f32,
    /** texture-alpha multiplier; under clip blend it shrinks the cutout, under alpha blend it fades */
    opacity: f32,
    /** drawn when nonzero */
    visible: f32,
    /** billboard orientation, a {@link SpriteBillboard} mode */
    billboard: u32,
    /** compositing, a {@link SpriteBlend} mode */
    blend: u32,
    /** leading fraction of the image shown, 0..1, along {@link fillMode} */
    fill: f32,
    /** fill direction, a {@link SpriteFill} mode */
    fillMode: u32,
};

// one sprite instance = the quad-local offset (-size·anchor) + size, the owning eid, the array
// layer, a packed sRGBA tint, and the packed fill (unorm16 amount | mode << 16). 32 bytes / two
// vec4 reads. Stride derived from the schema (a second hand-authored stride is layout drift
// waiting to happen).
export const SPRITE_BYTES = d.sizeOf(SpriteData);
const SPRITE_FLOATS = SPRITE_BYTES / 4;
/** initial instance capacity: the staging + GPU buffer double on demand */
export const INITIAL = 1 << 8;

/** six buckets, billboard-major: bucket = billboard * 2 + blend */
export const BUCKETS = 6;

interface Instance {
    eid: number;
    ox: number;
    oy: number;
    w: number;
    h: number;
    layer: number;
    color: number;
    fill: number;
}

interface SpritePackState {
    staging: ArrayBuffer;
    f32: Float32Array<ArrayBuffer>;
    u32: Uint32Array<ArrayBuffer>;
    dataCap: number;
    eids: Uint32Array<ArrayBuffer>;
    slotCap: number;
    count: number;
    byBucket: Instance[][];
    ranges: { start: number; count: number }[];
    bits: Float32Array;
    bitsU: Uint32Array;
}

const spritePackKey = { create: createSpritePackState };

function createSpritePackState(): SpritePackState {
    const staging = new ArrayBuffer(INITIAL * SPRITE_BYTES);
    const bits = new Float32Array(1);
    return {
        staging,
        f32: new Float32Array(staging),
        u32: new Uint32Array(staging),
        dataCap: INITIAL,
        eids: new Uint32Array(INITIAL),
        slotCap: INITIAL,
        count: 0,
        byBucket: Array.from({ length: BUCKETS }, () => []),
        ranges: Array.from({ length: BUCKETS }, () => ({ start: 0, count: 0 })),
        bits,
        bitsU: new Uint32Array(bits.buffer),
    };
}

function spritePackState(state: World): SpritePackState {
    return state.resource(spritePackKey);
}

function packFill(amount: number, mode: number): number {
    const a = Math.round(Math.min(1, Math.max(0, amount)) * 0xffff);
    return ((mode & 0xffff) << 16) | a;
}

function fbits(v: number, state: SpritePackState): number {
    state.bits[0] = v;
    return state.bitsU[0];
}
function fold(h: number, x: number): number {
    return Math.imul(h ^ x, 16777619);
}

// the dirty key: every visible sprite's layout-affecting state + membership, billboard + blend
// included (they pick the bucket). The transform is deliberately absent — it flows through the
// slab, so moving a sprite leaves the signature (and the instance buffer) untouched
export function signature(state: World): number {
    const scratch = spritePackState(state);
    let h = 0x811c9dc5 | 0;
    for (const eid of state.query([Sprite, GlobalTransform])) {
        if (!state.of(Sprite).visible.get(eid)) continue;
        h = fold(h, eid);
        h = fold(h, state.of(Sprite).image.get(eid));
        h = fold(h, fbits(state.of(Sprite).size.x.get(eid), scratch));
        h = fold(h, fbits(state.of(Sprite).size.y.get(eid), scratch));
        h = fold(h, fbits(state.of(Sprite).anchor.x.get(eid), scratch));
        h = fold(h, fbits(state.of(Sprite).anchor.y.get(eid), scratch));
        h = fold(h, state.of(Sprite).color.get(eid));
        h = fold(h, fbits(state.of(Sprite).opacity.get(eid), scratch));
        h = fold(h, state.of(Sprite).billboard.get(eid));
        h = fold(h, state.of(Sprite).blend.get(eid));
        h = fold(h, fbits(state.of(Sprite).fill.get(eid), scratch));
        h = fold(h, state.of(Sprite).fillMode.get(eid));
    }
    return h;
}

function growData(min: number, state: SpritePackState): void {
    let cap = state.dataCap;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * SPRITE_BYTES);
    new Uint8Array(next).set(new Uint8Array(state.staging, 0, state.dataCap * SPRITE_BYTES));
    state.staging = next;
    state.f32 = new Float32Array(next);
    state.u32 = new Uint32Array(next);
    state.dataCap = cap;
}

function growSlots(min: number, state: SpritePackState): void {
    let cap = state.slotCap;
    while (cap < min) cap *= 2;
    const next = new Uint32Array(cap);
    next.set(state.eids.subarray(0, state.slotCap));
    state.eids = next;
    state.slotCap = cap;
}

/** restore the staging to its initial capacity: the producer's `warm` reset */
export function resetPack(state: World): void {
    const pack = spritePackState(state);
    pack.dataCap = INITIAL;
    pack.staging = new ArrayBuffer(INITIAL * SPRITE_BYTES);
    pack.f32 = new Float32Array(pack.staging);
    pack.u32 = new Uint32Array(pack.staging);
    pack.slotCap = INITIAL;
    pack.eids = new Uint32Array(INITIAL);
    pack.count = 0;
}

export function packSprites(state: World): {
    ranges: { start: number; count: number }[];
    count: number;
    dataCap: number;
    f32: Float32Array<ArrayBuffer>;
    u32: Uint32Array<ArrayBuffer>;
    eids: Uint32Array<ArrayBuffer>;
} {
    const pack = spritePackState(state);
    for (const bucket of pack.byBucket) bucket.length = 0;

    let maxEid = -1;
    for (const eid of state.query([Sprite, GlobalTransform])) {
        if (!state.of(Sprite).visible.get(eid)) continue;
        const w = state.of(Sprite).size.x.get(eid);
        const h = state.of(Sprite).size.y.get(eid);
        const billboard = Math.min(state.of(Sprite).billboard.get(eid), 2);
        const blend = Math.min(state.of(Sprite).blend.get(eid), 1);
        if (eid > maxEid) maxEid = eid;
        pack.byBucket[billboard * 2 + blend].push({
            eid,
            ox: -w * state.of(Sprite).anchor.x.get(eid),
            oy: -h * state.of(Sprite).anchor.y.get(eid),
            w,
            h,
            layer: state.of(Sprite).image.get(eid),
            color: packColor(state.of(Sprite).color.get(eid), state.of(Sprite).opacity.get(eid)),
            fill: packFill(state.of(Sprite).fill.get(eid), state.of(Sprite).fillMode.get(eid)),
        });
    }

    let total = 0;
    for (const bucket of pack.byBucket) total += bucket.length;
    if (total > pack.slotCap) growSlots(total, pack);
    if (maxEid + 1 > pack.dataCap) growData(maxEid + 1, pack);

    let n = 0;
    for (let b = 0; b < BUCKETS; b++) {
        pack.ranges[b].start = n;
        for (const s of pack.byBucket[b]) {
            const o = s.eid * SPRITE_FLOATS;
            pack.f32[o] = s.ox;
            pack.f32[o + 1] = s.oy;
            pack.f32[o + 2] = s.w;
            pack.f32[o + 3] = s.h;
            pack.u32[o + 4] = s.eid;
            pack.u32[o + 5] = s.layer;
            pack.u32[o + 6] = s.color;
            pack.u32[o + 7] = s.fill;
            pack.eids[n] = s.eid;
            n++;
        }
        pack.ranges[b].count = n - pack.ranges[b].start;
    }
    pack.count = n;
    return {
        ranges: pack.ranges,
        count: pack.count,
        dataCap: pack.dataCap,
        f32: pack.f32,
        u32: pack.u32,
        eids: pack.eids,
    };
}

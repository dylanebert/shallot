// the Sprite component + the CPU half of the producer: bucket every visible sprite by
// (billboard, blend), pack the buckets contiguously into one shared staging buffer (each variant's
// draw indexes its range via firstInstance), and the FNV signature that gates the rebuild. Pure
// over State — no GPU — so the packing contract is what sprite.test.ts exercises directly.

import { f32, type State, sparse, u32, vec2 } from "../../engine";
import { packColor } from "../../engine/utils/core";
import { Transform } from "../../standard/transforms";

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
 * `clip` blend that shrinks the cutout (the sprite vanishes below 0.5, the gltf-clip convention);
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
    image: sparse(u32),
    /** quad size in world units, before the transform's scale */
    size: sparse(vec2),
    /** 0..1 pivot within the quad; 0.5 0.5 centers, 0.5 0 pins the bottom edge to the entity */
    anchor: sparse(vec2),
    /** hex sRGB tint multiplied into the texture */
    color: sparse(f32),
    /** texture-alpha multiplier; under clip blend it shrinks the cutout, under alpha blend it fades */
    opacity: sparse(f32),
    /** drawn when nonzero */
    visible: sparse(f32),
    /** billboard orientation, a {@link SpriteBillboard} mode */
    billboard: sparse(u32),
    /** compositing, a {@link SpriteBlend} mode */
    blend: sparse(u32),
    /** leading fraction of the image shown, 0..1, along {@link fillMode} */
    fill: sparse(f32),
    /** fill direction, a {@link SpriteFill} mode */
    fillMode: sparse(u32),
};

// one sprite instance = the quad-local offset (-size·anchor) + size, the owning eid, the array
// layer, a packed sRGBA tint, and the packed fill (unorm16 amount | mode << 16). 32 bytes / two
// vec4 reads
const SPRITE_FLOATS = 8;
export const SPRITE_BYTES = 32;
/** initial instance capacity: the staging + GPU buffer double on demand */
export const INITIAL = 1 << 8;

/** six buckets, billboard-major: bucket = billboard * 2 + blend */
export const BUCKETS = 6;

let _staging = new ArrayBuffer(INITIAL * SPRITE_BYTES);
let _f32 = new Float32Array(_staging);
let _u32 = new Uint32Array(_staging);
let _cap = INITIAL;
let _count = 0;
const _byBucket: Instance[][] = Array.from({ length: BUCKETS }, () => []);
const _ranges = Array.from({ length: BUCKETS }, () => ({ start: 0, count: 0 }));

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

function packFill(amount: number, mode: number): number {
    const a = Math.round(Math.min(1, Math.max(0, amount)) * 0xffff);
    return ((mode & 0xffff) << 16) | a;
}

const _bits = new Float32Array(1);
const _bitsU = new Uint32Array(_bits.buffer);
function fbits(v: number): number {
    _bits[0] = v;
    return _bitsU[0];
}
function fold(h: number, x: number): number {
    return Math.imul(h ^ x, 16777619);
}

// the dirty key: every visible sprite's layout-affecting state + membership, billboard + blend
// included (they pick the bucket). The transform is deliberately absent — it flows through the
// slab, so moving a sprite leaves the signature (and the instance buffer) untouched
export function signature(state: State): number {
    let h = 0x811c9dc5 | 0;
    for (const eid of state.query([Sprite, Transform])) {
        if (!Sprite.visible.get(eid)) continue;
        h = fold(h, eid);
        h = fold(h, Sprite.image.get(eid));
        h = fold(h, fbits(Sprite.size.x.get(eid)));
        h = fold(h, fbits(Sprite.size.y.get(eid)));
        h = fold(h, fbits(Sprite.anchor.x.get(eid)));
        h = fold(h, fbits(Sprite.anchor.y.get(eid)));
        h = fold(h, Sprite.color.get(eid));
        h = fold(h, fbits(Sprite.opacity.get(eid)));
        h = fold(h, Sprite.billboard.get(eid));
        h = fold(h, Sprite.blend.get(eid));
        h = fold(h, fbits(Sprite.fill.get(eid)));
        h = fold(h, Sprite.fillMode.get(eid));
    }
    return h;
}

function grow(min: number): void {
    let cap = _cap;
    while (cap < min) cap *= 2;
    const next = new ArrayBuffer(cap * SPRITE_BYTES);
    new Uint8Array(next).set(new Uint8Array(_staging, 0, _count * SPRITE_BYTES));
    _staging = next;
    _f32 = new Float32Array(next);
    _u32 = new Uint32Array(next);
    _cap = cap;
}

/** restore the staging to its initial capacity: the producer's `warm` reset */
export function resetPack(): void {
    _cap = INITIAL;
    _staging = new ArrayBuffer(INITIAL * SPRITE_BYTES);
    _f32 = new Float32Array(_staging);
    _u32 = new Uint32Array(_staging);
    _count = 0;
}

export function packSprites(state: State): {
    ranges: { start: number; count: number }[];
    count: number;
    cap: number;
    f32: Float32Array<ArrayBuffer>;
    u32: Uint32Array<ArrayBuffer>;
} {
    for (const bucket of _byBucket) bucket.length = 0;

    for (const eid of state.query([Sprite, Transform])) {
        if (!Sprite.visible.get(eid)) continue;
        const w = Sprite.size.x.get(eid);
        const h = Sprite.size.y.get(eid);
        const billboard = Math.min(Sprite.billboard.get(eid), 2);
        const blend = Math.min(Sprite.blend.get(eid), 1);
        _byBucket[billboard * 2 + blend].push({
            eid,
            ox: -w * Sprite.anchor.x.get(eid),
            oy: -h * Sprite.anchor.y.get(eid),
            w,
            h,
            layer: Sprite.image.get(eid),
            color: packColor(Sprite.color.get(eid), Sprite.opacity.get(eid)),
            fill: packFill(Sprite.fill.get(eid), Sprite.fillMode.get(eid)),
        });
    }

    let total = 0;
    for (const bucket of _byBucket) total += bucket.length;
    if (total > _cap) grow(total);

    let n = 0;
    for (let b = 0; b < BUCKETS; b++) {
        _ranges[b].start = n;
        for (const s of _byBucket[b]) {
            const o = n * SPRITE_FLOATS;
            _f32[o] = s.ox;
            _f32[o + 1] = s.oy;
            _f32[o + 2] = s.w;
            _f32[o + 3] = s.h;
            _u32[o + 4] = s.eid;
            _u32[o + 5] = s.layer;
            _u32[o + 6] = s.color;
            _u32[o + 7] = s.fill;
            n++;
        }
        _ranges[b].count = n - _ranges[b].start;
    }
    _count = n;
    return { ranges: _ranges, count: _count, cap: _cap, f32: _f32, u32: _u32 };
}

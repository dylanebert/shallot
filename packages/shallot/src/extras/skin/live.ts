import tgpu from "typegpu";
import * as d from "typegpu/data";
import type { Plugin, System } from "../../engine";
import { Compute, compose, decompose, multiply, vec4 } from "../../engine";
import { chunk, packColor4, spliceNs } from "../../engine/utils/core";
import { Color } from "../../standard/part";
import { RenderPlugin } from "../../standard/render";
import type { Binding } from "../../standard/render/core";
import { BeginFrameSystem, Render } from "../../standard/render/core";
import { PrepassSystem } from "../../standard/sear/core";
import { SlabPlugin, slab } from "../../standard/slab";

// The live joint-palette skinning substrate: a runtime paradigm the engine owns, not an importer's. A live
// skinned instance is posed each frame by a *producer* — a physics ragdoll, a scripted driver, the glTF
// importer's rig converter — which writes per-joint object-space transforms into a per-instance **palette**,
// and a surface's `vs` blends them per vertex. Because skinning is a VS warp, sear's shadow + prepass passes
// deform for free. The surface that reads the palette is declared by whoever owns the *material* path
// (`extras/gltf` registers the `skin-live` PBR trio over {@link LIVE_SKIN_VS}); this file owns the buffer,
// the layout, the pose-write API, and the WGSL the GPU reader splices.
//
// All the per-instance palettes and all the per-mesh joints/weights live in ONE storage binding — `skinData`
// — to stay under the 10-storage-buffer ceiling (gpu.md; the skin surface has zero headroom). The buffer is
// block-concatenated (gpu.md consolidation #4): region A holds the dynamic per-instance palette blocks at
// the front (so instance bases stay stable as it grows), region B the static per-mesh joints/weights after
// it. This file owns the CPU-side layout arithmetic + the pose-write API + the flush; the surface `vs` (the
// GPU reader) and the rig converter (the JW producer) build on it. The LBS blend math it must reproduce is
// the matrix linear-blend skinning `bakeVat` implements — the equivalence gate in the tests pins that.

/**
 * a skinned instance's per-entity animation state. One `slab(vec4)` published as `"skin"` — the CPU side of
 * the {@link skinBindings} `skin` binding — read by whichever skin surface the instance uses. Live
 * joint-palette path: lane x is the palette base (the vec4 index of the instance's block in `skinData`, from
 * {@link LiveSkin.alloc}), y the material index the surface's shading path reads, z unused, w 0. The glTF
 * VAT (baked-clip) path reuses the same lanes for its own meaning: x the play time in seconds, y the same
 * material index, z a per-instance phase offset (crowd variety), w the clip duration its `SkinSystem` loops
 * on — a `w` of 0 is what makes that system skip a live instance. A producer adds the component to each
 * instance it poses.
 */
export const Skin = { anim: slab(vec4, "skin") };

/** {@link Skin}'s traits: field defaults, derived (a producer writes the lanes, a scene never authors them).
 *  Shared by every plugin that registers `Skin` — the substrate and the glTF importer register the same
 *  object, so the traits live here once rather than diverging per plugin (ecs.md "Stable component ids"). */
export const skinTraits = {
    defaults: () => ({ anim: [0, 0, 0, 0] }),
    derived: true,
};

// skinData element = one 16-byte vec4 (u32 lanes for the header/JW, f32 lanes bitcast for the palette Xforms).
const VEC4_BYTES = 16;
// a palette entry is an Xform (the `Xform` schema: pos+pad, quat, scale+pad) = 12 floats = 3 vec4.
export const PALETTE_STRIDE = 3;
// each instance block leads with a header vec4 (packed color, jointCount, flags) — color folded here so the
// separate `color` slab binding can be dropped (consolidation #3), keeping the surface at 10 storage buffers.
export const HEADER_VEC4 = 1;
/** the per-skinned-mesh constants a live-skin surface decodes against: `jwBase` (the vec4 index the mesh's
 *  joints/weights block starts at in `skinData` region B — shifts when region A's palette capacity grows, so
 *  the producer rewrites it on a layoutDirty flush; {@link LiveSkin.jwBaseOf}) and `vertCount` (the block's
 *  vertex extent). Each skinned mesh binds its own `skinParams` per-draw via `Mesh.bindings` (so N live
 *  meshes coexist); `vatParams` is the precedent. The layout {@link skinBindings}' `skinParams` descriptor
 *  names, and the one {@link LiveSkin.writeParams}' staging indices derive from ({@link skinParamsWgsl} is
 *  the WGSL a surface splices). */
export const SkinParams = d.struct({
    jwBase: d.u32,
    vertCount: d.u32,
    pad0: d.u32,
    pad1: d.u32,
});

const SKIN_PARAMS_BYTES = d.sizeOf(SkinParams);

// the u32 index of one SkinParams field — writeParams fills a flat Uint32Array, so the schema stays the
// one source for where each value lands
const SKIN_PARAMS_AT = {
    jwBase: d.memoryLayoutOf(SkinParams, (p) => p.jwBase).offset / 4,
    vertCount: d.memoryLayoutOf(SkinParams, (p) => p.vertCount).offset / 4,
} as const;

// initial region capacities in vec4; both double on overflow.
const INITIAL_PALETTE_CAP = 64;
const INITIAL_JW_CAP = 64;
/** the {@link paletteEntry} residual above which a skin matrix isn't a similarity transform (shear /
 *  non-uniform-scale-under-rotation) and a decomposed Xform palette entry can't represent it faithfully.
 *  The documented bound on what the palette can hold — nothing on the pose-write path reads the residual
 *  (a per-frame check would cost more than the deformation it guards), so a producer that can produce
 *  sheared matrices checks it itself at rig-build time. */
export const SKIN_SHEAR_EPSILON = 1e-3;

/** the vec4 count an instance's palette block occupies: the header plus one {@link PALETTE_STRIDE}-vec4
 *  Xform per joint. */
export function blockVec4(jointCount: number): number {
    return HEADER_VEC4 + jointCount * PALETTE_STRIDE;
}

/** the vec4 count a mesh's joints/weights block occupies: two vertices per vec4 (8 B/vertex — a packed u32
 *  of 4 joint slots + a packed u32 of 4 weights). */
export function jwVec4(vertCount: number): number {
    return (vertCount + 1) >> 1;
}

/** write an instance block's header into `u32` (the skinData buffer's u32 view) at vec4 index `base`:
 *  `[packed sRGB color, jointCount, flags, pad]`. */
export function writeHeader(
    u32: Uint32Array,
    base: number,
    color: number,
    jointCount: number,
    flags: number,
): void {
    const o = base * 4;
    u32[o] = color;
    u32[o + 1] = jointCount;
    u32[o + 2] = flags;
    u32[o + 3] = 0;
}

const _trs = new Float32Array(10);
const _recompose = new Float32Array(16);

/** decompose a skin matrix (column-major mat4) into an Xform palette entry written to `f32` (the skinData
 *  buffer's f32 view) at float offset `off`, returning the decompose residual — the max abs element
 *  difference between the source and the recomposed T·R·S. A residual past {@link SKIN_SHEAR_EPSILON} means
 *  the matrix carries shear / non-uniform-scale-under-rotation a TRS triple can't hold, so the entry is a
 *  lossy approximation. {@link LiveSkin.writePalette} ignores the return — the check is the caller's, at
 *  rig-build time, never per frame. */
export function paletteEntry(m: Float32Array, f32: Float32Array, off: number): number {
    decompose(m, _trs);
    // Xform layout: pos.xyz + pad, quat.xyzw, scale.xyz + pad
    f32[off] = _trs[0];
    f32[off + 1] = _trs[1];
    f32[off + 2] = _trs[2];
    f32[off + 3] = 0;
    f32[off + 4] = _trs[3];
    f32[off + 5] = _trs[4];
    f32[off + 6] = _trs[5];
    f32[off + 7] = _trs[6];
    f32[off + 8] = _trs[7];
    f32[off + 9] = _trs[8];
    f32[off + 10] = _trs[9];
    f32[off + 11] = 0;
    // residual = how far the source strays from the recomposed T·R·S (shear / non-uniform-scale-under-
    // rotation a TRS triple can't hold). Translation is copied exact, so it only surfaces in the 3×3.
    recomposeTRS(_trs, m);
    let res = 0;
    for (let i = 0; i < 16; i++) res = Math.max(res, Math.abs(_recompose[i] - m[i]));
    return res;
}

// rebuild the column-major T·R·S matrix from a decomposed `[pos, quat, scale]` (translation taken from the
// source's own translation column) into `_recompose` — the inverse of `decompose`, for the residual check.
function recomposeTRS(trs: Float32Array, src: Float32Array): void {
    const qx = trs[3],
        qy = trs[4],
        qz = trs[5],
        qw = trs[6];
    const sx = trs[7],
        sy = trs[8],
        sz = trs[9];
    const x2 = qx + qx,
        y2 = qy + qy,
        z2 = qz + qz;
    const xx = qx * x2,
        xy = qx * y2,
        xz = qx * z2;
    const yy = qy * y2,
        yz = qy * z2,
        zz = qz * z2;
    const wx = qw * x2,
        wy = qw * y2,
        wz = qw * z2;
    const r = _recompose;
    r[0] = (1 - yy - zz) * sx;
    r[1] = (xy + wz) * sx;
    r[2] = (xz - wy) * sx;
    r[3] = 0;
    r[4] = (xy - wz) * sy;
    r[5] = (1 - xx - zz) * sy;
    r[6] = (yz + wx) * sy;
    r[7] = 0;
    r[8] = (xz + wy) * sz;
    r[9] = (yz - wx) * sz;
    r[10] = (1 - xx - yy) * sz;
    r[11] = 0;
    r[12] = src[12];
    r[13] = src[13];
    r[14] = src[14];
    r[15] = 1;
}

const _posed = new Float32Array(16);

/**
 * the object-space skin matrix a live-skin producer feeds {@link LiveSkin.writePalette} for one joint: the
 * posed (unit-scale) transform times a precomputed inverse-bind, `compose(pos, quat) · invBind` (column-major
 * mat4). This is the pose composition {@link LiveSkin}'s palette convention wants, one per joint. A physics
 * ragdoll passes each bone's root-relative pose plus the bone's own bind inverse, so the bind pose
 * (`pos`/`quat` equal to the bone's bind) returns identity and renders undeformed; the glTF importer's baked
 * clip takes the VAT path instead. `out` must not alias `invBind` (the matrix multiply isn't alias-safe).
 * @example
 * ```ts
 * const skin = new Float32Array(16 * jointCount);
 * for (let j = 0; j < jointCount; j++) skinMatrix(pos[j], quat[j], invBind[j], skin.subarray(j * 16));
 * LiveSkin.writePalette(eid, skin);
 * ```
 */
export function skinMatrix(
    pos: readonly [number, number, number],
    quat: readonly [number, number, number, number],
    invBind: Float32Array,
    out?: Float32Array,
): Float32Array {
    compose(pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3], 1, 1, 1, _posed);
    return multiply(_posed, invBind, out ?? new Float32Array(16));
}

// rotate `v` by quaternion `q` (xyzw) — a CPU-local twin of the codec's `xformQuat`.
function qRotate(
    qx: number,
    qy: number,
    qz: number,
    qw: number,
    vx: number,
    vy: number,
    vz: number,
    out: [number, number, number],
): void {
    const tx = 2 * (qy * vz - qz * vy);
    const ty = 2 * (qz * vx - qx * vz);
    const tz = 2 * (qx * vy - qy * vx);
    out[0] = vx + qw * tx + (qy * tz - qz * ty);
    out[1] = vy + qw * ty + (qz * tx - qx * tz);
    out[2] = vz + qw * tz + (qx * ty - qy * tx);
}

const _acc: [number, number, number] = [0, 0, 0];

/**
 * linear-blend skin a point through an instance's palette — the CPU twin of the `skin-live` surface `vs`,
 * and the executable spec for it. `p' = Σ wᵢ·xformPoint(palette[base+1+jᵢ], p)`, algebraically the matrix
 * LBS `bakeVat` bakes; the equivalence gate pins them equal. Weights are taken pre-normalized (the importer
 * renormalizes at decode, so the surface skips a runtime renorm). Reads the skinData f32 view directly.
 */
export function skinPoint(
    f32: Float32Array,
    base: number,
    joints: readonly number[],
    weights: readonly number[],
    px: number,
    py: number,
    pz: number,
    out: [number, number, number] = [0, 0, 0],
): [number, number, number] {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    for (let k = 0; k < 4; k++) {
        const w = weights[k];
        if (w === 0) continue;
        const o = (base + HEADER_VEC4 + joints[k] * PALETTE_STRIDE) * 4;
        qRotate(
            f32[o + 4],
            f32[o + 5],
            f32[o + 6],
            f32[o + 7],
            px * f32[o + 8],
            py * f32[o + 9],
            pz * f32[o + 10],
            _acc,
        );
        out[0] += w * (f32[o] + _acc[0]);
        out[1] += w * (f32[o + 1] + _acc[1]);
        out[2] += w * (f32[o + 2] + _acc[2]);
    }
    return out;
}

/**
 * linear-blend skin a normal — the CPU twin of the `skin-live` surface `vs` normal path. `n' =
 * normalize(Σ wᵢ·xformNormal(palette[base+1+jᵢ], n))`; normals blend as plain vec3 and renormalize, never
 * oct across the blend (gpu.md rule 9, the VAT lesson). `xformNormal` is the inverse-scale rotate
 * (`R·(n/s)`, the inverse-transpose for a TRS frame), the zero-scale lane dropped to 0.
 */
export function skinNormal(
    f32: Float32Array,
    base: number,
    joints: readonly number[],
    weights: readonly number[],
    nx: number,
    ny: number,
    nz: number,
    out: [number, number, number] = [0, 0, 0],
): [number, number, number] {
    out[0] = 0;
    out[1] = 0;
    out[2] = 0;
    for (let k = 0; k < 4; k++) {
        const w = weights[k];
        if (w === 0) continue;
        const o = (base + HEADER_VEC4 + joints[k] * PALETTE_STRIDE) * 4;
        const sx = f32[o + 8],
            sy = f32[o + 9],
            sz = f32[o + 10];
        qRotate(
            f32[o + 4],
            f32[o + 5],
            f32[o + 6],
            f32[o + 7],
            sx !== 0 ? nx / sx : 0,
            sy !== 0 ? ny / sy : 0,
            sz !== 0 ? nz / sz : 0,
            _acc,
        );
        out[0] += w * _acc[0];
        out[1] += w * _acc[1];
        out[2] += w * _acc[2];
    }
    const len = Math.hypot(out[0], out[1], out[2]) || 1;
    out[0] /= len;
    out[1] /= len;
    out[2] /= len;
    return out;
}

// a live instance's palette block: `base` is the vec4 index of its header in region A; `size` the block's
// vec4 count (cached so free returns it to the hole list without recomputing). `stamp` is the owning
// entity's create-stamp — a realias to a new same-jointCount instance would otherwise inherit this pose
// (alloc is idempotent on the block, not on membership; ecs.md "An eid is a borrow").
interface Block {
    base: number;
    jointCount: number;
    size: number;
    stamp: number;
}

const _color = new Float32Array(4);
const _params = new Uint32Array(SKIN_PARAMS_BYTES / 4);

// the initial region-A backing store, held in a local so the literal's f32 + u32 fields below are views of
// the SAME buffer — the invariant `flush` uploads through (it writes `palette.buffer`, so a `paletteU32`
// over its own buffer would drop every header write). `reset` / `growPalette` re-establish it.
const _initialAB = new ArrayBuffer(INITIAL_PALETTE_CAP * VEC4_BYTES);

/**
 * the live joint-palette substrate: a process singleton owning the `skinData` buffer, its CPU shadow, and
 * the block layout. Producers author through the eid-keyed pose-write API ({@link LiveSkin.alloc} /
 * {@link LiveSkin.writePalette} / {@link LiveSkin.free}) and register a mesh's joints/weights once
 * ({@link LiveSkin.registerMesh}); {@link LiveSkinSystem} flushes dirty blocks to the GPU each frame. Reset
 * on every build ({@link LiveSkin.reset}), so it survives a State rebuild (ecs.md reload-safety).
 */
export const LiveSkin = {
    // region A (palettes) shadow: [0, paletteEnd) vec4 used of paletteCap; f32 + u32 views of one buffer.
    paletteCap: INITIAL_PALETTE_CAP,
    paletteEnd: 0,
    paletteAB: _initialAB,
    palette: new Float32Array(_initialAB),
    paletteU32: new Uint32Array(_initialAB),
    // region B (joints/weights) shadow: [0, jwEnd) vec4 used of jwCap; region-B-local, uploaded at the
    // paletteCap offset (so jwBase = paletteCap + local shifts when region A's capacity grows).
    jwCap: INITIAL_JW_CAP,
    jwEnd: 0,
    jw: new Uint32Array(INITIAL_JW_CAP * 4),

    blocks: new Map<number, Block>(),
    holes: [] as Block[],
    meshes: new Map<number, { local: number; vertCount: number }>(),
    // meshId → its `skinParams` uniform (jwBase, vertCount): created lazily by {@link paramsBuffer}, owned
    // here so a palette-growth realloc (which shifts every jwBase) rewrites them in {@link flush} with no
    // consumer round-trip. The producer binds the returned buffer via `Mesh.bindings.skinParams`.
    params: new Map<number, GPUBuffer>(),

    buffer: null as GPUBuffer | null,
    bufferVec4: 0,
    // a zero `skinParams` published globally so a `skin-live` surface's no-op draws over non-live meshes (Part
    // registers one Draw per instanced-surface × mesh) resolve their bind group — the `fallbackVat` shape. A
    // real live mesh overrides it per-draw via `Mesh.bindings.skinParams`, so only the 0-instance draws read it.
    fallbackParams: null as GPUBuffer | null,
    dirty: new Set<number>(),
    // capacity changed → the GPU buffer reallocs and both regions re-upload (region B's offset moved, so
    // every mesh's jwBase changed — the producer rewrites its skinParams uniform on this).
    layoutDirty: true,
    // region B changed without a capacity change → re-upload region B alone.
    jwDirty: false,

    /** reset to the empty layout (a host plugin's `initialize`), preserving the singleton across a State
     *  rebuild. Drops the GPU buffer so the next flush reallocs + republishes into the wiped
     *  `Compute.buffers`. */
    reset(): void {
        this.paletteCap = INITIAL_PALETTE_CAP;
        this.paletteEnd = 0;
        this.paletteAB = new ArrayBuffer(INITIAL_PALETTE_CAP * VEC4_BYTES);
        this.palette = new Float32Array(this.paletteAB);
        this.paletteU32 = new Uint32Array(this.paletteAB);
        this.jwCap = INITIAL_JW_CAP;
        this.jwEnd = 0;
        this.jw = new Uint32Array(INITIAL_JW_CAP * 4);
        this.blocks.clear();
        this.holes.length = 0;
        this.meshes.clear();
        for (const b of this.params.values()) b.destroy();
        this.params.clear();
        this.fallbackParams?.destroy();
        this.fallbackParams = null;
        this.buffer?.destroy();
        this.buffer = null;
        this.bufferVec4 = 0;
        this.dirty.clear();
        this.layoutDirty = true;
        this.jwDirty = false;
    },

    /** free the GPU buffers (a host plugin's `dispose`). */
    dispose(): void {
        this.buffer?.destroy();
        this.buffer = null;
        for (const b of this.params.values()) b.destroy();
        this.params.clear();
        this.fallbackParams?.destroy();
        this.fallbackParams = null;
    },

    /** allocate `eid`'s palette block for `jointCount` joints, seeded to the rest (bind) pose — identity
     *  Xforms + a white header — so an unposed live instance renders the bind pose. Returns the block's
     *  `base` (the vec4 index of its header); the caller writes it into `Skin.anim.x` for the surface to
     *  read. Idempotent for an unchanged jointCount + `stamp`; a changed count OR a realias (a bumped
     *  create-stamp, `state.stamp(eid)` — a same-update destroy+create that kept the eid a live instance)
     *  frees + reallocates, reseeding the bind pose so the new instance never inherits the old one's pose. */
    alloc(eid: number, jointCount: number, stamp: number): number {
        const existing = this.blocks.get(eid);
        if (existing) {
            if (existing.jointCount === jointCount && existing.stamp === stamp)
                return existing.base;
            this.free(eid);
        }
        const size = blockVec4(jointCount);
        const base = this.take(size);
        this.blocks.set(eid, { base, jointCount, size, stamp });
        writeHeader(this.paletteU32, base, packColor4(1, 1, 1, 1), jointCount, 0);
        for (let j = 0; j < jointCount; j++) {
            const o = (base + HEADER_VEC4 + j * PALETTE_STRIDE) * 4;
            // identity Xform: pos 0, quat (0,0,0,1), scale 1
            this.palette.fill(0, o, o + 12);
            this.palette[o + 7] = 1;
            this.palette[o + 8] = 1;
            this.palette[o + 9] = 1;
            this.palette[o + 10] = 1;
        }
        this.dirty.add(eid);
        return base;
    },

    // reserve `size` vec4 in region A: reuse an exact-size hole (keeps bases stable for the common
    // spawn/despawn-same-rig pattern), else append + grow the capacity by doubling.
    take(size: number): number {
        for (let i = 0; i < this.holes.length; i++) {
            if (this.holes[i].size === size) {
                const base = this.holes[i].base;
                this.holes.splice(i, 1);
                return base;
            }
        }
        const base = this.paletteEnd;
        if (base + size > this.paletteCap) {
            let cap = this.paletteCap;
            while (base + size > cap) cap *= 2;
            this.growPalette(cap);
        }
        this.paletteEnd = base + size;
        return base;
    },

    growPalette(cap: number): void {
        const ab = new ArrayBuffer(cap * VEC4_BYTES);
        const f32 = new Float32Array(ab);
        f32.set(this.palette.subarray(0, this.paletteEnd * 4));
        this.paletteAB = ab;
        this.palette = f32;
        this.paletteU32 = new Uint32Array(ab);
        this.paletteCap = cap;
        this.layoutDirty = true; // region B's GPU offset (= paletteCap) moved → every jwBase changed
    },

    /** release `eid`'s block back to the hole list (or shrink the tail when it's the last block). Bases of
     *  other instances stay put. */
    free(eid: number): void {
        const block = this.blocks.get(eid);
        if (!block) return;
        this.blocks.delete(eid);
        this.dirty.delete(eid);
        if (block.base + block.size === this.paletteEnd) this.paletteEnd = block.base;
        else this.holes.push(block);
    },

    /** write `eid`'s pose: `matrices` is `jointCount` column-major skin matrices (object-space
     *  `rootInv·jointWorld·inverseBind`), each decomposed to its Xform palette entry. Marks the block dirty
     *  for the next flush. */
    writePalette(eid: number, matrices: Float32Array): void {
        const block = this.blocks.get(eid);
        if (!block) return;
        const n = block.jointCount;
        for (let j = 0; j < n; j++) {
            const off = (block.base + HEADER_VEC4 + j * PALETTE_STRIDE) * 4;
            paletteEntry(matrices.subarray(j * 16, j * 16 + 16), this.palette, off);
        }
        this.dirty.add(eid);
    },

    /** register a skinned mesh's per-vertex joints/weights once (region B). `jointsPacked[v]` is a u32 of 4
     *  u8 joint slots, `weightsPacked[v]` a u32 of 4 unorm8 weights (the producer quantizes to these).
     *  Returns the mesh's `jwBase` (the vec4 index its block starts at, for the surface's skinParams). */
    registerMesh(meshId: number, jointsPacked: Uint32Array, weightsPacked: Uint32Array): number {
        // one mesh registers once per build (a re-register of the same asset in one build resolves to the
        // same meshId — reuse its block rather than orphan a second copy in region B)
        if (this.meshes.has(meshId)) return this.jwBaseOf(meshId);
        const vertCount = jointsPacked.length;
        const size = jwVec4(vertCount);
        const local = this.jwEnd;
        if (local + size > this.jwCap) {
            let cap = this.jwCap;
            while (local + size > cap) cap *= 2;
            const grown = new Uint32Array(cap * 4);
            grown.set(this.jw.subarray(0, this.jwEnd * 4));
            this.jw = grown;
            this.jwCap = cap;
            this.layoutDirty = true; // buffer reallocs (grows region B)
        }
        for (let v = 0; v < vertCount; v++) {
            const o = (local + (v >> 1)) * 4 + (v & 1) * 2;
            this.jw[o] = jointsPacked[v];
            this.jw[o + 1] = weightsPacked[v];
        }
        this.jwEnd = local + size;
        this.meshes.set(meshId, { local, vertCount });
        this.jwDirty = true;
        return this.paletteCap + local;
    },

    /** a registered mesh's current `jwBase` (the vec4 index of its region-B block in the buffer). Shifts
     *  when region A's capacity grows, so a producer re-reads it after a `layoutDirty` flush. */
    jwBaseOf(meshId: number): number {
        const m = this.meshes.get(meshId);
        return m ? this.paletteCap + m.local : 0;
    },

    /** the mesh's `skinParams` uniform (`jwBase`, `vertCount`, pad, pad), created lazily + owned here so a
     *  palette-growth realloc (which shifts jwBase) rewrites it in {@link LiveSkin.flush} without the
     *  consumer re-reading. The producer binds the returned buffer via `Mesh.bindings.skinParams`.
     *  Call after {@link registerMesh} (the buffer seeds from the mesh's current jwBase). */
    paramsBuffer(device: GPUDevice, meshId: number): GPUBuffer {
        const existing = this.params.get(meshId);
        if (existing) return existing;
        const buf = device.createBuffer({
            label: `skin-params:${meshId}`,
            size: SKIN_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.params.set(meshId, buf);
        this.writeParams(device, meshId, buf);
        return buf;
    },

    // write `[jwBase, vertCount, 0, 0]` into a mesh's skinParams uniform (contents only — the buffer identity
    // is stable, so sear's bind-group cache survives the rewrite). flush calls this for every mesh after a
    // realloc, since region A's capacity growth shifts every jwBase.
    writeParams(device: GPUDevice, meshId: number, buf: GPUBuffer): void {
        const m = this.meshes.get(meshId);
        if (!m) return;
        _params[SKIN_PARAMS_AT.jwBase] = this.paletteCap + m.local;
        _params[SKIN_PARAMS_AT.vertCount] = m.vertCount;
        device.queue.writeBuffer(buf, 0, _params);
    },

    /** upload dirty blocks (LiveSkinSystem). Syncs each live block's header color from its `Color`
     *  component (the color fold), then reallocs + re-uploads both regions on a capacity change, or writes
     *  the changed palette/JW blocks otherwise. Publishes the buffer under `"skinData"`. No-op with no
     *  device. */
    flush(device: GPUDevice | null | undefined): void {
        if (!device) return;
        for (const [eid, block] of this.blocks) {
            Color.rgba.read(eid, _color);
            const packed = packColor4(_color[0], _color[1], _color[2], _color[3]);
            if (this.paletteU32[block.base * 4] !== packed) {
                this.paletteU32[block.base * 4] = packed;
                this.dirty.add(eid);
            }
        }

        const needVec4 = this.paletteCap + this.jwCap;
        if (!this.buffer || this.bufferVec4 < needVec4 || this.layoutDirty) {
            this.buffer?.destroy();
            this.buffer = device.createBuffer({
                label: "skin-data",
                size: needVec4 * VEC4_BYTES,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this.bufferVec4 = needVec4;
            if (this.paletteEnd > 0)
                device.queue.writeBuffer(
                    this.buffer,
                    0,
                    this.palette.buffer,
                    0,
                    this.paletteEnd * VEC4_BYTES,
                );
            if (this.jwEnd > 0)
                device.queue.writeBuffer(
                    this.buffer,
                    this.paletteCap * VEC4_BYTES,
                    this.jw.buffer,
                    0,
                    this.jwEnd * VEC4_BYTES,
                );
            Compute.buffers.set("skinData", this.buffer);
            // publish the global skinParams fallback into the (wiped-each-build) Compute.buffers, so a
            // skin-live no-op draw over a non-live mesh resolves; a real live mesh overrides it per-draw
            if (!this.fallbackParams)
                this.fallbackParams = device.createBuffer({
                    label: "skin-params-fallback",
                    size: SKIN_PARAMS_BYTES,
                    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                });
            Compute.buffers.set("skinParams", this.fallbackParams);
            // a realloc shifted region B (jwBase = paletteCap + local): every mesh's skinParams jwBase moved
            for (const [meshId, buf] of this.params) this.writeParams(device, meshId, buf);
            this.layoutDirty = false;
            this.jwDirty = false;
            this.dirty.clear();
            return;
        }
        if (this.jwDirty) {
            device.queue.writeBuffer(
                this.buffer,
                this.paletteCap * VEC4_BYTES,
                this.jw.buffer,
                0,
                this.jwEnd * VEC4_BYTES,
            );
            this.jwDirty = false;
        }
        for (const eid of this.dirty) {
            const block = this.blocks.get(eid);
            if (!block) continue;
            device.queue.writeBuffer(
                this.buffer,
                block.base * VEC4_BYTES,
                this.palette.buffer,
                block.base * VEC4_BYTES,
                block.size * VEC4_BYTES,
            );
        }
        this.dirty.clear();
    },
};

/**
 * flush the live joint palettes to the GPU each frame. Same schedule slot as physics' compose (`draw`,
 * `after: [BeginFrameSystem]`, `before: [PrepassSystem]`), so the deformation is current before every sear
 * geometry pass — prepass, shadow, and color — reads it.
 */
export const LiveSkinSystem: System = {
    name: "LiveSkin",
    group: "draw",
    annotations: { mode: "always" },
    after: [BeginFrameSystem],
    before: [PrepassSystem],
    update() {
        if (!Render.encoder) return;
        LiveSkin.flush(Compute.device);
    },
};

/**
 * the live joint-palette substrate as a plugin: registers {@link Skin}, runs {@link LiveSkinSystem}, and
 * owns {@link LiveSkin}'s per-build reset + teardown. Add it for a producer that poses a rig itself — a
 * physics ragdoll, a scripted driver — with no glTF asset in the scene.
 *
 * It provides the substrate, not a way to draw: palette + component + system + the pose-write API, plus the
 * WGSL a surface splices (`@dylanebert/shallot/skin/core`). The producer supplies the surface that reads it,
 * with whatever material path it wants — `extras/gltf`'s `skin-live` PBR trio (registered by `GltfPlugin`)
 * is one such consumer. `GltfPlugin` wires the same substrate itself for an imported rig, so a glTF app
 * needs neither this plugin nor a second copy of the schedule slot; having both is harmless, since the
 * system registers by identity and the component by name.
 *
 * @example
 * ```ts
 * // pose a hand-built rig, drawn through a surface you register yourself
 * plugins: [...DEFAULT_PLUGINS, SkinPlugin]
 * LiveSkin.registerMesh(meshId, jointsPacked, weightsPacked);
 * Skin.anim.x.set(eid, LiveSkin.alloc(eid, jointCount, state.stamp(eid)));
 * ```
 */
export const SkinPlugin: Plugin = {
    name: "Skin",
    dependencies: [RenderPlugin, SlabPlugin],
    components: { Skin },
    traits: { Skin: skinTraits },
    systems: [LiveSkinSystem],
    initialize() {
        LiveSkin.reset();
    },
    dispose() {
        LiveSkin.dispose();
    },
};

// ---- the WGSL a live-skin surface splices: the GPU reader of the palette substrate above. A surface owning
// a material path (`extras/gltf`'s `skin-live` PBR trio) declares {@link skinBindings} beside its own
// material bindings, splices the preamble chunks, and takes {@link LIVE_SKIN_VS} as its `vs`.

/** the bindings a live-skin surface declares for the substrate: the folded {@link Skin} slab (palette base
 *  in x, material index in y), the block-concat `skinData` buffer, and the per-mesh `skinParams` uniform.
 *  Two of the three are storage, so a surface taking these spends 2 of its `10 − SHARED_STORAGE_COUNT`
 *  own-storage budget here (gpu.md); the uniform is a separate limit. Declaration order is the binding
 *  order — a surface interleaves these with its own bindings deliberately, it doesn't just spread them.
 *  Frozen (deeply): a registered surface aliases these descriptors, so a mutation would retarget every
 *  shipped live-skin surface. */
export const skinBindings: Readonly<Record<string, Binding>> = Object.freeze({
    skin: Object.freeze({ type: "storage", element: "vec4<f32>" }) as Binding,
    skinData: Object.freeze({ type: "storage", element: "vec4<u32>" }) as Binding,
    skinParams: Object.freeze({ type: "uniform", struct: "SkinParams" }) as Binding,
});

/** the WGSL {@link SkinParams} struct a live-skin surface declares its `skinParams` uniform over. Splice at
 *  module scope; structs resolve order-free, so it may sit after the binding decl that references it. */
export const skinParamsWgsl = chunk("skinParamsWgsl", [SkinParams], spliceNs);

// the per-instance tint — WGSL-bodied, because it reads `skinData` and `skin` as module-scope globals the
// *consumer* declares by name (the relocatable contract every live-skin surface shares) and calls the
// sear-spliced `unpackLdrColor`. A TGSL body has no spelling for either until the surface contract itself
// is typed (4a).
const liveTint = tgpu
    .fn(
        [d.u32],
        d.vec4f,
    )(/* wgsl */ `(e: u32) -> vec4f {
    return unpackLdrColor(skinData[u32(skin[e].x)].x);
}`)
    .$name("liveTint");

/** the per-instance tint helper, read from the palette block's header (the color fold, gpu.md
 *  consolidation #3): `color` is packed into the header's first u32 — synced from the `Color` component by
 *  the flush — so a live-skin surface carries no separate `color` storage binding and stays at the
 *  10-storage ceiling. `skin[eid].x` is the header's vec4 base in `skinData`; `skin` / `skinData` /
 *  `unpackLdrColor` are all referenced by name (the latter sear-spliced for every surface). */
export const liveTintWgsl = chunk("liveTintWgsl", [liveTint], spliceNs);

/** the live-skin `vs`: decode this vertex's 4 joint influences from `skinData` region B (keyed by `vidx`, the
 *  skinned mesh's local vertex index — 2 verts per vec4, 8 B/vertex, gpu.md rule 6), then blend the
 *  instance's palette Xforms (region A, based at `skin[eid].x`). `p' = Σ wᵢ·xformPoint(palette[base+1+jᵢ],
 *  localPos)` — algebraically the matrix LBS `bakeVat` bakes (the equivalence gate pins them equal), so the
 *  palette entries being Xform-shaped lets the VS reuse the spliced xformWgsl() `xformPoint`/`xformNormal`
 *  verbatim (zero new transform WGSL). The normal blends as a plain vec3 and renormalizes — never oct across
 *  a blend (gpu.md rule 9, the VAT lesson). Palettes are object-space (root-relative), so the standard
 *  instance transform (`transforms[eid]`, applied here after the blend) still carries the skinned pose to
 *  world space — the instance's root stays the meaningful `Transform` in the firehose. Weights are
 *  pre-normalized at import, so there's no runtime renorm; a zero-weight influence skips its palette read
 *  (the memory-bound early-out, the {@link skinPoint} CPU twin's shape). */
export const LIVE_SKIN_VS = /* wgsl */ `
    let jwElem = skinData[skinParams.jwBase + (vidx >> 1u)];
    let jwPair = (vidx & 1u) * 2u;
    let js = jwElem[jwPair];
    let wt = unpack4x8unorm(jwElem[jwPair + 1u]);
    let joints = vec4<u32>(js & 0xffu, (js >> 8u) & 0xffu, (js >> 16u) & 0xffu, (js >> 24u) & 0xffu);
    let pbase = u32(skin[eid].x);
    var sp = vec3<f32>(0.0);
    var sn = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let w = wt[k];
        if (w == 0.0) { continue; }
        let po = pbase + ${HEADER_VEC4}u + joints[k] * ${PALETTE_STRIDE}u;
        let jx = Xform(
            bitcast<vec3<f32>>(skinData[po].xyz),
            bitcast<vec4<f32>>(skinData[po + 1u]),
            bitcast<vec3<f32>>(skinData[po + 2u].xyz));
        sp += w * xformPoint(jx, localPos);
        sn += w * xformNormal(jx, localNormal);
    }
    let xf = transforms[eid];
    world = vec4<f32>(xformPoint(xf, sp), 1.0);
    worldNormal = xformNormal(xf, normalize(sn));`;

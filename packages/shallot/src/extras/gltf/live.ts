import type { System } from "../../engine";
import { Compute, compose, decompose, multiply } from "../../engine";
import { packLdrColor } from "../../engine/utils/core";
import { Color } from "../../standard/part";
import type { Binding, Surface } from "../../standard/render/core";
import { BeginFrameSystem, Render, Surfaces } from "../../standard/render/core";
import { PrepassSystem } from "../../standard/sear/core";
import { ALBEDO_NAMES } from "./image";
import { materialPreamble } from "./shade";

// The live joint-palette skinning substrate — the runtime twin of the VAT (skin.ts / vat.ts). Where the VAT
// bakes a clip to two textures the skin surface samples by play-time, a *live* skinned instance is posed
// each frame by a producer (a physics ragdoll, a scripted driver): the producer writes per-joint object-space
// transforms into a per-instance **palette**, and the `skin-live` surface's `vs` blends them per vertex.
// Because skinning is a VS warp, sear's shadow + prepass passes deform for free, exactly like the VAT.
//
// All the per-instance palettes and all the per-mesh joints/weights live in ONE storage binding — `skinData`
// — to stay under the 10-storage-buffer ceiling (gpu.md; the skin surface has zero headroom). The buffer is
// block-concatenated (gpu.md consolidation #4): region A holds the dynamic per-instance palette blocks at
// the front (so instance bases stay stable as it grows), region B the static per-mesh joints/weights after
// it. This file owns the CPU-side layout arithmetic + the pose-write API + the flush; the surface `vs` (the
// GPU reader) and the importer (the JW producer) build on it. The LBS blend math it must reproduce is the
// matrix linear-blend skinning `bakeVat` implements — the equivalence gate in the tests pins that.

// skinData element = one 16-byte vec4 (u32 lanes for the header/JW, f32 lanes bitcast for the palette Xforms).
const VEC4_BYTES = 16;
// a palette entry is an Xform (XFORM_WGSL: pos+pad, quat, scale+pad) = 12 floats = 3 vec4.
export const PALETTE_STRIDE = 3;
// each instance block leads with a header vec4 (packed color, jointCount, flags) — color folded here so the
// separate `color` slab binding can be dropped (consolidation #3), keeping the surface at 10 storage buffers.
export const HEADER_VEC4 = 1;
// initial region capacities in vec4; both double on overflow.
const INITIAL_PALETTE_CAP = 64;
const INITIAL_JW_CAP = 64;
// residual above which a skin matrix isn't a similarity transform (shear / non-uniform-scale-under-rotation)
// and a decomposed Xform palette entry can't represent it faithfully — the import path warns past this.
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

/** read an instance block's header — the inverse of {@link writeHeader}. */
export function readHeader(
    u32: Uint32Array,
    base: number,
): { color: number; jointCount: number; flags: number } {
    const o = base * 4;
    return { color: u32[o], jointCount: u32[o + 1], flags: u32[o + 2] };
}

const _trs = new Float32Array(10);
const _recompose = new Float32Array(16);

/** decompose a skin matrix (column-major mat4) into an Xform palette entry written to `f32` (the skinData
 *  buffer's f32 view) at float offset `off`, returning the decompose residual — the max abs element
 *  difference between the source and the recomposed T·R·S. A residual past {@link SKIN_SHEAR_EPSILON} means
 *  the matrix carries shear / non-uniform-scale-under-rotation a TRS triple can't hold; the import path
 *  warns on it once per asset (never per frame — the residual is ignored on the pose-write path). */
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

// rotate `v` by quaternion `q` (xyzw) — the CPU twin of XFORM_WGSL's `xformQuat`.
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
const _params = new Uint32Array(4);

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
    paletteAB: new ArrayBuffer(INITIAL_PALETTE_CAP * VEC4_BYTES),
    palette: new Float32Array(INITIAL_PALETTE_CAP * 4),
    paletteU32: new Uint32Array(INITIAL_PALETTE_CAP * 4),
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
    // consumer round-trip. The importer / gym binds the returned buffer via `Mesh.bindings.skinParams`.
    params: new Map<number, GPUBuffer>(),

    buffer: null as GPUBuffer | null,
    bufferVec4: 0,
    // a zero `skinParams` published globally so a `skin-live` surface's no-op draws over non-live meshes (Part
    // registers one Draw per instanced-surface × mesh) resolve their bind group — the `fallbackVat` shape. A
    // real live mesh overrides it per-draw via `Mesh.bindings.skinParams`, so only the 0-instance draws read it.
    fallbackParams: null as GPUBuffer | null,
    dirty: new Set<number>(),
    // capacity changed → the GPU buffer reallocs and both regions re-upload (region B's offset moved, so
    // every mesh's jwBase changed — the importer rewrites its skinParams uniform on this).
    layoutDirty: true,
    // region B changed without a capacity change → re-upload region B alone.
    jwDirty: false,

    /** reset to the empty layout (GltfPlugin.initialize), preserving the singleton across a State rebuild.
     *  Drops the GPU buffer so the next flush reallocs + republishes into the wiped `Compute.buffers`. */
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

    /** free the GPU buffers (GltfPlugin.dispose). */
    dispose(): void {
        this.buffer?.destroy();
        this.buffer = null;
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
        writeHeader(this.paletteU32, base, packLdrColor(1, 1, 1, 1), jointCount, 0);
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
     *  u8 joint slots, `weightsPacked[v]` a u32 of 4 unorm8 weights (the importer quantizes to these).
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
     *  when region A's capacity grows, so the importer re-reads it after a `layoutDirty` flush. */
    jwBaseOf(meshId: number): number {
        const m = this.meshes.get(meshId);
        return m ? this.paletteCap + m.local : 0;
    },

    /** the mesh's `skinParams` uniform (`jwBase`, `vertCount`, pad, pad), created lazily + owned here so a
     *  palette-growth realloc (which shifts jwBase) rewrites it in {@link LiveSkin.flush} without the
     *  consumer re-reading. The importer / gym binds the returned buffer via `Mesh.bindings.skinParams`.
     *  Call after {@link registerMesh} (the buffer seeds from the mesh's current jwBase). */
    paramsBuffer(device: GPUDevice, meshId: number): GPUBuffer {
        const existing = this.params.get(meshId);
        if (existing) return existing;
        const buf = device.createBuffer({
            label: `gltf-skin-params:${meshId}`,
            size: 16,
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
        _params[0] = this.paletteCap + m.local;
        _params[1] = m.vertCount;
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
            const packed = packLdrColor(_color[0], _color[1], _color[2], _color[3]);
            if (this.paletteU32[block.base * 4] !== packed) {
                this.paletteU32[block.base * 4] = packed;
                this.dirty.add(eid);
            }
        }

        const needVec4 = this.paletteCap + this.jwCap;
        if (!this.buffer || this.bufferVec4 < needVec4 || this.layoutDirty) {
            this.buffer?.destroy();
            this.buffer = device.createBuffer({
                label: "gltf-skin-data",
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
                    label: "gltf-skin-params-fallback",
                    size: 16,
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

// ---- the `skin-live` surface trio: the GPU reader of the palette substrate above. The runtime-posed twin
// of the VAT skin surface (skin.ts) — same material `shadePbr` path, same alpha-mode split (opaque / MASK
// clip / BLEND); its vs blends the live joint palette each frame where the VAT vs samples two baked textures.
// Registered beside the VAT surfaces by `GltfPlugin.initialize`; because skinning is a VS warp, sear's
// prepass + shadow passes deform for free, exactly like the VAT.

// the per-skinned-mesh constants the surface decodes against: `jwBase` (the vec4 index the mesh's
// joints/weights block starts at in `skinData` region B — shifts when region A's palette capacity grows, so
// the importer rewrites it on a layoutDirty flush; {@link LiveSkin.jwBaseOf}) and `vertCount` (the block's
// vertex extent). Each skinned mesh binds its own `skinParams` per-draw via `Mesh.bindings` (so N live meshes
// coexist); `vatParams` is the precedent. Declared in the preamble after the binding decl that references it —
// module-scope structs resolve order-free (like VatParams / MaterialData).
const SKIN_PARAMS_WGSL = /* wgsl */ `
struct SkinParams {
    jwBase: u32,
    vertCount: u32,
    pad0: u32,
    pad1: u32,
}`;

// the per-instance tint, read from the palette block's header (the color fold, gpu.md consolidation #3):
// `color` is packed into the header's first u32 — synced from the `Color` component by the flush — so the
// surface carries no separate `color` storage binding and stays at the 10-storage ceiling. `skin[eid].x` is
// the header's vec4 base in `skinData`; `unpackLdrColor` is sear-spliced for every surface.
const LIVE_TINT_WGSL = /* wgsl */ `
fn liveTint(e: u32) -> vec4<f32> {
    return unpackLdrColor(skinData[u32(skin[e].x)].x);
}`;

// the bindings every live-skin surface declares: the instancing convention (eids + transforms) + the folded
// `skin` slab (palette base in x, material palette index in y) + the shared material palette + the block-
// concat `skinData` buffer + the material's texture arrays/sampler (shared with the textured + VAT paths) +
// the per-mesh `skinParams` uniform. Storage count is 5 (eids/transforms/skin/materialData/skinData) + sear's
// shared 5 = 10, the ceiling (gpu.md), zero headroom: folding `color` into the palette header
// (LIVE_TINT_WGSL) is what buys the room for `skinData` versus the VAT skin surface's `color` binding. The
// texture arrays + the `skinParams` uniform are separate limits, not storage.
const liveSkinBindings: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    skin: { type: "storage", element: "vec4<f32>" },
    materialData: { type: "storage", element: "MaterialData" },
    skinData: { type: "storage", element: "vec4<u32>" },
    ...Object.fromEntries(ALBEDO_NAMES.map((n) => [n, { type: "texture-2d-array" } as Binding])),
    mr: { type: "texture-2d-array" },
    normalTex: { type: "texture-2d-array" },
    occlusion: { type: "texture-2d-array" },
    emissive: { type: "texture-2d-array" },
    albedoSamp: { type: "sampler" },
    skinParams: { type: "uniform", struct: "SkinParams" },
};

// SkinParams + the tint helper + the material map-set helpers (shadePbr / sampleAlbedo, specialized per
// material map-set — the same specialization the VAT + textured surfaces use)
const liveSkinPreamble = (variant: number) =>
    SKIN_PARAMS_WGSL + LIVE_TINT_WGSL + materialPreamble(variant);

// decode this vertex's 4 joint influences from `skinData` region B (keyed by `vidx`, the skinned mesh's local
// vertex index — 2 verts per vec4, 8 B/vertex, gpu.md rule 6), then blend the instance's palette Xforms
// (region A, based at `skin[eid].x`): `p' = Σ wᵢ·xformPoint(palette[base+1+jᵢ], localPos)` — algebraically the
// matrix LBS `bakeVat` bakes (the equivalence gate in live.test.ts pins them equal), so the palette entries
// being Xform-shaped lets the VS reuse the spliced XFORM_WGSL `xformPoint`/`xformNormal` verbatim (zero new
// transform WGSL). The normal blends as a plain vec3 and renormalizes — never oct across a blend (gpu.md
// rule 9, the VAT lesson). Palettes are object-space (root-relative), so the standard instance transform
// (`transforms[eid]`, applied here after the blend) still carries the skinned pose to world space — the
// instance's root stays the meaningful `Transform` in the firehose. Weights are pre-normalized at import, so
// there's no runtime renorm; a zero-weight influence skips its palette read (the memory-bound early-out, the
// `skinPoint` CPU twin's shape).
const LIVE_SKIN_VS = /* wgsl */ `
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

// the three alpha-mode surfaces share LIVE_SKIN_VS (the palette blend) + the `shadePbr` material path; only
// the blend mode + cutout discard differ, exactly like the VAT `skin*` trio. `mid` is the folded `skin[eid].y`
// palette index; the per-instance tint comes from the header (`liveTint`), not a `color` binding.
const liveSkinSurfaces: Surface[] = [
    {
        name: "skin-live",
        bindings: liveSkinBindings,
        specialize: (variant) => ({ preamble: liveSkinPreamble(variant) }),
        vs: LIVE_SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let base = sampleAlbedo(mid, uv).rgb * liveTint(eid).rgb;
        col = vec4<f32>(shadePbr(mid, uv, base, normalize(worldNormal), world), 1.0);`,
    },
    {
        name: "skin-live-clip",
        blend: "clip",
        bindings: liveSkinBindings,
        specialize: (variant) => ({ preamble: liveSkinPreamble(variant) }),
        vs: LIVE_SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let tex = sampleAlbedo(mid, uv);
        let c = liveTint(eid);
        let rgb = shadePbr(mid, uv, tex.rgb * c.rgb, normalize(worldNormal), world);
        if (tex.a * c.a < materialData[mid].cutoff) { discard; }
        col = vec4<f32>(rgb, 1.0);`,
    },
    {
        name: "skin-live-blend",
        blend: "alpha",
        bindings: liveSkinBindings,
        specialize: (variant) => ({ preamble: liveSkinPreamble(variant) }),
        vs: LIVE_SKIN_VS,
        fs: /* wgsl */ `
        let mid = u32(skin[eid].y);
        let tex = sampleAlbedo(mid, uv) * liveTint(eid);
        col = vec4<f32>(shadePbr(mid, uv, tex.rgb, normalize(worldNormal), world), tex.a);`,
    },
];

/**
 * register the three alpha-mode live-skin surfaces — opaque `skin-live` / MASK `skin-live-clip` (cutout →
 * holed shadows) / BLEND `skin-live-blend`. The runtime-posed twin of the VAT `registerSkinSurfaces`: the
 * same material `shadePbr` path + alpha split, over the live joint palette ({@link LiveSkin}) where the VAT
 * samples baked textures. Called by `GltfPlugin.initialize` beside the VAT surfaces.
 */
export function registerLiveSkinSurfaces(): void {
    for (const s of liveSkinSurfaces) Surfaces.register(s);
}

/** the live-skin surface name per glTF alphaMode — the importer routes each live instance by its
 *  material's mode, the `skinSurface` twin. */
export function liveSkinSurface(alphaMode: "OPAQUE" | "MASK" | "BLEND"): string {
    return alphaMode === "MASK"
        ? "skin-live-clip"
        : alphaMode === "BLEND"
          ? "skin-live-blend"
          : "skin-live";
}

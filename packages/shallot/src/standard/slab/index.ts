import type { TgpuBindGroup, TgpuBuffer, TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import {
    Compute,
    capacity,
    f32,
    type Pair,
    type Plugin,
    type Quad,
    type Single,
    type System,
    type Type,
    type TypedArray,
} from "../../engine";
import { entries } from "../../engine/ecs/core";
import { precompile } from "../../engine/runtime";
import { allocMembership, MembershipSystem } from "./membership";
import {
    compiled,
    elementBytes,
    elementOf,
    resetPipelines,
    scatterKey,
    scatterPipeline,
} from "./scatter";

// Toji's persistent-staging pattern — packing dirty bits straight into a mapped buffer, then scattering
// them on the GPU, beats a per-slot `writeBuffer` at every K measured: CPU-side pack+encode+submit is
// 0.020 ms vs 0.240 ms at K=1024 and 0.68 ms vs 59.7 ms at K=65536 (lovelace, 2026-07-29, a throwaway
// gym microbench). Validated on the real GPU by the `render` gym scenario (transport round-trip assert
// + `slab:flush` span).

const warned = new Set<string>();

// WebGPU constraint: MAP_WRITE buffers can only combine with COPY_SRC. The
// stager can't itself be a STORAGE binding, hence the separate scatter
// sources we copy into.
function createStager(device: GPUDevice, bytes: number): GPUBuffer {
    return device.createBuffer({
        label: "slab-staging",
        size: bytes,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
    });
}

/**
 * GPU-mirrored per-entity storage. parameterized by {@link Type}: scalar
 * types yield a {@link Single}, `vec2` yields a {@link Pair}, `vec4` yields
 * a {@link Quad}. {@link SlabSystem} flushes dirty slots into the canonical
 * GPU buffer once per frame via Toji's persistent-staging + scatter compute
 * (far cheaper than a per-slot `writeBuffer` at any K; the `render` gym
 * scenario exercises this)
 *
 * write-only by design. GPU→CPU readback is a different shape
 * (frame-stamped, opt-in, per-consumer extract) with its own primitive —
 * don't reuse for it. Purely GPU-derived per-entity data (no CPU writes
 * ever) shouldn't use it either; allocate a `capacity`-sized buffer at the
 * consumer level instead
 *
 * Types not native to WGSL (`u8`, `u16`) warn at construction and stay
 * CPU-only — pack into `u32` manually and use `slab(u32)` for GPU upload
 *
 * Constructed lazily, like {@link sparse}: the constructor allocates nothing,
 * so `slab(vec4)` is callable at module scope (a component field declared
 * inline, no placeholder). `SlabPlugin.initialize` walks the registered
 * components and {@link Slab.alloc}s every slab field once `capacity` is fixed
 */
export class Slab {
    private static _all: Slab[] = [];

    /** type descriptor — drives ctor, lanes, encode/decode, GPU element size */
    readonly type: Type;
    /** optional binding name — registers `.gpu` in `Compute.buffers` at warm */
    readonly name: string | null;
    /** capacity * lanes CPU storage; assigned by {@link alloc} at build time */
    array!: TypedArray;
    /** one bit per slot — word w covers eids w*32..w*32+31; cleared by flush */
    dirty!: Uint32Array;
    /** false for `u8`, `u16` — no native WGSL storage type, no GPU buffer */
    readonly gpuSupported: boolean;

    /** lane Singles for `vec2`/`vec4` types; sharing the master's storage + dirty bitmap */
    readonly x!: Single;
    readonly y!: Single;
    readonly z!: Single;
    readonly w!: Single;

    /** canonical GPU buffer; null until the first flush prepares it */
    gpu: GPUBuffer | null = null;
    /** the typed twin of {@link gpu} — published to `Compute.typed` under {@link name} */
    typed: TgpuBuffer<d.AnyWgslData> | null = null;
    private _slots: TgpuBuffer<d.AnyWgslData> | null = null;
    private _values: TgpuBuffer<d.AnyWgslData> | null = null;
    private _rawSlots: GPUBuffer | null = null;
    private _rawValues: GPUBuffer | null = null;
    private _bindGroup: TgpuBindGroup | null = null;
    // the pipeline with this slab's bind group already bound. `.with(...)` returns a fresh wrapper each
    // call, so binding once at prepare keeps the per-frame flush to one allocation per dispatch
    private _bound: TgpuComputePipeline | null = null;
    private readonly _stagingPool: GPUBuffer[] = [];
    // bumped by release(): a stager whose mapAsync resolves after its epoch ended belongs to a
    // torn-down build (prior size, possibly prior device) and must be destroyed, not re-pooled
    private _epoch = 0;

    constructor(type: Type = f32, name: string | null = null) {
        this.type = type;
        this.name = name;
        this.gpuSupported = type.wgsl !== null;

        if (!this.gpuSupported && !warned.has(type.name)) {
            warned.add(type.name);
            const packFactor = type.name === "u8" ? 4 : 2;
            console.warn(
                `[slab] "${type.name}" is not a WGSL storage type. Slab stays CPU-only — ` +
                    `pack ${packFactor} ${type.name} values into one u32 manually and use slab(u32) for GPU upload.`,
            );
        }

        // lanes read `array`/`dirty` off `this` each call (arrows capture the
        // instance), not a captured local — the storage is (re)allocated per build
        // by `alloc`, so a captured array would go stale after the first build.
        const stride = type.lanes;
        if (stride >= 2) {
            const enc = type.encode;
            const dec = type.decode;
            const lane = (offset: number): Single => ({
                set: enc
                    ? (eid, v) => {
                          this.array[eid * stride + offset] = enc(v);
                          this.dirty[eid >>> 5] |= 1 << (eid & 31);
                      }
                    : (eid, v) => {
                          this.array[eid * stride + offset] = v;
                          this.dirty[eid >>> 5] |= 1 << (eid & 31);
                      },
                get: dec
                    ? (eid) => dec(this.array[eid * stride + offset])
                    : (eid) => this.array[eid * stride + offset],
                type,
                gpu: null,
            });
            (this as { x: Single }).x = lane(0);
            (this as { y: Single }).y = lane(1);
            if (stride === 4) {
                (this as { z: Single }).z = lane(2);
                (this as { w: Single }).w = lane(3);
            }
        }
    }

    /**
     * allocate the CPU storage at the now-fixed `capacity`. Called per build by
     * {@link Slab.collect} (for every slab field of a registered component);
     * reallocates fresh, so a rebuild starts from zeroed data. The `.gpu` mirror
     * is created separately by {@link prepare} at warm.
     */
    alloc(): void {
        this.array = new this.type.ctor(capacity * this.type.lanes);
        this.dirty = new Uint32Array((capacity + 31) >>> 5);
    }

    /**
     * write a slot and mark it dirty. arity matches lane count: scalar takes
     * 1 value, `vec2` takes 2, `vec4` takes 4
     */
    set(eid: number, x: number, y?: number, z?: number, w?: number): void {
        const enc = this.type.encode;
        const lanes = this.type.lanes;
        const base = eid * lanes;
        const array = this.array;
        array[base] = enc ? enc(x) : x;
        if (lanes >= 2) array[base + 1] = enc ? enc(y as number) : (y as number);
        if (lanes === 4) {
            array[base + 2] = enc ? enc(z as number) : (z as number);
            array[base + 3] = enc ? enc(w as number) : (w as number);
        }
        this.dirty[eid >>> 5] |= 1 << (eid & 31);
    }

    /** scalar read — meaningful for `lanes === 1` slabs only */
    get(eid: number): number {
        const dec = this.type.decode;
        const v = this.array[eid * this.type.lanes];
        return dec ? dec(v) : v;
    }

    /** bulk read for `vec2`/`vec4` slabs — copies all lanes into `out` */
    read(eid: number, out: Float32Array): Float32Array {
        const dec = this.type.decode;
        const lanes = this.type.lanes;
        const base = eid * lanes;
        const array = this.array;
        for (let i = 0; i < lanes; i++) {
            const v = array[base + i];
            out[i] = dec ? dec(v) : v;
        }
        return out;
    }

    prepare(): void {
        if (this.gpu || !this.gpuSupported) return;
        const element = elementOf(this.type)!;
        const ctx = compiled(this.type);
        if (!ctx) {
            throw new Error(
                `[slab] scatter pipeline for "${this.type.name}" not compiled — ` +
                    `declare SlabPlugin as a dependency so warm() compiles the pipeline.`,
            );
        }
        const root = Compute.root;
        const values = d.arrayOf(element, capacity);
        this.typed = root
            .createBuffer(values)
            .$usage("storage")
            .$name(`slab-canonical-${this.type.name}`);
        this._slots = root
            .createBuffer(d.arrayOf(d.u32, capacity + 1))
            .$usage("storage")
            .$name(`slab-slots-${this.type.name}`);
        this._values = root
            .createBuffer(values)
            .$usage("storage")
            .$name(`slab-values-${this.type.name}`);
        this.gpu = root.unwrap(this.typed);
        this._rawSlots = root.unwrap(this._slots);
        this._rawValues = root.unwrap(this._values);
        // the layout and these buffers are built from the same `element`, so they agree by construction —
        // but `element` is a runtime value, so TS sees a bare `AnyWgslData` against the layout's inferred
        // element union and can't check the pairing.
        this._bindGroup = root.createBindGroup(ctx.layout, {
            slots: this._slots,
            values: this._values,
            canonical: this.typed,
        } as never);
        this._bound = ctx.pipeline.with(this._bindGroup);
        if (this.name) {
            Compute.buffers.set(this.name, this.gpu);
            Compute.typed.set(this.name, this.typed);
        }
    }

    // Multi-lane types copy `lanes` consecutive CPU elements per slot — any lane set dirties the whole
    // slot. A packed type (`type.gpu`) instead folds its lanes into `gpu.bytes / 4` u32 words per slot
    // (`srgb8x4` → 1, `f16x4` → 2) — the CPU array stays the full lossless `lanes`, only the mirror packs.
    private pack(stager: GPUBuffer): number {
        const lanes = this.type.lanes;
        const gpu = this.type.gpu;
        const range = stager.getMappedRange();
        const slotView = new Uint32Array(range, 0, capacity + 1);
        const valueOffset = (capacity + 1) * 4;
        const dirty = this.dirty;
        const array = this.array;
        let count = 0;
        if (gpu) {
            const words = gpu.bytes >>> 2; // u32 words per slot: srgb8x4 → 1, f16x4 → 2
            const valueView = new Uint32Array(range, valueOffset, capacity * words);
            for (let w = 0; w < dirty.length; w++) {
                let bits = dirty[w];
                if (bits === 0) continue;
                const base = w << 5;
                while (bits !== 0) {
                    const lsb = bits & -bits;
                    const eid = base + (31 - Math.clz32(lsb));
                    const src = eid * lanes;
                    slotView[1 + count] = eid;
                    gpu.pack(
                        valueView,
                        count * words,
                        array[src],
                        array[src + 1],
                        array[src + 2],
                        array[src + 3],
                    );
                    count++;
                    bits ^= lsb;
                }
                dirty[w] = 0;
            }
            slotView[0] = count;
            stager.unmap();
            return count;
        }
        const valueElements = capacity * lanes;
        const Ctor = this.type.ctor as unknown as new (
            buffer: ArrayBuffer,
            byteOffset: number,
            length: number,
        ) => TypedArray;
        const valueView = new Ctor(range, valueOffset, valueElements);
        for (let w = 0; w < dirty.length; w++) {
            let bits = dirty[w];
            if (bits === 0) continue;
            const base = w << 5;
            while (bits !== 0) {
                const lsb = bits & -bits;
                const bit = 31 - Math.clz32(lsb);
                const eid = base + bit;
                slotView[1 + count] = eid;
                if (lanes === 1) {
                    valueView[count] = array[eid];
                } else {
                    const src = eid * lanes;
                    const dst = count * lanes;
                    for (let l = 0; l < lanes; l++) valueView[dst + l] = array[src + l];
                }
                count++;
                bits ^= lsb;
            }
            dirty[w] = 0;
        }
        slotView[0] = count;
        stager.unmap();
        return count;
    }

    private release(): void {
        if (this.name && Compute.buffers?.get(this.name) === this.gpu) {
            Compute.buffers.delete(this.name);
            Compute.typed?.delete(this.name);
        }
        this.typed?.destroy();
        this._slots?.destroy();
        this._values?.destroy();
        for (const s of this._stagingPool) s.destroy();
        this.gpu = null;
        this.typed = null;
        this._slots = null;
        this._values = null;
        this._rawSlots = null;
        this._rawValues = null;
        this._bindGroup = null;
        this._bound = null;
        this._stagingPool.length = 0;
        this._epoch++;
    }

    static reset(): void {
        for (const s of Slab._all) s.release();
        Slab._all.length = 0;
        warned.clear();
        // Pipelines bind to the device they were compiled against; clearing
        // on reset forces recompile when a new build comes up with a fresh
        // device (every test).
        resetPipelines();
    }

    /**
     * the per-build slab roster: release the prior build's slabs, then walk the
     * registered components and `alloc` every slab field at the now-fixed
     * `capacity`, collecting them into `_all` for flush + prepare. Slab lifetime
     * tracks component registration — a component declared inline (`pos:
     * slab(vec4)`) is allocated iff its plugin is registered, and `clear()`
     * (between tests) drops the registry so the next build starts clean.
     */
    static collect(): void {
        Slab.reset();
        for (const { component } of entries()) {
            for (const field of Object.values(component)) {
                if (field instanceof Slab) {
                    field.alloc();
                    Slab._all.push(field);
                }
            }
        }
    }

    /** allocate the canonical buffer + scatter bind group for every live slab, then queue one forced
     *  compile per element type — typegpu creates pipelines synchronously and Dawn defers the real
     *  compile to the first dispatch, so without this the first frame pays it */
    static prepareAll(): void {
        for (const s of Slab._all) if (s.gpuSupported) s.prepare();
        const forced = new Set<string>();
        for (const s of Slab._all) {
            const key = scatterKey(s.type);
            if (!s._bound || forced.has(key)) continue;
            forced.add(key);
            const bound = s._bound;
            precompile(`slab-scatter-${key}`, () => {
                bound.dispatchWorkgroups(0);
                return bound;
            });
        }
    }

    /** unique gpu-supported types across every live slab — used at warm time */
    static gpuTypes(): Type[] {
        const seen = new Map<string, Type>();
        for (const s of Slab._all) {
            if (s.gpuSupported) seen.set(scatterKey(s.type), s.type);
        }
        return [...seen.values()];
    }

    static flush(): void {
        if (Slab._all.length === 0) return;
        const device = Compute.device;
        const encoder = device.createCommandEncoder({ label: "slab-flush" });
        const used: { slab: Slab; stager: GPUBuffer; count: number }[] = [];

        for (const slab of Slab._all) {
            if (!slab.gpu) continue;
            const dirty = slab.dirty;
            let anyDirty = false;
            for (let w = 0; w < dirty.length; w++) {
                if (dirty[w] !== 0) {
                    anyDirty = true;
                    break;
                }
            }
            if (!anyDirty) continue;
            const bytes = elementBytes(slab.type)!;
            const stagerBytes = (capacity + 1) * 4 + capacity * bytes;
            const stager = slab._stagingPool.pop() ?? createStager(device, stagerBytes);
            const count = slab.pack(stager);
            encoder.copyBufferToBuffer(stager, 0, slab._rawSlots!, 0, (count + 1) * 4);
            encoder.copyBufferToBuffer(
                stager,
                (capacity + 1) * 4,
                slab._rawValues!,
                0,
                count * bytes,
            );
            used.push({ slab, stager, count });
        }

        if (used.length === 0) return;

        // One compute pass for all slabs — each dispatch rebinds its own group; the pass is shared, which
        // saves N-1 beginComputePass/endPass round-trips.
        const pass = encoder.beginComputePass({
            label: "slab-scatter",
            timestampWrites: Compute.span?.("slab:flush"),
        });
        for (const { slab, count } of used) {
            slab._bound!.with(pass).dispatchWorkgroups(Math.ceil(count / 64));
        }
        pass.end();

        device.queue.submit([encoder.finish()]);
        for (const { slab, stager } of used) {
            const epoch = slab._epoch;
            stager
                .mapAsync(GPUMapMode.WRITE)
                .then(() => {
                    if (slab._epoch === epoch) slab._stagingPool.push(stager);
                    else stager.destroy();
                })
                .catch(() => {});
        }
    }
}

/**
 * typed slab factory: mirrors `sparse(...)` so swapping `sparse(f32)` for
 * `slab(f32)` is a one-token change. Scalar types return a {@link Single};
 * `vec2` returns a {@link Pair}; `vec4` returns a {@link Quad}. Bulk `set`
 * matches the lane count; partial writes go through the lane accessors.
 * Pass an optional `name` to publish the canonical GPU buffer under that
 * name in `Compute.buffers` once allocated — and its typed twin under the
 * same name in `Compute.typed`, for a consumer binding through a schema;
 * surfaces resolve bindings against that registry, so named slabs become
 * shader-visible by name
 *
 * @example
 * const Health = { current: slab(f32), max: slab(f32) };
 * Health.current.set(eid, 100);
 *
 * @example
 * const pos = slab(vec4);
 * pos.set(eid, 1.5, 0, 0, 1);  // typed bulk write — one fn call, one dirty bit
 * pos.x.set(eid, 2.0);          // per-lane (parser path, partial updates)
 * pos.gpu                       // canonical vec4 buffer for surface binding
 *
 * @example
 * const Pulse = { value: slab(f32, "pulse") };
 * // Surface { bindings: { pulse: { type: "storage", element: "f32" } } }
 * // resolves `pulse` to `Pulse.value.gpu` via Compute.buffers
 */
export function slab(type: Type & { readonly lanes: 1 }, name?: string): Single;
export function slab(type: Type & { readonly lanes: 2 }, name?: string): Pair;
export function slab(type: Type & { readonly lanes: 4 }, name?: string): Quad;
export function slab(type: Type, name?: string): Single | Pair | Quad {
    return new Slab(type, name ?? null) as unknown as Single | Pair | Quad;
}

/**
 * per-frame flush of every slab. Runs at the head of the draw group so any
 * draw-group consumer sees the just-uploaded canonical buffer this frame.
 * `mode: "always"`; a live authoring host builds with `mode: "edit"`, and the transform
 * compose firehose this feeds must upload there too, or the viewport renders nothing.
 */
export const SlabSystem: System = {
    group: "draw",
    annotations: { mode: "always" },
    first: true,
    update() {
        Slab.flush();
    },
};

/**
 * owns the scatter pipeline and per-frame slab flush. Runs first among plugins
 * (everyone declares `dependencies: [SlabPlugin]`), so `initialize` allocates
 * every registered component's slab fields before any other plugin's
 * `initialize` reads or seeds them.
 */
export const SlabPlugin: Plugin = {
    name: "Slab",
    systems: [SlabSystem, MembershipSystem],

    initialize() {
        Slab.collect();
    },

    // `warm` is the GPU-setup phase, so a device is assumed here — every call below allocates against
    // it, and a headless build with a gpu-supported slab registered fails loud at the first one
    warm(state) {
        for (const t of Slab.gpuTypes()) scatterPipeline(t);
        Slab.prepareAll();
        allocMembership(state);
    },

    dispose() {
        Slab.reset();
    },
};

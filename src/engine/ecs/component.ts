import { packColor4 } from "../utils";
import type { Entity } from "./entity";

/** SoA component schema: each field names a type; worlds own the columns. */
export type Component = Record<string, unknown>;

const FIELD = Symbol("component field");

/** A component field's immutable schema. It carries no entity data or GPU handle. */
export interface FieldSchema<T extends Type = Type> {
    readonly [FIELD]: true;
    readonly type: T;
}

export type SchemaField<T extends Type> = FieldSchema<T> &
    (T["lanes"] extends 1 ? Single : T["lanes"] extends 2 ? Pair : Quad);

let activeState: { of(component: Component): Record<string, unknown> } | undefined;
const owners = new WeakMap<FieldSchema, { component: Component; name: string }>();

/** @internal Set the world used by legacy component handles during an engine callback. */
export function useState(state: { of(component: Component): Record<string, unknown> }): void {
    activeState = state;
}

/** @internal Bind a schema's field handles to the component that declares them. */
export function bindFields(component: Component): void {
    for (const name of Object.keys(component)) {
        const field = component[name];
        if (isFieldSchema(field)) {
            Object.freeze(field);
            owners.set(field, { component, name });
        }
    }
    Object.freeze(component);
}

function bound(field: FieldSchema): Single | Pair | Quad {
    const owner = owners.get(field);
    if (!owner || !activeState) throw new Error("Component fields require state.of(Component)");
    return activeState.of(owner.component)[owner.name] as Single | Pair | Quad;
}

/** create a schema descriptor; values and columns remain owned by its State. @internal */
export function fieldSchema<T extends Type>(type: T): SchemaField<T> {
    let descriptor!: SchemaField<T>;
    descriptor = {
        [FIELD]: true as const,
        type,
        write(eids: Uint32Array, source: TypedArray) {
            bound(descriptor).write(eids, source);
        },
        set(eid: number, x: number, y = 0, z = 0, w = 0) {
            const storage = bound(descriptor);
            if (descriptor.type.lanes === 1) {
                (storage as Single).set(eid, x);
            } else if (descriptor.type.lanes === 2) {
                (storage as Pair).set(eid, x, y);
            } else {
                (storage as Quad).set(eid, x, y, z, w);
            }
        },
        get get() {
            return (bound(descriptor) as Single).get;
        },
        read(eid: number, out: Float32Array) {
            return (bound(descriptor) as Pair | Quad).read(eid, out);
        },
        get x(): Single {
            return (bound(descriptor) as Pair | Quad).x;
        },
        get y(): Single {
            return (bound(descriptor) as Pair | Quad).y;
        },
        get z(): Single {
            return (bound(descriptor) as Quad).z;
        },
        get w(): Single {
            return (bound(descriptor) as Quad).w;
        },
        get column() {
            return (bound(descriptor) as Single | Pair | Quad).column;
        },
        markChanged(eid: number) {
            bound(descriptor).markChanged(eid);
        },
    };
    return descriptor;
}

/** true for a field descriptor, not a world-bound column. @internal */
export function isFieldSchema(value: unknown): value is FieldSchema {
    return !!value && typeof value === "object" && (value as FieldSchema)[FIELD] === true;
}

/** typed-array element backing for component columns. */
export type TypedArray = Float32Array | Int32Array | Uint32Array | Uint16Array | Uint8Array;

/**
 * typed-array storage descriptor. Shared between {@link Single}/{@link Pair}/{@link Quad}
 * fields so a consumer can change the type without
 * changing the type spelling. Metadata only. Descriptors don't carry state.
 *
 * @expand
 */
export interface Type<TArray extends TypedArray = TypedArray> {
    /** typed-array constructor used to back CPU storage */
    readonly ctor: {
        readonly BYTES_PER_ELEMENT: number;
        new (length: number): TArray;
    };
    /** scalar = 1, vec2 = 2, vec4 = 4. stride into the backing array per eid */
    readonly lanes: 1 | 2 | 4;
    /** debug label */
    readonly name: string;
    /** WGSL element type for GPU storage bindings. null for types without native WGSL storage (u8, u16) */
    readonly wgsl: string | null;
    /** JS number → array-slot value. omit for identity-mapped types */
    readonly encode?: (v: number) => number;
    /** array-slot value → JS number. omit for identity-mapped types */
    readonly decode?: (raw: number) => number;
    /**
     * a packed GPU mirror: the CPU storage stays the full `ctor`×`lanes` (so `.set` / lane accessors /
     * `read` / serialize see lossless floats). A GPU producer may use the `pack(...)` form:
     * what the per-lane {@link encode} can't express, since it folds across lanes (4 lanes → an `srgb8x4`
     * u32, or → an `f16x4` `vec2<u32>` pair). Quantization is a storage-boundary concern: the
     * pack runs once at the per-frame flush, the CPU side never sees it. The reader shader binds `wgsl`.
     */
    readonly gpu?: {
        readonly wgsl: string;
        readonly bytes: number;
        /** write `bytes / 4` u32 words at `out[at..]` from the four CPU lanes */
        pack(out: Uint32Array, at: number, x: number, y: number, z: number, w: number): void;
    };
}

/** 32-bit IEEE float. */
export const f32: Type<Float32Array> & { readonly lanes: 1 } = {
    ctor: Float32Array,
    lanes: 1,
    name: "f32",
    wgsl: "f32",
};

/** 32-bit signed integer. */
export const i32: Type<Int32Array> & { readonly lanes: 1 } = {
    ctor: Int32Array,
    lanes: 1,
    name: "i32",
    wgsl: "i32",
};

/** 32-bit unsigned integer. */
export const u32: Type<Uint32Array> & { readonly lanes: 1 } = {
    ctor: Uint32Array,
    lanes: 1,
    name: "u32",
    wgsl: "u32",
};

/**
 * a u32 that holds an entity id, a `@name` reference in scene files (`Joint.a`,
 * `Spring.b`). Storage is identical to {@link u32}; the distinct descriptor lets the field
 * declare itself a ref, so `serialize` round-trips it by the target's scene id rather than the
 * recycled, creation-order eid, with no side list to keep in sync. {@link refs} enumerates them.
 */
export const entity: Type<Uint32Array> & { readonly lanes: 1 } = {
    ctor: Uint32Array,
    lanes: 1,
    name: "entity",
    wgsl: "u32",
};

/** @internal compare storage and conversion semantics, never a Type's debug label. */
export function sameTypeLayout(a: Type, b: Type): boolean {
    if (
        a.ctor !== b.ctor ||
        a.ctor.BYTES_PER_ELEMENT !== b.ctor.BYTES_PER_ELEMENT ||
        a.lanes !== b.lanes ||
        a.wgsl !== b.wgsl ||
        a.encode !== b.encode ||
        a.decode !== b.decode
    ) {
        return false;
    }
    const ag = a.gpu;
    const bg = b.gpu;
    if (ag?.wgsl !== bg?.wgsl || ag?.bytes !== bg?.bytes || ag?.pack !== bg?.pack) return false;
    return a !== entity && b !== entity ? true : a === b;
}

/**
 * 8-bit unsigned CPU column. WGSL has no u8 storage type; a GPU record uses u32.
 */
export const u8: Type<Uint8Array> & { readonly lanes: 1 } = {
    ctor: Uint8Array,
    lanes: 1,
    name: "u8",
    wgsl: null,
};

/**
 * 16-bit unsigned CPU column. WGSL has no u16 storage type; a GPU record uses u32.
 */
export const u16: Type<Uint16Array> & { readonly lanes: 1 } = {
    ctor: Uint16Array,
    lanes: 1,
    name: "u16",
    wgsl: null,
};

// IEEE 754 binary16 codec — scratch buffer aliases an f32 over a u32 for the
// bit-pattern extraction. Module-scoped to avoid per-call allocation.
const F16_BUF = new ArrayBuffer(4);
const F16_F32 = new Float32Array(F16_BUF);
const F16_U32 = new Uint32Array(F16_BUF);

function f16encode(x: number): number {
    F16_F32[0] = x;
    const bits = F16_U32[0];
    const sign = (bits >>> 16) & 0x8000;
    const exp = ((bits >>> 23) & 0xff) - 127 + 15;
    const mantissa = bits & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return sign;
        const m = (mantissa | 0x800000) >>> (1 - exp);
        return sign | (m >>> 13);
    }
    if (exp >= 31) {
        // A finite f32 overflowing f16 range saturates to signed infinity;
        // only a genuine NaN/Infinity input carries its payload through.
        if (!Number.isFinite(x)) return sign | 0x7c00 | (mantissa ? 1 : 0);
        return sign | 0x7c00;
    }
    return sign | (exp << 10) | (mantissa >>> 13);
}

function f16decode(bits: number): number {
    const sign = bits & 0x8000;
    const exp = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exp === 0) {
        if (mantissa === 0) return sign ? -0 : 0;
        return (sign ? -1 : 1) * mantissa * 2 ** -24;
    }
    if (exp === 31) return mantissa ? Number.NaN : sign ? -Infinity : Infinity;
    return (sign ? -1 : 1) * (1 + mantissa / 1024) * 2 ** (exp - 15);
}

/**
 * 16-bit IEEE float. CPU storage uses `Uint16Array` of bit patterns; reads
 * and writes go through the half-float codec. The reader shader binds a native
 * `f16`, which needs an `enable f16` directive — `shader-f16` is NOT on the
 * platform floor, so a native-f16 GPU consumer declares it in its own
 * `Plugin.features`. For four half lanes with no feature at all, use {@link f16x4}.
 */
export const f16: Type<Uint16Array> & { readonly lanes: 1 } = {
    ctor: Uint16Array,
    lanes: 1,
    name: "f16",
    wgsl: "f16",
    encode: f16encode,
    decode: f16decode,
};

/** two f32 lanes. */
export const vec2: Type<Float32Array> & { readonly lanes: 2 } = {
    ctor: Float32Array,
    lanes: 2,
    name: "vec2",
    wgsl: "vec2<f32>",
};

/**
 * four f32 lanes. use for any 3-or-4-lane data: `vec3<f32>` is clobbered to
 * stride 16 in WebGPU storage anyway, so a true 3-lane type wouldn't save
 * memory. put something useful in `.w` (mass paired with position, opacity
 * with RGB) or leave it 0.
 */
export const vec4: Type<Float32Array> & { readonly lanes: 4 } = {
    ctor: Float32Array,
    lanes: 4,
    name: "vec4",
    wgsl: "vec4<f32>",
};

/**
 * a GPU mirror of four lanes as two u32 words holding two f16 each (16 B → 8 B) — the byte layout of
 * WebGPU's `float16x4`. The CPU surface is identical to {@link vec4} and sees lossless f32 (`set`,
 * `.x/.y/.z/.w`, `read`, serialize); only the mirror packs to half-floats at flush, and the reader shader
 * binds `vec2<u32>` and decodes with `unpack2x16float` — core WGSL, no `enable f16` and no `shader-f16`
 * feature (those gate the `f16` *type*, not the pack/unpack builtins). HDR-capable (range to 65504) and
 * finer than unorm8 across [0,1] (~15k representable values vs 256), so it suits PBR material params
 * (metallic / roughness / occlusion) alongside an unbounded emissive glow strength.
 */
export const f16x4: Type<Float32Array> & { readonly lanes: 4 } = {
    ctor: Float32Array,
    lanes: 4,
    name: "f16x4",
    wgsl: "vec2<u32>",
    gpu: {
        wgsl: "vec2<u32>",
        bytes: 8,
        // low 16 bits of a word hold the earlier of its two lanes, so the reader's
        // `unpack2x16float(word0)` recovers lanes 0+1 and `unpack2x16float(word1)` lanes 2+3
        pack: (out, at, x, y, z, w) => {
            out[at] = (f16encode(x) | (f16encode(y) << 16)) >>> 0;
            out[at + 1] = (f16encode(z) | (f16encode(w) << 16)) >>> 0;
        },
    },
};

/**
 * an LDR color mirrored to one u32: four 8-bit lanes, sRGB transfer on rgb + linear alpha (WebGPU's
 * `rgba8unorm-srgb` semantics), 16 B → 4 B. The CPU surface is identical to {@link vec4} and sees
 * lossless linear floats; only the GPU mirror packs (sRGB-encoding rgb on store), and the reader shader
 * binds a `u32` and decodes with `unpackLdrColor` (`engine/utils/encode.ts`). For `Part.Color` and any
 * LDR per-entity color: sRGB storage keeps perceptual precision in 8 bits.
 */
export const srgb8x4: Type<Float32Array> & { readonly lanes: 4 } = {
    ctor: Float32Array,
    lanes: 4,
    name: "srgb8x4",
    wgsl: "u32",
    gpu: {
        wgsl: "u32",
        bytes: 4,
        pack: (out, at, r, g, b, a) => {
            out[at] = packColor4(r, g, b, a);
        },
    },
};

/**
 * per-entity scalar storage. one value per entity, read/written by eid.
 * Component fields that need dirty tracking, GPU mirroring, or other
 * lifecycle behavior expose this instead of a bare typed array. `state.add`
 * routes default values through `.set`, so defaults flow into dirty bits
 * automatically. GPU consumers declare record tables separately.
 */
export interface Single {
    /** Copy encoded typed rows in eid order; source must match this lane's element type. */
    write(eids: Uint32Array, source: TypedArray): void;
    set(eid: number, value: number): void;
    get(eid: number): number;
    /** type descriptor — needed for surface binding (WGSL element type) */
    readonly type: Type;
    /** world-owned CPU column, including each vector lane in field order */
    readonly column: TypedArray;
    /** Publish this eid after a raw column write. Resolve column again after growth. */
    markChanged(eid: number): void;
}

/**
 * per-entity 2-lane storage. one vec2 per entity. `set` writes both lanes at
 * once (AoS), the perf-friendly path. `x` and `y` are per-lane
 * {@link Single} accessors sharing the master's storage; partial writes go
 * through them and dirty the whole slot. `read` copies both lanes into an
 * out param without allocation
 */
export interface Pair {
    /** Copy encoded two-lane typed rows in eid order and mark each entity changed. */
    write(eids: Uint32Array, source: TypedArray): void;
    set(eid: number, x: number, y: number): void;
    read(eid: number, out: Float32Array): Float32Array;
    readonly x: Single;
    readonly y: Single;
    readonly type: Type;
    readonly column: TypedArray;
    /** Publish this eid after a raw column write. Resolve column again after growth. */
    markChanged(eid: number): void;
}

/**
 * per-entity 4-lane storage. one vec4 per entity. shape matches {@link Pair}
 * with two more lanes
 */
export interface Quad {
    /** Copy encoded four-lane typed rows in eid order and mark each entity changed. */
    write(eids: Uint32Array, source: TypedArray): void;
    set(eid: number, x: number, y: number, z: number, w: number): void;
    read(eid: number, out: Float32Array): Float32Array;
    readonly x: Single;
    readonly y: Single;
    readonly z: Single;
    readonly w: Single;
    readonly type: Type;
    readonly column: TypedArray;
    /** Publish this eid after a raw column write. Resolve column again after growth. */
    markChanged(eid: number): void;
}

/**
 * structural lane-count detector. Returns 1 / 2 / 4 for a {@link Single} /
 * {@link Pair} / {@link Quad}; 0 for anything else (TypedArray, plain Array,
 * non-storage object, primitive, null). Discriminates by shape so lane
 * `Single`s of a parent Quad (which inherit the parent's `type.lanes`)
 * report as `Single` (1), not their parent's lane count
 */
export function lanes(value: unknown): 0 | 1 | 2 | 4 {
    if (isFieldSchema(value)) return value.type.lanes;
    if (!value || typeof value !== "object") return 0;
    const v = value as Record<string, unknown>;
    if (typeof v.set !== "function") return 0;
    // Classify by an actual lane handle, not just a property name.
    if (v.z != null && v.w != null) return 4;
    if (v.x != null && v.y != null) return 2;
    if (typeof v.get === "function") return 1;
    return 0;
}

/**
 * a component's typed storage fields: each {@link Single} / {@link Pair} /
 * {@link Quad} store paired with its declared name, in declaration order. The
 * canonical enumerator of a clean component's stores, for a serializer, a
 * reflection reader, or a schema walk. Keys with no typed layout (a GPU-buffer getter)
 * report {@link lanes} 0 and are skipped.
 */
export function fields(component: Component): { name: string; field: FieldSchema }[] {
    const out: { name: string; field: FieldSchema }[] = [];
    for (const name of Object.keys(component)) {
        const field = component[name];
        if (isFieldSchema(field)) out.push({ name, field });
    }
    return out;
}

/** @internal compare component storage shape independently of its field declaration order. */
export function sameComponentSchema(a: Component, b: Component): boolean {
    const af = fields(a).sort((left, right) => left.name.localeCompare(right.name));
    const bf = fields(b).sort((left, right) => left.name.localeCompare(right.name));
    if (af.length !== bf.length) return false;
    for (let i = 0; i < af.length; i++) {
        const left = af[i];
        const right = bf[i];
        if (left.name !== right.name || !sameTypeLayout(left.field.type, right.field.type)) {
            return false;
        }
    }
    return true;
}

/**
 * the fields holding an entity ref: those declared `field(entity)`.
 * `serialize` reads it to emit each as `@<id>`; the ref-ness lives on the field's type, so it
 * can't drift from a separate list. A sibling of {@link fields}.
 */
export function refs(component: Component): string[] {
    const out: string[] = [];
    for (const name of Object.keys(component)) {
        if (isFieldSchema(component[name]) && component[name].type === entity) out.push(name);
    }
    return out;
}

// Stable component identity. A component's id is interned by name at
// registration (`intern`) and resolves back to the same id when a reloaded
// module hands in a fresh component object under the same name — so membership,
// queries, and storage re-attach across a hot swap, the component object being
// the one thing a module reload recreates. An unregistered component (a bare
// test marker) auto-mints an anonymous id on first sight, stable for the
// object's lifetime. Process-global and monotonic: ids never reset, so no id is
// ever reused for a different name (a `clear()` between sessions leaves them intact).
// The WeakMap keeps identity off the frozen schema object and out of field walks.
const _idByComponent = new WeakMap<object, number>();
const _idByName = new Map<string, number>();
let _nextId = 0;

/**
 * the component's stable numeric id, the key for membership and query
 * structures. Auto-mints an anonymous id for an unregistered component;
 * {@link intern} binds it by name at registration so a reloaded handle (a fresh
 * object) resolves to the same id.
 */
export function idOf(component: object): number {
    const id = _idByComponent.get(component);
    if (id !== undefined) return id;
    const next = _nextId++;
    _idByComponent.set(component, next);
    return next;
}

/**
 * intern the stable id for `name` on `component`: first sight assigns one
 * (adopting an id the component auto-minted while bare), and re-registration
 * under the same name resolves to it, the reload contract that re-attaches a
 * fresh module object. Called by `register`.
 */
export function intern(component: object, name: string): number {
    let id = _idByName.get(name);
    if (id === undefined) {
        id = _idByComponent.get(component) ?? _nextId++;
        _idByName.set(name, id);
    }
    _idByComponent.set(component, id);
    return id;
}

const BITS_PER_GEN = 31;

/**
 * read access to the component-membership bitset, exposed as `state.membership`.
 * A GPU producer that scans a buffer by index gates on it instead of a per-field
 * sentinel: `(membershipWord & mask) != 0` is the authoritative "does eid carry
 * this component" test. The standard membership mirror uploads the bitset each
 * frame; consumers read the published `"membership"` buffer.
 */
export interface Membership {
    /**
     * a component's gate coordinates — `gen` selects the membership word,
     * `mask` is the bit to test within it. Assigns a bit on first use
     */
    bit(component: object): { gen: number; mask: number };
    /** membership words per entity (31 components each); fixes the mirror size */
    readonly generations: number;
    /**
     * lock the generation count after `build` has assigned every registered
     * component its bit. A later component that would require a new generation
     * is refused rather than silently outsizing the fixed GPU mirror.
     */
    freeze(): void;
    /**
     * report every entity whose membership changed since the last call, then
     * clear the pending set; returns false when nothing changed. The standard
     * membership mirror is the sole caller — it copies each `(eid, gen, word)`
     * into its GPU staging
     */
    drain(visit: (eid: number, gen: number, word: number) => void): boolean;
}

/** per-entity component membership, packed as bitsets across generations of 31-bit masks */
export class Components implements Membership {
    private _nextBit = 0;
    private _gen = 0;
    private _frozen = false;
    // keyed by component id (idOf), not the object — a reloaded component handle
    // re-attaches by id. Array-by-id, since ids are small and monotonic.
    private _meta: ({ gen: number; bit: number } | undefined)[] = [];
    private _masks: number[][] = [[]];
    /** eids whose membership changed since the last {@link drain} */
    private _dirty = new Set<number>();

    has(eid: Entity, component: any): boolean {
        const m = this._meta[idOf(component)];
        if (!m) return false;
        return ((this._masks[m.gen][eid] ?? 0) & m.bit) !== 0;
    }

    add(eid: Entity, component: any): boolean {
        const m = this.ensure(component);
        const prev = this._masks[m.gen][eid] ?? 0;
        if (prev & m.bit) return false;
        this._masks[m.gen][eid] = prev | m.bit;
        this._dirty.add(eid);
        return true;
    }

    remove(eid: Entity, component: any): boolean {
        const m = this._meta[idOf(component)];
        if (!m) return false;
        const prev = this._masks[m.gen][eid] ?? 0;
        if (!(prev & m.bit)) return false;
        this._masks[m.gen][eid] = prev & ~m.bit;
        this._dirty.add(eid);
        return true;
    }

    clear(eid: Entity): void {
        for (let g = 0; g <= this._gen; g++) this._masks[g][eid] = 0;
        this._dirty.add(eid);
    }

    /** {@inheritDoc Membership.bit} */
    bit(component: any): { gen: number; mask: number } {
        const m = this.ensure(component);
        return { gen: m.gen, mask: m.bit };
    }

    /** {@inheritDoc Membership.generations} */
    get generations(): number {
        return this._gen + 1;
    }

    /** {@inheritDoc Membership.freeze} */
    freeze(): void {
        this._frozen = true;
    }

    /** {@inheritDoc Membership.drain} */
    drain(visit: (eid: number, gen: number, word: number) => void): boolean {
        if (this._dirty.size === 0) return false;
        const gens = this._gen + 1;
        for (const eid of this._dirty) {
            for (let g = 0; g < gens; g++) visit(eid, g, this._masks[g][eid] ?? 0);
        }
        this._dirty.clear();
        return true;
    }

    private ensure(component: any) {
        const id = idOf(component);
        const existing = this._meta[id];
        if (existing) return existing;
        if (this._nextBit >= BITS_PER_GEN) {
            if (this._frozen) {
                throw new Error(
                    `membership: generation count is frozen at ${this._gen + 1} after build; ` +
                        `a new component would require generation ${this._gen + 2} — ` +
                        `rebuild to register more than ${(this._gen + 1) * BITS_PER_GEN} components`,
                );
            }
            this._gen++;
            this._nextBit = 0;
            this._masks.push([]);
        }
        const m = { gen: this._gen, bit: 1 << this._nextBit++ };
        this._meta[id] = m;
        return m;
    }
}

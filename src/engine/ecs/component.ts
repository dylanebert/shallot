import type { Entity } from "./entity";

/** SoA component schema: each field names a type; worlds own the columns. */
export type Component = Record<string, unknown>;

/** Optional starting values for a component's declared fields. */
export type ComponentValues<T> = {
    [K in keyof T as T[K] extends FieldType ? K : never]?: T[K] extends FieldType
        ? T[K]["lanes"] extends 1
            ? number
            : T[K]["lanes"] extends 2
              ? readonly [number, number]
              : readonly [number, number, number, number]
        : never;
};

/** @internal Freeze a component's shared declarations, never its world's storage. */
export function freezeComponent(component: Component): void {
    for (const value of Object.values(component)) {
        if (isType(value)) Object.freeze(value);
    }
    Object.freeze(component);
}

/** @internal Recognize a bare CPU storage type, not a world-bound field handle. */
export function isType(value: unknown): value is FieldType {
    if (!value || typeof value !== "object") return false;
    const type = value as FieldType;
    return (
        typeof type.ctor === "function" &&
        typeof type.ctor.BYTES_PER_ELEMENT === "number" &&
        (type.lanes === 1 || type.lanes === 2 || type.lanes === 4) &&
        typeof type.name === "string" &&
        (type.wgsl === null || typeof type.wgsl === "string")
    );
}

/** typed-array element backing for component columns. */
export type TypedArray = Float32Array | Int32Array | Uint32Array | Uint16Array | Uint8Array;

/**
 * typed-array storage descriptor. Shared between {@link ScalarField}/{@link Vector2Field}/{@link Vector4Field}
 * fields so a consumer can change the type without
 * changing the type spelling. Metadata only. Descriptors don't carry state.
 */
export interface FieldType<TArray extends TypedArray = TypedArray> {
    /** typed-array constructor used to back CPU storage */
    readonly ctor: {
        readonly BYTES_PER_ELEMENT: number;
        new (length: number): TArray;
    };
    /** scalar = 1, vec2 = 2, vec4 = 4. stride into the backing array per eid */
    readonly lanes: 1 | 2 | 4;
    readonly name: string;
    /** WGSL element type, or null for u8 and u16, which WGSL cannot store. Tables bind TypeGPU
     * records, not this; it is part of the layout that hot reload compares. */
    readonly wgsl: string | null;
    /** JS number → array-slot value. omit for identity-mapped types */
    readonly encode?: (v: number) => number;
    /** array-slot value → JS number. omit for identity-mapped types */
    readonly decode?: (raw: number) => number;
}

/** 32-bit IEEE float. */
export const f32: FieldType<Float32Array> & { readonly lanes: 1 } = {
    ctor: Float32Array,
    lanes: 1,
    name: "f32",
    wgsl: "f32",
};

/** 32-bit signed integer. */
export const i32: FieldType<Int32Array> & { readonly lanes: 1 } = {
    ctor: Int32Array,
    lanes: 1,
    name: "i32",
    wgsl: "i32",
};

/** 32-bit unsigned integer. */
export const u32: FieldType<Uint32Array> & { readonly lanes: 1 } = {
    ctor: Uint32Array,
    lanes: 1,
    name: "u32",
    wgsl: "u32",
};

/**
 * a u32 that holds an entity id (`Joint.a`, `Spring.b`). Storage is identical to {@link u32}.
 */
export const entity: FieldType<Uint32Array> & { readonly lanes: 1 } = {
    ctor: Uint32Array,
    lanes: 1,
    name: "entity",
    wgsl: "u32",
};

/** @internal compare storage and conversion semantics, never a FieldType's debug label. */
export function sameTypeLayout(a: FieldType, b: FieldType): boolean {
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
    return a !== entity && b !== entity ? true : a === b;
}

/**
 * 8-bit unsigned CPU column. WGSL has no u8 storage type; a GPU record uses u32.
 */
export const u8: FieldType<Uint8Array> & { readonly lanes: 1 } = {
    ctor: Uint8Array,
    lanes: 1,
    name: "u8",
    wgsl: null,
};

/**
 * 16-bit unsigned CPU column. WGSL has no u16 storage type; a GPU record uses u32.
 */
export const u16: FieldType<Uint16Array> & { readonly lanes: 1 } = {
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
 * 16-bit IEEE float, stored as `Uint16Array` bit patterns; reads and writes
 * convert through the half-float codec. A GPU table refuses to bind it, as it
 * refuses every encoded field.
 */
export const f16: FieldType<Uint16Array> & { readonly lanes: 1 } = {
    ctor: Uint16Array,
    lanes: 1,
    name: "f16",
    wgsl: "f16",
    encode: f16encode,
    decode: f16decode,
};

/** two f32 lanes. */
export const vec2: FieldType<Float32Array> & { readonly lanes: 2 } = {
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
export const vec4: FieldType<Float32Array> & { readonly lanes: 4 } = {
    ctor: Float32Array,
    lanes: 4,
    name: "vec4",
    wgsl: "vec4<f32>",
};

/**
 * per-entity scalar storage. one value per entity, read/written by eid.
 * Component fields expose this instead of a bare typed array so writes
 * publish change marks and survive storage growth. `world.add`
 * writes defaults through `.set`, so defaults publish marks too. GPU
 * consumers declare record tables separately.
 */
export interface ScalarField {
    /** Copy encoded typed rows in eid order; source must match this lane's element type. */
    writeEncoded(eids: Uint32Array, source: TypedArray): void;
    set(eid: number, value: number): void;
    get(eid: number): number;
    readonly type: FieldType;
    /** world-owned CPU column, including each vector lane in field order */
    readonly column: TypedArray;
    /** Publish this eid after a raw column write. Resolve column again after growth. */
    markChanged(eid: number): void;
}

/**
 * per-entity 2-lane storage. one vec2 per entity. `set` writes both lanes at
 * once (AoS), the perf-friendly path. `x` and `y` are per-lane
 * {@link ScalarField} accessors sharing the master's storage; partial writes go
 * through them and dirty the whole slot. `read` copies both lanes into an
 * out param without allocation
 */
export interface Vector2Field {
    /** Copy encoded two-lane typed rows in eid order and mark each entity changed. */
    writeEncoded(eids: Uint32Array, source: TypedArray): void;
    set(eid: number, x: number, y: number): void;
    read(eid: number, out: Float32Array): Float32Array;
    readonly x: ScalarField;
    readonly y: ScalarField;
    readonly type: FieldType;
    readonly column: TypedArray;
    /** Publish this eid after a raw column write. Resolve column again after growth. */
    markChanged(eid: number): void;
}

/**
 * per-entity 4-lane storage. one vec4 per entity. shape matches {@link Vector2Field}
 * with two more lanes
 */
export interface Vector4Field {
    /** Copy encoded four-lane typed rows in eid order and mark each entity changed. */
    writeEncoded(eids: Uint32Array, source: TypedArray): void;
    set(eid: number, x: number, y: number, z: number, w: number): void;
    read(eid: number, out: Float32Array): Float32Array;
    readonly x: ScalarField;
    readonly y: ScalarField;
    readonly z: ScalarField;
    readonly w: ScalarField;
    readonly type: FieldType;
    readonly column: TypedArray;
    /** Publish this eid after a raw column write. Resolve column again after growth. */
    markChanged(eid: number): void;
}

/**
 * structural lane-count detector. Returns 1 / 2 / 4 for a {@link ScalarField} /
 * {@link Vector2Field} / {@link Vector4Field}; 0 for anything else (TypedArray, plain Array,
 * non-storage object, primitive, null). Discriminates by shape so lane
 * `ScalarField`s of a parent Vector4Field (which inherit the parent's `type.lanes`)
 * report as `ScalarField` (1), not their parent's lane count
 */
export function lanes(value: unknown): 0 | 1 | 2 | 4 {
    if (isType(value)) return value.lanes;
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
 * A component's declared bare storage types paired with their names, in declaration order.
 * This enumerates metadata for reflection and schema comparison, not a world's field handles.
 * Keys with no CPU storage type are skipped; resolve entity data with `world.storage(component)`.
 */
export function fields(component: Component): { name: string; field: FieldType }[] {
    const out: { name: string; field: FieldType }[] = [];
    for (const name of Object.keys(component)) {
        const field = component[name];
        if (isType(field)) out.push({ name, field });
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
        if (left.name !== right.name || !sameTypeLayout(left.field, right.field)) {
            return false;
        }
    }
    return true;
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

/** per-entity component membership, packed as bitsets across generations of 31-bit masks */
export class Components {
    private _nextBit = 0;
    private _gen = 0;
    // keyed by component id (idOf), not the object — a reloaded component handle
    // re-attaches by id. Array-by-id, since ids are small and monotonic.
    private _meta: ({ gen: number; bit: number } | undefined)[] = [];
    private _masks: number[][] = [[]];

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
        return true;
    }

    remove(eid: Entity, component: any): boolean {
        const m = this._meta[idOf(component)];
        if (!m) return false;
        const prev = this._masks[m.gen][eid] ?? 0;
        if (!(prev & m.bit)) return false;
        this._masks[m.gen][eid] = prev & ~m.bit;
        return true;
    }

    clear(eid: Entity): void {
        for (let g = 0; g <= this._gen; g++) this._masks[g][eid] = 0;
    }

    private ensure(component: any) {
        const id = idOf(component);
        const existing = this._meta[id];
        if (existing) return existing;
        if (this._nextBit >= BITS_PER_GEN) {
            this._gen++;
            this._nextBit = 0;
            this._masks.push([]);
        }
        const m = { gen: this._gen, bit: 1 << this._nextBit++ };
        this._meta[id] = m;
        return m;
    }
}

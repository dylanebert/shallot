import { linearToSrgb, srgbToLinear } from "../utils";
import { CHUNK_MASK, CHUNK_SHIFT, type ArrayKind, type Buf } from "./capacity";
import { toKebabCase } from "./strings";

/** SoA component — keys map to typed arrays indexed by entity */
export type Component = Record<string, unknown>;

/** computed field derived from other parsed fields */
export interface Derived {
    get(parsed: Record<string, number>): number;
    set(value: number, parsed: Record<string, number>): Record<string, number>;
}

/** component metadata — defaults, dependencies, parse/format */
export interface Traits {
    requires?: Component[];
    defaults?: () => Record<string, number>;
    parse?: Record<string, (value: string) => number | undefined>;
    format?: Record<string, (value: number) => string | undefined>;
    enums?: Record<string, Record<string, number>>;
    annotations?: Record<string, unknown>;
}

export function parseEnum(enumObj: Record<string, number>): (value: string) => number | undefined {
    const lookup = new Map<string, number>();
    for (const [key, val] of Object.entries(enumObj)) {
        lookup.set(toKebabCase(key), val);
    }
    return (value: string) => lookup.get(value);
}

export function formatEnum(enumObj: Record<string, number>): (value: number) => string | undefined {
    const lookup = new Map<number, string>();
    for (const [key, val] of Object.entries(enumObj)) {
        lookup.set(val, toKebabCase(key));
    }
    return (value: number) => lookup.get(value);
}

const traitsMap = new WeakMap<Component, Traits>();

/** attach metadata to a component */
export function traits(component: Component, t: Traits): void {
    if (t.enums) {
        const parse = t.parse ?? {};
        const format = t.format ?? {};
        for (const [field, enumObj] of Object.entries(t.enums)) {
            if (!parse[field]) parse[field] = parseEnum(enumObj);
            if (!format[field]) format[field] = formatEnum(enumObj);
        }
        t.parse = parse;
        t.format = format;
    }
    traitsMap.set(component, t);
}

export function getTraits(component: Component): Traits | undefined {
    return traitsMap.get(component);
}

export interface ComponentEntry {
    readonly component: Component;
    readonly name: string;
    readonly traits?: Traits;
}

const registry = new Map<string, ComponentEntry>();

export function registerComponent(name: string, component: Component): void {
    const kebabName = toKebabCase(name);
    const traits = traitsMap.get(component);
    registry.set(kebabName, { component, name: kebabName, traits });
}

export function getComponent(name: string): ComponentEntry | undefined {
    return registry.get(toKebabCase(name));
}

export function getComponents(): ComponentEntry[] {
    return [...registry.values()];
}

export function getComponentName(component: Component): string | undefined {
    for (const [name, entry] of registry) {
        if (entry.component === component) return name;
    }
    return undefined;
}

export function clearRegistry(): void {
    registry.clear();
}

export interface FieldProxy extends Array<number> {
    get(eid: number): number;
    set(eid: number, value: number): void;
}

/**
 * memory layout of a typed-array-backed field. external runtimes use this to
 * address chunk slots directly: `chunks[eid >>> CHUNK_SHIFT][(eid & CHUNK_MASK) * stride + offset]`.
 */
export interface FieldLayout {
    bufId: number;
    array: ArrayKind;
    stride: number;
    offset: number;
}

const proxyLayouts = new WeakMap<FieldProxy, FieldLayout>();

export function getFieldLayout(proxy: FieldProxy): FieldLayout | undefined {
    return proxyLayouts.get(proxy);
}

function recordLayout(proxy: FieldProxy, ref: Buf, stride: number, offset: number): void {
    proxyLayouts.set(proxy, { bufId: ref.id, array: ref.kind, stride, offset });
}

export function isStringField(component: Component, field: string): boolean {
    const val = component[field];
    if (val == null) return false;
    if (ArrayBuffer.isView(val) || Array.isArray(val)) return false;
    return typeof val === "object";
}

export function createFieldProxy(ref: Buf, stride: number, offset: number): FieldProxy {
    const chunks = ref.chunks;
    function getValue(eid: number): number {
        return chunks[eid >>> CHUNK_SHIFT][(eid & CHUNK_MASK) * stride + offset];
    }

    function setValue(eid: number, value: number): void {
        chunks[eid >>> CHUNK_SHIFT][(eid & CHUNK_MASK) * stride + offset] = value;
    }

    const proxy = new Proxy([] as unknown as FieldProxy, {
        get(_, prop) {
            if (prop === "get") return getValue;
            if (prop === "set") return setValue;
            const eid = Number(prop);
            if (Number.isNaN(eid)) return undefined;
            return getValue(eid);
        },
        set(_, prop, value) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return false;
            setValue(eid, value);
            return true;
        },
    });
    recordLayout(proxy, ref, stride, offset);
    return proxy;
}

export function createColorProxy(
    ref: Buf<Float32Array>,
    stride: number,
    offset: number,
): FieldProxy {
    const chunks = ref.chunks;
    function getValue(eid: number): number {
        const chunk = chunks[eid >>> CHUNK_SHIFT];
        const o = (eid & CHUNK_MASK) * stride + offset;
        const r = Math.round(linearToSrgb(chunk[o]) * 255);
        const g = Math.round(linearToSrgb(chunk[o + 1]) * 255);
        const b = Math.round(linearToSrgb(chunk[o + 2]) * 255);
        return (r << 16) | (g << 8) | b;
    }

    function setValue(eid: number, value: number): void {
        const chunk = chunks[eid >>> CHUNK_SHIFT];
        const o = (eid & CHUNK_MASK) * stride + offset;
        chunk[o] = srgbToLinear(((value >> 16) & 0xff) / 255);
        chunk[o + 1] = srgbToLinear(((value >> 8) & 0xff) / 255);
        chunk[o + 2] = srgbToLinear((value & 0xff) / 255);
        chunk[o + 3] = 1;
    }

    const proxy = new Proxy([] as unknown as FieldProxy, {
        get(_, prop) {
            if (prop === "get") return getValue;
            if (prop === "set") return setValue;
            const eid = Number(prop);
            if (Number.isNaN(eid)) return undefined;
            return getValue(eid);
        },
        set(_, prop, value) {
            const eid = Number(prop);
            if (Number.isNaN(eid)) return false;
            setValue(eid, value);
            return true;
        },
    });
    recordLayout(proxy, ref, stride, offset);
    return proxy;
}

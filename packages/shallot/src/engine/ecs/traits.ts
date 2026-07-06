import type { Alias, Input } from "../utils";
import type { Component, Pair, Quad, Single } from "./component";
import { fields, idOf, intern, lanes } from "./component";
import { kebab } from "./reflection";

/** parse-time metadata declared per component */
export interface Traits {
    requires?: Component[];
    /**
     * components this one stands in for — an entity carrying it satisfies another component's
     * `requires` of any listed component, without holding that component itself. `Body.provides =
     * [Transform]` (physics owns the entity's world transform, so `Body` excludes `Transform` yet a
     * `Part` on the same entity still renders). Directional (the counterpart to `requires`), read by
     * scene validation only, not enforced at `state.add`
     */
    provides?: Component[];
    /** one instance per scene (lights, the active camera). Informational — surfaced as editor
     * metadata, not enforced at `state.add` */
    singleton?: boolean;
    /**
     * runtime-derived decoration — a system owns its membership and values (the glTF route sync's
     * `Textured` / `Skin`), so scenes never author it: `serialize` skips it, the editor's add-component
     * picker and derived-section append hide it, and `diagnose` flags an authored attr. Registration
     * still allocates its storage (a slab field needs it), and an always-mode system may add/remove it
     * freely — the exemption from the edit-mode "never add/remove components" contract, since nothing
     * document-facing can see it
     */
    derived?: boolean;
    /**
     * components that cannot coexist on the same entity. Symmetric — declaring
     * `A.excludes = [B]` is equivalent to declaring `B.excludes = [A]`; both
     * directions are enforced at `state.add` and during scene validation
     */
    excludes?: Component[];
    /**
     * default field values, applied on `state.add`. Values are scalars for
     * Single fields and per-lane arrays for direct {@link Pair}/{@link Quad}
     * fields (`{ pos: [0, 0, 0, 0] }`). Dotted keys (`{ "pos.x": 0 }`)
     * address a single lane of a parent Pair/Quad
     */
    defaults?: () => Record<string, number | readonly number[]>;
    /** per-field authoring aliases — a stored vector field edited in an alternate representation */
    aliases?: Record<string, Alias>;
    parse?: Record<string, (value: string) => number | undefined>;
    format?: Record<string, (value: number) => string | undefined>;
    enums?: Record<string, Record<string, number>>;
    /** per-field editor input widget — a stored field shown through a richer control (a `toggle`
     * checkbox, an `angle` unit switcher). Display-only; storage is unchanged */
    inputs?: Record<string, Input>;
    annotations?: Record<string, unknown>;
}

interface DefaultsPlan {
    arrs: Array<number[] | Float32Array | Uint32Array>;
    arrVals: number[];
    fields: Single[];
    fieldVals: number[];
    pairs: Pair[];
    pairVals: Float32Array;
    quads: Quad[];
    quadVals: Float32Array;
}

interface Entry {
    component: Component;
    name: string;
    traits?: Traits;
    /** lazy-compiled defaults writer. undefined = unbuilt, null = no defaults. */
    plan?: DefaultsPlan | null;
}

const byName = new Map<string, Entry>();
// keyed by stable component id, so a handle held across any number of reloads
// resolves the current registration
const byId = new Map<number, Entry>();
// derived from the registered traits, keyed by stable component id, never
// accumulated — rebuilt lazily after any registration so a reload's removed
// exclude stops being enforced and a stale pre-reload handle resolves the
// current set through its id
let exclusions: Map<number, Set<Component>> | null = null;

function buildExclusions(): Map<number, Set<Component>> {
    const map = new Map<number, Set<Component>>();
    const link = (id: number, other: Component) => {
        let set = map.get(id);
        if (!set) map.set(id, (set = new Set()));
        set.add(other);
    };
    for (const entry of byName.values()) {
        for (const declared of entry.traits?.excludes ?? []) {
            const other = byId.get(idOf(declared))?.component ?? declared;
            link(idOf(entry.component), other);
            link(idOf(other), entry.component);
        }
    }
    return map;
}

function expandEnums(t: Traits): Traits {
    if (!t.enums) return t;
    const parse: NonNullable<Traits["parse"]> = { ...t.parse };
    const format: NonNullable<Traits["format"]> = { ...t.format };
    for (const [field, enumObj] of Object.entries(t.enums)) {
        const fwd = new Map<string, number>();
        const rev = new Map<number, string>();
        for (const [key, val] of Object.entries(enumObj)) {
            const k = kebab(key);
            fwd.set(k, val);
            rev.set(val, k);
        }
        parse[field] ??= (value: string) => fwd.get(value);
        format[field] ??= (value: number) => rev.get(value);
    }
    return { ...t, parse, format };
}

/** register a component under a name, with optional traits */
export function register(name: string, component: Component, traits?: Traits): void {
    const k = kebab(name);
    const prev = byName.get(k);
    if (prev && prev.component !== component) {
        // reload contract: a fresh component object handed in under an existing
        // name adopts the prior registration's stores, so runtime data and GPU
        // buffers survive the module swap. The id is reused by name below, so
        // membership + queries re-attach to the same slots.
        for (const f of fields(prev.component)) {
            if (f.name in component) (component as Record<string, unknown>)[f.name] = f.store;
        }
    }
    const id = intern(component, k);
    const expanded = traits ? expandEnums(traits) : undefined;
    const entry: Entry = expanded
        ? { component, name: k, traits: expanded }
        : { component, name: k };
    byName.set(k, entry);
    byId.set(id, entry);
    exclusions = null;
}

/** components that may not coexist with `component`. Symmetric over all declarations */
export function getExclusions(component: Component): ReadonlySet<Component> | undefined {
    exclusions ??= buildExclusions();
    return exclusions.get(idOf(component));
}

/** the registered component handle for a name, or `undefined` if none is registered under it */
export function getComponent(name: string): Component | undefined {
    return byName.get(kebab(name))?.component;
}

/** the parse-time `Traits` registered with a component name, or `undefined` if none */
export function getTraits(name: string): Traits | undefined {
    return byName.get(kebab(name))?.traits;
}

export function getName(component: Component): string | undefined {
    return byId.get(idOf(component))?.name;
}

/** iterate every registered component with its name and traits */
export function entries(): IterableIterator<{
    component: Component;
    name: string;
    traits?: Traits;
}> {
    return byName.values();
}

/** write default field values for an entity that just received `component` */
export function applyDefaults(component: Component, eid: number): void {
    const entry = byId.get(idOf(component));
    if (!entry) return;
    let plan = entry.plan;
    if (plan === undefined) {
        plan = entry.plan = compilePlan(entry);
    }
    if (plan === null) return;
    const { arrs, arrVals, fields, fieldVals, pairs, pairVals, quads, quadVals } = plan;
    for (let i = 0; i < arrs.length; i++) {
        arrs[i][eid] = arrVals[i];
    }
    for (let i = 0; i < fields.length; i++) {
        fields[i].set(eid, fieldVals[i]);
    }
    for (let i = 0; i < pairs.length; i++) {
        const o = i * 2;
        pairs[i].set(eid, pairVals[o], pairVals[o + 1]);
    }
    for (let i = 0; i < quads.length; i++) {
        const o = i * 4;
        quads[i].set(eid, quadVals[o], quadVals[o + 1], quadVals[o + 2], quadVals[o + 3]);
    }
}

const LANE_INDEX: Record<string, number> = { x: 0, y: 1, z: 2, w: 3 };

function compilePlan(entry: Entry): DefaultsPlan | null {
    const defaults = entry.traits?.defaults;
    if (!defaults) return null;
    const dict = defaults();
    const data = entry.component as Record<string, unknown>;
    const arrs: DefaultsPlan["arrs"] = [];
    const arrVals: number[] = [];
    const fields: Single[] = [];
    const fieldVals: number[] = [];
    const pairs: Pair[] = [];
    const pairValsList: number[] = [];
    const quads: Quad[] = [];
    const quadValsList: number[] = [];

    // dotted keys gather per-parent and resolve at the end so lane writes on
    // the same parent merge into one Pair/Quad bulk set
    const dotted = new Map<string, number[]>();

    for (const field in dict) {
        const value = dict[field];

        const dotIdx = field.indexOf(".");
        if (dotIdx !== -1) {
            const base = field.slice(0, dotIdx);
            const laneKey = field.slice(dotIdx + 1);
            const parent = data[base];
            const parentLanes = lanes(parent);
            if (parentLanes !== 2 && parentLanes !== 4) continue;
            if (typeof value !== "number") continue;
            const idx = LANE_INDEX[laneKey];
            if (idx === undefined || idx >= parentLanes) continue;
            let arr = dotted.get(base);
            if (!arr) {
                arr = new Array(parentLanes).fill(0);
                dotted.set(base, arr);
            }
            arr[idx] = value;
            continue;
        }

        const target = data[field];
        if (target == null) continue;
        const n = lanes(target);

        if (Array.isArray(value)) {
            if (n === 4) {
                quads.push(target as Quad);
                quadValsList.push(value[0] ?? 0, value[1] ?? 0, value[2] ?? 0, value[3] ?? 0);
            } else if (n === 2) {
                pairs.push(target as Pair);
                pairValsList.push(value[0] ?? 0, value[1] ?? 0);
            } else if (n === 1) {
                fields.push(target as Single);
                fieldVals.push(value[0] ?? 0);
            } else if (ArrayBuffer.isView(target) || Array.isArray(target)) {
                arrs.push(target as number[] | Float32Array | Uint32Array);
                arrVals.push(value[0] ?? 0);
            }
            continue;
        }

        if (typeof value !== "number") continue;

        // typed arrays also expose `.set`, so gate on ArrayBuffer.isView first
        if (ArrayBuffer.isView(target) || Array.isArray(target)) {
            arrs.push(target as number[] | Float32Array | Uint32Array);
            arrVals.push(value);
        } else if (typeof (target as Single).set === "function") {
            fields.push(target as Single);
            fieldVals.push(value);
        }
    }

    for (const [base, arr] of dotted) {
        const target = data[base];
        const n = lanes(target);
        if (n === 4) {
            quads.push(target as Quad);
            quadValsList.push(arr[0], arr[1], arr[2], arr[3]);
        } else if (n === 2) {
            pairs.push(target as Pair);
            pairValsList.push(arr[0], arr[1]);
        }
    }

    if (arrs.length === 0 && fields.length === 0 && pairs.length === 0 && quads.length === 0) {
        return null;
    }
    return {
        arrs,
        arrVals,
        fields,
        fieldVals,
        pairs,
        pairVals: new Float32Array(pairValsList),
        quads,
        quadVals: new Float32Array(quadValsList),
    };
}

/** wipe every registration — used between editor sessions and tests */
export function clear(): void {
    byName.clear();
    byId.clear();
    exclusions = null;
}

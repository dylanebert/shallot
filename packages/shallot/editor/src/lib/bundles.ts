import { dependencies, getComponent, kebab } from "@dylanebert/shallot/ecs/core";
import type { Node } from "@dylanebert/shallot/editor";
import { normalizeAttr } from "@dylanebert/shallot/scene/core";
import {
    Box,
    Camera,
    Circle,
    Diamond,
    Gamepad2,
    Lightbulb,
    Pill,
    Square,
    Sun,
    Sunrise,
} from "lucide-static";
import { CATEGORIES, type Category } from "./components";

/** one component of a {@link Bundle}: a registered component `name` (any casing — kebabed on use) and
 * the authoring `override` string a hand-author would type (`""` for a bare component). */
type Part = { name: string; override: string };

/**
 * a named cluster of components with authored defaults — the outliner Add menu's primitives. `instantiate`
 * turns one into an ordinary scene {@link Node}, so a bundle drops a ready-to-use entity in without the
 * author knowing which components to add. Pure data; the output is indistinguishable from hand-authored.
 */
export type Bundle = {
    /** menu label, e.g. `"Point Light"` */
    label: string;
    /** outliner id base, deduped against the document (`box`, `box-2`); omit for a nameless entity */
    id?: string;
    /** lucide-static icon markup */
    icon: string;
    /** palette token for the icon tint */
    color: string;
    /** category id for menu grouping, matching `components.ts` CATEGORIES; omit to sit ungrouped first */
    category?: string;
    /** the components, in emit order; missing dependencies fill after (bare) */
    parts: Part[];
};

const geometry = "var(--cat-rendering)";
const lighting = "var(--cat-lighting)";
const camera = "var(--cat-camera)";

/** the starter set. Field overrides match the zoo scenes' authored look; each placeable bundle carries
 * its own placement — geometry rests on the Ground bundle's top plane (y = 0), so the set composes into
 * a sensible scene as-added. */
export const BUNDLES: Bundle[] = [
    { label: "Empty", icon: Diamond, color: "var(--accent)", parts: [] },

    {
        label: "Box",
        id: "box",
        icon: Box,
        color: geometry,
        category: "geometry",
        parts: [
            { name: "part", override: "" },
            { name: "color", override: "rgba: 0.85 0.55 0.35" },
            { name: "transform", override: "pos: 0 0.5 0" },
        ],
    },
    {
        label: "Sphere",
        id: "sphere",
        icon: Circle,
        color: geometry,
        category: "geometry",
        parts: [
            { name: "part", override: "mesh: sphere" },
            { name: "color", override: "rgba: 0.45 0.65 0.85" },
            { name: "transform", override: "pos: 0 0.5 0" },
        ],
    },
    {
        label: "Capsule",
        id: "capsule",
        icon: Pill,
        color: geometry,
        category: "geometry",
        parts: [
            { name: "part", override: "mesh: capsule" },
            { name: "color", override: "rgba: 0.65 0.55 0.85" },
            // capsule spans y ±(halfHeight 0.5 + radius 0.5) — bottom cap kisses y = 0
            { name: "transform", override: "pos: 0 1 0" },
        ],
    },
    {
        label: "Ground",
        id: "ground",
        icon: Square,
        color: geometry,
        category: "geometry",
        parts: [
            { name: "part", override: "" },
            { name: "color", override: "rgba: 0.35 0.37 0.4" },
            { name: "transform", override: "pos: 0 -0.1 0; scale: 10 0.2 10" },
        ],
    },

    {
        label: "Point Light",
        id: "point-light",
        icon: Lightbulb,
        color: lighting,
        category: "lighting",
        parts: [
            { name: "point-light", override: "intensity: 1; range: 15" },
            { name: "transform", override: "pos: 0 3 0" },
        ],
    },
    {
        label: "Directional Light",
        id: "sun",
        icon: Sunrise,
        color: lighting,
        category: "lighting",
        parts: [
            {
                name: "directional-light",
                override: "direction: -0.4 -1 -0.55; color: 0xfff4e0; intensity: 1.2",
            },
            { name: "shadow", override: "distance: 20" },
        ],
    },
    {
        label: "Ambient Light",
        id: "ambient",
        icon: Sun,
        color: lighting,
        category: "lighting",
        parts: [{ name: "ambient-light", override: "intensity: 0.6" }],
    },

    {
        label: "Camera",
        id: "camera",
        icon: Camera,
        color: camera,
        category: "camera",
        parts: [
            { name: "camera", override: "" },
            { name: "sear", override: "" },
            { name: "transform", override: "pos: 0 2 6" },
        ],
    },

    {
        label: "Player",
        id: "player",
        icon: Gamepad2,
        color: "var(--cat-gameplay)",
        category: "gameplay",
        parts: [{ name: "player", override: "" }],
    },
];

/** a bundle is offered only when every component it names is registered — a plugin toggled off (physics,
 * player) hides its bundles with no second enablement source. */
export function available(bundle: Bundle): boolean {
    return bundle.parts.every((p) => getComponent(p.name) !== undefined);
}

/** a menu section: `category` is null for the leading ungrouped bundles (Empty), else its group header. */
export type BundleGroup = { category: Category | null; items: Bundle[] };

/** the available bundles grouped for the Add menu — ungrouped (Empty) first, then one section per
 * category in CATEGORIES order. A category with no available bundle is dropped, so the menu reflects the
 * live plugin set. */
export function menuGroups(): BundleGroup[] {
    const avail = BUNDLES.filter(available);
    const groups: BundleGroup[] = [];
    const ungrouped = avail.filter((b) => !b.category);
    if (ungrouped.length) groups.push({ category: null, items: ungrouped });
    for (const cat of CATEGORIES) {
        const items = avail.filter((b) => b.category === cat.id);
        if (items.length) groups.push({ category: cat, items });
    }
    return groups;
}

/**
 * turn a bundle into a scene node: canonicalize each override the way the scene formatter does (minimal
 * attrs, no default-noise), fill missing dependencies (transitively, bare), and mint a deduped id.
 *
 * @example
 * instantiate(BUNDLES[1], doc); // { id: "box", attrs: [part, color, transform], children: [] }
 */
export function instantiate(bundle: Bundle, doc: { nodes: Node[] }): Node {
    const attrs: { name: string; value: string }[] = [];
    const added = new Set<string>();
    const add = (name: string, value: string) => {
        if (added.has(name)) return;
        added.add(name);
        attrs.push({ name, value });
    };

    for (const part of bundle.parts) {
        const name = kebab(part.name);
        add(name, part.override ? (normalizeAttr(name, part.override) ?? part.override) : "");
    }

    // dependency closure, breadth-first from the authored components — a bare attr for each missing
    // requirement (the fill `addComponent` does, extended to the transitive case, e.g. player → body → transform)
    for (let i = 0; i < attrs.length; i++) {
        for (const dep of dependencies(attrs[i].name)) if (!added.has(dep)) add(dep, "");
    }

    const node: Node = { attrs, children: [] };
    if (bundle.id) node.id = mintId(bundle.id, doc);
    return node;
}

/** an id base, suffixed (`box`, `box-2`, `box-3`) until it clears every id in the document — plus any
 *  extra `taken` ids (a multi-node mint dedupes against its own batch; the minted id is added for the
 *  caller). Shared by bundle instantiation and the model import's node minting. */
export function mintId(base: string, doc: { nodes: Node[] }, taken?: Set<string>): string {
    const ids = new Set(taken);
    for (const n of doc.nodes) if (n.id) ids.add(n.id);
    let id = base;
    for (let n = 2; ids.has(id); n++) id = `${base}-${n}`;
    taken?.add(id);
    return id;
}

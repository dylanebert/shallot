import { beforeEach, describe, expect, test } from "bun:test";
import {
    CharacterPlugin,
    DEFAULT_PLUGINS,
    f32,
    MirrorPlugin,
    PhysicsPlugin,
    PlayerPlugin,
    type Plugin,
    parse,
    sparse,
    stringify,
} from "@dylanebert/shallot";
import { clear, dependencies, register } from "@dylanebert/shallot/ecs/core";
import type { Node } from "@dylanebert/shallot/editor";
import { Meshes, Surfaces } from "@dylanebert/shallot/render/core";
import { normalizeAttr } from "@dylanebert/shallot/scene/core";
import { available, BUNDLES, type Bundle, instantiate, menuGroups } from "./bundles";

// register every plugin's components + traits the way `build()` does (registration is order-independent —
// requires resolve as long as the target component objects are registered). No GPU: `register` stages the
// traits without calling `defaults()`, so this runs at the `bun test` tier.
function registerSet(plugins: readonly Plugin[]): void {
    for (const p of plugins) {
        const comps = p.components ?? {};
        const traits = p.traits ?? {};
        for (const [name, comp] of Object.entries(comps)) register(name, comp, traits[name]);
    }
}

// `mesh()` no-ops without a GPU device, so warm never registers the built-in geometry — stub the names the
// Part `mesh`/`surface` codec resolves, the runtime registry state a bundle override round-trips against.
function stubRegistries(): void {
    Meshes.clear();
    Surfaces.clear();
    for (const name of ["cube", "sphere", "capsule"])
        Meshes.register({ name, indexBase: 0, indexCount: 0 } as never);
    for (const name of ["default", "unlit", "vertex"]) Surfaces.register({ name } as never);
}

function doc(nodes: Node[] = []): { nodes: Node[] } {
    return { nodes };
}

// bundles whose components resolve under DEFAULT_PLUGINS (Player needs physics — its own describe).
const core = () => BUNDLES.filter((b) => b.label !== "Player");

describe("bundles — DEFAULT_PLUGINS", () => {
    beforeEach(() => {
        clear();
        registerSet(DEFAULT_PLUGINS);
        stubRegistries();
    });

    test("every core bundle's components resolve; Player is filtered out", () => {
        for (const b of core()) expect(available(b)).toBe(true);
        expect(available(BUNDLES.find((b) => b.label === "Player")!)).toBe(false);
    });

    test("every field override is codec-canonical (formatFields∘parseFields fixed point)", () => {
        for (const b of core()) {
            for (const part of b.parts) {
                if (!part.override) continue;
                const once = normalizeAttr(part.name, part.override);
                expect(once).not.toBeNull();
                // idempotent → the authored override already is the minimal canonical form
                expect(normalizeAttr(part.name, once!)).toBe(once);
            }
        }
    });

    test("instantiate emits minimal attrs — every value is canonical, no default-noise", () => {
        for (const b of core()) {
            const node = instantiate(b, doc());
            for (const attr of node.attrs) {
                if (attr.value === "") continue; // bare component
                expect(normalizeAttr(attr.name, attr.value)).toBe(attr.value);
            }
        }
    });

    test("Box authors a warm cube resting on the ground plane", () => {
        const node = instantiate(BUNDLES.find((b) => b.label === "Box")!, doc());
        expect(node.id).toBe("box");
        expect(node.attrs.map((a) => a.name)).toEqual(["part", "color", "transform"]);
        expect(node.attrs.find((a) => a.name === "color")!.value).toBe("rgba: 0.85 0.55 0.35");
        expect(node.attrs.find((a) => a.name === "part")!.value).toBe("");
        expect(node.attrs.find((a) => a.name === "transform")!.value).toBe("pos: 0 0.5 0");
    });

    test("every geometry bundle sits on the Ground bundle's top plane (y = 0)", () => {
        // unit primitives: box/sphere half-extent 0.5, capsule half-height + radius = 1
        const rest: Record<string, number> = { Box: 0.5, Sphere: 0.5, Capsule: 1 };
        for (const [label, y] of Object.entries(rest)) {
            const node = instantiate(BUNDLES.find((b) => b.label === label)!, doc());
            expect(node.attrs.find((a) => a.name === "transform")!.value).toBe(`pos: 0 ${y} 0`);
        }
    });

    test("Sphere points Part at the sphere mesh by name", () => {
        const node = instantiate(BUNDLES.find((b) => b.label === "Sphere")!, doc());
        expect(node.attrs.find((a) => a.name === "part")!.value).toBe("mesh: sphere");
    });

    test("Ground carries its own placement", () => {
        const node = instantiate(BUNDLES.find((b) => b.label === "Ground")!, doc());
        expect(node.attrs.find((a) => a.name === "transform")!.value).toBe(
            "pos: 0 -0.1 0; scale: 10 0.2 10",
        );
    });

    test("Point Light fills its required transform", () => {
        const node = instantiate(BUNDLES.find((b) => b.label === "Point Light")!, doc());
        expect(node.attrs.some((a) => a.name === "point-light")).toBe(true);
        expect(node.attrs.some((a) => a.name === "transform")).toBe(true);
    });

    test("Empty is a nameless bare entity", () => {
        const node = instantiate(BUNDLES.find((b) => b.label === "Empty")!, doc());
        expect(node.id).toBeUndefined();
        expect(node.attrs).toEqual([]);
    });

    test("id dedupes against an occupied document", () => {
        const box = BUNDLES.find((b) => b.label === "Box")!;
        const occupied = doc([
            { id: "box", attrs: [], children: [] },
            { id: "box-2", attrs: [], children: [] },
        ]);
        expect(instantiate(box, occupied).id).toBe("box-3");
        expect(instantiate(box, doc()).id).toBe("box");
    });

    test("menuGroups leads with Empty (ungrouped) and drops the unavailable Player group", () => {
        const groups = menuGroups();
        expect(groups[0].category).toBeNull();
        expect(groups[0].items.map((b) => b.label)).toEqual(["Empty"]);
        const all = groups.flatMap((g) => g.items.map((b) => b.label));
        expect(all).toContain("Box");
        expect(all).not.toContain("Player"); // physics off → filtered
        // grouped sections carry a header category
        expect(groups.slice(1).every((g) => g.category !== null)).toBe(true);
    });

    test("a bundle entity round-trips as an ordinary scene entity", () => {
        for (const b of core()) {
            const node = instantiate(b, doc());
            const [back] = parse(stringify([node]));
            expect(back.id).toBe(node.id);
            expect(back.attrs).toEqual(node.attrs);
        }
    });
});

describe("bundles — dependency closure", () => {
    beforeEach(() => clear());

    test("Player resolves and its component closure fills once physics is loaded", () => {
        registerSet([
            ...DEFAULT_PLUGINS,
            MirrorPlugin,
            PhysicsPlugin,
            CharacterPlugin,
            PlayerPlugin,
        ]);
        const player = BUNDLES.find((b) => b.label === "Player")!;
        expect(available(player)).toBe(true);
        const names = instantiate(player, doc()).attrs.map((a) => a.name);
        expect(names).toContain("player");
        expect(names).toContain("body");
        expect(names).toContain("character");
    });

    test("dependencies fill transitively (a → b → c)", () => {
        const C = { z: sparse(f32) };
        const B = { y: sparse(f32) };
        const A = { x: sparse(f32) };
        register("cc", C, { defaults: () => ({ z: 0 }) });
        register("bb", B, { requires: [C], defaults: () => ({ y: 0 }) });
        register("aa", A, { requires: [B], defaults: () => ({ x: 0 }) });
        expect(dependencies("aa")).toEqual(["bb"]); // guards the one-level baseline the fill extends

        const bundle: Bundle = {
            label: "A",
            icon: "",
            color: "",
            parts: [{ name: "aa", override: "" }],
        };
        expect(instantiate(bundle, doc()).attrs.map((a) => a.name)).toEqual(["aa", "bb", "cc"]);
    });
});

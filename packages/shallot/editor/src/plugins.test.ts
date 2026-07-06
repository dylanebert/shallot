import { describe, expect, test } from "bun:test";
import {
    CharacterPlugin,
    FogPlugin,
    InputPlugin,
    MirrorPlugin,
    OrbitPlugin,
    OutlinePlugin,
    PhysicsPlugin,
    PlayerPlugin,
    type Plugin,
    RenderPlugin,
} from "@dylanebert/shallot";
import {
    compose,
    EDITOR_SUBSTRATES,
    enabledPlugins,
    entriesFor,
    isTooling,
    localPlugins,
    SHALLOT_PLUGINS,
    STANDARD_PLUGINS,
    TOOLING_PLUGINS,
} from "./plugins";
import type { Manifest } from "./project/manifest";

const RENDER = RenderPlugin.name; // a default
const ORBIT = OrbitPlugin.name; // an extra
const spin = { name: "Spin", systems: [{ update() {} }] } as unknown as Plugin;
const locals = [{ name: "Spin", plugin: spin }];

describe("entriesFor", () => {
    test("every catalog plugin appears with its source + enabled state, plus declared locals", () => {
        const entries = entriesFor({ plugins: { Spin: "./src/spin" } }, locals);
        const by = new Map(entries.map((e) => [e.name, e]));
        expect(by.get(RENDER)?.source).toBe("default");
        expect(by.get(RENDER)?.enabled).toBe(true); // default on
        expect(by.get(ORBIT)?.source).toBe("extra");
        expect(by.get(ORBIT)?.enabled).toBe(false); // extra off unless enabled
        expect(by.get("Spin")).toMatchObject({ source: "project", enabled: true, plugin: spin });
    });

    test("a disabled default and an enabled extra resolve to their objects", () => {
        const manifest: Manifest = { plugins: { [RENDER]: false, [ORBIT]: true } };
        const by = new Map(entriesFor(manifest, []).map((e) => [e.name, e]));
        expect(by.get(RENDER)?.enabled).toBe(false);
        expect(by.get(ORBIT)?.enabled).toBe(true);
        expect(by.get(ORBIT)?.plugin).toBe(OrbitPlugin);
    });

    test("an arbitrary engine plugin resolves to its object — physics by name, no hand-listing", () => {
        // SHALLOT_PLUGINS derives from the barrel, so a manifest enables any `*Plugin` by name and the
        // editor builds it — the substrate the physics-using showcases (sandbox) stand on.
        const by = new Map(entriesFor({ plugins: { Physics: true } }, []).map((e) => [e.name, e]));
        expect(by.get("Physics")?.source).toBe("extra");
        expect(by.get("Physics")?.enabled).toBe(true);
        expect(by.get("Physics")?.plugin).toBe(PhysicsPlugin);
    });
});

describe("SHALLOT_PLUGINS (barrel-derived extras)", () => {
    test("includes every non-default engine plugin, so a manifest can name it", () => {
        // the derive replaces a hand list — the plugins the showcases need are present without an edit
        for (const p of [PhysicsPlugin, CharacterPlugin, PlayerPlugin, FogPlugin, OutlinePlugin]) {
            expect(SHALLOT_PLUGINS).toContain(p);
        }
    });

    test("excludes the defaults — they're STANDARD_PLUGINS, never doubled as an extra", () => {
        const extras = new Set(SHALLOT_PLUGINS.map((p) => p.name));
        for (const d of STANDARD_PLUGINS) expect(extras.has(d.name)).toBe(false);
    });
});

describe("enabledPlugins", () => {
    test("returns the enabled objects — defaults on, extras only when enabled, locals when declared", () => {
        const out = enabledPlugins(
            entriesFor({ plugins: { [ORBIT]: true, Spin: "./src/spin" } }, locals),
        );
        expect(out).toContain(RenderPlugin); // a default
        expect(out).toContain(OrbitPlugin); // enabled extra
        expect(out).toContain(spin); // enabled local
        expect(out.length).toBe(STANDARD_PLUGINS.length + 2);
    });

    test("a disabled default drops out of the built set", () => {
        const out = enabledPlugins(entriesFor({ plugins: { [RENDER]: false } }, []));
        expect(out).not.toContain(RenderPlugin);
        expect(out.length).toBe(STANDARD_PLUGINS.length - 1);
    });
});

describe("localPlugins", () => {
    test("returns only enabled project-source plugins — the swap's prev/next", () => {
        const out = localPlugins(
            entriesFor({ plugins: { Spin: "./src/spin", [ORBIT]: true } }, locals),
        );
        expect(out).toEqual([spin]);
        for (const std of STANDARD_PLUGINS) expect(out).not.toContain(std);
        expect(out).not.toContain(OrbitPlugin); // an engine extra, not a local
    });

    test("a disabled local is excluded", () => {
        const out = localPlugins(entriesFor({ plugins: { Spin: ["./src/spin", false] } }, locals));
        expect(out).toEqual([]);
    });
});

describe("layer axis", () => {
    const tooling = {
        name: "T",
        systems: [{ annotations: { layer: "tooling" }, update() {} }],
    } as unknown as Plugin;
    const app = { name: "A", systems: [{ update() {} }] } as unknown as Plugin;

    test("isTooling is true only when a system declares the tooling layer", () => {
        expect(isTooling(tooling)).toBe(true);
        expect(isTooling(app)).toBe(false);
        expect(isTooling({ name: "N" } as Plugin)).toBe(false);
    });

    test("the editor's tooling plugins are all tooling-layer", () => {
        expect(TOOLING_PLUGINS.length).toBeGreaterThan(0);
        for (const p of TOOLING_PLUGINS) expect(isTooling(p)).toBe(true);
    });

    test("the app-base engine plugins are not tooling — they ship with the game", () => {
        for (const p of [InputPlugin, RenderPlugin, OrbitPlugin]) expect(isTooling(p)).toBe(false);
    });
});

describe("compose", () => {
    const A = { name: "a", systems: [{ update() {} }] } as unknown as Plugin;

    test("play builds the app set verbatim — nothing forced on", () => {
        expect(compose("play", [A])).toEqual([A]);
        expect(compose("play", [])).toEqual([]);
    });

    test("edit composes the editor foundation over the app", () => {
        const names = compose("edit", [A]).map((p) => p.name);
        expect(names).toContain("a");
        for (const p of EDITOR_SUBSTRATES) expect(names).toContain(p.name);
        expect(names).toContain(OrbitPlugin.name);
        expect(names).toContain(MirrorPlugin.name);
        for (const p of TOOLING_PLUGINS) expect(names).toContain(p.name);
    });

    test("edit shares a substrate by reference — never doubles one the app already declares", () => {
        const out = compose("edit", [RenderPlugin, A]);
        expect(out.filter((p) => p.name === RenderPlugin.name)).toHaveLength(1);
        expect(out.map((p) => p.name)).toContain(InputPlugin.name);
    });

    test("play keeps the app's declared orbit but composes no tooling", () => {
        const names = compose("play", [OrbitPlugin, A]).map((p) => p.name);
        expect(names).toEqual([OrbitPlugin.name, "a"]);
        for (const p of TOOLING_PLUGINS) expect(names).not.toContain(p.name);
    });
});

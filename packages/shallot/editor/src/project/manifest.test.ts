import { describe, expect, test } from "bun:test";
import { type Manifest, normalize, resolve, serialize, setEnabled } from "./manifest";

// synthetic engine name sets — the resolver is pure over these, so it never imports a plugin object
const DEFAULTS = ["Slab", "Render", "Glaze"] as const;
const EXTRAS = ["Orbit", "Tween"] as const;

const enabledOf = (m: Manifest) =>
    new Set(
        resolve(m, DEFAULTS, EXTRAS)
            .entries.filter((e) => e.enabled)
            .map((e) => e.name),
    );

describe("normalize", () => {
    test("parses a valid manifest", () => {
        const m = normalize(`{ "scene": "s.scene", "plugins": { "Orbit": true } }`);
        expect(m.scene).toBe("s.scene");
        expect(m.plugins).toEqual({ Orbit: true });
    });

    test("tolerates corrupt or absent storage", () => {
        expect(normalize(null)).toEqual({});
        expect(normalize("not json")).toEqual({});
        expect(normalize("[1,2,3]")).toEqual({}); // a non-object root
    });

    test("drops a non-string scene and a non-object plugins", () => {
        expect(normalize(`{ "scene": 5, "plugins": [1] }`)).toEqual({});
    });

    test("parses a numeric capacity (the serialized Config field) and drops a non-numeric one", () => {
        expect(normalize(`{ "scene": "s.scene", "capacity": 512 }`).capacity).toBe(512);
        expect(normalize(`{ "capacity": "lots" }`).capacity).toBeUndefined();
    });

    test("preserves $schema across a normalize → serialize round-trip (editor writes keep the pointer)", () => {
        const schema = "./node_modules/@dylanebert/shallot/shallot.schema.json";
        const m = normalize(`{ "$schema": "${schema}", "plugins": { "Orbit": true } }`);
        expect(m.$schema).toBe(schema);
        // a plugin toggle (setEnabled) then serialize must not drop it
        const out = serialize(setEnabled(m, "Profile", true, "extra"));
        expect(out).toContain(`"$schema": "${schema}"`);
    });
});

describe("resolve", () => {
    test("a default is on unless disabled; an extra is off unless enabled", () => {
        const { entries } = resolve({}, DEFAULTS, EXTRAS);
        const by = new Map(entries.map((e) => [e.name, e]));
        expect(by.get("Render")?.enabled).toBe(true); // default on
        expect(by.get("Render")?.source).toBe("default");
        expect(by.get("Orbit")?.enabled).toBe(false); // extra off
        expect(by.get("Orbit")?.source).toBe("extra");
    });

    test("a default disabled by false, an extra enabled by true", () => {
        expect(enabledOf({ plugins: { Glaze: false, Orbit: true } })).toEqual(
            new Set(["Slab", "Render", "Orbit"]),
        );
    });

    test("a string value declares an enabled local plugin (source project)", () => {
        const { entries } = resolve({ plugins: { Spin: "./src/spin" } }, DEFAULTS, EXTRAS);
        const spin = entries.find((e) => e.name === "Spin");
        expect(spin).toEqual({
            name: "Spin",
            source: "project",
            spec: "./src/spin",
            enabled: true,
        });
    });

    test("a local kept but disabled via the [spec, false] tuple retains its spec", () => {
        const { entries } = resolve({ plugins: { Spin: ["./src/spin", false] } }, DEFAULTS, EXTRAS);
        const spin = entries.find((e) => e.name === "Spin");
        expect(spin).toEqual({
            name: "Spin",
            source: "project",
            spec: "./src/spin",
            enabled: false,
        });
    });

    test("a bool naming an unknown plugin is a diagnostic, not an entry", () => {
        const { entries, diagnostics } = resolve({ plugins: { Ghost: true } }, DEFAULTS, EXTRAS);
        expect(entries.some((e) => e.name === "Ghost")).toBe(false);
        expect(diagnostics).toHaveLength(1);
        expect(diagnostics[0].source).toBe("Ghost");
    });

    test("a specifier on an engine plugin name is a diagnostic (still resolved by name)", () => {
        const { entries, diagnostics } = resolve(
            { plugins: { Orbit: "@scope/orbit" } },
            DEFAULTS,
            EXTRAS,
        );
        // Orbit stays an extra resolved by name; the stray specifier is flagged, the extra left off
        expect(entries.find((e) => e.name === "Orbit")?.source).toBe("extra");
        expect(diagnostics[0].source).toBe("Orbit");
    });
});

describe("setEnabled", () => {
    test("disabling a default writes false; re-enabling drops the key (minimal diff)", () => {
        const off = setEnabled({}, "Render", false, "default");
        expect(off.plugins).toEqual({ Render: false });
        expect(setEnabled(off, "Render", true, "default").plugins).toEqual({});
    });

    test("enabling an extra writes true; disabling drops the key", () => {
        const on = setEnabled({}, "Orbit", true, "extra");
        expect(on.plugins).toEqual({ Orbit: true });
        expect(setEnabled(on, "Orbit", false, "extra").plugins).toEqual({});
    });

    test("a local toggles losslessly — disabled keeps the spec, re-enabled is the shorthand", () => {
        const m: Manifest = { plugins: { Spin: "./src/spin" } };
        const off = setEnabled(m, "Spin", false, "project", "./src/spin");
        expect(off.plugins).toEqual({ Spin: ["./src/spin", false] });
        expect(setEnabled(off, "Spin", true, "project", "./src/spin").plugins).toEqual({
            Spin: "./src/spin",
        });
    });

    test("toggling a local without its specifier throws", () => {
        expect(() => setEnabled({}, "Spin", false, "project")).toThrow();
    });
});

describe("round-trip", () => {
    test("setEnabled then resolve reproduces the intended enabled set", () => {
        // the persistence contract: an editor toggle → manifest write → re-derive lands on the same set
        let m: Manifest = {};
        m = setEnabled(m, "Glaze", false, "default"); // turn a default off
        m = setEnabled(m, "Orbit", true, "extra"); // turn an extra on
        expect(enabledOf(m)).toEqual(new Set(["Slab", "Render", "Orbit"]));
    });

    test("serialize emits stable 2-space JSON ending in a newline", () => {
        const text = serialize({ scene: "s.scene", plugins: { Orbit: true } });
        expect(text).toBe(`{\n  "scene": "s.scene",\n  "plugins": {\n    "Orbit": true\n  }\n}\n`);
        expect(normalize(text)).toEqual({ scene: "s.scene", plugins: { Orbit: true } });
    });
});

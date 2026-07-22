import { describe, expect, test } from "bun:test";
import { generateModule, plan } from "./generate";
import type { Manifest } from "./manifest";

const DIR = "/proj";

describe("plan", () => {
    test("an empty manifest enables every default, no locals", () => {
        const { engine, locals } = plan({}, DIR);
        expect(engine).toEqual(["Slab", "Transforms", "Input", "Render", "Part", "Sear", "Glaze"]);
        expect(locals).toEqual([]);
    });

    test("a disabled default drops out; an enabled extra joins the engine set", () => {
        const { engine } = plan({ plugins: { Glaze: false, Orbit: true } }, DIR);
        expect(engine).toContain("Orbit");
        expect(engine).not.toContain("Glaze");
    });

    test("a local specifier resolves project-relative → project-absolute", () => {
        const { locals } = plan({ plugins: { Spin: "./src/spin" } }, DIR);
        expect(locals).toEqual([{ name: "Spin", path: "/proj/src/spin" }]);
    });

    test("a disabled local is not imported; a bare package passes through", () => {
        const { locals } = plan({ plugins: { Off: ["./src/off", false], Pkg: "@scope/foo" } }, DIR);
        expect(locals).toEqual([{ name: "Pkg", path: "@scope/foo" }]);
    });

    test("an arbitrary engine plugin (true, not a default) joins the engine set by name", () => {
        // the generator trusts any `true` name as an engine plugin — no catalog gate, since it runs
        // headless (no plugin objects to check against). Where it imports FROM (barrel vs. a
        // backend plugin's own subpath) is generateModule's concern, not plan's — see below.
        const { engine } = plan({ plugins: { Foo: true } }, DIR);
        expect(engine).toContain("Foo");
    });
});

describe("generateModule", () => {
    test("emits a lean named barrel import for enabled engine plugins", () => {
        const src = generateModule({ plugins: { Orbit: true } }, DIR, []);
        expect(src).toContain(`import { SlabPlugin, `);
        expect(src).toContain(`OrbitPlugin } from "@dylanebert/shallot";`);
        expect(src).not.toContain(`@dylanebert/shallot/orbit`); // not a subpath specifier
    });

    test("routes a subpath-only engine plugin (e.g. Avbd) to its own import, not the barrel", () => {
        // AvbdPlugin isn't on the main barrel (exports.md) — the generator must import it from
        // @dylanebert/shallot/avbd or `shallot dev`/`build` throws "does not provide an export named
        // AvbdPlugin" at runtime (the bug this test pins).
        const src = generateModule({ plugins: { Avbd: true } }, DIR, []);
        expect(src).toContain(`import { AvbdPlugin } from "@dylanebert/shallot/avbd";`);
        expect(src).not.toContain(`AvbdPlugin } from "@dylanebert/shallot";`);
        expect(src).toContain(`const engine = [`);
        expect(src).toContain(`AvbdPlugin`);
    });

    test("emits a default import + a loud guard for a local plugin, and no HMR self-accept", () => {
        const manifest: Manifest = { scene: "scenes/s.scene", plugins: { Spin: "./src/spin" } };
        const src = generateModule(manifest, DIR, ["public/scenes/s.scene"]);
        expect(src).toContain(`import _l0 from "/proj/src/spin";`);
        expect(src).toContain(`const scene = "scenes/s.scene";`);
        // the runtime guard fails loud when a module doesn't default-export a Plugin
        expect(src).toContain(`its module must default-export a Plugin`);
        // no self-accept: a local plugin edit full-reloads the page (dev and a build agree)
        expect(src).not.toContain(`import.meta.hot`);
        expect(src).toContain(`export default project;`);
    });

    test("a scene-only project (empty manifest) still resolves the defaults + a null scene", () => {
        const src = generateModule({}, DIR, ["public/scenes/a.scene"]);
        expect(src).toContain(`import { SlabPlugin,`);
        expect(src).toContain(`const scene = null;`);
        expect(src).toContain(`const scenes = ["public/scenes/a.scene"];`);
    });

    test("a null dir (no project) emits dir null + the defaults", () => {
        const src = generateModule({}, null, []);
        expect(src).toContain(`const dir = null;`);
        expect(src).toContain(`import { SlabPlugin,`);
    });

    test("capacity threads into the project object — the manifest's value, else null", () => {
        expect(generateModule({ capacity: 512 }, DIR, [])).toContain(`const capacity = 512;`);
        expect(generateModule({}, DIR, [])).toContain(`const capacity = null;`);
        // it rides the project object the boot reads, beside scene
        expect(generateModule({ capacity: 512 }, DIR, [])).toContain(`scene, capacity, scenes`);
    });
});

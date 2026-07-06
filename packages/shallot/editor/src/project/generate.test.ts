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

    test("an arbitrary engine plugin (true, not a default) joins the barrel-import set by name", () => {
        // the generator trusts any `true` name as an engine `${name}Plugin` barrel import — no catalog gate,
        // so a physics-using project resolves the same headless as in the editor
        const { engine } = plan({ plugins: { Physics: true } }, DIR);
        expect(engine).toContain("Physics");
    });
});

describe("generateModule", () => {
    test("emits a lean named barrel import for enabled engine plugins", () => {
        const src = generateModule({ plugins: { Orbit: true } }, DIR, []);
        expect(src).toContain(`import { SlabPlugin, `);
        expect(src).toContain(`OrbitPlugin } from "@dylanebert/shallot";`);
        expect(src).not.toContain(`@dylanebert/shallot/orbit`); // not a subpath specifier
    });

    test("emits a default import + a per-local HMR accept + a loud guard for a local plugin", () => {
        const manifest: Manifest = { scene: "scenes/s.scene", plugins: { Spin: "./src/spin" } };
        const src = generateModule(manifest, DIR, ["public/scenes/s.scene"]);
        expect(src).toContain(`import _l0 from "/proj/src/spin";`);
        expect(src).toContain(`const scene = "scenes/s.scene";`);
        // the runtime guard fails loud when a module doesn't default-export a Plugin
        expect(src).toContain(`its module must default-export a Plugin`);
        expect(src).toContain(`import.meta.hot.accept(["/proj/src/spin"], (mods) =>`);
        expect(src).toContain(`m.default`);
        expect(src).toContain(`emitProjectReload(`);
        expect(src).toContain(`export default project;`);
    });

    test("a project with no locals emits no HMR accept block", () => {
        const src = generateModule({ plugins: { Orbit: true } }, DIR, []);
        expect(src).not.toContain(`import.meta.hot.accept`);
    });

    test("hot=false (a production build) emits no HMR block + no editor reload import", () => {
        const src = generateModule({ plugins: { Spin: "./src/spin" } }, DIR, [], false);
        expect(src).toContain(`import _l0 from "/proj/src/spin";`); // still imports the local
        expect(src).not.toContain(`import.meta.hot`);
        expect(src).not.toContain(`/src/project/reload`);
    });

    test("a scene-only project (empty manifest) still resolves the defaults + a null scene", () => {
        const src = generateModule({}, DIR, ["public/scenes/a.scene"]);
        expect(src).toContain(`import { SlabPlugin,`);
        expect(src).toContain(`const scene = null;`);
        expect(src).toContain(`const scenes = ["public/scenes/a.scene"];`);
    });

    test("a null dir (editor with no project) emits dir null + the defaults", () => {
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

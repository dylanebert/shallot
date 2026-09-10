import { describe, expect, test } from "bun:test";
import { generateModule } from "./generate";
import type { Manifest } from "./manifest";

// `plan` moved to the project host (`host.ts`) with the A1 seam — its own arms live in `host.test.ts`.
// What stays here is the emission half: a plan (however it was produced) becoming module source.

const DIR = "/proj";

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

    test("capacity and pixel ratio thread into the project object — manifest values, else null", () => {
        const configured = generateModule({ capacity: 512, pixelRatio: "auto" }, DIR, []);
        expect(configured).toContain(`const capacity = 512;`);
        expect(configured).toContain(`const pixelRatio = "auto";`);
        expect(generateModule({}, DIR, [])).toContain(`const capacity = null;`);
        expect(generateModule({}, DIR, [])).toContain(`const pixelRatio = null;`);
        // they ride the project object the boot reads, beside scene
        expect(configured).toContain(`scene, capacity, pixelRatio, scenes`);
    });
});

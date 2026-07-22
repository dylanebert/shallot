import { describe, expect, test } from "bun:test";
import type { GltfScene } from "../src/extras/gltf/gltf";
import {
    type CorpusModel,
    corpusPresent,
    entryOf,
    features,
    type Matrix,
    pickVariants,
    status,
    walkCorpus,
} from "./gltf-corpus";
import matrixJson from "./gltf-matrix.json";

// the glTF conformance regression gate (roadmap "glTF import — conformance + regression suite"). Walks the
// deviceless importer over the Khronos corpus and pins each (model, variant) to `gltf-matrix.json`: the
// skipped-feature set, the decoded geometry count, and the derived status. A regression (a feature we silently
// stop decoding) or a new capability (a feature we start handling) reads as a red row here, then as a
// reviewable diff through `scripts/gltf-conformance.ts`. The corpus is a submodule outside the package, so the
// walk is presence-gated with a loud, announced skip — never a hidden green.

// the pure derivation, pinned corpus-independently so it's covered when the submodule is absent and its
// unsupported (zero-mesh) branch — which no real corpus model hits — is still exercised.
describe("gltf conformance derivation", () => {
    const scene = (meshes: number, feats: string[]): GltfScene => ({
        meshes: new Array(meshes) as GltfScene["meshes"],
        instances: [],
        materials: [],
        images: [],
        skinInputs: [],
        live: false,
        unsupported: feats.map((feature) => ({ feature })),
    });

    test("status reflects geometry + skipped features", () => {
        expect(status(scene(2, []))).toBe("supported");
        expect(status(scene(2, ["KHR_materials_clearcoat"]))).toBe("partial");
        expect(status(scene(0, []))).toBe("unsupported"); // no geometry decoded — never seen in the corpus
    });

    test("features are sorted stable keys, detail dropped", () => {
        const s = scene(1, []);
        s.unsupported = [{ feature: "skin", detail: "2 skins" }, { feature: "animation" }];
        expect(features(s)).toEqual(["animation", "skin"]);
    });

    test("pickVariants takes both base paths plus every Draco / KTX codec", () => {
        const model = (variants: string[]): CorpusModel => ({
            label: "m",
            name: "m",
            variants: Object.fromEntries(variants.map((v) => [v, `${v}.gltf`])),
        });
        expect(
            pickVariants(model(["glTF", "glTF-Binary", "glTF-Draco", "glTF-KTX-BasisU"])),
        ).toEqual(["glTF", "glTF-Binary", "glTF-Draco", "glTF-KTX-BasisU"]);
        expect(pickVariants(model(["glTF-Embedded"]))).toEqual(["glTF-Embedded"]); // self-contained fallback
        expect(pickVariants(model(["glTF-Binary"]))).toEqual(["glTF-Binary"]);
    });
});

const matrix = matrixJson as Matrix;

if (!corpusPresent()) {
    const required = process.env.GLTF_CORPUS_REQUIRED === "1";
    describe("gltf corpus conformance", () => {
        test(
            required
                ? "corpus REQUIRED (GLTF_CORPUS_REQUIRED=1) but absent"
                : "corpus absent — suite skipped (init the gltf-sample-assets submodule to run it)",
            () => {
                const bar = "=".repeat(78);
                console.warn(
                    `\n${bar}\n[gltf-corpus] glTF-Sample-Assets submodule not checked out — conformance walk SKIPPED.\n  run: git submodule update --init reference/gltf-sample-assets\n${bar}\n`,
                );
                if (required) throw new Error("[gltf-corpus] corpus required but absent");
            },
        );
    });
} else {
    const entries = await walkCorpus();
    describe("gltf corpus conformance", () => {
        for (const e of entries) {
            test(`${e.model} / ${e.variant}`, () => {
                if (e.error) throw new Error(`parse failed: ${e.error}`);
                const pinned = matrix[e.model]?.[e.variant];
                if (!pinned)
                    throw new Error(
                        `${e.model}/${e.variant} not in gltf-matrix.json — a new corpus model/variant; run: bun run scripts/gltf-conformance.ts --write`,
                    );
                const got = entryOf(e.scene!);
                expect(got.unsupported).toEqual(pinned.unsupported);
                expect(got.meshes).toBe(pinned.meshes);
                expect(got.status).toBe(pinned.status);
            });
        }
    });
}

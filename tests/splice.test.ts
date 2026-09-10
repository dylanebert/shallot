import { describe, expect, test } from "bun:test";
import {
    hdrColorPackWgsl,
    hdrColorUnpackWgsl,
    ldrColorPackWgsl,
    ldrColorUnpackWgsl,
    octEncodeWgsl,
    posQuantPackWgsl,
    posQuantWgsl,
    quatSnorm16x4Wgsl,
    smallest3Wgsl,
    xformWgsl,
} from "../src/engine/utils/core";
import { materialDataWgsl } from "../src/extras/gltf/palette";
import { liveTintWgsl, skinParamsWgsl } from "../src/extras/skin/core";
import {
    boxBoxWgsl,
    helpersWgsl,
    hullCoreWgsl,
    hullSatWgsl,
    roundedPolyWgsl,
    roundedWgsl,
} from "../src/standard/avbd/collide";
import { bvhRootWgsl, bvhTraverseWgsl } from "../src/standard/bvh/core";
import { fogInScatterWgsl, fogMarchWgsl, fogStructWgsl } from "../src/standard/fog/core";
import { tonemapWgsl } from "../src/standard/glaze/tonemap";
import {
    frameWgsl,
    lightingWgsl,
    linearToSrgbWgsl,
    pointLightsWgsl,
} from "../src/standard/render/core";
import {
    casterWgsl,
    lightEvalWgsl,
    pointShadowWgsl,
    sunShadowWgsl,
    sunStructWgsl,
} from "../src/standard/sear/core";
// the PBR lobe is spliced by sear itself, not by a relocatable consumer, so it stays internal — the
// structural tests reach it directly, like `surfaceCode`
import { pbrWgsl } from "../src/standard/sear/shade";

// Every relocatable WGSL chunk shares one dedup namespace (`spliceNs`, engine/utils/tgsl.ts), so a
// dependency two chunks both need is emitted into whichever resolves FIRST — and a consumer splices
// several of them into one module. Two failures live here and nowhere else:
//
//   - a definition emitted by two chunks is a duplicate declaration, which Dawn rejects at pipeline
//     creation with nothing pointing at the chunk that caused it;
//   - a definition emitted by NO chunk (because the one that owns it was never resolved) is a missing
//     identifier, which only the consumer that skipped the base chunk sees.
//
// Each module's own test file covers its own chunks; this is the cross-module matrix, which is the scope
// the namespace actually spans. Add a chunk here when you add one to `spliceNs`.

const chunks: [string, () => string][] = [
    ["frameWgsl", frameWgsl],
    ["lightingWgsl", lightingWgsl],
    ["octEncodeWgsl", octEncodeWgsl],
    ["quatSnorm16x4Wgsl", quatSnorm16x4Wgsl],
    ["smallest3Wgsl", smallest3Wgsl],
    ["posQuantWgsl", posQuantWgsl],
    ["posQuantPackWgsl", posQuantPackWgsl],
    ["xformWgsl", xformWgsl],
    ["ldrColorUnpackWgsl", ldrColorUnpackWgsl],
    ["ldrColorPackWgsl", ldrColorPackWgsl],
    ["hdrColorUnpackWgsl", hdrColorUnpackWgsl],
    ["hdrColorPackWgsl", hdrColorPackWgsl],
    ["pointLightsWgsl", pointLightsWgsl],
    ["lightEvalWgsl", lightEvalWgsl],
    ["pbrWgsl", pbrWgsl],
    ["casterWgsl", casterWgsl],
    ["pointShadowWgsl", pointShadowWgsl],
    ["sunStructWgsl", sunStructWgsl],
    ["sunShadowWgsl", sunShadowWgsl],
    ["fogStructWgsl", fogStructWgsl],
    ["fogMarchWgsl", fogMarchWgsl],
    ["fogInScatterWgsl", fogInScatterWgsl],
    ["tonemapWgsl", tonemapWgsl],
    ["linearToSrgbWgsl", linearToSrgbWgsl],
    ["skinParamsWgsl", skinParamsWgsl],
    ["liveTintWgsl", liveTintWgsl],
    ["materialDataWgsl", materialDataWgsl],
    ["bvhRootWgsl", bvhRootWgsl],
    ["bvhTraverseWgsl", bvhTraverseWgsl],
    ["helpersWgsl", helpersWgsl],
    ["boxBoxWgsl", boxBoxWgsl],
    ["roundedWgsl", roundedWgsl],
    ["hullCoreWgsl", hullCoreWgsl],
    ["hullSatWgsl", hullSatWgsl],
    ["roundedPolyWgsl", roundedPolyWgsl],
];

const defs = (wgsl: string): string[] =>
    [...wgsl.matchAll(/^(?:fn|struct)\s+([A-Za-z0-9_]+)/gm)].map((m) => m[1]);

describe("splice chunks", () => {
    test("every pair is duplicate-free, so any combination of them splices", () => {
        for (let i = 0; i < chunks.length; i++) {
            for (let j = i + 1; j < chunks.length; j++) {
                const [an, a] = chunks[i];
                const [bn, b] = chunks[j];
                const all = [...defs(a()), ...defs(b())];
                const dupes = all.filter((x, k) => all.indexOf(x) !== k);
                expect(dupes, `${an} + ${bn}`).toEqual([]);
            }
        }
    });

    test("no chunk emits an anonymous or suffixed definition", () => {
        // typegpu names an unnamed factory-built schema `item`, and suffixes a name that collides with
        // one already emitted in the namespace (`PointCasters_1`) — either way a raw splice site that
        // declares the authored name references nothing, and only a real shader compile would say so
        for (const [name, chunk] of chunks) {
            for (const def of defs(chunk())) {
                expect(def, `${name} emits ${def}`).not.toMatch(/^item/);
                expect(def, `${name} emits ${def}`).not.toMatch(/_\d+$/);
            }
        }
    });

    test("the dependent chunks name their base's definitions, never redefine them", () => {
        // the base-forcing rule made concrete: a chunk that calls into another one's function emits the
        // call, and exactly one chunk emits the definition (the pair matrix above proves the "exactly")
        expect(defs(pointLightsWgsl())).toContain("PointLightGpu");
        expect(lightEvalWgsl()).toContain("light: PointLightGpu");
        expect(defs(lightEvalWgsl())).not.toContain("PointLightGpu");

        expect(defs(octEncodeWgsl())).toContain("octDecodeNormal");
        expect(lightEvalWgsl()).toContain("octDecodeNormal(");
        expect(defs(lightEvalWgsl())).not.toContain("octDecodeNormal");

        expect(defs(casterWgsl())).toEqual(
            expect.arrayContaining(["PointCaster", "PointCasters", "TileRects"]),
        );
        expect(pointShadowWgsl()).toContain("pointShadows.casters[k]");
        expect(defs(pointShadowWgsl())).not.toContain("PointCasters");

        expect(defs(sunStructWgsl())).toEqual(expect.arrayContaining(["Cascade", "SunShadow"]));
        expect(sunShadowWgsl()).toContain("sunShadow.cascades[ci]");
        expect(defs(sunShadowWgsl())).not.toContain("SunShadow");

        // fog's in-scatter is the deepest dependent: it evaluates the same lit cones sear does, so the
        // light struct comes from render's chunk and the falloff + cone from sear's, never a second copy
        expect(fogInScatterWgsl()).toContain("light: PointLightGpu");
        expect(defs(fogInScatterWgsl())).not.toContain("PointLightGpu");
        expect(defs(lightEvalWgsl())).toEqual(
            expect.arrayContaining(["distanceAttenuation", "spotFactor"]),
        );
        expect(fogInScatterWgsl()).toContain("distanceAttenuation(");
        expect(defs(fogInScatterWgsl())).not.toContain("distanceAttenuation");
        expect(defs(fogInScatterWgsl())).not.toContain("spotFactor");

        // the two sRGB encodes are distinct functions on purpose — the vec3 present-gamma one glaze writes
        // the swapchain through, and the scalar one the LDR color codec packs each channel with
        expect(defs(linearToSrgbWgsl())).toEqual(["linearToSrgb"]);
        expect(defs(ldrColorPackWgsl())).toContain("linearToSrgb1");

        // a live-skin surface splices the schema chunk beside the tint fn; the tint reads `unpackLdrColor`
        // (sear-spliced for every surface) by name from a raw body, so it must define neither
        expect(defs(skinParamsWgsl())).toEqual(["SkinParams"]);
        expect(liveTintWgsl()).toContain("unpackLdrColor(");
        expect(defs(liveTintWgsl())).toEqual(["liveTint"]);
    });
});

import { beforeEach, describe, expect, test } from "bun:test";
import { body } from "../../../tests/wgsl";
import { State } from "../../engine";
import { Surfaces } from "../../standard/render/core";
import { TypedSurfaces } from "../../standard/sear/contract";
import { engineLayout } from "../../standard/sear/engine";
import { typedPrepassWgsl, typedShadowWgsl, typedSurfaceWgsl } from "../../standard/sear/pipelines";
import { registerTexturedSurfaces } from "./assets";
import { liveSkinSurface, registerLiveSkinSurfaces } from "./live";
import { MAP_ALL } from "./shade";
import { registerSkinSurfaces, skinSurface } from "./skin";

const names = [
    "gltf-albedo",
    "gltf-albedo-clip",
    "gltf-albedo-blend",
    "skin",
    "skin-clip",
    "skin-blend",
    "skin-live",
    "skin-live-clip",
    "skin-live-blend",
] as const;

function registerAll(state = new State()): State {
    registerTexturedSurfaces(state);
    registerSkinSurfaces(state);
    registerLiveSkinSurfaces(state);
    return state;
}

describe("typed glTF surface trios", () => {
    beforeEach(() => {
        Surfaces.clear();
        TypedSurfaces.clear();
        registerAll();
    });

    test("all nine production shaders register typed while the Part compatibility entries stay data-only", () => {
        expect(TypedSurfaces.size).toBe(9);
        expect(Surfaces.size).toBe(9);
        for (const name of names) {
            expect(TypedSurfaces.get(name)).toBeDefined();
            const compatibility = Surfaces.get(name)!;
            expect(compatibility.vs).toBeUndefined();
            expect(compatibility.fs).toBeUndefined();
            expect(compatibility.preamble).toBeUndefined();
        }
    });

    test("alpha-mode routing and blend state stay fixed", () => {
        expect([skinSurface("OPAQUE"), skinSurface("MASK"), skinSurface("BLEND")]).toEqual([
            "skin",
            "skin-clip",
            "skin-blend",
        ]);
        expect([
            liveSkinSurface("OPAQUE"),
            liveSkinSurface("MASK"),
            liveSkinSurface("BLEND"),
        ]).toEqual(["skin-live", "skin-live-clip", "skin-live-blend"]);
        expect(TypedSurfaces.get("gltf-albedo")!.blend).toBeUndefined();
        expect(TypedSurfaces.get("gltf-albedo-clip")!.blend).toBe("clip");
        expect(TypedSurfaces.get("gltf-albedo-blend")!.blend).toBe("alpha");
    });

    test("material specialization folds absent map samples and keeps albedo bucket routing as an if-chain", () => {
        const surface = TypedSurfaces.get("gltf-albedo")!;
        const sparse = typedSurfaceWgsl(surface, 0);
        const full = typedSurfaceWgsl(surface, MAP_ALL);
        const sparseShade = body(sparse, "fn shadePbr(");
        const fullShade = body(full, "fn shadePbr(");
        for (const texture of ["mr", "normalTex", "occlusion", "emissive"]) {
            expect(sparseShade).not.toContain(`textureSample(${texture}`);
            expect(fullShade).toContain(`textureSample(${texture}`);
        }
        const albedo = body(full, "fn sampleAlbedo(");
        expect(albedo).toContain("if ((md.albedoBucket == 0u))");
        expect(albedo).toContain("if ((md.albedoBucket == 1u))");
        expect(albedo).toContain("if ((md.albedoBucket == 2u))");
        expect(albedo).not.toContain("switch");
    });

    test("VAT skin samples the baked position and normal using the vertex index", () => {
        const code = typedSurfaceWgsl(TypedSurfaces.get("skin")!, 0);
        expect(code).toContain("f32(vsIn.vidx)");
        expect(code).toContain("textureSampleLevel(vatPos");
        expect(code).toContain("textureSampleLevel(vatNorm");
        expect(code).toContain("xformPoint(xf, p)");
        expect(code).toContain("xformNormal(xf, n)");
    });

    test("live skin retains the sanctioned dynamic vec4 palette lane and tint header", () => {
        const code = typedSurfaceWgsl(TypedSurfaces.get("skin-live")!, 0);
        const vs = body(code, "fn skinLiveVs(");
        expect(vs).toContain("skinParams.jwBase + (vidx >> 1u)");
        expect(vs).toContain("let jwPair = (vidx & 1u) * 2u;");
        expect(vs).toContain("unpack4x8unorm");
        expect(vs).toContain("xformPoint(jx, localPos)");
        expect(vs).toContain("xformNormal(jx, localNormal)");
        const tint = body(code, "fn liveTint(");
        expect(tint).toContain("skinData[base]");
        expect(tint).toContain("unpackLdrColor");
        expect(code).not.toContain("var<storage, read> color:");
    });

    test("clip executes its material cutoff in color, prepass, and both shadow atlases", () => {
        const clip = TypedSurfaces.get("skin-live-clip")!;
        const color = typedSurfaceWgsl(clip, 0);
        const prepass = typedPrepassWgsl(clip, 0);
        const shadow = typedShadowWgsl(clip, 0);
        for (const code of [color, prepass[""], prepass.tag, shadow.point, shadow.cascade]) {
            expect(code).toContain(".cutoff");
            expect(code).toContain("discard;");
        }
        for (const code of [prepass[""], prepass.tag, shadow.point, shadow.cascade]) {
            expect(code).toContain("decodeUv(");
            expect(code).not.toContain("let uv = vec2f(0, 0);");
            expect(code).not.toContain("let uv = vec2f();");
        }
    });
});

describe("storage-ceiling audit — every typed glTF surface fits the 10-storage stage limit", () => {
    const Ceiling = 10;
    const engineStorage = Object.values(engineLayout.entries).filter(
        (entry) => "storage" in entry && entry.visibility.includes("vertex"),
    ).length;

    beforeEach(() => {
        Surfaces.clear();
        TypedSurfaces.clear();
        registerAll();
    });

    test("every trio has at most five own storage bindings", () => {
        let checked = 0;
        for (const surface of TypedSurfaces) {
            const surfaceStorage = Object.values(surface.layout.entries).filter(
                (entry) => "storage" in entry && entry.visibility.includes("vertex"),
            ).length;
            expect([surface.name, engineStorage + surfaceStorage <= Ceiling]).toEqual([
                surface.name,
                true,
            ]);
            checked++;
        }
        expect(checked).toBe(9);
    });

    test("live skin stays exactly at the ceiling", () => {
        const surface = TypedSurfaces.get("skin-live")!;
        const surfaceStorage = Object.values(surface.layout.entries).filter(
            (entry) => "storage" in entry && entry.visibility.includes("vertex"),
        ).length;
        expect(engineStorage).toBe(4);
        expect(engineStorage + surfaceStorage).toBe(Ceiling);
    });
});

test("typed ownership follows its State and cannot remove a later registration", () => {
    const first = registerAll();
    const firstSpec = TypedSurfaces.get("gltf-albedo")!;
    const second = registerAll();
    const secondSpec = TypedSurfaces.get("gltf-albedo")!;
    expect(secondSpec).not.toBe(firstSpec);

    first.dispose();
    for (const name of names) expect(TypedSurfaces.get(name)).toBeDefined();
    expect(TypedSurfaces.get("gltf-albedo")).toBe(secondSpec);
    second.dispose();
    expect(TypedSurfaces.size).toBe(0);
});

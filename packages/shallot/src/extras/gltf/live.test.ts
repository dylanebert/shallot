import { beforeEach, describe, expect, test } from "bun:test";
import { Surfaces } from "../../standard/render/core";
import { SHARED_STORAGE_COUNT, surfaceCode } from "../../standard/sear/forward";
import { registerTexturedSurfaces } from "./assets";
import { registerLiveSkinSurfaces } from "./live";
import { registerSkinSurfaces } from "./skin";

// Structural validation of the `skin-live` surface's WGSL codegen (`surfaceCode` is pure — no GPU, the
// sprite.test.ts pattern). The vs must decode the packed joints/weights, blend the Xform-shaped palette via
// the spliced `xformWgsl()` chunk, and apply the instance transform; the fs reads the tint from the palette header
// (the color fold), never a `color` binding. The substrate those chunks come from (`extras/skin`) is spec'd
// in its own tests — the equivalence gate that licenses the blend math included. Real-GPU compile + the
// deform/shadow/bounds behavior live in the gym `render` `skin-live` mode.

describe("skin-live surface trio", () => {
    beforeEach(() => registerLiveSkinSurfaces());

    test("the vs decodes packed JW and blends the palette through the spliced xformWgsl()", () => {
        const code = surfaceCode(Surfaces.get("skin-live")!);
        // JW region B is keyed by vidx, 2 verts per vec4 (gpu.md rule 6): the mesh's jwBase + the vertex's
        // half, then the 4×u8 joint unpack + the 4×unorm8 weight unpack
        expect(code).toContain("let jwElem = skinData[skinParams.jwBase + (vidx >> 1u)];");
        expect(code).toContain("let jwPair = (vidx & 1u) * 2u;");
        expect(code).toContain("let wt = unpack4x8unorm(jwElem[jwPair + 1u]);");
        expect(code).toContain(
            "let joints = vec4<u32>(js & 0xffu, (js >> 8u) & 0xffu, (js >> 16u) & 0xffu, (js >> 24u) & 0xffu);",
        );
        // the palette entry is Xform-shaped, read at header + jointCount stride, blended with the spliced
        // xformPoint / xformNormal (zero new transform WGSL) — the equivalence gate licenses this reuse
        expect(code).toContain("let po = pbase + 1u + joints[k] * 3u;");
        expect(code).toContain("bitcast<vec4<f32>>(skinData[po + 1u])");
        expect(code).toContain("sp += w * xformPoint(jx, localPos);");
        expect(code).toContain("sn += w * xformNormal(jx, localNormal);");
        // object-space palette → the standard instance transform still carries the pose to world; the normal
        // blends as a plain vec3 and renormalizes (never oct across the blend, gpu.md rule 9)
        expect(code).toContain("world = vec4<f32>(xformPoint(xf, sp), 1.0);");
        expect(code).toContain("worldNormal = xformNormal(xf, normalize(sn));");
    });

    test("the fs reads the tint from the palette header — no separate color storage binding", () => {
        const code = surfaceCode(Surfaces.get("skin-live")!);
        // the tint helper is a WGSL-bodied `tgpu.fn` (it reads `skin` / `skinData` as consumer-declared
        // globals), so its signature carries the short type aliases resolution requires
        expect(code).toContain("fn liveTint(e: u32) -> vec4f {");
        expect(code).toContain("return unpackLdrColor(skinData[u32(skin[e].x)].x);");
        expect(code).toContain("sampleAlbedo(mid, uv).rgb * liveTint(eid).rgb");
        // the VAT skin surface binds `color` as storage; the live path folds it into the header, so no
        // `color` storage binding survives (that's what buys the room for `skinData` at the ceiling)
        expect(code).not.toContain("var<storage, read> color:");
    });

    test("the five own storage bindings declare at SURFACE_BASE in order, then the skinParams uniform", () => {
        const code = surfaceCode(Surfaces.get("skin-live")!);
        expect(code).toContain("@group(0) @binding(8) var<storage, read> eids: array<u32>;");
        expect(code).toContain(
            "@group(0) @binding(9) var<storage, read> transforms: array<Xform>;",
        );
        expect(code).toContain("@group(0) @binding(10) var<storage, read> skin: array<vec4<f32>>;");
        expect(code).toContain(
            "@group(0) @binding(11) var<storage, read> materialData: array<MaterialData>;",
        );
        expect(code).toContain(
            "@group(0) @binding(12) var<storage, read> skinData: array<vec4<u32>>;",
        );
        // the per-mesh constants ride a uniform (a separate limit, the vatParams precedent), its struct in
        // the preamble (module-scope structs resolve order-free)
        expect(code).toContain("var<uniform> skinParams: SkinParams;");
        expect(code).toContain("struct SkinParams {");
    });

    test("clip discards on the material cutoff; blend writes the tinted alpha", () => {
        const clip = surfaceCode(Surfaces.get("skin-live-clip")!);
        expect(clip).toContain("if (tex.a * c.a < materialData[mid].cutoff) { discard; }");
        // clip is opaque (writes depth + casts) → its empty-lane prepass fragment runs the discard so the
        // shadow map cuts holes; a blend surface has no prepass at all
        expect(surfaceCode(Surfaces.get("skin-live-clip")!, "prepass")).toContain("fn fsPrepass");
        const blend = surfaceCode(Surfaces.get("skin-live-blend")!);
        expect(blend).toContain(
            "col = vec4<f32>(shadePbr(mid, uv, tex.rgb, normalize(worldNormal), world), tex.a);",
        );
        expect(surfaceCode(Surfaces.get("skin-live-blend")!, "prepass")).not.toContain(
            "fn fsPrepass",
        );
    });
});

// The storage-ceiling audit (gpu.md's 10-per-stage limit): every registered surface's own storage bindings
// plus sear's shared 5 (vertices/pointLights/lightGrid/lightIndices/meshQuant) must be ≤ 10 — the counter
// gpu.md says to run by hand. Registering the full gltf surface set (the ceiling-critical family: textured +
// VAT skin + live skin each carry 5 own storage → exactly 10) and iterating the registry runs it for real,
// so a future binding that breaks the ceiling fails here, not at pipeline creation with no diagnostic (Chrome
// fails silently past 10). The register functions are pure (no device), so this stays CPU-only per
// testing.md; the sear default materials carry ≤ 4 own storage (forward.ts) — sub-ceiling by construction.
describe("storage-ceiling audit — every surface fits the 10-storage stage limit", () => {
    // sear's shared per-stage storage bindings — imported, not mirrored, so a sixth goes red here
    const Shared = SHARED_STORAGE_COUNT;
    const Ceiling = 10;
    const ownStorage = (name: string) =>
        Object.values(Surfaces.get(name)!.bindings ?? {}).filter((b) => b.type === "storage")
            .length;

    test("every gltf surface's own storage + sear's shared 5 stays within the ceiling", () => {
        Surfaces.clear();
        registerTexturedSurfaces();
        registerSkinSurfaces();
        registerLiveSkinSurfaces();
        let checked = 0;
        for (const s of Surfaces) {
            const own = Object.values(s.bindings ?? {}).filter((b) => b.type === "storage").length;
            expect([s.name, Shared + own <= Ceiling]).toEqual([s.name, true]);
            checked++;
        }
        expect(checked).toBe(9); // 3 textured + 3 VAT skin + 3 live skin
    });

    test("skin-live sits exactly at the ceiling (5 own storage bindings)", () => {
        registerLiveSkinSurfaces();
        // eids, transforms, skin, materialData, skinData — the color fold keeps it at 5, not 6
        expect(ownStorage("skin-live")).toBe(5);
        expect(Shared + ownStorage("skin-live")).toBe(Ceiling);
    });
});

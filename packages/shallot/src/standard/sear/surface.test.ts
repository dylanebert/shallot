import { describe, expect, test } from "bun:test";
import { mesh, packMeshes, quantizeMeshes } from "../render/mesh";
import { backgroundCode, surfaceCode } from "./forward";

// Structural validation of sear's per-surface WGSL codegen. `surfaceCode` is
// pure (no GPU), so the contract — explicit `col` output, the `lit` /
// `lightFactor` lighting helpers, custom interpolators, and the binding layout —
// is checked by reading the generated source. GPU compilation of these pipelines
// lives in `bun bench`.
describe("sear surfaceCode", () => {
    test("emits the explicit-color contract, not a forced lighting wrapper", () => {
        const code = surfaceCode({ name: "t", fs: "col = vec4<f32>(1.0);" });
        // surfaces write `col`; sear returns it verbatim from a single-target color fragment (no MRT)
        expect(code).toContain("var col = vec4<f32>(0.0, 0.0, 0.0, 1.0);");
        expect(code).toContain("fn fs(fin: VertexOut) -> @location(0) vec4<f32> {");
        expect(code).toContain("return col;");
        // lighting is opt-in via helpers, not applied around the chunk
        expect(code).toContain("fn lit(");
        expect(code).toContain("fn lightFactor(");
        expect(code).not.toContain("fn lambert(");
        expect(code).not.toContain("struct Surface");
    });

    test("lit is base color times lightFactor; the sun-shadow seam lives in lightFactor", () => {
        const code = surfaceCode({ name: "t" });
        // lit itself is unchanged — shading is base color times the light factor
        expect(code).toContain("return baseColor * lightFactor(normal);");
        // the sun term (not ambient) is the half-Lambert diffuse times the sampled sun visibility
        expect(code).toContain("sunColor.rgb * sun * sunVisibility");
        // sunVisibility defaults to fully lit, so the vertex stage + shadowless frames no-op
        expect(code).toContain("var<private> sunVisibility: f32 = 1.0;");
    });

    test("the metallic-roughness PBR model is available alongside the diffuse helpers", () => {
        const code = surfaceCode({ name: "t" });
        // litPbr + the Cook-Torrance pieces ride the same preamble as lit, so any surface can shade PBR
        expect(code).toContain("fn litPbr(");
        expect(code).toContain("struct Pbr {");
        expect(code).toContain("fn brdf(");
        expect(code).toContain("fn distributionGGX(");
        expect(code).toContain("fn fresnelSchlick(");
        // view-dependent specular reads the camera world position from the View uniform
        expect(code).toContain("eye: vec4<f32>");
        expect(code).toContain("view.eye.xyz");
        // the engine-default flat look: dielectric F0 0 → f90 0 → zero specular at metallic 0, so a bare
        // material reduces to the diffuse lit() — the f90-from-F0 (Frostbite) term is the discriminator
        expect(code).toContain("let f90 = saturate(dot(f0, vec3<f32>(50.0 / 3.0)));");
    });

    test("the default diffuse cosine is Valve half-Lambert, the soft happy-path look", () => {
        const code = surfaceCode({ name: "t" });
        // remap [-1,1]→[0,1] then square: light wraps fully around, the terminator softens
        expect(code).toContain("fn halfLambert(ndl: f32) -> f32 {");
        expect(code).toContain("let h = ndl * 0.5 + 0.5;");
        expect(code).toContain("return h * h;");
        // wired into every diffuse term — the sun, the clustered point lights, and the PBR diffuse lobe
        expect(code).toContain("let sun = halfLambert(dot(normal, L));");
        expect(code).toContain("let diff = halfLambert(");
        expect(code).toContain("s.albedo / PI * halfLambert(d)");
        // the specular keeps the physical clamped cosine, so it vanishes on back faces and metals
        // / glTF dielectrics are unchanged
        expect(code).toContain("spec * ndl");
    });

    test("the fs samples the sun shadow inline from the group-1 shadow map", () => {
        const code = surfaceCode({ name: "t" });
        // the shadow map + comparison sampler + light params are their own bind group, so the per-draw
        // group 0 stays camera-independent (the tag + depth pipelines omit group 1)
        expect(code).toContain("@group(1) @binding(0) var shadowMap: texture_depth_2d;");
        expect(code).toContain("@group(1) @binding(1) var shadowSamp: sampler_comparison;");
        expect(code).toContain("@group(1) @binding(2) var<uniform> sunShadow: SunShadow;");
        // the fragment projects the fragment's world position into the map + PCF-compares
        expect(code).toContain("sunVisibility = sampleSunShadow(world, worldNormal);");
        expect(code).toContain("textureSampleCompareLevel(shadowMap, shadowSamp,");
    });

    // normal-offset bias (the primary acne fix, matching Bevy): the receiver shifts along its world
    // normal by normalBias shadow texels of world size before the depth compare, so grazing faces get
    // the most offset. The structural contract is that the offset is applied off the per-pixel normal
    test("sampleSunShadow offsets the receiver along its world normal", () => {
        const code = surfaceCode({ name: "t" });
        // the params carry the global multiplier knob + each cascade's own texel world size (the scale)
        expect(code).toContain("normalBias: f32,");
        expect(code).toContain("texelWorld: f32,");
        // the per-cascade sample shifts the receiver along the surface normal before projecting
        expect(code).toContain(
            "fn sampleSunShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> f32 {",
        );
        expect(code).toContain("worldPos + normalize(normal) * (sunShadow.normalBias");
        expect(code).toContain("c.texelWorld)"); // per-cascade texel world size
        // the CSM receiver selects a cascade by linear view-z (Bevy get_cascade_index) then blends
        expect(code).toContain("if (viewZ < sunShadow.cascades[i].far)");
        expect(code).toContain("let nextNear = (1.0 - sunShadow.overlap) * thisFar;");
        // the fs feeds the fragment's world normal in (per-pixel) — per-vertex shading reads the 1.0 default
        expect(code).toContain("sunVisibility = sampleSunShadow(world, worldNormal);");
        // the residual constant lift toward the light (reverse-Z: adds, since the light is at greater depth)
        expect(code).toContain("let receiver = l.z + sunShadow.depthBias;");
    });

    // point/spot shadows bias the receiver in linear depth (the peter-panning fix), not by subtracting
    // depthBias from the hyperbolic perspective depth — both branches route through one helper
    test("point/spot shadow receivers bias in linear depth, not hyperbolic NDC", () => {
        const code = surfaceCode({ name: "t" });
        expect(code).toContain(
            "fn pointReceiver(z: f32, near: f32, far: f32, depthBias: f32) -> f32 {",
        );
        expect(code).toContain("receiver = pointReceiver(z, c.nf.x, c.nf.y, c.nf.z);");
        // the pre-fix form subtracted the bias straight from the hyperbolic NDC depth — gone
        expect(code).not.toContain("(z * (c.nf.y - c.nf.x)) - c.nf.z");
    });

    test("the fs chunk is spliced and varyings are rebound as locals", () => {
        const code = surfaceCode({ name: "t", fs: "col = vec4<f32>(uv, 0.0, 1.0);" });
        expect(code).toContain("col = vec4<f32>(uv, 0.0, 1.0);");
        // worldNormal crosses as a plain vec3, renormalized per fragment — NOT oct-encoded (oct
        // interpolation is invalid across the octahedral seam; see builtinFields)
        expect(code).toContain("let worldNormal = normalize(fin.worldNormal);");
        // uv crosses because the fs reads it; world always crosses (the scaffold's sampleSunShadow)
        expect(code).toContain("let uv = fin.uv;");
        expect(code).toContain("let world = fin.world;");
    });

    // the always-present built-ins are worldNormal + eid + world; uv / localPos prune
    // when the fs doesn't read them, so a chunk-free surface carries world at @location(2), not 4
    test("world position threads to the fs at the last built-in location", () => {
        const code = surfaceCode({ name: "t" });
        expect(code).toContain("@location(2) world: vec3<f32>,");
        expect(code).toContain("out.world = world.xyz;");
        // uv + localPos pruned (unread) — only worldNormal, eid, world cross
        expect(code).not.toContain(" uv: vec2<f32>,");
        expect(code).not.toContain(" localPos: vec3<f32>,");
    });

    test("a custom interpolator threads vs → fs after the built-in varyings", () => {
        const code = surfaceCode({
            name: "t",
            interpolators: { litFactor: "vec3<f32>" },
            vs: "litFactor = lightFactor(worldNormal);",
            fs: "col = vec4<f32>(litFactor, 1.0);",
        });
        // the fs reads neither uv nor localPos, so the built-ins are worldNormal(0), eid(1), world(2);
        // the custom follows at 3
        expect(code).toContain("@location(3) litFactor: vec3<f32>,");
        expect(code).toContain("var litFactor: vec3<f32>;");
        expect(code).toContain("out.litFactor = litFactor;");
        expect(code).toContain("let litFactor = fin.litFactor;");
    });

    test("integer interpolators are flat (the rasterizer can't lerp them)", () => {
        const code = surfaceCode({ name: "t", interpolators: { kind: "u32" } });
        expect(code).toContain("@location(3) @interpolate(flat) kind: u32,");
    });

    test("float interpolators interpolate", () => {
        const code = surfaceCode({ name: "t", interpolators: { fog: "f32" } });
        expect(code).toContain("@location(3) fog: f32,");
        expect(code).not.toContain("@interpolate(flat) fog");
    });

    test("multiple interpolators get sequential locations in declaration order", () => {
        const code = surfaceCode({ name: "t", interpolators: { a: "vec3<f32>", b: "f32" } });
        expect(code).toContain("@location(3) a: vec3<f32>,");
        expect(code).toContain("@location(4) b: f32,");
    });

    test("an interpolator named like a built-in local is rejected", () => {
        expect(() =>
            surfaceCode({ name: "t", interpolators: { worldNormal: "vec3<f32>" } }),
        ).toThrow(/reserved name/);
    });

    // the scaffold introduces `sunVisibility` + the inline-shadow helpers (`sampleSunShadow` +
    // its `shadowMap` / `shadowSamp` / `sunShadow` bindings) and the `tag` fs local; an interpolator
    // colliding with one would redeclare an in-scope name, failing to compile confusingly, so they're
    // reserved. Entry-point names (fs / fsTag / fsDepth) aren't reserved — an interpolator may
    // shadow one harmlessly
    test.each([
        "sunVisibility",
        "sampleSunShadow",
        "shadowMap",
        "shadowSamp",
        "sunShadow",
        "tag",
    ])("the fragment-scaffold name %s is reserved", (name) => {
        expect(() => surfaceCode({ name: "t", interpolators: { [name]: "f32" } })).toThrow(
            /reserved name/,
        );
    });

    test("overflowing the inter-stage budget is rejected", () => {
        // 3 built-ins (worldNormal, eid, world) for a chunk-free surface, cap 16 → 13 customs fit, 14 overflows
        const interpolators: Record<string, string> = {};
        for (let i = 0; i < 14; i++) interpolators[`v${i}`] = "f32";
        expect(() => surfaceCode({ name: "t", interpolators })).toThrow(/inter-stage limit/);
    });

    test("sampled-texture bindings emit texture_2d + sampler declarations", () => {
        const code = surfaceCode({
            name: "t",
            bindings: { albedo: { type: "texture-2d" }, samp: { type: "sampler" } },
            fs: "col = textureSample(albedo, samp, uv);",
        });
        // surface bindings start after frame/view/lighting (0..2), the vertex stream (3),
        // the point-light list (4..6), and the per-mesh quant table (7)
        expect(code).toContain("@group(0) @binding(8) var albedo: texture_2d<f32>;");
        expect(code).toContain("@group(0) @binding(9) var samp: sampler;");
        expect(code).toContain("col = textureSample(albedo, samp, uv);");
    });

    test("texture-2d-array bindings emit a texture_2d_array declaration", () => {
        const code = surfaceCode({
            name: "t",
            bindings: { albedo: { type: "texture-2d-array" }, samp: { type: "sampler" } },
            fs: "col = textureSample(albedo, samp, uv, 0);",
        });
        expect(code).toContain("@group(0) @binding(8) var albedo: texture_2d_array<f32>;");
        expect(code).toContain("@group(0) @binding(9) var samp: sampler;");
    });

    test("buffer and texture bindings coexist in declaration order", () => {
        const code = surfaceCode({
            name: "t",
            bindings: {
                tint: { type: "storage", element: "vec4<f32>" },
                albedo: { type: "texture-2d" },
                samp: { type: "sampler" },
            },
        });
        expect(code).toContain("@group(0) @binding(8) var<storage, read> tint: array<vec4<f32>>;");
        expect(code).toContain("@group(0) @binding(9) var albedo: texture_2d<f32>;");
        expect(code).toContain("@group(0) @binding(10) var samp: sampler;");
    });

    // the quantized vertex contract (gpu.md rule 6): the color VS decodes pos + oct normal + uv from the
    // 16 B main stream (`vertices`, vec4<u32>) against the meshId-selected MeshQuant; the prepass/shadow VS
    // decodes only position from the 8 B stream (`position`, vec2<u32>) — the depth passes' reduced read
    test("the color VS decodes the quantized vertex from the 16 B main stream + quant table", () => {
        const code = surfaceCode({ name: "t" });
        expect(code).toContain("var<storage, read> vertices: array<vec4<u32>>;");
        expect(code).toContain("var<storage, read> meshQuant: array<MeshQuant>;");
        expect(code).toContain("let v = vertices[vidx];");
        expect(code).toContain("var localPos = decodePos(v.x, v.y, mq);");
        expect(code).toContain("var localNormal = octDecodeNormal(v.z);");
        expect(code).toContain("var uv = decodeUv(v.w, mq);");
        // the retired 32 B f32 contract is gone
        expect(code).not.toContain("struct Vertex {");
        expect(code).not.toContain("v.posU.xyz");
    });

    test("the prepass VS reads position only from the 8 B stream", () => {
        const code = surfaceCode({ name: "t" }, "prepass");
        // binding 3 is the 8 B position stream here, not the 16 B main stream
        expect(code).toContain("var<storage, read> position: array<vec2<u32>>;");
        expect(code).not.toContain("array<vec4<u32>>");
        expect(code).toContain("let v = position[vidx];");
        expect(code).toContain("var localPos = decodePos(v.x, v.y, meshQuant[meshIdOf(v.y)]);");
        // depth needs no normal / uv — they default (the shadow render samples neither)
        expect(code).toContain("var localNormal = vec3<f32>(0.0, 0.0, 1.0);");
    });

    // color is a single target (no MRT); the prepass lanes ride a separate single-sample pass — and
    // a separate shader **module** (the `"prepass"` variant), so the prepass entries compile against
    // shadow stubs and never reference group 1. The id lane is its own `fsPrepassTag` entry point
    // returning the surface tag at location 0. An opaque surface's empty-lane prepass variant is
    // position-only (no `fn fsPrepass(`), so the only prepass fragment is the id one
    test("color is single-target (fs); the id lane is the prepass variant's fsPrepassTag entry", () => {
        const color = surfaceCode({ name: "t", fs: "col = vec4<f32>(1.0);" });
        // one color target, no MRT struct, no second color attachment, no prepass entries
        expect(color).toContain("fn fs(fin: VertexOut) -> @location(0) vec4<f32> {");
        expect(color).not.toContain("struct FragOut {");
        expect(color).not.toContain("@location(1) tag: u32,");
        expect(color).not.toContain("fn fsPrepass");
        expect(color).toContain("col = vec4<f32>(1.0);");
        // the id lane is its own opaque-only prepass entry point returning u32
        const prepass = surfaceCode({ name: "t", fs: "col = vec4<f32>(1.0);" }, "prepass");
        expect(prepass).toContain("fn fsPrepassTag(fin: VertexOut) -> @location(0) u32 {");
        expect(prepass).toContain("return tag;");
        expect(prepass).not.toContain("fn fs(fin: VertexOut) -> @location(0) vec4<f32> {");
        // an opaque surface's empty-lane-set prepass is position-only — no discard fragment
        expect(prepass).not.toContain("fn fsPrepass(fin: VertexOut) {");
    });

    // the prepass pipelines bind group 0 alone, so their module must never declare group-1
    // bindings — the chunk's `lit()` statically reaches the shadow helpers, which the prepass
    // variant therefore ships as stubs (return 1.0). The atlas render also draws through these
    // pipelines, so a real atlas read here would sample the texture being written
    test("the prepass variant stubs the shadow helpers and declares no group-1 bindings", () => {
        const prepass = surfaceCode(
            { name: "t", fs: "col = vec4<f32>(lit(col.rgb, worldNormal), 1.0);" },
            "prepass",
        );
        expect(prepass).not.toContain("@group(1)");
        expect(prepass).toContain(
            "fn sampleSunShadow(worldPos: vec3<f32>, normal: vec3<f32>) -> f32 { return 1.0; }",
        );
        expect(prepass).toContain(
            "fn pointShadowOf(light: PointLightGpu, normal: vec3<f32>, fragWorld: vec3<f32>) -> f32 { return 1.0; }",
        );
    });

    // the clustered loop's point-light term is shadowed per light: the compacted entry's color.a
    // carries the source eid, matched against the PointCaster slots (pos.w; -1 = empty). Only the
    // color variant carries the real atlas bindings + face projection
    test("the color variant samples point shadows inside the cluster loop", () => {
        const code = surfaceCode({ name: "t" });
        expect(code).toContain("@group(1) @binding(3) var pointAtlas: texture_depth_2d;");
        expect(code).toContain("@group(1) @binding(4) var<uniform> pointShadows: PointCasters;");
        expect(code).toContain(
            "* diff * spotFactor(light, L) * pointShadowOf(light, normal, fragWorld))",
        );
        expect(code).toContain("if (c.pos.w != light.color.a) { continue; }");
        expect(code).toContain("fn pointFaceOf(d: vec3<f32>) -> PointFace {");
        expect(code).toContain("textureSampleCompareLevel(pointAtlas, shadowSamp,");
    });

    // `tag` is a mutable fs local symmetric with `col`: an instanced surface defaults it to the
    // instance's eid; a non-instanced one (a world-space producer like terrain) to TAG_NONE
    // (4294967295) — the cleared-background value a reader takes as "no surface"
    test("the tag local defaults to eid for an instanced surface", () => {
        const code = surfaceCode({
            name: "t",
            bindings: {
                eids: { type: "storage", element: "u32" },
                transforms: { type: "storage", element: "Xform" },
            },
        });
        expect(code).toContain("var tag: u32 = eid;");
    });

    test("the tag local defaults to TAG_NONE for a non-instanced surface", () => {
        const code = surfaceCode({ name: "t" });
        expect(code).toContain("var tag: u32 = 4294967295u;");
    });

    // surface-authored: the fs chunk overrides `tag` (terrain → `capacity + cell`), and the override
    // lands in the prepass id entry. Symmetric with how the chunk writes `col` for the color pass
    test("the fs chunk can override the tag local", () => {
        const code = surfaceCode({ name: "t", fs: "tag = 100u + u32(uv.x);" }, "prepass");
        // the override is reachable from the id entry (the chunk splices into fsPrepassTag)
        const tagFn = code.slice(code.indexOf("fn fsPrepassTag"));
        expect(tagFn).toContain("tag = 100u + u32(uv.x);");
        expect(tagFn).toContain("return tag;");
    });

    // the prepass binds group 0 only — fsPrepassTag must not sample the shadow map, so the id lane
    // carries no lighting dependency (the group-1 shadow bindings stay unreferenced from this entry)
    test("fsPrepassTag does not sample the sun shadow", () => {
        const code = surfaceCode({ name: "t" }, "prepass");
        const tagFn = code.slice(
            code.indexOf("fn fsPrepassTag"),
            code.indexOf("fn fsPrepassTag") + 400,
        );
        expect(tagFn).not.toContain("sampleSunShadow");
    });

    // a `blend` (alpha) surface renders only in the non-opaque color pass: one color target, no prepass
    // lanes at all (a transparent pixel has no single owner / writes no prepass depth / casts nothing)
    test("a blend surface has a color target but no prepass lanes", () => {
        const code = surfaceCode({
            name: "t",
            blend: "alpha",
            fs: "col = vec4<f32>(1.0, 0.0, 0.0, 0.5);",
        });
        expect(code).toContain("fn fs(fin: VertexOut) -> @location(0) vec4<f32> {");
        expect(code).toContain("return col;");
        expect(code).not.toContain("fn fsPrepass");
        // the chunk + the sun-shadow sample still splice — a blended surface can be lit
        expect(code).toContain("col = vec4<f32>(1.0, 0.0, 0.0, 0.5);");
        expect(code).toContain("sunVisibility = sampleSunShadow(world, worldNormal);");
    });

    // clip is orthogonal to blend — a `blend` surface's fs may `discard` (a hard-edged decal),
    // which the non-opaque pass runs for free (no prepass to hole)
    test("a clip chunk's discard splices into the blend fragment", () => {
        const code = surfaceCode({
            name: "t",
            blend: "alpha",
            fs: "if (uv.x < 0.5) { discard; } col = vec4<f32>(1.0);",
        });
        expect(code).toContain("discard;");
        expect(code).toContain("fn fs(fin: VertexOut) -> @location(0) vec4<f32> {");
    });

    // a `clip` (masked-opaque cutout) surface is opaque: a single-target color fragment + the id lane,
    // plus the empty-lane-set prepass fragment (`fsPrepass`) that runs the chunk's `discard` so the
    // shadowmap + a depth-only prepass cut the holes (and it casts a holed shadow)
    test("a clip surface emits color + id lane + a discard-only depth fragment", () => {
        const spec = {
            name: "t",
            blend: "clip" as const,
            fs: "if (uv.x < 0.5) { discard; } col = vec4<f32>(1.0);",
        };
        // opaque single-target color (not the alpha-only path) in the color variant
        const color = surfaceCode(spec);
        expect(color).toContain("fn fs(fin: VertexOut) -> @location(0) vec4<f32> {");
        expect(color).not.toContain("struct FragOut {");
        // the id lane (hover target) + the empty-lane-set prepass fragment that runs the chunk for
        // its discard, no return — unlike an opaque surface, a clip one emits it (the position-only
        // variant would skip the discard)
        const prepass = surfaceCode(spec, "prepass");
        expect(prepass).toContain("fn fsPrepassTag(fin: VertexOut) -> @location(0) u32 {");
        expect(prepass).toContain("fn fsPrepass(fin: VertexOut) {");
        expect(prepass).toContain("discard;");
        // the prepass module binds group 0 only — no group-1 shadow declarations at all
        expect(prepass).not.toContain("@group(1)");
    });

    // adding a lane (normal / motion) must stay a COLOR_LANES table row + the subset codegen, never a
    // new pass: the prepass fragments are generated by iterating the lane subsets. With only the id lane
    // active, the requestable subsets are {} (depth-only) and {id} — so an opaque surface emits exactly
    // one prepass fragment (the id one), proving the subset machinery doesn't over-generate
    test("the prepass fragments come from the lane subsets, not a fixed list", () => {
        const code = surfaceCode({ name: "t" }, "prepass");
        const entries = [...code.matchAll(/fn fsPrepass\w*\(/g)].map((m) => m[0]);
        expect(entries).toEqual(["fn fsPrepassTag("]);
    });

    // a screen-space surface (lines) projects its own endpoints: the vs chunk writes clipPos and sear
    // emits `out.clip = clipPos`. A world-space surface (the default) projects view.viewProj * world
    // *after* the chunk, so a chunk that displaces world still projects correctly (the fountain case)
    test("a screen surface emits out.clip = clipPos and exposes a clipPos local", () => {
        const code = surfaceCode({ name: "t", screen: true, vs: "clipPos = vec4<f32>(1.0);" });
        expect(code).toContain("var clipPos = vec4<f32>(0.0);");
        expect(code).toContain("out.clip = clipPos;");
        expect(code).not.toContain("out.clip = view.viewProj * world;");
    });

    test("a world-space surface projects view.viewProj * world and has no clipPos local", () => {
        const code = surfaceCode({ name: "t" });
        expect(code).toContain("out.clip = view.viewProj * world;");
        expect(code).not.toContain("var clipPos");
    });

    test("clipPos is reserved so an interpolator can't shadow the screen-space override", () => {
        expect(() => surfaceCode({ name: "t", interpolators: { clipPos: "vec4<f32>" } })).toThrow(
            /reserved name/,
        );
    });

    // the View struct carries resolution (pixels) so a screen-space producer sizes constant-pixel geometry
    test("the View struct carries resolution", () => {
        const code = surfaceCode({ name: "t" });
        expect(code).toContain("resolution: vec2<f32>,");
    });

    // a specializing surface (the glTF importer) splices `specialize(variant)`'s preamble/fs per material
    // map-set; sear compiles one pipeline per variant a scene draws (Bevy's on-demand specialize), so a
    // sparse-map material samples only the maps it carries
    test("specialize splices the variant's preamble + fs; the variant arg routes through", () => {
        const spec = {
            name: "t",
            specialize: (v: number) => ({
                preamble: `// PRE${v}`,
                fs: `col = vec4<f32>(f32(${v}u));`,
            }),
        };
        const v2 = surfaceCode(spec, "color", 2);
        expect(v2).toContain("// PRE2");
        expect(v2).toContain("col = vec4<f32>(f32(2u));");
        const v5 = surfaceCode(spec, "color", 5);
        expect(v5).toContain("// PRE5");
        expect(v5).not.toContain("// PRE2");
        // default variant is 0 (a non-specializing surface ignores it entirely)
        expect(surfaceCode(spec)).toContain("// PRE0");
    });

    // specialize returning only a `preamble` (the glTF case) overrides `surface.preamble` but leaves
    // `surface.fs` standing — so the importer specializes the map-set helpers without restating the fs
    test("specialize.preamble overrides surface.preamble; a missing specialize.fs keeps surface.fs", () => {
        const code = surfaceCode({
            name: "t",
            preamble: "// BASE",
            fs: "col = vec4<f32>(uv, 0.0, 1.0);",
            specialize: () => ({ preamble: "// SPECIAL" }),
        });
        expect(code).toContain("// SPECIAL");
        expect(code).not.toContain("// BASE");
        expect(code).toContain("col = vec4<f32>(uv, 0.0, 1.0);");
    });
});

// the vertices buffer is `array<Vertex>` (posU + normalV = 8 floats). A length
// that isn't a whole number of records would misalign every vertex past the
// first — a silent CPU→GPU corruption, so `mesh()` rejects it up front
describe("mesh() vertex contract", () => {
    test("rejects a vertices length that isn't a whole number of Vertex records", () => {
        expect(() =>
            mesh({ name: "bad", vertices: new Float32Array(7), indices: new Uint32Array([0]) }),
        ).toThrow(/not a multiple of 8/);
    });

    // static meshes pack into one shared buffer; each mesh's indices are shifted
    // by its vertex base so the shared index stream is absolute. Pure-tested
    // because the shift is the one place this silently corrupts geometry.
    test("packMeshes concatenates and shifts indices by vertex base", () => {
        const packed = packMeshes([
            {
                name: "a",
                vertices: new Float32Array(4 * 8),
                indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
            },
            { name: "b", vertices: new Float32Array(3 * 8), indices: new Uint32Array([0, 1, 2]) },
        ]);
        // a unshifted (base 0); b shifted by a's 4 vertices. The per-mesh vertex range
        // (base + count) keys the quant table — a vertex belongs to exactly one mesh.
        expect([...packed.indices]).toEqual([0, 1, 2, 0, 2, 3, 4, 5, 6]);
        expect(packed.vertices.length).toBe((4 + 3) * 8);
        expect(packed.slices).toEqual([
            { name: "a", indexBase: 0, indexCount: 6, vertexBase: 0, vertexCount: 4 },
            { name: "b", indexBase: 6, indexCount: 3, vertexBase: 4, vertexCount: 3 },
        ]);
    });

    // The quantized GPU streams (gpu.md rule 6). The CPU encode here pairs with the WGSL decode
    // (POS_QUANT_WGSL) — the lattice must match, so this pins the round-trip bound (extent/65535
    // per axis, derived) the shader decodes against, plus the meshId pack + the position-stream
    // mirror. Two meshes in one family so meshId 0/1 both exercise; non-trivial position + uv AABBs.
    test("quantizeMeshes round-trips pos+uv within extent/65535, packs meshId, mirrors the position stream", () => {
        const packed = packMeshes([
            {
                name: "a",
                // px py pz u  nx ny nz v — a spread position + uv box, +z normal
                vertices: new Float32Array([
                    -2, 0.5, -10, 0, 0, 0, 1, 0, 3, 1.5, 7, 4, 0, 0, 1, 8, 0, 1, -1, 2, 0, 0, 1, 4,
                ]),
                indices: new Uint32Array([0, 1, 2]),
            },
            {
                name: "b",
                vertices: new Float32Array([
                    1, 1, 1, 0.25, 1, 0, 0, 0.5, 2, 2, 2, 0.75, 1, 0, 0, 1.5,
                ]),
                indices: new Uint32Array([0, 1]),
            },
        ]);
        const { main, position, quant } = quantizeMeshes(packed.vertices, packed.slices);

        for (const [meshId, s] of packed.slices.entries()) {
            const o = meshId * 12;
            const [pminX, pminY, pminZ, uminX] = quant.slice(o, o + 4);
            const [pextX, pextY, pextZ, uminY] = quant.slice(o + 4, o + 8);
            const [uextX, uextY] = quant.slice(o + 8, o + 10);
            for (let v = 0; v < s.vertexCount; v++) {
                const vi = s.vertexBase + v;
                const src = vi * 8;
                const w0 = main[vi * 4];
                const w1 = main[vi * 4 + 1];
                const w3 = main[vi * 4 + 3];
                expect(w1 >>> 16).toBe(meshId); // meshId packs into w1's high half
                expect(position[vi * 2]).toBe(w0); // the 8 B depth stream mirrors w0/w1
                expect(position[vi * 2 + 1]).toBe(w1);
                const px = pminX + ((w0 & 0xffff) / 65535) * pextX;
                const py = pminY + ((w0 >>> 16) / 65535) * pextY;
                const pz = pminZ + ((w1 & 0xffff) / 65535) * pextZ;
                expect(Math.abs(px - packed.vertices[src])).toBeLessThanOrEqual(pextX / 65535);
                expect(Math.abs(py - packed.vertices[src + 1])).toBeLessThanOrEqual(pextY / 65535);
                expect(Math.abs(pz - packed.vertices[src + 2])).toBeLessThanOrEqual(pextZ / 65535);
                const u = uminX + ((w3 & 0xffff) / 65535) * uextX;
                const vv = uminY + ((w3 >>> 16) / 65535) * uextY;
                expect(Math.abs(u - packed.vertices[src + 3])).toBeLessThanOrEqual(
                    uextX / 65535 + 1e-9,
                );
                expect(Math.abs(vv - packed.vertices[src + 7])).toBeLessThanOrEqual(
                    uextY / 65535 + 1e-9,
                );
            }
        }
    });
});

// Structural validation of the backdrop codegen. `backgroundCode` is pure (no GPU), so the contract — a
// fullscreen-triangle VS at the reverse-Z far plane, the per-pixel view-ray reconstruction (not an
// interpolator), the HDR `col` output, and bindings starting after frame/view/lighting — is checked by
// reading the generated source; the real-GPU compile + the backdrop-fills-only-background-pixels behavior
// live in the gym `render` `background` mode.
describe("sear backgroundCode", () => {
    test("emits a fullscreen triangle at the reverse-Z far plane", () => {
        const code = backgroundCode({ name: "t", fs: "col = vec3<f32>(dir);" });
        // no vertex pull — the three corners come from the vertex index
        expect(code).toContain("fn vs(@builtin(vertex_index) vidx: u32) -> VertexOut {");
        expect(code).toContain("(vidx << 1u) & 2u");
        // clip z = 0 is the reverse-Z far plane, so the depth-equal test admits only un-rendered pixels
        expect(code).toContain("out.clip = vec4<f32>(c * 2.0 - 1.0, 0.0, 1.0);");
    });

    test("reconstructs the world-space view ray per-pixel from invViewProj, not an interpolator", () => {
        const code = backgroundCode({ name: "t", fs: "col = vec3<f32>(dir);" });
        // derived from @builtin(position) + the inverse view-projection (gpu.md rule 9) — the VertexOut
        // carries only the clip position, never a `dir` varying
        expect(code).toContain("let uv = fin.clip.xy / view.resolution;");
        expect(code).toContain("vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0)");
        expect(code).toContain("view.invViewProj * vec4<f32>(ndc, 1.0)");
        expect(code).toContain("let dir = normalize(far.xyz / far.w - view.eye.xyz);");
        expect(code).not.toContain("@location(0) dir");
        // the chunk writes an HDR vec3; sear returns it opaque. No surface scaffolding (vertex pull, lit, shadows)
        expect(code).toContain("var col = vec3<f32>(0.0);");
        expect(code).toContain("{ col = vec3<f32>(dir); }");
        expect(code).toContain("return vec4<f32>(col, 1.0);");
        expect(code).not.toContain("vertices[");
        expect(code).not.toContain("sampleSunShadow");
    });

    test("frame/view/lighting are group-0 0/1/2; the background's own bindings follow at BG_BASE (3)", () => {
        const code = backgroundCode({
            name: "t",
            bindings: { sky: { type: "uniform", struct: "Sky" } },
            fs: "col = sky.zenith;",
        });
        expect(code).toContain("@group(0) @binding(0) var<uniform> frame: Frame;");
        expect(code).toContain("@group(0) @binding(1) var<uniform> view: View;");
        expect(code).toContain("@group(0) @binding(2) var<uniform> lighting: Lighting;");
        expect(code).toContain("@group(0) @binding(3) var<uniform> sky: Sky;");
    });

    test("a preamble is spliced at module scope and an f16 binding enables the directive", () => {
        const code = backgroundCode({
            name: "t",
            bindings: { tint: { type: "storage", element: "vec4<f16>" } },
            preamble: "fn helper() -> f32 { return 1.0; }",
            fs: "col = vec3<f32>(helper());",
        });
        expect(code.startsWith("enable f16;")).toBe(true);
        expect(code).toContain("fn helper() -> f32 { return 1.0; }");
    });
});

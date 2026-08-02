import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { body, noIntegerDivision } from "../../../tests/wgsl";
import { Compute, type Plugin, Registry } from "../../engine";
import { precompile, precompileAll, requestGPU } from "../../engine/runtime";
import { unpackLdrColor, Xform } from "../../engine/utils/core";
import { publishPartDraws } from "../part/part";
import { RenderPlugin } from "../render";
import type { Binding, Draw, Mesh, Surface } from "../render/core";
import { Draws, Meshes } from "../render/core";
import { shadowLayoutTyped } from "./atlas";
import { backgroundCode, pointShadowCode, surfaceCode } from "./codegen";
import type { TypedBackground } from "./contract";
import {
    BgCtx,
    backgroundLayout as bgLayout,
    fsCtxSchema,
    surfaceLayout as layout,
    TypedBackgrounds,
    type TypedSurface,
    VsIn,
    vsPatchSchema,
} from "./contract";
import { litPbr } from "./engine";
import {
    type Background,
    Backgrounds,
    precompileTypedVariants,
    registerBackground,
    SearPlugin,
} from "./forward";
import {
    clearGroups,
    compileTypedVariant,
    getCompiledTyped,
    getTypedGroup,
    knownTypedVariants,
    setTypedGroup,
    surfacePrimitive,
    type TypedGroupEntry,
    typedBgWgsl,
    typedPrepassWgsl,
    typedShadowWgsl,
    typedSurfaceWgsl,
} from "./pipelines";
import { COMBO_SHIFT, EID_MASK } from "./regather";
import { Pbr } from "./shade";
import {
    cascadeAtlasSize,
    MAX_CASCADES,
    pointAtlasSize,
    pointCasters,
    sunCascades,
    sunResolution,
} from "./shadows";

// The typed pipeline builder (4a-ii-c-2 template stage, extended at c-3): `compileTypedVariant`
// (device-bound — see pipelines.ts, exercised for real only by `bun bench` warming `SearPlugin`) and its
// device-free resolve seam `typedSurfaceWgsl`. This file pins the WGSL differential against the string
// path's `surfaceCode` for `default` (the load-bearing gate) and `unlit` (c-3's second template
// extension), plus the remaining boundary guards and the generic `"clip"` / `specialize` support used by
// the typed glTF surfaces. `"alpha"` is a carried capability (the transparent pipeline).
//
// `default`'s raw + typed shapes are both authored fresh here, matching `forward.ts`'s `litBindings` /
// `PbrPreamble` / `fs` and `typedDefaultLayout` / `typedDefaultFs` respectively — a second, independent
// copy of each, so a drift between the real registration and this pin shows up as a real failure.

const litBindingsRaw: Record<string, Binding> = {
    eids: { type: "storage", element: "u32" },
    transforms: { type: "storage", element: "Xform" },
    color: { type: "storage", element: "u32" },
    material: { type: "storage", element: "vec2<u32>" },
};

const PbrPreambleRaw = /* wgsl */ `
fn matOf(eid: u32) -> Pbr {
    let m = material[eid];
    let mr = unpack2x16float(m.x);
    let eo = unpack2x16float(m.y);
    return Pbr(unpackLdrColor(color[eid]).rgb, mr.x, mr.y, eo.y, 0.0);
}
fn emissiveOf(eid: u32) -> vec3<f32> {
    return unpackLdrColor(color[eid]).rgb * unpack2x16float(material[eid].y).x;
}`;

const rawDefaultSurface: Surface = {
    name: "default",
    bindings: litBindingsRaw,
    preamble: PbrPreambleRaw,
    fs: /* wgsl */ `col = vec4<f32>(litPbr(matOf(eid), worldNormal, world) + emissiveOf(eid), 1.0);`,
};

const typedDefaultLayout = layout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    color: { type: "storage", element: d.u32 },
    material: { type: "storage", element: d.vec2u },
});

const typedDefaultFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const m = typedDefaultLayout.$.material[ctx.eid];
    const mr = std.unpack2x16float(m.x);
    const eo = std.unpack2x16float(m.y);
    const albedo = unpackLdrColor(typedDefaultLayout.$.color[ctx.eid]).xyz;
    const pbr = Pbr({ albedo, metallic: mr.x, roughness: mr.y, occlusion: eo.y, dielectric: 0 });
    const emissive = std.mul(albedo, eo.x);
    return d.vec4f(std.add(litPbr(pbr, ctx.worldNormal, ctx.world), emissive), 1);
});

const typedDefaultSurface = { name: "default", layout: typedDefaultLayout, fs: typedDefaultFs };

describe('typedSurfaceWgsl(default) vs surfaceCode(default, "color") — the load-bearing differential', () => {
    const rawWgsl = surfaceCode(rawDefaultSurface, "color", 0);
    const typedWgsl = typedSurfaceWgsl(typedDefaultSurface);

    test("the typed fs computes the same Pbr fields, same operand order, as matOf/litPbr/emissiveOf", () => {
        // metallic/roughness from word x, emissive/occlusion from word y — same unpack calls. `m` reads
        // as a pointer (the array-element-read law, 3b-ii), so the deref shows as `(*m).x` — accounted
        expect(typedWgsl).toContain("unpack2x16float((*m).x)");
        expect(typedWgsl).toContain("unpack2x16float((*m).y)");
        // Pbr(albedo, metallic, roughness, occlusion, dielectric) — same field values as matOf's
        // `Pbr(unpackLdrColor(color[eid]).rgb, mr.x, mr.y, eo.y, 0.0)`
        expect(typedWgsl).toMatch(/Pbr\(albedo, mr\.x, mr\.y, eo\.y, 0f\)/);
        expect(typedWgsl).toContain("let emissive = (albedo * eo.x);");
        // the raw text calls unpackLdrColor(color[eid]) twice (matOf + emissiveOf, two chunks); the typed
        // fs computes `albedo` once and reuses it for `emissive` — a real, accounted dedup (DXC/Naga CSE
        // this regardless), flagged in the executor's report as a new deviation class, not an established one
        expect((rawWgsl.match(/unpackLdrColor\(color\[eid\]\)/g) ?? []).length).toBe(2);
        expect((typedWgsl.match(/unpackLdrColor\(color\[ctx\.eid\]\)/g) ?? []).length).toBe(1);
    });

    test("both call litPbr(pbr, worldNormal, world) — the shared engine scaffold, not a re-derivation", () => {
        expect(rawWgsl).toContain("litPbr(matOf(eid), worldNormal, world)");
        expect(typedWgsl).toContain("litPbr(pbr, ctx.worldNormal, ctx.world)");
    });

    test("both decode the same four vertex-pull steps: main-stream read, meshQuant lookup, position + oct-normal decode", () => {
        for (const wgsl of [rawWgsl, typedWgsl]) {
            expect(wgsl).toContain("meshIdOf(");
            expect(wgsl).toContain("decodePos(");
            expect(wgsl).toContain("octDecodeNormal(");
        }
    });

    test("both apply the standard instance transform (eids/transforms declared → instanced)", () => {
        expect(rawWgsl).toContain("eid = eids[iid];");
        expect(rawWgsl).toContain("xformPoint(xf, world.xyz)");
        expect(rawWgsl).toContain("xformNormal(xf, worldNormal)");
        // typed: the resolved transform is copied into the VsIn-carried value a deforming surface reuses
        expect(typedWgsl).toContain("eid = eids[iid];");
        expect(typedWgsl).toContain("xformPoint(xform, world.xyz)");
        expect(typedWgsl).toContain("xformNormal(xform, worldNormal)");
    });

    test("the raw path prunes uv/localPos (default's fs reads neither, gpu.md rule 9); the typed path now crosses both for real (c-3)", () => {
        expect(rawWgsl).not.toMatch(/@location\(\d+\) uv: vec2<f32>/);
        expect(rawWgsl).not.toMatch(/@location\(\d+\) localPos: vec3<f32>/);
        // the typed vs→fs interstage struct: five `@location` slots (worldNormal, eid, world, uv,
        // localPos) — c-2 zero-filled uv/localPos on the fs side; c-3 wires the real crossing
        // unconditionally (every typed surface pays the two slots this stage, a disclosed regression from
        // the raw path's per-surface prune — `compileTypedVariant`'s module header)
        const interstage = typedWgsl.slice(
            typedWgsl.indexOf("_Output {"),
            typedWgsl.indexOf("}", typedWgsl.indexOf("_Output {")),
        );
        expect((interstage.match(/@location\(/g) ?? []).length).toBe(5);
        expect(interstage).toContain("uv");
        expect(interstage).toContain("localPos");
        // and the ctx construction genuinely reads the crossed fs-input values, not a zero-fill constant
        expect(typedWgsl).toMatch(
            /CtxSchema\(_arg_0\.eid, _arg_0\.world, worldNormal, _arg_0\.uv, _arg_0\.localPos\)/,
        );
    });

    test("idiv audit: the newly-authored vs/fs entry bodies carry no division at all", () => {
        // scoped to `typedColorVs`/`typedColorFs`'s own emitted bodies — the transitively-pulled
        // dependencies (`decodePos`'s unorm dequant, `pointShadowOf`'s PCF loop) are already-audited,
        // already-shipped code from earlier stages (1b/2a), not new c-2 code, and running the audit over
        // the whole resolved text re-flags their pre-existing legitimate float divisions as false positives
        noIntegerDivision(body(typedWgsl, "fn defaultVs("));
        noIntegerDivision(body(typedWgsl, "fn defaultFs("));
    });
});

describe("compileTypedVariant — boundaries and glTF support", () => {
    test("a `varyings`-declaring surface with no `vs` throws — a varying can only be written by the surface's own vs chunk", () => {
        const l = layout({ items: { type: "storage", element: d.f32 } });
        const fs = tgpu.fn(
            [fsCtxSchema({ tint: d.vec3f })],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        expect(() =>
            compileTypedVariant({
                name: `varyings-guard-${Math.random()}`,
                layout: l,
                varyings: { tint: d.vec3f },
                fs,
            }),
        ).toThrow(/varyings/);
    });

    test("a `screen` surface with no `vs` throws — nothing but its own vs chunk can supply the clip position", () => {
        const l = layout({});
        const fs = tgpu.fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        expect(() =>
            compileTypedVariant({
                name: `screen-guard-${Math.random()}`,
                layout: l,
                fs,
                screen: true,
            }),
        ).toThrow(/screen surface with no vs/);
    });

    test('a `blend: "alpha"` surface no longer throws at the guard — it proceeds to pipeline creation (device-bound, `bun bench` gate)', () => {
        const l = layout({});
        const fs = tgpu.fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        // `bun test` binds no device (testing.md — never bind a device here), so past the now-dropped
        // guard the call reaches `Compute.root` and throws on the missing device, not on a boundary
        // check — the positive proof (a real transparent pipeline compiles) is `bun bench`'s job.
        expect(() =>
            compileTypedVariant({
                name: `alpha-past-guard-${Math.random()}`,
                layout: l,
                fs,
                blend: "alpha",
            }),
        ).not.toThrow(/blend/);
    });

    test("specialize replaces the emitted fs for the requested variant", () => {
        const l = layout({});
        const baseFs = tgpu
            .fn(
                [fsCtxSchema()],
                d.vec4f,
            )(() => {
                "use gpu";
                return d.vec4f(0);
            })
            .$name("specializeBaseFs");
        const variantFs = tgpu
            .fn(
                [fsCtxSchema()],
                d.vec4f,
            )(() => {
                "use gpu";
                return d.vec4f(1);
            })
            .$name("specializeVariantFs");
        const surface = {
            name: "specialize-probe",
            layout: l,
            fs: baseFs,
            specialize: (variant: number) => ({ fs: variant === 7 ? variantFs : baseFs }),
        };
        expect(typedSurfaceWgsl(surface, 0)).toContain("fn specializeBaseFs(");
        const variant = typedSurfaceWgsl(surface, 7);
        expect(variant).toContain("fn specializeVariantFs(");
        expect(variant).not.toContain("fn specializeBaseFs(");
    });

    test("a same-name surface replacement recompiles the warmed typed variant", () => {
        const name = `replacement-probe-${Math.random()}`;
        const first = {
            name,
            layout: layout({}),
            fs: tgpu.fn(
                [fsCtxSchema()],
                d.vec4f,
            )(() => {
                "use gpu";
                return d.vec4f(0);
            }),
        };
        const replacement = {
            name,
            layout: layout({}),
            fs: tgpu.fn(
                [fsCtxSchema()],
                d.vec4f,
            )(() => {
                "use gpu";
                return d.vec4f(1);
            }),
        };

        // Pipeline wrappers are lazy; this root stub lets the cache contract stay a device-free unit
        // test while the real WGSL/device validation remains the typed-variant bench gate.
        const originalRoot = Compute.root;
        const fakeRoot = {
            with: () => fakeRoot,
            createRenderPipeline: () => ({
                $name() {
                    return this;
                },
            }),
        };
        (Compute as unknown as { root: typeof fakeRoot }).root = fakeRoot;
        try {
            const firstCompiled = compileTypedVariant(first);
            const firstGroup = {
                owner: first,
                layout: first.layout,
                quant: {} as GPUBuffer,
                color: {} as GPUBindGroup,
                depth: null,
                point: null,
                cascade: null,
                engineCache: new Map(),
                atlasG0: {} as GPUBindGroup,
                resources: [],
            } as unknown as TypedGroupEntry;
            setTypedGroup(name, firstGroup);

            // The unchanged owner keeps both identity caches hot.
            expect(compileTypedVariant(first)).toBe(firstCompiled);
            expect(getTypedGroup(name, first)).toBe(firstGroup);

            const replacementCompiled = compileTypedVariant(replacement);

            expect(replacementCompiled).not.toBe(firstCompiled);
            expect(replacementCompiled.owner).toBe(replacement);
            expect(replacementCompiled.layout).toBe(replacement.layout);
            expect(getCompiledTyped(name)).toBe(replacementCompiled);
            // The new spec can reuse the same GPU resources, but never the old layout-bound group.
            expect(getTypedGroup(name, replacement)).toBeUndefined();
            expect(getTypedGroup(name)).toBeUndefined();
        } finally {
            (Compute as unknown as { root: typeof originalRoot }).root = originalRoot;
        }
    });

    test("clip runs the authored discard in depth, tag, point, and cascade fragments", () => {
        const l = layout({
            eids: { type: "storage", element: d.u32 },
            transforms: { type: "storage", element: Xform },
        });
        const fs = tgpu
            .fn(
                [fsCtxSchema()],
                d.vec4f,
            )((ctx) => {
                "use gpu";
                if (ctx.eid === 0) std.discard();
                return d.vec4f(1);
            })
            .$name("clipProbeFs");
        const surface = { name: "clip-probe", layout: l, fs, blend: "clip" as const };
        const prepass = typedPrepassWgsl(surface);
        const shadow = typedShadowWgsl(surface);
        for (const wgsl of [prepass[""], prepass.tag, shadow.point, shadow.cascade]) {
            expect(wgsl).toContain("fn clipProbeFs(");
            expect(wgsl).toContain("discard;");
        }
    });

    test("clip carries its declared varying through depth, tag, point, and cascade cutoff fragments", () => {
        const varyings = { cutoffWeight: d.f32 };
        const l = layout({
            eids: { type: "storage", element: d.u32 },
            transforms: { type: "storage", element: Xform },
        });
        const Patch = vsPatchSchema(varyings);
        const vs = tgpu
            .fn(
                [VsIn],
                Patch,
            )((input) => {
                "use gpu";
                return Patch({
                    world: input.world,
                    worldNormal: input.worldNormal,
                    clip: d.vec4f(0),
                    cutoffWeight: input.uv.x,
                });
            })
            .$name("clipVaryingProbeVs");
        const fs = tgpu
            .fn(
                [fsCtxSchema(varyings)],
                d.vec4f,
            )((ctx) => {
                "use gpu";
                if (ctx.cutoffWeight < 0.5) std.discard();
                return d.vec4f(1);
            })
            .$name("clipVaryingProbeFs");
        const surface = {
            name: "clip-varying-probe",
            layout: l,
            varyings,
            vs,
            fs,
            blend: "clip" as const,
        };
        const prepass = typedPrepassWgsl(surface);
        const shadow = typedShadowWgsl(surface);
        for (const wgsl of [prepass[""], prepass.tag, shadow.point, shadow.cascade]) {
            expect(wgsl).toMatch(/@location\(5\) cutoffWeight: f32/);
            expect(wgsl).toMatch(/@location\(5\) v0: f32/);
            expect(wgsl).toContain(
                "let ctx = Ctx(eid, world, normalize(worldNormalIn), uv, localPos, v0);",
            );
            expect(wgsl).toContain("world = patched.world;");
            expect(wgsl).toContain("worldNormal = patched.worldNormal;");
            expect(wgsl).toContain("fn clipVaryingProbeFs(");
            expect(wgsl).toContain("discard;");
        }
    });

    test("a non-instanced clip calls its cutoff with material eid zero but writes TAG_NONE", () => {
        const l = layout({});
        const fs = tgpu
            .fn(
                [fsCtxSchema()],
                d.vec4f,
            )((ctx) => {
                "use gpu";
                if (ctx.eid !== 0) std.discard();
                return d.vec4f(1);
            })
            .$name("nonInstancedClipFs");
        const prepass = typedPrepassWgsl({
            name: "non-instanced-clip",
            layout: l,
            fs,
            blend: "clip",
        });
        for (const wgsl of Object.values(prepass)) {
            expect(wgsl).toContain("let eid = 0u;");
            expect(wgsl).toContain("let ctx = Ctx(_arg_0.eid");
            expect(wgsl).toContain("nonInstancedClipFs(ctx);");
        }
        expect(prepass.tag).toContain("return vec4u(4294967295, 0, 0, 0);");
    });

    test("specialize replaces the vs in color, prepass, point, and cascade", () => {
        const l = layout({
            eids: { type: "storage", element: d.u32 },
            transforms: { type: "storage", element: Xform },
        });
        const fs = tgpu.fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        const Patch = vsPatchSchema();
        const variantVs = tgpu
            .fn(
                [VsIn],
                Patch,
            )((input) => {
                "use gpu";
                return Patch({
                    world: input.world,
                    worldNormal: input.worldNormal,
                    clip: d.vec4f(0),
                } as any);
            })
            .$name("specializeVariantVs");
        const surface = {
            name: "specialize-vs-probe",
            layout: l,
            fs,
            specialize: () => ({ fs, vs: variantVs }),
        };

        expect(typedSurfaceWgsl(surface, 7)).toContain("fn specializeVariantVs(");
        for (const wgsl of Object.values(typedPrepassWgsl(surface, 7))) {
            expect(wgsl).toContain("fn specializeVariantVs(");
        }
        for (const wgsl of Object.values(typedShadowWgsl(surface, 7))) {
            expect(wgsl).toContain("fn specializeVariantVs(");
        }
    });
});

test("known typed specialization variants come from registered draw/mesh pairs and dedupe", () => {
    const buffer = {} as GPUBuffer;
    for (const [name, variant] of [
        ["warm-mesh-a", 3],
        ["warm-mesh-b", 3],
        ["warm-mesh-c", 7],
    ] as const) {
        Meshes.register({
            name,
            vertices: buffer,
            position: buffer,
            quant: buffer,
            indices: buffer,
            indexBase: 0,
            indexCount: 3,
            variant,
        });
        Draws.register({
            name: `warm-draw-${name}`,
            surface: "warm-probe",
            mesh: name,
            args: { indirect: buffer },
        });
    }
    const surface = {
        name: "warm-probe",
        layout: layout({}),
        fs: typedDefaultFs,
        specialize: () => ({ fs: typedDefaultFs }),
    };
    expect(knownTypedVariants(surface).sort((a, b) => a - b)).toEqual([3, 7]);
});

test("Sear warms Part-published specializing variants; post-build variants stay lazy", async () => {
    const warmMesh = `warm-mesh-${Math.random()}`;
    const lateMesh = `late-mesh-${Math.random()}`;
    const surfaceName = `specializing-surface-${Math.random()}`;
    const warmVariant = 7;
    const lateVariant = 11;
    const buffer = {} as GPUBuffer;
    const surfaces = new Registry<Surface>();
    const typedSurfaces = new Registry<TypedSurface>();
    const meshes = new Registry<Mesh>();
    const draws = new Registry<Draw>();
    const registries = { surfaces, meshes, draws };
    const typedLayout = layout({});
    const fs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    });
    surfaces.register({
        name: surfaceName,
        bindings: {
            eids: { type: "storage", element: "u32" },
            transforms: { type: "storage", element: "Xform" },
        },
    });
    typedSurfaces.register({
        name: surfaceName,
        layout: typedLayout,
        fs,
        specialize: () => ({ fs }),
    });
    const mesh = (name: string, variant: number) =>
        meshes.register({
            name,
            vertices: buffer,
            position: buffer,
            quant: buffer,
            indices: buffer,
            indexBase: 0,
            indexCount: 3,
            variant,
        });
    const producer: Plugin = {
        name: `VariantProducer${Math.random()}`,
        dependencies: [RenderPlugin],
        initialize() {
            mesh(warmMesh, warmVariant);
        },
    };
    const saved = { ...Compute };
    try {
        await requestGPU({
            queue: { onSubmittedWorkDone: async () => {} },
            pushErrorScope: () => {},
            popErrorScope: async () => null,
        } as unknown as GPUDevice);
        await producer.initialize?.({} as never);

        const events: string[] = [];
        const warmed = new Set<string>();
        precompileTypedVariants(
            (surface, variant) => {
                const key = `${surface.name}#${variant}`;
                events.push(`sear:${key}`);
                warmed.add(key);
                return true;
            },
            typedSurfaces,
            (surface) => knownTypedVariants(surface, draws, meshes),
        );
        precompile("kitchen-part-count", () => {
            events.push("part");
            publishPartDraws(buffer, surfaces.size, surfaces.size * meshes.size, registries);
            return true;
        });
        await precompileAll();

        expect(events[0]).toBe("part");
        expect(warmed).toContain(`${surfaceName}#${warmVariant}`);

        mesh(lateMesh, lateVariant);
        publishPartDraws(buffer, surfaces.size, surfaces.size * meshes.size, registries);
        expect(warmed).not.toContain(`${surfaceName}#${lateVariant}`);
        expect(knownTypedVariants(typedSurfaces.get(surfaceName)!, draws, meshes)).toContain(
            lateVariant,
        );
    } finally {
        Object.assign(Compute, saved);
    }
});

describe('typedSurfaceWgsl(unlit) vs surfaceCode(unlit, "color") — the second template extension (c-3)', () => {
    const rawUnlitSurface: Surface = {
        name: "unlit",
        bindings: {
            eids: { type: "storage", element: "u32" },
            transforms: { type: "storage", element: "Xform" },
            color: { type: "storage", element: "u32" },
        },
        fs: /* wgsl */ `col = vec4<f32>(unpackLdrColor(color[eid]).rgb, 1.0);`,
    };
    const typedUnlitLayout = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
        color: { type: "storage", element: d.u32 },
    });
    const typedUnlitFs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(unpackLdrColor(typedUnlitLayout.$.color[ctx.eid]).xyz, 1);
    });
    const typedUnlitSurface = { name: "unlit", layout: typedUnlitLayout, fs: typedUnlitFs };

    const rawWgsl = surfaceCode(rawUnlitSurface, "color", 0);
    const typedWgsl = typedSurfaceWgsl(typedUnlitSurface);

    test("the typed fs reads unpackLdrColor(color[eid]).xyz verbatim, no lighting call", () => {
        expect(rawWgsl).toContain("unpackLdrColor(color[eid]).rgb");
        expect(typedWgsl).toContain("unpackLdrColor(color[ctx.eid]).xyz");
        expect(typedWgsl).not.toContain("litPbr(");
    });

    test("both apply the standard instance transform (eids/transforms declared → instanced)", () => {
        expect(rawWgsl).toContain("eid = eids[iid];");
        expect(typedWgsl).toContain("eid = eids[iid];");
    });

    test("idiv audit: no division in either newly-authored body", () => {
        noIntegerDivision(body(typedWgsl, "fn unlitVs("));
        noIntegerDivision(body(typedWgsl, "fn unlitFs("));
    });
});

describe("compileTypedVariant — the transparent twin (c-3): shares typedColorVs/typedColorFs verbatim", () => {
    test("blend has no effect on the emitted vs/fs text — only the pipeline's blend/depth state differs, proven device-side by `bun bench`", () => {
        const l = layout({ items: { type: "storage", element: d.f32 } });
        const fs = tgpu.fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        const opaque = typedSurfaceWgsl({ name: "blend-probe", layout: l, fs });
        const alpha = typedSurfaceWgsl({
            name: "blend-probe",
            layout: l,
            fs,
            blend: "alpha",
        });
        // `typedColorVs`/`typedColorFs` never read `surface.blend` — the field only steers
        // `compileTypedVariant`'s pipeline-creation branch (color vs. transparent target/depth state)
        expect(alpha).toBe(opaque);
    });
});

describe('typedSurfaceWgsl(vertex) vs surfaceCode(vertex, "color") — the varyings mechanism (c-3a), litColor crossing vs→fs', () => {
    const rawVertexSurface: Surface = {
        name: "vertex",
        bindings: litBindingsRaw,
        preamble: PbrPreambleRaw,
        interpolators: { litColor: "vec3<f32>" },
        vs: /* wgsl */ `litColor = litPbr(matOf(eid), normalize(worldNormal), world.xyz) + emissiveOf(eid);`,
        fs: /* wgsl */ `col = vec4<f32>(litColor, 1.0);`,
    };

    const typedVertexVaryings = { litColor: d.vec3f };
    const typedVertexLayout = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
        color: { type: "storage", element: d.u32 },
        material: { type: "storage", element: d.vec2u },
    });
    const VertexPatch = vsPatchSchema(typedVertexVaryings);
    const typedVertexVs = tgpu.fn(
        [VsIn],
        VertexPatch,
    )((vsIn) => {
        "use gpu";
        const m = typedVertexLayout.$.material[vsIn.eid];
        const mr = std.unpack2x16float(m.x);
        const eo = std.unpack2x16float(m.y);
        const albedo = unpackLdrColor(typedVertexLayout.$.color[vsIn.eid]).xyz;
        const pbr = Pbr({
            albedo,
            metallic: mr.x,
            roughness: mr.y,
            occlusion: eo.y,
            dielectric: 0,
        });
        const emissive = std.mul(albedo, eo.x);
        const litColor = std.add(
            litPbr(pbr, std.normalize(vsIn.worldNormal), vsIn.world.xyz),
            emissive,
        );
        return VertexPatch({
            world: vsIn.world,
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(0),
            litColor,
        });
    });
    const typedVertexFs = tgpu.fn(
        [fsCtxSchema(typedVertexVaryings)],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(ctx.litColor, 1);
    });
    const typedVertexSurface = {
        name: "vertex",
        layout: typedVertexLayout,
        varyings: typedVertexVaryings,
        vs: typedVertexVs,
        fs: typedVertexFs,
    };

    const rawWgsl = surfaceCode(rawVertexSurface, "color", 0);
    const typedWgsl = typedSurfaceWgsl(typedVertexSurface);

    test("the typed vs runs litPbr once per vertex (not per fragment) and crosses litColor as a real varying", () => {
        expect(rawWgsl).toContain(
            "litColor = litPbr(matOf(eid), normalize(worldNormal), world.xyz) + emissiveOf(eid);",
        );
        expect(typedWgsl).toContain("litPbr(pbr, normalize(vsIn.worldNormal), vsIn.world.xyz)");
        // litColor rides the interstage struct as a real @location, not a zero-fill or a base fsCtxSchema field
        expect(typedWgsl).toMatch(/@location\(\d+\) litColor: vec3f/);
    });

    test("the typed fs writes vec4(ctx.litColor, 1) verbatim, no re-derivation of the lighting math", () => {
        expect(rawWgsl).toContain("col = vec4<f32>(litColor, 1.0);");
        expect(typedWgsl).toMatch(/vec4f\(ctx\.litColor, 1f?\)/);
        expect(typedWgsl).not.toContain("litPbr(pbr, ctx.worldNormal");
    });

    test("the vs copier's result binds to a local before crossing the entry boundary (the mechanism-lock law)", () => {
        expect(typedWgsl).toContain("let r = vertexCopier(vidx, iid);");
        expect(typedWgsl).toContain(
            "return vertexVs_Output(r.pos, r.worldNormal, r.eid, r.world, r.uv, r.localPos, r.litColor);",
        );
    });

    test("the fs entry crosses litColor through a fixed internal slot (v0), positionally into the real ctx field", () => {
        expect(typedWgsl).toContain("@location(5) v0: vec3f,");
        expect(typedWgsl).toContain("let col = vertexFsCopier(pos, _arg_0.worldNormal");
        expect(typedWgsl).toContain("_arg_0.v0");
        expect(typedWgsl).toMatch(/Ctx\(eid, world, worldNormal, uv, localPos, v0\)/);
    });

    test("idiv audit: no division in the copier bodies", () => {
        noIntegerDivision(body(typedWgsl, "fn vertexCopier("));
        noIntegerDivision(body(typedWgsl, "fn vertexFsCopier("));
    });

    test("the copier applies the vs chunk's world/worldNormal override before building out.pos/out.world/out.worldNormal — mirroring typedColorVs's order, not the pre-patch locals", () => {
        const copierBody = body(typedWgsl, "fn vertexCopier(");
        // `vs` resolves to the surface's own `$uses`-bound function name (`typedVertexVs` here), not the
        // literal WGSL-template identifier `vs` — match the call by its `VsIn(` argument, not the callee name.
        const patchMatch = copierBody.match(/let patched = \w+\(VsIn\(/);
        const patchIdx = patchMatch ? (patchMatch.index ?? -1) : -1;
        const worldIdx = copierBody.indexOf("world = patched.world;");
        const normalIdx = copierBody.indexOf("worldNormal = patched.worldNormal;");
        const posIdx = copierBody.indexOf("out.pos = view.viewProj * world;");
        const outWorldIdx = copierBody.indexOf("out.world = world.xyz;");
        expect(patchIdx).toBeGreaterThan(-1);
        expect(worldIdx).toBeGreaterThan(patchIdx);
        expect(normalIdx).toBeGreaterThan(patchIdx);
        expect(posIdx).toBeGreaterThan(worldIdx);
        expect(outWorldIdx).toBeGreaterThan(worldIdx);
    });
});

// The 1–4 varying arity dispatch (4a-ii-d-1). The vs side was already N-general (the copier templates its
// `out.<key> = patched.<key>` lines over `Object.keys`, locations over `VARYING_BASE + i`); the bound was
// the FRAGMENT entry's, whose transpiled body must statically name `input.v<i>`. These fixtures declare
// mixed-width varyings on purpose — the copier's parameter list is per-type templated, so a same-type set
// would not prove the templating.
describe("typedVaryingFs — the 1-to-4 varying arity dispatch (gpu.md rule 9's custom interpolator budget)", () => {
    const arityLayout = layout({ items: { type: "storage", element: d.f32 } });

    // two: vec4f + vec2f — the lines consumer's locked authoring shape (`lineRgba` + `edge`)
    const v2 = { a: d.vec4f, b: d.vec2f };
    const P2 = vsPatchSchema(v2);
    const vs2 = tgpu.fn(
        [VsIn],
        P2,
    )((vsIn) => {
        "use gpu";
        return P2({
            world: vsIn.world,
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(0),
            a: d.vec4f(vsIn.localPos, 1),
            b: vsIn.uv,
        });
    });
    const fs2 = tgpu.fn(
        [fsCtxSchema(v2)],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(ctx.a.xyz, ctx.b.x);
    });

    const v3 = { a: d.vec4f, b: d.vec2f, c: d.f32 };
    const P3 = vsPatchSchema(v3);
    const vs3 = tgpu.fn(
        [VsIn],
        P3,
    )((vsIn) => {
        "use gpu";
        return P3({
            world: vsIn.world,
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(0),
            a: d.vec4f(vsIn.localPos, 1),
            b: vsIn.uv,
            c: vsIn.localPos.x,
        });
    });
    const fs3 = tgpu.fn(
        [fsCtxSchema(v3)],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(ctx.a.xyz, ctx.b.x + ctx.c);
    });

    const v4 = { a: d.vec4f, b: d.vec2f, c: d.f32, e: d.vec3f };
    const P4 = vsPatchSchema(v4);
    const vs4 = tgpu.fn(
        [VsIn],
        P4,
    )((vsIn) => {
        "use gpu";
        return P4({
            world: vsIn.world,
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(0),
            a: d.vec4f(vsIn.localPos, 1),
            b: vsIn.uv,
            c: vsIn.localPos.x,
            e: vsIn.localNormal,
        });
    });
    const fs4 = tgpu.fn(
        [fsCtxSchema(v4)],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(std.add(ctx.a.xyz, ctx.e), ctx.b.x + ctx.c);
    });

    // resolved inside each test, not at describe scope: a regressed arity bound throws, and a throw at
    // describe scope reads as an unhandled error rather than a named failing test
    const wgsl2 = () =>
        typedSurfaceWgsl({ name: "arity2", layout: arityLayout, varyings: v2, vs: vs2, fs: fs2 });
    const wgsl3 = () =>
        typedSurfaceWgsl({ name: "arity3", layout: arityLayout, varyings: v3, vs: vs3, fs: fs3 });
    const wgsl4 = () =>
        typedSurfaceWgsl({ name: "arity4", layout: arityLayout, varyings: v4, vs: vs4, fs: fs4 });

    test("every varying pins to its own slot from VARYING_BASE up, on BOTH sides — the authored name on the vs, the fixed v<i> on the fs", () => {
        for (const [wgsl, types] of [
            [wgsl2(), ["vec4f", "vec2f"]],
            [wgsl3(), ["vec4f", "vec2f", "f32"]],
            [wgsl4(), ["vec4f", "vec2f", "f32", "vec3f"]],
        ] as const) {
            const names = ["a", "b", "c", "e"];
            types.forEach((type, i) => {
                expect(wgsl).toContain(`@location(${5 + i}) ${names[i]}: ${type}`);
                expect(wgsl).toContain(`@location(${5 + i}) v${i}: ${type}`);
            });
            // and nothing past the declared count — the next slot is free
            expect(wgsl).not.toContain(`@location(${5 + types.length}) `);
        }
    });

    test("the fs entry passes every slot positionally, and the copier lands them in the ctx's real fields", () => {
        expect(wgsl2()).toContain("_arg_0.localPos, _arg_0.v0, _arg_0.v1)");
        expect(wgsl2()).toContain("Ctx(eid, world, worldNormal, uv, localPos, v0, v1)");
        expect(wgsl3()).toContain("_arg_0.localPos, _arg_0.v0, _arg_0.v1, _arg_0.v2)");
        expect(wgsl3()).toContain("Ctx(eid, world, worldNormal, uv, localPos, v0, v1, v2)");
        expect(wgsl4()).toContain("_arg_0.localPos, _arg_0.v0, _arg_0.v1, _arg_0.v2, _arg_0.v3)");
        expect(wgsl4()).toContain("Ctx(eid, world, worldNormal, uv, localPos, v0, v1, v2, v3)");
    });

    test("the copier declares each slot at its authored type — the parameter list is per-type templated, not a fixed vec3f superset", () => {
        expect(wgsl2()).toContain("localPos: vec3f, v0: vec4f, v1: vec2f) -> vec4f");
        expect(wgsl3()).toContain("localPos: vec3f, v0: vec4f, v1: vec2f, v2: f32) -> vec4f");
        expect(wgsl4()).toContain(
            "localPos: vec3f, v0: vec4f, v1: vec2f, v2: f32, v3: vec3f) -> vec4f",
        );
    });

    test("the vs copier assigns every authored varying from the patch, in declaration order", () => {
        const copier4 = body(wgsl4(), "fn arity4Copier(");
        const order = ["out.a = patched.a;", "out.b = patched.b;", "out.c = patched.c;"];
        let at = -1;
        for (const line of [...order, "out.e = patched.e;"]) {
            const idx = copier4.indexOf(line);
            expect(idx).toBeGreaterThan(at);
            at = idx;
        }
    });

    test("the forcing touch survives every arm — each arity's fs still declares group 1's shadow bindings (gpu.md group-count compatibility, the recurring gap per commit 564e3913)", () => {
        for (const wgsl of [wgsl2(), wgsl3(), wgsl4()]) {
            expect(wgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var pointAtlas/);
            expect(wgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var shadowSamp/);
        }
    });

    test("a fifth varying throws — gpu.md rule 9's hard budget, loud rather than a silent slot overflow", () => {
        const v5 = { a: d.vec4f, b: d.vec2f, c: d.f32, e: d.vec3f, f: d.f32 };
        const P5 = vsPatchSchema(v5);
        const vs5 = tgpu.fn(
            [VsIn],
            P5,
        )((vsIn) => {
            "use gpu";
            return P5({
                world: vsIn.world,
                worldNormal: vsIn.worldNormal,
                clip: d.vec4f(0),
                a: d.vec4f(0),
                b: d.vec2f(0),
                c: 0,
                e: d.vec3f(0),
                f: 0,
            });
        });
        const fs5 = tgpu.fn(
            [fsCtxSchema(v5)],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            return d.vec4f(ctx.a.xyz, ctx.f);
        });
        expect(() =>
            typedSurfaceWgsl({
                name: "arity5",
                layout: arityLayout,
                varyings: v5,
                vs: vs5,
                fs: fs5,
            }),
        ).toThrow(/declares 5 varyings/);
    });
});

// `screen` surfaces (4a-ii-d-1): the raw path's `out.clip = clipPos` (codegen.ts's `surfaceCode`) — the
// vs chunk projects its own clip-space geometry, and back-face culling is off because those quads have no
// consistent winding. Both typed vs shapes (the shared TGSL body and the per-surface WGSL copier) honor it.
describe("screen surfaces — the vs chunk's patch.clip IS the clip position", () => {
    const screenLayout = layout({ items: { type: "storage", element: d.f32 } });
    const Patch = vsPatchSchema();
    const screenVs = tgpu.fn(
        [VsIn],
        Patch,
    )((vsIn) => {
        "use gpu";
        // the no-varyings default's `[x: string]: never` index signature — the same escape the
        // `specialize-vs-probe` fixture above uses
        return Patch({
            world: vsIn.world,
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(vsIn.localPos, 1),
        } as any);
    });
    const screenFs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    });
    const screenSurface = {
        name: "screenProbe",
        layout: screenLayout,
        vs: screenVs,
        fs: screenFs,
        screen: true,
    };
    // the same chunks minus `screen` — the control that proves the difference is the flag, not the fixture
    const worldSurface = {
        name: "worldProbe",
        layout: screenLayout,
        vs: screenVs,
        fs: screenFs,
    };

    test("the color vs returns the patch's clip verbatim and never projects — the world-space twin does the opposite", () => {
        const screenWgsl = typedSurfaceWgsl(screenSurface);
        expect(body(screenWgsl, "fn screenProbeVs(")).toContain("clip = patched.clip;");
        expect(screenWgsl).not.toContain("view.viewProj * world");
        // and the projection matrix isn't even declared — a screen module reads no view
        expect(screenWgsl).not.toContain("var<uniform> view: View;");

        const worldWgsl = typedSurfaceWgsl(worldSurface);
        expect(body(worldWgsl, "fn worldProbeVs(")).toContain("clip = (view.viewProj * world);");
        expect(worldWgsl).not.toContain("clip = patched.clip;");
    });

    test("the prepass + tag vs entries honor it too — the raw path splices one `out.clip` for every pass", () => {
        const prepass = typedPrepassWgsl(screenSurface);
        for (const wgsl of [prepass[""], prepass.tag]) {
            expect(wgsl).toContain("clip = patched.clip;");
            expect(wgsl).not.toContain("view.viewProj * world");
        }
    });

    test("the varyings copier honors it — the per-surface raw body, the other of the two vs shapes", () => {
        const varyings = { tint: d.vec3f };
        const VaryPatch = vsPatchSchema(varyings);
        const varyVs = tgpu.fn(
            [VsIn],
            VaryPatch,
        )((vsIn) => {
            "use gpu";
            return VaryPatch({
                world: vsIn.world,
                worldNormal: vsIn.worldNormal,
                clip: d.vec4f(vsIn.localPos, 1),
                tint: vsIn.localNormal,
            });
        });
        const varyFs = tgpu.fn(
            [fsCtxSchema(varyings)],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            return d.vec4f(ctx.tint, 1);
        });
        const wgsl = typedSurfaceWgsl({
            name: "screenVary",
            layout: screenLayout,
            varyings,
            vs: varyVs,
            fs: varyFs,
            screen: true,
        });
        expect(body(wgsl, "fn screenVaryCopier(")).toContain("out.pos = patched.clip;");
        expect(wgsl).not.toContain("view.viewProj * world");
    });

    test("a screen surface rasterizes un-culled; every world-space surface keeps back-face culling", () => {
        // one source of truth for the color, single-sample, and prepass pipelines — `compileVariant`'s
        // own law, and load-bearing for lines (its clip-space quads flip winding with segment direction)
        expect(surfacePrimitive(true).cullMode).toBe("none");
        expect(surfacePrimitive(false).cullMode).toBe("back");
        expect(surfacePrimitive(undefined).cullMode).toBe("back");
        expect(surfacePrimitive(true).frontFace).toBe("ccw");
        expect(surfacePrimitive(true).topology).toBe("triangle-list");
    });
});

describe('typedPrepassWgsl(unlit) vs surfaceCode(unlit, "prepass") — the typed prepass pipelines (4a-ii-c-3a-2)', () => {
    const rawUnlitSurface: Surface = {
        name: "unlit",
        bindings: {
            eids: { type: "storage", element: "u32" },
            transforms: { type: "storage", element: "Xform" },
            color: { type: "storage", element: "u32" },
        },
        fs: /* wgsl */ `col = vec4<f32>(unpackLdrColor(color[eid]).rgb, 1.0);`,
    };
    const typedUnlitLayout = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
        color: { type: "storage", element: d.u32 },
    });
    const typedUnlitFs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(unpackLdrColor(typedUnlitLayout.$.color[ctx.eid]).xyz, 1);
    });
    const typedUnlitSurface = { name: "unlit", layout: typedUnlitLayout, fs: typedUnlitFs };

    const rawDepthWgsl = surfaceCode(rawUnlitSurface, "prepass", 0);
    const typedWgsl = typedPrepassWgsl(typedUnlitSurface);

    test('the raw empty-lane-set pipeline (no "clip") has no fragment stage at all — a plain opaque surface never discards in the depth-only pass', () => {
        expect(rawDepthWgsl).not.toContain("fn fsPrepass(");
    });

    test("the typed depth-only pipeline is vertex-only too — no fragment fn in its emitted text", () => {
        expect(typedWgsl[""]).not.toMatch(/@fragment/);
    });

    test("both decode position ALONE from the 8 B stream — the vs body calls decodePos but never octDecodeNormal (normal defaults +Z; octDecodeNormal still appears module-wide as spotFactor's general-purpose definition, so the check is scoped to the vs body, not full-text)", () => {
        const rawVs = body(rawDepthWgsl, "fn vs(");
        expect(rawVs).toContain("decodePos(v.x, v.y, meshQuant[meshIdOf(v.y)])");
        expect(rawVs).not.toContain("octDecodeNormal(");
        const typedVs = body(typedWgsl[""], "fn unlitPrepassVs(");
        expect(typedVs).toContain("decodePos((*v).x, (*v).y, (*mq))");
        expect(typedVs).not.toContain("octDecodeNormal(");
    });

    test("both apply the standard instance transform (eids/transforms declared → instanced)", () => {
        expect(rawDepthWgsl).toContain("eid = eids[iid];");
        expect(rawDepthWgsl).toContain("xformPoint(xf, world.xyz)");
        expect(rawDepthWgsl).toContain("xformNormal(xf, worldNormal)");
        // typed copies the depth layout's transform into the VsIn-carried value
        expect(typedWgsl[""]).toContain("eid = eids[iid];");
        expect(typedWgsl[""]).toContain("xformPoint(xform, world.xyz)");
        expect(typedWgsl[""]).toContain("xformNormal(xform, worldNormal)");
    });

    test("the raw path's VertexOut always carries worldNormal/eid/world (one struct shared with the color pass's own entries) even on the unused empty-lane pipeline; the typed depth-only pipeline compiles its own minimal struct — a real, disclosed pruning deviation, not an established one", () => {
        expect(rawDepthWgsl).toContain("out.worldNormal = normalize(worldNormal);");
        expect(rawDepthWgsl).toContain("out.eid = eid;");
        const interstage = typedWgsl[""].slice(
            typedWgsl[""].indexOf("_Output {"),
            typedWgsl[""].indexOf("}", typedWgsl[""].indexOf("_Output {")),
        );
        expect((interstage.match(/@location\(/g) ?? []).length).toBe(0);
        expect(interstage).not.toContain("worldNormal");
        expect(interstage).not.toContain("eid");
    });

    test('the raw tag-lane fs writes a mutable "tag" local defaulting to eid, then returns it as a bare u32; the typed tag fs returns vec4u(eid,0,0,0) — typegpu constrains every fragment color output to a vec4 family type (probed live: a bare `d.u32` fails FragmentOutConstrained before the body resolves), which WebGPU accepts against the single-channel r32uint target (excess output components are dropped per spec) — a real, disclosed WGSL-shape deviation', () => {
        expect(rawDepthWgsl).toContain("fn fsPrepassTag(fin: VertexOut) -> @location(0) u32 {");
        expect(rawDepthWgsl).toContain("var tag: u32 = eid;");
        expect(rawDepthWgsl).toContain("return tag;");
        expect(typedWgsl.tag).toMatch(
            /@fragment fn unlitPrepassTagFs\(.*\) -> @location\(0\) vec4u/,
        );
        expect(typedWgsl.tag).toContain("return vec4u(_arg_0.eid, 0u, 0u, 0u);");
    });

    test("idiv audit: no division in the newly-authored prepass vs/fs bodies", () => {
        noIntegerDivision(body(typedWgsl[""], "fn unlitPrepassVs("));
        noIntegerDivision(body(typedWgsl.tag, "fn unlitPrepassTagVs("));
        noIntegerDivision(body(typedWgsl.tag, "fn unlitPrepassTagFs("));
    });
});

describe("typed prepass — the `vs`-chunk / non-instanced branches (review-caught: red-first)", () => {
    // `codegen.ts`'s raw prepass module keeps `localNormal` at `vec3<f32>(0.0, 0.0, 1.0)` for the vs's
    // whole life (`INSTANCE_VS` mutates only `world`/`worldNormal`), so a `vs` chunk reading `localNormal`
    // must see that pinned default, never the transformed normal — the `unlit` differential above declares
    // no `vs` chunk, so it can't exercise this branch on its own.
    const rawSurfaceWithVs: Surface = {
        name: "prepassVsProbe",
        bindings: {
            eids: { type: "storage", element: "u32" },
            transforms: { type: "storage", element: "Xform" },
        },
        vs: /* wgsl */ `world = world + vec4<f32>(localNormal, 0.0);`,
        fs: /* wgsl */ `col = vec4<f32>(1.0);`,
    };
    const typedInstancedLayout = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
    });
    const patchSchema = vsPatchSchema();
    const typedVsProbe = tgpu.fn(
        [VsIn],
        patchSchema,
    )((vsIn) => {
        "use gpu";
        return patchSchema({
            world: std.add(vsIn.world, d.vec4f(vsIn.localNormal, 0)),
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(0),
        } as any);
    });
    const typedFsProbe = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    });
    const typedSurfaceWithVs = {
        name: "prepassVsProbe",
        layout: typedInstancedLayout,
        vs: typedVsProbe,
        fs: typedFsProbe,
    };

    test("the raw prepass module never mutates localNormal — INSTANCE_VS + a vs chunk both read the pinned (0,0,1) default", () => {
        const rawWgsl = surfaceCode(rawSurfaceWithVs, "prepass", 0);
        const rawVsBody = body(rawWgsl, "fn vs(");
        expect(rawVsBody).toContain("var localNormal = vec3<f32>(0.0, 0.0, 1.0);");
        // localNormal is read (by the spliced vs chunk) but never assigned again anywhere in the body
        expect((rawVsBody.match(/localNormal\s*=/g) ?? []).length).toBe(1); // only its own declaration
    });

    test("the typed depth-only + tag prepass vs entries pass the pinned localNormal (not the transformed worldNormal) into VsIn — the fixed bug", () => {
        const wgsl = typedPrepassWgsl(typedSurfaceWithVs);
        for (const key of ["", "tag"] as const) {
            const vsBody = body(
                wgsl[key],
                key === "" ? "fn prepassVsProbePrepassVs(" : "fn prepassVsProbePrepassTagVs(",
            );
            expect(vsBody).toContain("let localNormal = vec3f(0, 0, 1);");
            // localNormal is declared once and never reassigned (worldNormal is, by xformNormal)
            expect((vsBody.match(/\blocalNormal\b/g) ?? []).length).toBeGreaterThanOrEqual(2); // decl + VsIn use
            expect(vsBody).toContain(
                "VsIn(localPos, localNormal, uv, vidx, eid, iid, xform, world, worldNormal)",
            );
        }
    });

    test("a non-instanced typed surface's tag entry defaults eid to TAG_NONE, matching COLOR_LANES' raw default", () => {
        const nonInstancedLayout = layout({});
        const surface = {
            name: `tagNoneProbe${Math.random()}`,
            layout: nonInstancedLayout,
            fs: typedFsProbe,
        };
        const wgsl = typedPrepassWgsl(surface);
        expect(wgsl.tag).toMatch(/eid = 4294967295u;/);
    });
});

describe("typedShadowWgsl(unlit) vs pointShadowCode(unlit) — the typed point/cascade shadow-atlas pipelines (4a-ii-c-3a-3)", () => {
    const rawUnlitSurface: Surface = {
        name: "unlit",
        bindings: {
            eids: { type: "storage", element: "u32" },
            transforms: { type: "storage", element: "Xform" },
            color: { type: "storage", element: "u32" },
        },
        fs: /* wgsl */ `col = vec4<f32>(unpackLdrColor(color[eid]).rgb, 1.0);`,
    };
    const typedUnlitLayout = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
        color: { type: "storage", element: d.u32 },
    });
    const typedUnlitFs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(unpackLdrColor(typedUnlitLayout.$.color[ctx.eid]).xyz, 1);
    });
    const typedUnlitSurface = { name: "unlit", layout: typedUnlitLayout, fs: typedUnlitFs };

    const rawPointWgsl = pointShadowCode(rawUnlitSurface, 0);
    const rawCascadeWgsl = pointShadowCode(rawUnlitSurface, 0, true);
    const typedWgsl = typedShadowWgsl(typedUnlitSurface);

    test("the typed point/cascade layouts carry schema-sized FaceVPs, ComboMeta, and TileRects uniforms", () => {
        const pointSlots = 6 * pointCasters();
        for (const [wgsl, slots] of [
            [typedWgsl.point, pointSlots],
            [typedWgsl.cascade, MAX_CASCADES],
        ] as const) {
            expect(wgsl).toContain(`m: array<mat4x4f, ${slots}>`);
            expect(wgsl).toContain(`m: array<vec4u, ${slots}>`);
            expect(wgsl).toContain(`rects: array<vec4f, ${slots}>`);
            expect(wgsl).toContain("@group(1) @binding(0) var<uniform> faceVP: FaceVPs;");
            expect(wgsl).toContain("@group(1) @binding(1) var<uniform> comboMeta: ComboMeta;");
            expect(wgsl).toContain("@group(1) @binding(2) var<uniform> tileRects: TileRects;");
        }
    });

    test("both read the re-gathered (combo << COMBO_SHIFT) | eid packed instance at the eids lane, same mask/shift", () => {
        expect(rawPointWgsl).toContain("let packed = eids[iid];");
        expect(rawPointWgsl).toContain(`var eid: u32 = packed & ${EID_MASK}u;`);
        expect(rawPointWgsl).toContain(`let combo = packed >> ${COMBO_SHIFT}u;`);
        expect(typedWgsl.point).toContain("let packed = eids[iid];");
        expect(typedWgsl.point).toContain(`let eid = (packed & ${EID_MASK}u);`);
        expect(typedWgsl.point).toContain(`let combo = (packed >> ${COMBO_SHIFT}u);`);
        // the cascade VS reads the identical packed split — only the rect index formula + atlas differ
        expect(typedWgsl.cascade).toContain(`let combo = (packed >> ${COMBO_SHIFT}u);`);
    });

    test("point indexes tileRects by caster·6+face (m.x*6u+m.y); cascade indexes by the cascade index alone (m.x) — pointShadowCode's rectExpr split", () => {
        expect(rawPointWgsl).toContain("tileRects.rects[m.x * 6u + m.y]");
        expect(rawCascadeWgsl).toContain("tileRects.rects[m.x];");
        expect(typedWgsl.point).toContain("tileRects.rects[(((*m).x * 6u) + (*m).y)]");
        expect(typedWgsl.cascade).toContain("tileRects.rects[(*m).x]");
    });

    test("both project by the combo's tile-folded viewProj and scale tileBox by their OWN atlas's pixel size (point ≠ cascade)", () => {
        const pointAtlas = pointAtlasSize();
        const cascadeAtlas = cascadeAtlasSize(sunResolution(), sunCascades());
        expect(rawPointWgsl).toContain("out.clip = faceVP.m[combo] * world;");
        expect(rawPointWgsl).toContain(
            `out.tileBox = vec4<f32>(rect.xy * ${pointAtlas}.0, rect.z * ${pointAtlas}.0, 0.0);`,
        );
        expect(rawCascadeWgsl).toContain(
            `out.tileBox = vec4<f32>(rect.xy * ${cascadeAtlas}.0, rect.z * ${cascadeAtlas}.0, 0.0);`,
        );
        expect(typedWgsl.point).toContain("let clip = (faceVP.m[combo] * world);");
        expect(typedWgsl.point).toContain(
            `let tileBox = vec4f((${pointAtlas}f * (*rect).xy), ((*rect).z * ${pointAtlas}f), 0f);`,
        );
        expect(typedWgsl.cascade).toContain(
            `let tileBox = vec4f((${cascadeAtlas}f * (*rect).xy), ((*rect).z * ${cascadeAtlas}f), 0f);`,
        );
        expect(pointAtlas).not.toBe(cascadeAtlas); // else the two atlas-scale assertions above prove nothing
    });

    test("the fs discards outside the tile bounds and writes no color output (Void) — the raw path's depth-only `targets: []` shape; ONE shared fs instance serves both atlases (their fsPoint text is identical past the tile-box compare, codegen.ts)", () => {
        expect(rawPointWgsl).toContain("fn fsPoint(fin: VertexOut) {");
        expect(rawPointWgsl).toContain("discard;");
        expect(rawCascadeWgsl).toContain("fn fsPoint(fin: VertexOut) {");
        const rawFsBody = (raw: string) => body(raw, "fn fsPoint(");
        expect(rawFsBody(rawPointWgsl)).toBe(rawFsBody(rawCascadeWgsl));
        expect(typedWgsl.point).toMatch(/@fragment fn shadowAtlasFs\(/);
        expect(typedWgsl.point).toContain("discard;");
        const typedFsBody = (typed: string) => body(typed, "fn shadowAtlasFs(");
        expect(typedFsBody(typedWgsl.point)).toBe(typedFsBody(typedWgsl.cascade));
    });

    test("idiv audit: no division in the newly-authored shadow-atlas vs/fs bodies", () => {
        noIntegerDivision(body(typedWgsl.point, "fn unlitPointVs("));
        noIntegerDivision(body(typedWgsl.cascade, "fn unlitCascadeVs("));
        noIntegerDivision(body(typedWgsl.point, "fn shadowAtlasFs("));
    });
});

describe("typed shadow atlas — the `vs`-chunk override (mirrors typedPrepassVs's pinned localNormal law)", () => {
    // point/cascade pull position-only (localNormal defaults +Z, uv 0 — codegen.ts's pointShadowCode), so a
    // surface's own vs chunk must see that pinned default, never a transformed value — same law
    // typedPrepassVs's review-caught fix pins; this surface's vs chunk reads localNormal to prove it.
    const rawSurfaceWithVs: Surface = {
        name: "shadowVsProbe",
        bindings: {
            eids: { type: "storage", element: "u32" },
            transforms: { type: "storage", element: "Xform" },
        },
        vs: /* wgsl */ `world = world + vec4<f32>(localNormal, 0.0);`,
        fs: /* wgsl */ `col = vec4<f32>(1.0);`,
    };
    const typedInstancedLayout = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
    });
    const patchSchema = vsPatchSchema();
    const typedVsProbe = tgpu.fn(
        [VsIn],
        patchSchema,
    )((vsIn) => {
        "use gpu";
        return patchSchema({
            world: std.add(vsIn.world, d.vec4f(vsIn.localNormal, 0)),
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(0),
        } as any);
    });
    const typedFsProbe = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    });
    const typedSurfaceWithVs = {
        name: "shadowVsProbe",
        layout: typedInstancedLayout,
        vs: typedVsProbe,
        fs: typedFsProbe,
    };

    test("the raw point/cascade VS splices the surface's own vs chunk, which can override world before projection", () => {
        const rawWgsl = pointShadowCode(rawSurfaceWithVs, 0);
        const vsBody = body(rawWgsl, "fn vs(");
        expect(vsBody).toContain("{ world = world + vec4<f32>(localNormal, 0.0); }");
        expect(vsBody.indexOf("localNormal, 0.0); }")).toBeLessThan(
            vsBody.indexOf("out.clip = faceVP.m[combo] * world;"),
        );
    });

    test("the typed point/cascade VS calls the surface's vs chunk, applies its world override before projecting, and the vs chunk sees the pinned (0,0,1) localNormal default", () => {
        const wgsl = typedShadowWgsl(typedSurfaceWithVs);
        for (const [key, name] of [
            ["point", "shadowVsProbePointVs"],
            ["cascade", "shadowVsProbeCascadeVs"],
        ] as const) {
            const vsBody = body(wgsl[key], `fn ${name}(`);
            expect(vsBody).toContain("let localNormal = vec3f(0, 0, 1);");
            expect(vsBody).toContain(
                "VsIn(localPos, localNormal, uv, vidx, eid, iid, xform, world, worldNormal)",
            );
            expect(vsBody).toMatch(/world = patch\w*\.world;/);
            const patchIdx = vsBody.search(/world = patch\w*\.world;/);
            expect(patchIdx).toBeGreaterThan(0);
            expect(patchIdx).toBeLessThan(vsBody.indexOf("let clip"));
        }
    });
});

// The typed `Backgrounds` contract's pipeline builder (4a-ii-c-3a-4, the Backgrounds bindings lock):
// `typedBgVs`/`typedBgFs` (device-bound only via `bun bench` warming a registered `TypedBackgrounds` entry
// — none is registered in production code this stage, so `bun bench` does not reach a typed bg compile;
// this file is the device-free proof the WGSL is right) and their device-free resolve seam `typedBgWgsl`.
// Pins the view-ray reconstruct + the fullscreen-triangle vs operand-for-operand against `backgroundCode`.

const rawBgProbe: Background = {
    name: "bg-probe",
    bindings: { tint: { type: "uniform", struct: "vec4<f32>" } },
    fs: /* wgsl */ `col = mix(tint.rgb, dir * 0.5 + 0.5, 0.5);`,
};

// the typed contract's `uniform` binding takes a WGSL struct schema (`Binding`'s `struct: AnyWgslStruct`),
// unlike the legacy contract's bare type-string — `Tint` is the one-field struct twin of the raw `vec4<f32>`
const Tint = d.struct({ value: d.vec4f }).$name("Tint");
const typedBgProbeLayout = bgLayout({ tint: { type: "uniform", struct: Tint } });
const typedBgProbeFs = tgpu.fn(
    [BgCtx],
    d.vec3f,
)((ctx) => {
    "use gpu";
    return std.mix(
        typedBgProbeLayout.$.tint.value.xyz,
        std.add(std.mul(ctx.dir, 0.5), d.vec3f(0.5)),
        0.5,
    );
});
const typedBgProbe: TypedBackground = {
    name: "bg-probe",
    layout: typedBgProbeLayout,
    fs: typedBgProbeFs,
};

describe("typedBgWgsl(bg-probe) vs backgroundCode(bg-probe) — the Backgrounds bindings lock differential (4a-ii-c-3a-4)", () => {
    const rawWgsl = backgroundCode(rawBgProbe);
    const typedWgsl = typedBgWgsl(typedBgProbe);

    test("both reconstruct the view ray operand-for-operand: uv, ndc, invViewProj mul, xyz/w divide, subtract eye, normalize", () => {
        expect(rawWgsl).toContain("let uv = fin.clip.xy / view.resolution;");
        expect(typedWgsl).toContain("let uv = (pos.xy / view.resolution);");
        expect(rawWgsl).toContain("let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0);");
        expect(typedWgsl).toContain("let ndc = vec3f(((uv.x * 2f) - 1f), (1f - (uv.y * 2f)), 0f);");
        expect(rawWgsl).toContain("let far = view.invViewProj * vec4<f32>(ndc, 1.0);");
        expect(typedWgsl).toContain("let far = (view.invViewProj * vec4f(ndc, 1f));");
        expect(rawWgsl).toContain("let dir = normalize(far.xyz / far.w - view.eye.xyz);");
        expect(typedWgsl).toContain("let dir = normalize(((far.xyz / far.w) - view.eye.xyz));");
    });

    test("both draw the fullscreen triangle from vertex_index alone, no vertex pull — same bit ops, same operand order", () => {
        expect(rawWgsl).toContain("let c = vec2<f32>(f32((vidx << 1u) & 2u), f32(vidx & 2u));");
        // a real, disclosed deviation: the typed vs writes `d.vec4f(c.x * 2 - 1, c.y * 2 - 1, 0, 1)`
        // (per-component) where the raw path does `c * 2.0 - 1.0` (vector-wide) — same values, same
        // operand order per component, no reassociation; typegpu emits the component form as authored
        const vsBody = body(typedWgsl, "fn bgVs(");
        expect(vsBody).toContain("let c = vec2f(f32(((vidx << 1u) & 2u)), f32((vidx & 2u)));");
        expect(vsBody).toContain("vec4f(((c.x * 2f) - 1f), ((c.y * 2f) - 1f), 0f, 1f)");
    });

    test("both declare the tint uniform binding at the surface/background group and read its color field", () => {
        expect(rawWgsl).toContain("var<uniform> tint: vec4<f32>;");
        expect(rawWgsl).toContain("tint.rgb");
        expect(typedWgsl).toContain("@group(2) @binding(0) var<uniform> tint: Tint;");
        expect(typedWgsl).toContain("tint.value.xyz");
    });

    test("group 1 (shadowLayoutTyped) is forced into scope though the background never shades with it — the forcedZero precedent, preserving compileBackground's group-count-compatibility reason", () => {
        expect(typedWgsl).toContain("@group(1) @binding(0) var shadowMap: texture_depth_2d;");
        expect(typedWgsl).toContain(
            "@group(1) @binding(4) var<uniform> pointShadows: PointCasters;",
        );
        expect(typedWgsl).toMatch(/let forcedZero = .*\* 0f\);/);
    });

    test("both write the opaque HDR fill vec4(col, 1) — the typed fs's col is the forcedZero-touched sum, a no-op addition", () => {
        expect(rawWgsl).toContain("return vec4<f32>(col, 1.0);");
        expect(typedWgsl).toContain("return vec4f((col + vec3f(forcedZero)), 1f);");
    });

    test("both call the same mix(tint, dir*0.5+0.5, 0.5) fs body, same operand order", () => {
        expect(rawWgsl).toContain("mix(tint.rgb, dir * 0.5 + 0.5, 0.5)");
        expect(typedWgsl).toContain("mix(tint.value.xyz, ((ctx.dir * 0.5f) + vec3f(0.5)), 0.5f)");
    });

    test("idiv audit: the newly-authored vs/fs entry bodies carry no division at all", () => {
        noIntegerDivision(body(typedWgsl, "fn bgVs("));
        noIntegerDivision(body(typedWgsl, "fn bgprobeFs("));
    });
});

describe("registerBackground — dual-accept discrimination (forward.ts's register() precedent)", () => {
    test("a legacy (string-contract) spec lands in the real `Backgrounds` registry, unchanged", () => {
        const name = `legacy-bg-${Math.random()}`;
        registerBackground({ name, fs: "col = vec3<f32>(1.0);" });
        expect(Backgrounds.get(name)).toBeDefined();
        expect(TypedBackgrounds.get(name)).toBeUndefined();
    });

    test("a typed spec lands in `TypedBackgrounds`, not the legacy `Backgrounds` registry", () => {
        const name = `typed-bg-${Math.random()}`;
        registerBackground({ name, layout: typedBgProbeLayout, fs: typedBgProbeFs });
        expect(TypedBackgrounds.get(name)).toBeDefined();
        expect(Backgrounds.get(name)).toBeUndefined();
    });

    test("a legacy spec carrying a stray `layout` key still lands in the legacy registry (fs's string shape is the guard)", () => {
        const name = `legacy-bg-stray-layout-${Math.random()}`;
        const legacyWithStrayLayout = {
            name,
            fs: "col = vec3<f32>(1.0);",
            layout: "some-incidental-string-field",
        } as unknown as Parameters<typeof registerBackground>[0];
        registerBackground(legacyWithStrayLayout);
        expect(Backgrounds.get(name)).toBeDefined();
        expect(TypedBackgrounds.get(name)).toBeUndefined();
    });
});

describe("draw wiring (4a-ii-c-3b) — the depth-pipeline receiver stub + the color pipeline's real receiver", () => {
    // a vs-chunk surface whose chunk reaches litPbr → pointShadowOf, the shape that forces the stub:
    // the raw path's prepass/shadow modules splice SHADOW_STUB_WGSL for exactly this reach
    const varyings = { litColor: d.vec3f };
    const lay = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
        color: { type: "storage", element: d.u32 },
    });
    const Patch = vsPatchSchema(varyings);
    const vs = tgpu.fn(
        [VsIn],
        Patch,
    )((vsIn) => {
        "use gpu";
        const albedo = unpackLdrColor(lay.$.color[vsIn.eid]).xyz;
        const pbr = Pbr({ albedo, metallic: 0, roughness: 1, occlusion: 1, dielectric: 0 });
        const litColor = litPbr(pbr, std.normalize(vsIn.worldNormal), vsIn.world.xyz);
        return Patch({
            world: vsIn.world,
            worldNormal: vsIn.worldNormal,
            clip: d.vec4f(0),
            litColor,
        });
    });
    const fs = tgpu.fn(
        [fsCtxSchema(varyings)],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(ctx.litColor, 1);
    });
    const surface = { name: "stubProbe", layout: lay, varyings, vs, fs };

    const colorWgsl = typedSurfaceWgsl(surface);
    const prepassWgsl = typedPrepassWgsl(surface);
    const shadowWgsl = typedShadowWgsl(surface);

    test("the color module emits the REAL receiver (samples the atlas) and declares its group-1 free names", () => {
        const receiver = body(colorWgsl, "fn pointShadowOf(");
        expect(receiver).toContain("textureSampleCompareLevel(pointAtlas");
        // typedVaryingFs's forcedZero fold is what declares them (invisible to the call-graph walk)
        expect(colorWgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var pointAtlas/);
        expect(colorWgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var shadowSamp/);
    });

    test("the prepass modules emit the STUB receiver — no atlas sampling, no undeclared free names", () => {
        for (const wgsl of [prepassWgsl[""], prepassWgsl.tag]) {
            const receiver = body(wgsl, "fn pointShadowOf(");
            expect(receiver).toContain("return 1f;");
            expect(receiver).not.toContain("textureSampleCompareLevel");
            expect(wgsl).not.toContain("pointAtlas");
            expect(wgsl).not.toContain("pointShadows");
        }
    });

    test("the point/cascade modules emit the STUB receiver — the real one would sample the atlas being written", () => {
        for (const wgsl of [shadowWgsl.point, shadowWgsl.cascade]) {
            const receiver = body(wgsl, "fn pointShadowOf(");
            expect(receiver).toContain("return 1f;");
            expect(receiver).not.toContain("textureSampleCompareLevel");
            expect(wgsl).not.toContain("pointAtlas");
        }
    });

    test("the prepass modules keep the group set dense — a group-1 touch, never a hole typegpu's group-indexed pipeline layout would emit", () => {
        for (const wgsl of [prepassWgsl[""], prepassWgsl.tag]) {
            expect(wgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var<uniform> tileRects/);
        }
    });

    test("the varying crosses at an explicit interstage slot — vs `litColor` and fs `v0` pinned to the same @location", () => {
        expect(colorWgsl).toContain("@location(5) litColor: vec3f");
        expect(colorWgsl).toContain("@location(5) v0: vec3f");
    });
});

describe("draw wiring (4a-ii-c-3b) — shadowLayoutTyped visibility mirrors the raw _shadowBgl", () => {
    test("sampler + point-shadow entries are vertex-visible (a per-vertex vs chunk reaches pointShadowOf); sun map + params stay fragment-only", () => {
        const entries = shadowLayoutTyped.entries as Record<string, { visibility?: string[] }>;
        expect(entries.shadowSamp.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.pointAtlas.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.pointShadows.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.tileRects.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.shadowMap.visibility).toEqual(["fragment"]);
        expect(entries.sunShadow.visibility).toEqual(["fragment"]);
    });
});

describe("draw wiring (4a-ii-c-3b) — the Backdrop typed id space", () => {
    const traits = (SearPlugin.traits as Record<string, unknown>).Backdrop as {
        parse: { name: (v: string) => number | undefined };
        format: { name: (v: number) => string | undefined };
    };
    const TypedBase = 0x80000000;

    test("a legacy background parses to its raw registry id; a typed-only one to its id offset by the typed base — and both format back", () => {
        Backgrounds.register({ name: "c3b-legacy-bg", fs: "col = vec3<f32>(dir.y);" });
        const bgFs = tgpu.fn(
            [BgCtx],
            d.vec3f,
        )((ctx) => {
            "use gpu";
            return d.vec3f(ctx.dir.y);
        });
        registerBackground({ name: "c3b-typed-bg", layout: bgLayout({}), fs: bgFs });
        const legacyId = traits.parse.name("c3b-legacy-bg")!;
        const typedId = traits.parse.name("c3b-typed-bg")!;
        expect(legacyId).toBe(Backgrounds.id("c3b-legacy-bg")!);
        expect(legacyId).toBeLessThan(TypedBase);
        expect(typedId).toBe(TypedBackgrounds.id("c3b-typed-bg")! + TypedBase);
        expect(traits.format.name(legacyId)).toBe("c3b-legacy-bg");
        expect(traits.format.name(typedId)).toBe("c3b-typed-bg");
        expect(traits.parse.name("c3b-not-registered")).toBeUndefined();
    });
});

describe("draw wiring (4a-ii-c-3b review) — typed group-state eviction", () => {
    test("clearGroups drops a cached typed entry (the re-gather realloc + rebuild eviction path, and the engine-group lifetime rides the entry)", () => {
        const entry = {
            engineCache: new Map([[0, {} as GPUBindGroup]]),
            resources: [],
        } as unknown as TypedGroupEntry;
        setTypedGroup("c3b-evict-probe", entry);
        expect(getTypedGroup("c3b-evict-probe")).toBe(entry);
        clearGroups();
        // the entry — and with it the per-entry engineCache holding the quant-keyed group-0 instances —
        // is unreachable after the clear; nothing module-scoped retains a churned quant buffer
        expect(getTypedGroup("c3b-evict-probe")).toBeUndefined();
    });
});

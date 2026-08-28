import { describe, expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { body, noIntegerDivision } from "../../../tests/wgsl";
import { Compute, type Plugin, Registry, State } from "../../engine";
import { precompile, precompileAll, requestGPU } from "../../engine/runtime";
import { unpackLdrColor, Xform } from "../../engine/utils/core";
import { publishPartDraws } from "../part/part";
import { RenderPlugin } from "../render";
import type { Draw, Mesh } from "../render/core";
import { DrawIndexedIndirect, Draws, Meshes } from "../render/core";
import { shadowLayout } from "./atlas";
import {
    type Background,
    Backgrounds,
    BgCtx,
    backgroundLayout as bgLayout,
    fsCtxSchema,
    surfaceLayout as layout,
    registerBackground,
    type Surface,
    VsIn,
    vsPatchSchema,
} from "./contract";
import { litPbr } from "./engine";
import { precompileVariants, SearPlugin } from "./forward";
import {
    backgroundWgsl,
    clearGroups,
    compileBackground,
    compileVariant,
    getBackground,
    getCompiledSurface,
    getGroup,
    knownVariants,
    prepassWgsl,
    type SurfaceGroupEntry,
    setGroup,
    shadowWgsl,
    surfacePrimitive,
    surfaceWgsl,
} from "./pipelines";
import { Pbr } from "./shade";

// The schema-backed pipeline builder: device-bound compilation is exercised by `bun bench`; these tests
// pin its device-free WGSL, specialization, cache identity, and draw-wiring seams.

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

describe("compileVariant — boundaries and glTF support", () => {
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
            compileVariant({
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
            compileVariant({
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
        // `bun test` binds no device (testing.md — never bind a device in a default-suite `bun test`
        // file), so past the now-dropped
        // guard the call reaches `Compute.root` and throws on the missing device, not on a boundary
        // check — the positive proof (a real transparent pipeline compiles) is `bun bench`'s job.
        expect(() =>
            compileVariant({
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
        expect(surfaceWgsl(surface, 0)).toContain("fn specializeBaseFs(");
        const variant = surfaceWgsl(surface, 7);
        expect(variant).toContain("fn specializeVariantFs(");
        expect(variant).not.toContain("fn specializeBaseFs(");
    });

    test("specialize returning a foreign-copy fn throws at typedVariant — the second seam a consumer-built fn enters through, `registerSurface` can't reach it", () => {
        const l = layout({});
        const fs = tgpu.fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(0);
        });
        // the same foreign-copy shape `contract.test.ts` covers at `registerSurface`/`registerBackground`
        // — right `resourceType`, a foreign `$internal` symbol a real second copy presents identically.
        const foreignInternal = Symbol("typegpu:0.11.9:$internal");
        const foreignFs = {
            resourceType: "function",
            [foreignInternal]: true,
        } as unknown as typeof fs;
        const surface = {
            name: "specialize-foreign-probe",
            layout: l,
            fs,
            specialize: () => ({ fs: foreignFs }),
        };
        expect(() => surfaceWgsl(surface, 1)).toThrow(
            /surface "specialize-foreign-probe" variant 1 fs:.*foreign copy of typegpu/s,
        );
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
            const firstCompiled = compileVariant(first);
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
            } as unknown as SurfaceGroupEntry;
            setGroup(name, firstGroup);

            // The unchanged owner keeps both identity caches hot.
            expect(compileVariant(first)).toBe(firstCompiled);
            expect(getGroup(name, first)).toBe(firstGroup);

            const replacementCompiled = compileVariant(replacement);

            expect(replacementCompiled).not.toBe(firstCompiled);
            expect(replacementCompiled.owner).toBe(replacement);
            expect(replacementCompiled.layout).toBe(replacement.layout);
            expect(getCompiledSurface(name)).toBe(replacementCompiled);
            // The new spec can reuse the same GPU resources, but never the old layout-bound group.
            expect(getGroup(name, replacement)).toBeUndefined();
            expect(getGroup(name)).toBeUndefined();
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
        const prepass = prepassWgsl(surface);
        const shadow = shadowWgsl(surface);
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
        const prepass = prepassWgsl(surface);
        const shadow = shadowWgsl(surface);
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
        const prepass = prepassWgsl({
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

    test("an authored tag returning the renderer default preserves eid parity without calling the color fs", () => {
        const Ctx = fsCtxSchema();
        const fs = tgpu
            .fn(
                [Ctx],
                d.vec4f,
            )(() => {
                "use gpu";
                return d.vec4f(1);
            })
            .$name("defaultParityColorFs");
        const tag = tgpu
            .fn(
                [Ctx, d.u32],
                d.u32,
            )((_ctx, defaultTag) => {
                "use gpu";
                return defaultTag;
            })
            .$name("defaultParityTag");
        const surface = {
            name: "default-parity-tag",
            layout: layout({
                eids: { type: "storage", element: d.u32 },
                transforms: { type: "storage", element: Xform },
            }),
            fs,
            tag,
        };

        const wgsl = prepassWgsl(surface).tag;
        expect(wgsl).toContain("fn defaultParityTag(");
        expect(wgsl).not.toContain("fn defaultParityColorFs(");
        expect(wgsl).toMatch(/defaultParityTag\(\w+, \w+\.eid\)/);
    });

    test("an authored tag receives requested localPos through the tag prepass", () => {
        const Ctx = fsCtxSchema();
        const localTagLayout = layout({
            eids: { type: "storage", element: d.u32 },
            transforms: { type: "storage", element: Xform },
            tagBase: { type: "storage", element: d.u32 },
        });
        const fs = tgpu.fn(
            [Ctx],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        const tag = tgpu
            .fn(
                [Ctx, d.u32],
                d.u32,
            )((ctx) => {
                "use gpu";
                return localTagLayout.$.tagBase[ctx.eid] + d.u32(ctx.localPos.x > 0 ? 17 : 23);
            })
            .$name("localPosTag");
        const surface = {
            name: "local-pos-tag",
            layout: localTagLayout,
            fragmentInputs: { localPos: true } as const,
            fs,
            tag,
        };

        const wgsl = prepassWgsl(surface).tag;
        expect(wgsl).toContain("fn localPosTag(");
        expect(wgsl).toMatch(/@location\(\d+\) localPos: vec3f/);
        expect(wgsl).toContain("_arg_0.localPos");
        expect(wgsl).toContain("tagBase");
    });

    test("a non-instanced authored tag receives localPos and the TAG_NONE renderer default", () => {
        const Ctx = fsCtxSchema();
        const Patch = vsPatchSchema();
        const fs = tgpu.fn(
            [Ctx],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        const vs = tgpu.fn(
            [VsIn],
            Patch,
        )((input) => {
            "use gpu";
            return Patch({
                world: input.world,
                worldNormal: input.worldNormal,
                clip: d.vec4f(0),
            } as any);
        });
        const tag = tgpu
            .fn(
                [Ctx, d.u32],
                d.u32,
            )((ctx, defaultTag) => {
                "use gpu";
                if (ctx.localPos.x < 0) return 71;
                return defaultTag;
            })
            .$name("nonInstancedLocalTag");

        const wgsl = prepassWgsl({
            name: "non-instanced-authored-tag",
            layout: layout({}),
            fragmentInputs: { localPos: true },
            vs,
            fs,
            tag,
        }).tag;
        expect(wgsl).toContain("let eid = 0u;");
        expect(wgsl).toMatch(/@location\(\d+\) localPos: vec3f/);
        expect(wgsl).toContain("nonInstancedLocalTag(ctx, 4294967295u)");
    });

    test("an authored tag receives the surface's custom varying and renderer default", () => {
        const varyings = { packedTag: d.f32 };
        const Ctx = fsCtxSchema(varyings);
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
                    packedTag: 41,
                });
            })
            .$name("varyingTagVs");
        const fs = tgpu.fn(
            [Ctx],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        const tag = tgpu
            .fn(
                [Ctx, d.u32],
                d.u32,
            )((ctx, defaultTag) => {
                "use gpu";
                if (ctx.packedTag > 0) return d.u32(ctx.packedTag);
                return defaultTag;
            })
            .$name("varyingTag");
        const surface = {
            name: "varying-tag",
            layout: layout({
                eids: { type: "storage", element: d.u32 },
                transforms: { type: "storage", element: Xform },
            }),
            varyings,
            vs,
            fs,
            tag,
        };

        const wgsl = prepassWgsl(surface).tag;
        expect(wgsl).toContain("fn varyingTag(");
        expect(wgsl).toMatch(/@location\(5\) packedTag: f32/);
        expect(wgsl).toMatch(/@location\(5\) v0: f32/);
        expect(wgsl).toContain(
            "let ctx = Ctx(eid, world, normalize(worldNormalIn), uv, localPos, v0);",
        );
    });

    test("an authored tag carries the distinct two-varying tree arm positionally", () => {
        const varyings = { treeColor: d.vec3f, treeCell: d.f32 };
        const Ctx = fsCtxSchema(varyings);
        const Patch = vsPatchSchema(varyings);
        const vs = tgpu.fn(
            [VsIn],
            Patch,
        )((input) => {
            "use gpu";
            return Patch({
                world: input.world,
                worldNormal: input.worldNormal,
                clip: d.vec4f(0),
                treeColor: d.vec3f(0, 1, 0),
                treeCell: 73,
            });
        });
        const fs = tgpu.fn(
            [Ctx],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        const tag = tgpu
            .fn(
                [Ctx, d.u32],
                d.u32,
            )((ctx, defaultTag) => {
                "use gpu";
                if (ctx.treeColor.g > 0) return d.u32(ctx.treeCell);
                return defaultTag;
            })
            .$name("twoVaryingTreeTag");

        const wgsl = prepassWgsl({
            name: "two-varying-tree-tag",
            layout: layout({
                eids: { type: "storage", element: d.u32 },
                transforms: { type: "storage", element: Xform },
            }),
            varyings,
            vs,
            fs,
            tag,
        }).tag;
        expect(wgsl).toMatch(/@location\(5\) treeColor: vec3f/);
        expect(wgsl).toMatch(/@location\(6\) treeCell: f32/);
        expect(wgsl).toMatch(/@location\(5\) v0: vec3f/);
        expect(wgsl).toMatch(/@location\(6\) v1: f32/);
        expect(wgsl).toContain(
            "let ctx = Ctx(eid, world, normalize(worldNormalIn), uv, localPos, v0, v1);",
        );
        expect(wgsl).toContain("twovaryingtreetagTagCopier(");
        expect(wgsl).toContain("_arg_0.eid, _arg_0.v0, _arg_0.v1);");
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

        expect(surfaceWgsl(surface, 7)).toContain("fn specializeVariantVs(");
        for (const wgsl of Object.values(prepassWgsl(surface, 7))) {
            expect(wgsl).toContain("fn specializeVariantVs(");
        }
        for (const wgsl of Object.values(shadowWgsl(surface, 7))) {
            expect(wgsl).toContain("fn specializeVariantVs(");
        }
    });
});

test("known typed specialization variants come from registered draw/mesh pairs and dedupe", () => {
    const buffer = {} as never;
    const drawBuffer = {
        dataType: DrawIndexedIndirect,
        usableAsIndirect: true,
    } as never;
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
            args: { indirect: drawBuffer },
        });
    }
    const surface = {
        name: "warm-probe",
        layout: layout({}),
        fs: typedDefaultFs,
        specialize: () => ({ fs: typedDefaultFs }),
    };
    expect(knownVariants(surface).sort((a, b) => a - b)).toEqual([3, 7]);
});

test("Sear warms Part-published specializing variants; post-build variants stay lazy", async () => {
    const warmMesh = `warm-mesh-${Math.random()}`;
    const lateMesh = `late-mesh-${Math.random()}`;
    const surfaceName = `specializing-surface-${Math.random()}`;
    const warmVariant = 7;
    const lateVariant = 11;
    const buffer = {} as never;
    const surfaces = new Registry<Surface>();
    const meshes = new Registry<Mesh>();
    const draws = new Registry<Draw>();
    const registries = { surfaces, meshes, draws };
    const fs = tgpu.fn(
        [fsCtxSchema()],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(1);
    });
    surfaces.register({
        name: surfaceName,
        layout: layout({
            eids: { type: "storage", element: d.u32 },
            transforms: { type: "storage", element: Xform },
        }),
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
        precompileVariants(
            (surface, variant) => {
                const key = `${surface.name}#${variant}`;
                events.push(`sear:${key}`);
                warmed.add(key);
                return true;
            },
            surfaces,
            (surface) => knownVariants(surface, draws, meshes),
        );
        precompile("shallot-part-count", () => {
            events.push("part");
            publishPartDraws(buffer, surfaces.size, surfaces.size * meshes.size, registries);
            return [];
        });
        await precompileAll();

        expect(events[0]).toBe("part");
        expect(warmed).toContain(`${surfaceName}#${warmVariant}`);

        mesh(lateMesh, lateVariant);
        publishPartDraws(buffer, surfaces.size, surfaces.size * meshes.size, registries);
        expect(warmed).not.toContain(`${surfaceName}#${lateVariant}`);
        expect(knownVariants(surfaces.get(surfaceName)!, draws, meshes)).toContain(lateVariant);
    } finally {
        Object.assign(Compute, saved);
    }
});

test("a specializing surface compiled at two distinct variants emits two distinct pipeline labels", async () => {
    // Assert the real observable — the labels `device.createRenderPipeline` actually receives are
    // distinct per variant — not `recordCompile`'s own merge/overwrite logic, which a same-label
    // mutant would satisfy trivially.
    const surface: Surface = {
        name: `variant-label-${Math.random()}`,
        layout: layout({
            eids: { type: "storage", element: d.u32 },
            transforms: { type: "storage", element: Xform },
        }),
        fs: typedDefaultFs,
        specialize: () => ({ fs: typedDefaultFs }),
    };
    const labels: string[] = [];
    const saved = { ...Compute };
    try {
        await requestGPU({
            queue: { onSubmittedWorkDone: async () => {} },
            pushErrorScope: () => {},
            popErrorScope: async () => null,
            // typegpu 0.12's shader-module resolution reads `device.limits` to warn on overflow
            // (core/pipeline/limitsOverflow.js) — absent on 0.11, so this mock had none.
            limits: { maxUniformBuffersPerShaderStage: 12, maxStorageBuffersPerShaderStage: 10 },
            createShaderModule: () => ({}),
            createBindGroupLayout: () => ({}),
            createPipelineLayout: () => ({}),
            createRenderPipeline: (desc: GPURenderPipelineDescriptor) => {
                labels.push(desc.label ?? "");
                return {};
            },
        } as unknown as GPUDevice);

        // `Compute.root.createRenderPipeline(...)` is lazy — typegpu defers the real, labeled
        // `device.createRenderPipeline` call to first `unwrap()` (`preparePipelines`'s own comment: "typegpu
        // defers both [resolve + the sync constructor] to first use"). Force it the same way
        // `unwrapVariant` (forward.ts, `precompileVariants`'s production `warm`) does.
        const force = (variant: number) => {
            const t = compileVariant(surface, variant);
            for (const p of [t.color, t.transparent, t.point, t.cascade, ...t.prepass.values()]) {
                if (p) Compute.root.unwrap(p);
            }
        };
        force(0);
        const afterFirst = [...labels];
        force(1);
        const secondBatch = labels.slice(afterFirst.length);

        // both variants compile the same pipeline set (color + prepass ×2 + shadow ×2 — the surface
        // declares `eids`/`transforms`, so it's instanced and casts) — same count, no label in common.
        expect(afterFirst.length).toBeGreaterThan(0);
        expect(secondBatch.length).toBe(afterFirst.length);
        for (const label of secondBatch) expect(afterFirst).not.toContain(label);
    } finally {
        Object.assign(Compute, saved);
    }
});

describe("compileVariant — the transparent twin: shares color entries verbatim", () => {
    test("blend has no effect on the emitted vs/fs text — only the pipeline's blend/depth state differs, proven device-side by `bun bench`", () => {
        const l = layout({ items: { type: "storage", element: d.f32 } });
        const fs = tgpu.fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        });
        const opaque = surfaceWgsl({ name: "blend-probe", layout: l, fs });
        const alpha = surfaceWgsl({
            name: "blend-probe",
            layout: l,
            fs,
            blend: "alpha",
        });
        // the color entries never read `surface.blend` — the field only steers
        // `compileVariant`'s pipeline-creation branch (color vs. transparent target/depth state)
        expect(alpha).toBe(opaque);
    });
});

describe("fragment input specialization — gpu.md rule 9", () => {
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

    test("uv/localPos cross only when the surface fragment declares them", () => {
        const none = surfaceWgsl({ name: "input-none", layout: l, fs });
        expect(none).not.toContain("@location(3) uv:");
        expect(none).not.toContain("@location(4) localPos:");

        const uv = surfaceWgsl({
            name: "input-uv",
            layout: l,
            fragmentInputs: { uv: true },
            fs,
        });
        const localPos = surfaceWgsl({
            name: "input-local-pos",
            layout: l,
            fragmentInputs: { localPos: true },
            fs,
        });

        const both = surfaceWgsl({
            name: "input-both",
            layout: l,
            fragmentInputs: { uv: true, localPos: true },
            fs,
        });
        expect(both).toContain("@location(3) uv: vec2f");
        expect(both).toContain("@location(4) localPos: vec3f");

        for (const [name, wgsl] of [
            ["inputnone", none],
            ["inputuv", uv],
            ["inputlocalpos", localPos],
            ["inputboth", both],
        ] as const) {
            expect(wgsl).toContain(`fn ${name}Vertex(`);
            expect(wgsl).not.toContain(`fn ${name}Copier(`);
        }
    });

    test("clip color, prepass, and shadow modules share the same declaration", () => {
        const surface = {
            name: "input-clip",
            layout: l,
            fragmentInputs: { uv: true } as const,
            fs,
            blend: "clip" as const,
        };
        const modules = [
            surfaceWgsl(surface),
            ...Object.values(prepassWgsl(surface)),
            ...Object.values(shadowWgsl(surface)),
        ];
        for (const wgsl of modules) {
            expect(wgsl).toMatch(/@location\(\d+\) uv: vec2f/);
            expect(wgsl).not.toMatch(/@location\(\d+\) localPos:/);
        }
    });
});

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
        surfaceWgsl({
            name: "arity2",
            layout: arityLayout,
            fragmentInputs: { uv: true, localPos: true },
            varyings: v2,
            vs: vs2,
            fs: fs2,
        });
    const wgsl3 = () =>
        surfaceWgsl({
            name: "arity3",
            layout: arityLayout,
            fragmentInputs: { uv: true, localPos: true },
            varyings: v3,
            vs: vs3,
            fs: fs3,
        });
    const wgsl4 = () =>
        surfaceWgsl({
            name: "arity4",
            layout: arityLayout,
            fragmentInputs: { uv: true, localPos: true },
            varyings: v4,
            vs: vs4,
            fs: fs4,
        });

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
        expect(wgsl2()).toContain("fn arity2Copier(");
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
            surfaceWgsl({
                name: "arity5",
                layout: arityLayout,
                varyings: v5,
                vs: vs5,
                fs: fs5,
            }),
        ).toThrow(/declares 5 varyings/);
    });
});

// `screen` surfaces: the raw path's `out.clip = clipPos` — the vs chunk projects its own clip-space
// geometry, and back-face culling is off because those quads have no consistent winding. Both typed vs
// shapes (the shared TGSL body and the per-surface WGSL copier) honor it.
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
        const screenWgsl = surfaceWgsl(screenSurface);
        expect(body(screenWgsl, "fn screenProbeVertex(")).toContain("pos = patched.clip;");
        expect(screenWgsl).not.toContain("view.viewProj * world");
        // and the projection matrix isn't even declared — a screen module reads no view
        expect(screenWgsl).not.toContain("var<uniform> view: View;");

        const worldWgsl = surfaceWgsl(worldSurface);
        expect(body(worldWgsl, "fn worldProbeVertex(")).toContain("pos = (view.viewProj * world);");
        expect(worldWgsl).not.toContain("pos = patched.clip;");
    });

    test("the prepass + tag vs entries honor it too — the raw path splices one `out.clip` for every pass", () => {
        const prepass = prepassWgsl(screenSurface);
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
        const wgsl = surfaceWgsl({
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

describe("typed prepass — the `vs`-chunk / non-instanced branches (review-caught: red-first)", () => {
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

    test("the typed depth-only + tag prepass vs entries pass the pinned localNormal (not the transformed worldNormal) into VsIn — the fixed bug", () => {
        const wgsl = prepassWgsl(typedSurfaceWithVs);
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
        const wgsl = prepassWgsl(surface);
        expect(wgsl.tag).toMatch(/eid = 4294967295u;/);
    });
});

describe("typed shadow atlas — the `vs`-chunk override (mirrors typedPrepassVs's pinned localNormal law)", () => {
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

    test("the typed point/cascade VS calls the surface's vs chunk, applies its world override before projecting, and the vs chunk sees the pinned (0,0,1) localNormal default", () => {
        const wgsl = shadowWgsl(typedSurfaceWithVs);
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

// A schema-backed background fixture for cache-identity tests.
const Tint = d.struct({ value: d.vec4f }).$name("Tint");
const typedBgProbeLayout = bgLayout({ tint: { type: "uniform", struct: Tint } });
const typedBgProbeFs = tgpu
    .fn(
        [BgCtx],
        d.vec3f,
    )((ctx) => {
        "use gpu";
        return std.mix(
            typedBgProbeLayout.$.tint.value.xyz,
            std.add(std.mul(ctx.dir, 0.5), d.vec3f(0.5)),
            0.5,
        );
    })
    .$name("typedBgProbeChunk");

test("final surface, prepass, shadow, and background entry bodies preserve integer division discipline", () => {
    const l = layout({
        eids: { type: "storage", element: d.u32 },
        transforms: { type: "storage", element: Xform },
    });
    const fs = tgpu
        .fn(
            [fsCtxSchema()],
            d.vec4f,
        )(() => {
            "use gpu";
            return d.vec4f(1);
        })
        .$name("integerAuditChunk");
    const surface = { name: "integer-audit", layout: l, fs };

    const color = surfaceWgsl(surface);
    for (const signature of [
        "fn integerauditVertex(",
        "fn integerauditVs(",
        "fn integerauditFs(",
    ]) {
        noIntegerDivision(body(color, signature));
    }

    const prepass = prepassWgsl(surface);
    noIntegerDivision(body(prepass[""], "fn integerauditPrepassVs("));
    noIntegerDivision(body(prepass.tag, "fn integerauditPrepassTagVs("));
    noIntegerDivision(body(prepass.tag, "fn integerauditPrepassTagFs("));

    const shadow = shadowWgsl(surface);
    noIntegerDivision(body(shadow.point, "fn integerauditPointVs("));
    noIntegerDivision(body(shadow.point, "fn shadowAtlasFs("));
    noIntegerDivision(body(shadow.cascade, "fn integerauditCascadeVs("));
    noIntegerDivision(body(shadow.cascade, "fn shadowAtlasFs("));

    const background = backgroundWgsl({
        name: "integer-audit-bg",
        layout: bgLayout({}),
        fs: typedBgProbeFs,
    });
    noIntegerDivision(body(background, "fn bgVs("));
    noIntegerDivision(body(background, "fn integerauditbgFs("));
});
test("typed background pipeline and bind-group caches invalidate on exact spec/layout replacement", () => {
    const name = `replace-typed-bg-${Math.random()}`;
    const first: Background = { name, layout: bgLayout({}), fs: typedBgProbeFs };
    // First replacement deliberately shares the layout, isolating spec identity from layout identity.
    const replacement: Background = { name, layout: first.layout, fs: typedBgProbeFs };

    // Pipeline wrappers are lazy; this root stub keeps the cache-identity contract device-free while
    // the sky bench proves the replacement-capable builder on a real device.
    const originalRoot = Compute.root;
    const fakeRoot = {
        createRenderPipeline: () => ({
            $name() {
                return this;
            },
        }),
    };
    (Compute as unknown as { root: typeof fakeRoot }).root = fakeRoot;
    try {
        const firstCompiled = compileBackground(first);
        firstCompiled.group2 = {
            group: {} as GPUBindGroup,
            resources: [{} as GPUBuffer],
        };
        firstCompiled.engineCache.set(0, {} as GPUBindGroup);

        expect(compileBackground(first)).toBe(firstCompiled);
        expect(getBackground(name, first)).toBe(firstCompiled);

        const replacementCompiled = compileBackground(replacement);
        expect(replacementCompiled).not.toBe(firstCompiled);
        expect(replacementCompiled.owner).toBe(replacement);
        expect(replacementCompiled.layout).toBe(replacement.layout);
        expect(getBackground(name, replacement)).toBe(replacementCompiled);
        expect(replacementCompiled.group2).toBeNull();
        expect(replacementCompiled.engineCache.size).toBe(0);

        // Then mutate only the current owner's layout, isolating the second half of the cache key.
        replacementCompiled.group2 = {
            group: {} as GPUBindGroup,
            resources: [{} as GPUBuffer],
        };
        replacementCompiled.engineCache.set(0, {} as GPUBindGroup);
        replacement.layout = bgLayout({});
        const relaidCompiled = compileBackground(replacement);
        expect(relaidCompiled).not.toBe(replacementCompiled);
        expect(relaidCompiled.owner).toBe(replacement);
        expect(relaidCompiled.layout).toBe(replacement.layout);
        expect(relaidCompiled.group2).toBeNull();
        expect(relaidCompiled.engineCache.size).toBe(0);
    } finally {
        (Compute as unknown as { root: typeof originalRoot }).root = originalRoot;
    }
});

describe("draw wiring — the depth-pipeline receiver stub + the color pipeline's real receiver", () => {
    // a vs-chunk surface whose chunk reaches litPbr → pointShadowOf, the shape that forces the stub:
    // the raw path's prepass/shadow modules bind `pointShadowStub` for exactly this reach
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

    const colorWgsl = surfaceWgsl(surface);
    const emittedPrepass = prepassWgsl(surface);
    const emittedShadow = shadowWgsl(surface);

    test("the color module emits the REAL receiver (samples the atlas) and declares its group-1 free names", () => {
        const receiver = body(colorWgsl, "fn pointShadowOf(");
        expect(receiver).toContain("textureSampleCompareLevel(pointAtlas");
        // typedVaryingFs's forcedZero fold is what declares them (invisible to the call-graph walk)
        expect(colorWgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var pointAtlas/);
        expect(colorWgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var shadowSamp/);
    });

    test("the prepass modules emit the STUB receiver — no atlas sampling, no undeclared free names", () => {
        for (const wgsl of [emittedPrepass[""], emittedPrepass.tag]) {
            const receiver = body(wgsl, "fn pointShadowOf(");
            expect(receiver).toContain("return 1f;");
            expect(receiver).not.toContain("textureSampleCompareLevel");
            expect(wgsl).not.toContain("pointAtlas");
            expect(wgsl).not.toContain("pointShadows");
        }
    });

    test("the point/cascade modules emit the STUB receiver — the real one would sample the atlas being written", () => {
        for (const wgsl of [emittedShadow.point, emittedShadow.cascade]) {
            const receiver = body(wgsl, "fn pointShadowOf(");
            expect(receiver).toContain("return 1f;");
            expect(receiver).not.toContain("textureSampleCompareLevel");
            expect(wgsl).not.toContain("pointAtlas");
        }
    });

    test("the prepass modules keep the group set dense — a group-1 touch, never a hole typegpu's group-indexed pipeline layout would emit", () => {
        for (const wgsl of [emittedPrepass[""], emittedPrepass.tag]) {
            expect(wgsl).toMatch(/@group\(1\)\s*@binding\(\d+\)\s*var<uniform> tileRects/);
        }
    });

    test("the varying crosses at an explicit interstage slot — vs `litColor` and fs `v0` pinned to the same @location", () => {
        expect(colorWgsl).toContain("@location(5) litColor: vec3f");
        expect(colorWgsl).toContain("@location(5) v0: vec3f");
    });
});

describe("draw wiring — shadowLayout visibility mirrors the raw _shadowBgl", () => {
    test("sampler + point-shadow entries are vertex-visible (a per-vertex vs chunk reaches pointShadowOf); sun map + params stay fragment-only", () => {
        const entries = shadowLayout.entries as Record<string, { visibility?: string[] }>;
        expect(entries.shadowSamp.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.pointAtlas.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.pointShadows.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.tileRects.visibility).toEqual(["vertex", "fragment"]);
        expect(entries.shadowMap.visibility).toEqual(["fragment"]);
        expect(entries.sunShadow.visibility).toEqual(["fragment"]);
    });
});

describe("draw wiring — the Backdrop registry id space", () => {
    const traits = (SearPlugin.traits as Record<string, unknown>).Backdrop as {
        parse: { name: (v: string) => number | undefined };
        format: { name: (v: number) => string | undefined };
    };
    test("a registered background parses to its registry id and formats back", () => {
        const bgFs = tgpu.fn(
            [BgCtx],
            d.vec3f,
        )((ctx) => {
            "use gpu";
            return d.vec3f(ctx.dir.y);
        });
        const name = `backdrop-${Math.random()}`;
        const state = new State();
        registerBackground(state, { name, layout: bgLayout({}), fs: bgFs });
        const id = traits.parse.name(name)!;
        expect(id).toBe(Backgrounds.id(name)!);
        expect(traits.format.name(id)).toBe(name);
        expect(traits.parse.name("c3b-not-registered")).toBeUndefined();
        state.dispose();
    });
});

describe("draw wiring — typed group-state eviction", () => {
    test("clearGroups drops a cached typed entry (the re-gather realloc + rebuild eviction path, and the engine-group lifetime rides the entry)", () => {
        const entry = {
            engineCache: new Map([[0, {} as GPUBindGroup]]),
            resources: [],
        } as unknown as SurfaceGroupEntry;
        setGroup("c3b-evict-probe", entry);
        expect(getGroup("c3b-evict-probe")).toBe(entry);
        clearGroups();
        // the entry — and with it the per-entry engineCache holding the quant-keyed group-0 instances —
        // is unreachable after the clear; nothing module-scoped retains a churned quant buffer
        expect(getGroup("c3b-evict-probe")).toBeUndefined();
    });
});

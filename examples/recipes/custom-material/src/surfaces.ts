import { PartPlugin, type Plugin, SearPlugin } from "@dylanebert/shallot";
import {
    BgCtx,
    backgroundLayout,
    engineLayout,
    fsCtxSchema,
    lit,
    registerBackground,
    registerSurface,
    sunVisibility,
    surfaceLayout,
} from "@dylanebert/shallot/sear/core";
import { unpackLdrColor, Xform } from "@dylanebert/shallot/utils/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

const instancedLayout = surfaceLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
});
const colorLayout = surfaceLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    color: { type: "storage", element: d.u32 },
});

const checkerFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const cell = std.floor(ctx.uv.x * 4) + std.floor(ctx.uv.y * 4);
    const c = std.fract(cell * 0.5) * 2;
    return d.vec4f(lit(std.mix(d.vec3f(0.15), d.vec3f(0.85), c), ctx.worldNormal), 1);
});

const glassFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const c = unpackLdrColor(colorLayout.$.color[ctx.eid]);
    return d.vec4f(lit(c.xyz, ctx.worldNormal), c.w);
});

const orenNayar = tgpu.fn(
    [d.vec3f, d.vec3f, d.vec3f, d.f32],
    d.f32,
)((normal, view, light, sigma) => {
    "use gpu";
    const s2 = sigma * sigma;
    const a0 = 1 - (0.5 * s2) / (s2 + 0.33);
    const b0 = (0.45 * s2) / (s2 + 0.09);
    const ndl = std.max(std.dot(normal, light), 0);
    const ndv = std.max(std.dot(normal, view), 0);
    const a = std.max(std.acos(ndl), std.acos(ndv));
    const b = std.min(std.acos(ndl), std.acos(ndv));
    const g = std.max(
        std.dot(
            std.normalize(std.sub(view, std.mul(normal, ndv))),
            std.normalize(std.sub(light, std.mul(normal, ndl))),
        ),
        0,
    );
    return ndl * (a0 + b0 * g * std.sin(a) * std.tan(b));
});

const clayFs = tgpu.fn(
    [fsCtxSchema()],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const base = unpackLdrColor(colorLayout.$.color[ctx.eid]).xyz;
    const view = std.normalize(std.sub(engineLayout.$.view.eye.xyz, ctx.world));
    const light = std.neg(engineLayout.$.lighting.sunDirection.xyz);
    const ambient = std.mul(
        engineLayout.$.lighting.ambientColor.rgb,
        engineLayout.$.lighting.ambientColor.a,
    );
    const sun = std.mul(
        engineLayout.$.lighting.sunColor.rgb,
        orenNayar(ctx.worldNormal, view, light, 0.7) * sunVisibility.$,
    );
    return d.vec4f(std.mul(base, std.add(ambient, sun)), 1);
});

const gradientFs = tgpu.fn(
    [BgCtx],
    d.vec3f,
)((ctx) => {
    "use gpu";
    const t = std.clamp(ctx.dir.y * 0.5 + 0.5, 0, 1);
    return std.mix(d.vec3f(0.02, 0.03, 0.06), d.vec3f(0.1, 0.16, 0.28), t);
});

// A surface is a schema-backed TGSL recipe registered by name; `Part.surface` selects it per entity.
// `surfaceLayout()` declares its group-2 resources, `fsCtxSchema()` types the fragment inputs, and the
// TGSL function returns the final RGBA color. Geometry is per-entity (`Part.mesh`), so one surface shades
// any mesh; `fragmentInputs` declares the optional mesh fields that actually cross the rasterizer.
const ShallotSurfaces = {
    name: "ShallotSurfaces",
    // RenderPlugin clears the surface registry and SearPlugin the backdrop registry on (re)build, so
    // register after both
    dependencies: [PartPlugin, SearPlugin],
    initialize(state) {
        // A procedural UV checker shaded through sear's `lit(base, normal)` diffuse helper. Declaring the
        // `eids` + `transforms` bindings is all instancing needs: sear applies the standard per-instance
        // transform, no `vs` chunk.
        registerSurface(state, {
            name: "checker",
            layout: instancedLayout,
            fragmentInputs: { uv: true },
            fs: checkerFs,
        });

        // `blend: "alpha"` routes the surface to sear's non-opaque pass. Per-instance opacity rides
        // `color.a`, so the scene tints and sets opacity with one `color` attribute.
        registerSurface(state, {
            name: "glass",
            blend: "alpha",
            layout: colorLayout,
            fs: glassFs,
        });

        // sear's `lit()` isn't the only shading model, so write your own. This is Oren-Nayar diffuse (the
        // rough-matte clay / moon look): the TGSL `orenNayar` helper rides `clayFs`'s call graph, while
        // `clayFs` closes over `engineLayout` for the sun and view directions. `sunVisibility.$` is filled
        // by the color scaffold before the function runs, so even a custom BRDF gets sun shadows for free.
        registerSurface(state, {
            name: "clay",
            layout: colorLayout,
            fs: clayFs,
        });

        // A backdrop fills the un-rendered (background) pixels: a view-ray → HDR color recipe registered
        // with `Backgrounds`, selected per camera by the `Backdrop` component. Its schema-backed TGSL
        // `fs` receives `BgCtx.dir`, the world-space view ray Sear reconstructs per pixel, and returns
        // the HDR RGB color.
        registerBackground(state, {
            name: "gradient",
            layout: backgroundLayout({}),
            fs: gradientFs,
        });
    },
} satisfies Plugin;

export default ShallotSurfaces;

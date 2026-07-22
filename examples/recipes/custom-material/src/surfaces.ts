import { PartPlugin, type Plugin, SearPlugin } from "@dylanebert/shallot";
import { Surfaces } from "@dylanebert/shallot/render/core";
import { Backgrounds } from "@dylanebert/shallot/sear/core";

// A surface is a WGSL shading program registered by name; `Part.surface` selects it per entity. Sear
// splices the chunk into its fragment shader: the chunk reads sear's locals (`uv`, `worldNormal`, the
// per-instance `color`) and writes the final `col`. Geometry is per-entity (`Part.mesh`), so one surface
// shades any mesh.
const KitchenSurfaces = {
    name: "KitchenSurfaces",
    // RenderPlugin clears the surface registry and SearPlugin the backdrop registry on (re)build, so
    // register after both
    dependencies: [PartPlugin, SearPlugin],
    initialize() {
        // A procedural UV checker shaded through sear's `lit(base, normal)` diffuse helper. Declaring the
        // `eids` + `transforms` bindings is all instancing needs: sear applies the standard per-instance
        // transform, no `vs` chunk.
        Surfaces.register({
            name: "checker",
            bindings: {
                eids: { type: "storage", element: "u32" },
                transforms: { type: "storage", element: "Xform" },
            },
            fs: /* wgsl */ `
                let c = (i32(floor(uv.x * 4.0)) + i32(floor(uv.y * 4.0))) & 1;
                col = vec4<f32>(lit(mix(vec3<f32>(0.15), vec3<f32>(0.85), f32(c)), worldNormal), 1.0);
            `,
        });

        // `blend: "alpha"` routes the surface to sear's non-opaque pass. Per-instance opacity rides
        // `color.a`, so the scene tints and sets opacity with one `color` attribute.
        Surfaces.register({
            name: "glass",
            blend: "alpha",
            bindings: {
                eids: { type: "storage", element: "u32" },
                transforms: { type: "storage", element: "Xform" },
                color: { type: "storage", element: "u32" },
            },
            fs: /* wgsl */ `
                let c = unpackLdrColor(color[eid]);
                col = vec4<f32>(lit(c.rgb, worldNormal), c.a);
            `,
        });

        // sear's `lit()` isn't the only shading model, so write your own. This is Oren-Nayar diffuse (the
        // rough-matte clay / moon look): `preamble` carries the BRDF, `fs` calls it with the sun and view
        // directions from `lighting` and `view.eye`. `sunVisibility` is filled by the color scaffold before
        // the chunk runs, so even a hand-written BRDF gets sun shadows for free.
        Surfaces.register({
            name: "clay",
            bindings: {
                eids: { type: "storage", element: "u32" },
                transforms: { type: "storage", element: "Xform" },
                color: { type: "storage", element: "u32" },
            },
            // Oren-Nayar 1994, the qualitative model (1/pi dropped to match sear's non-physical diffuse)
            preamble: /* wgsl */ `
                fn orenNayar(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, sigma: f32) -> f32 {
                    let s2 = sigma * sigma;
                    let A = 1.0 - 0.5 * s2 / (s2 + 0.33);
                    let B = 0.45 * s2 / (s2 + 0.09);
                    let ndl = max(dot(N, L), 0.0);
                    let ndv = max(dot(N, V), 0.0);
                    let a = max(acos(ndl), acos(ndv));
                    let b = min(acos(ndl), acos(ndv));
                    let g = max(dot(normalize(V - N * ndv), normalize(L - N * ndl)), 0.0);
                    return ndl * (A + B * g * sin(a) * tan(b));
                }
            `,
            fs: /* wgsl */ `
                let base = unpackLdrColor(color[eid]).rgb;
                let V = normalize(view.eye.xyz - world);
                let L = -lighting.sunDirection.xyz;
                let ambient = lighting.ambientColor.rgb * lighting.ambientColor.a;
                let sun = lighting.sunColor.rgb * orenNayar(worldNormal, V, L, 0.7) * sunVisibility;
                col = vec4<f32>(base * (ambient + sun), 1.0);
            `,
        });

        // A backdrop fills the un-rendered (background) pixels: a view-ray → HDR color recipe registered
        // with `Backgrounds`, selected per camera by the `Backdrop` component. The `fs` writes `col` from
        // `dir`, the world-space view ray sear reconstructs per pixel.
        Backgrounds.register({
            name: "gradient",
            fs: /* wgsl */ `
                let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
                col = mix(vec3<f32>(0.02, 0.03, 0.06), vec3<f32>(0.10, 0.16, 0.28), t);
            `,
        });
    },
} satisfies Plugin;

export default KitchenSurfaces;

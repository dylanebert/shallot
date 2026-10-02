import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Xform, xformPoint } from "../../engine/utils";
import { fsCtxSchema, surfaceLayout, VsIn, vsPatchSchema } from "../../standard/rendering";
import { Glyph, sdfToSignedDistance, textSrgbToLinear } from "./glyph";

export const atlasName = (id: number) => `textAtlas${id}`;

// the two custom interstage slots (within the 4-slot custom budget): `uvSize` folds the mixed atlas uv
// (`.xy`) and the world quad size (`.zw`, what the fs's AA math scales `fwidth(localPos)` by) into one
// vec4 — `vsPatchSchema` has no `uv` field to override (only `world`/`worldNormal`/`clip` + varyings), so
// the atlas uv can't ride the built-in. `gcolor` unpacks `unpack4x8unorm` in the vs (a per-instance
// constant, so it interpolates exactly) rather than crossing the packed u32 and unpacking per-fragment.
export const textVaryings = { uvSize: d.vec4f, gcolor: d.vec4f };

// per-font typed surface: a fresh `surfaceLayout` per id (the atlas texture binding's name carries the
// id, so each font gets its own layout object, and a vs/fs built against one layout can't be shared with
// another's). localPos.xy is the quad corner (0,0)..(1,1); signed-distance edge AA decodes the SDF to a
// world-space signed distance, faded over one screen-space derivative either side of the glyph edge
// (Valve "Improved Alpha-Tested Magnification"); fully-transparent texels discard before the blend
/** @internal Shader factory shared by registration and placement verification. */
export function textSurface(id: number) {
    const atlasKey = atlasName(id);
    const layout = surfaceLayout({
        textGlyphs: { type: "storage", element: Glyph },
        globalTransforms: { type: "storage", element: Xform },
        globalTransformRows: { type: "storage", element: d.u32 },
        textSamp: { type: "sampler" },
        [atlasKey]: { type: "texture-2d" },
    });
    // `vsPatchSchema`/`fsCtxSchema` are plain host functions (no "use gpu"), so they must be called OUTSIDE
    // any traced body — a call from inside a "use gpu" closure throws "not marked with the 'use gpu'
    // directive" at pipeline-resolution time (`standard/rendering/forward.ts`'s `typedVertexPatch` is the
    // reference pattern). Hoisted once here, the vs body below references the constructor only
    const VertexPatch = vsPatchSchema(textVaryings);
    const vs = tgpu
        .fn(
            [VsIn],
            VertexPatch,
        )((vsIn) => {
            "use gpu";
            const g = Glyph(layout.$.textGlyphs[vsIn.iid]);
            const encodedRow = layout.$.globalTransformRows[g.eid];
            // Like the instance packer, omit geometry without a placement. All corners collapse.
            if (encodedRow === 0) {
                return VertexPatch({
                    world: d.vec4f(0),
                    worldNormal: d.vec3f(0),
                    clip: d.vec4f(0),
                    uvSize: d.vec4f(0),
                    gcolor: d.vec4f(0),
                } as never);
            }
            const x = Xform(layout.$.globalTransforms[encodedRow - 1]);
            const corner = vsIn.localPos.xy;
            const gp = d.vec3f(
                g.pos.x + corner.x * g.size.x,
                g.pos.y + corner.y * g.size.y,
                g.pos.z,
            );
            const uv = std.mix(g.uvRect.xy, g.uvRect.zw, corner);
            return VertexPatch({
                world: d.vec4f(xformPoint(x, gp), 1),
                worldNormal: vsIn.worldNormal,
                clip: d.vec4f(0),
                uvSize: d.vec4f(uv, g.size),
                gcolor: std.unpack4x8unorm(g.color),
            } as never);
        })
        .$name(`text${id}Vs`);

    const fs = tgpu
        .fn(
            [fsCtxSchema(textVaryings)],
            d.vec4f,
        )((ctx) => {
            "use gpu";
            // the atlas texture key is per-font (computed), so `layout.$`'s mapped type can't narrow it
            // the way a fixed key like `layout.$.textSamp` resolves automatically — one cast to the
            // runtime texture-sample representation `$` exposes for a fixed `texture-2d` binding. Read
            // here, inside the traced body: `layout.$[atlasKey]` executes the TypeGPU view accessor for
            // real, which only resolves inside an active codegen/dispatch context — reading it at
            // factory-call time (module JS, before any trace) throws "outside of codegen mode" on a real
            // device (the untyped resolve path bun test exercises doesn't reach the accessor at all).
            // Passed straight into the call, never bound to a `const` first — a texture/sampler handle's
            // snippet origin is untyped-pointer-incompatible ("handle"), and TypeGPU's const-declaration
            // codegen tries to take a pointer to any aliased (non-copyable) RHS, so a `const atlas = ...`
            // binding throws "Creating pointer type from origin handle" at pipeline-resolution time.
            const sdf = std.textureSample(
                (layout.$ as unknown as Record<string, d.texture2d<d.F32>>)[atlasKey],
                layout.$.textSamp,
                ctx.uvSize.xy,
            ).x;
            const gsize = ctx.uvSize.zw;
            const maxDim = std.max(gsize.x, gsize.y);
            const signedDist = sdfToSignedDistance(sdf, maxDim);
            const aa = std.length(std.fwidth(std.mul(ctx.localPos.xy, gsize))) * 0.5;
            const alpha = std.smoothstep(aa, -aa, signedDist);
            if (alpha < 0.01) {
                std.discard();
            }
            return d.vec4f(textSrgbToLinear(ctx.gcolor.xyz), ctx.gcolor.w * alpha);
        })
        .$name(`text${id}Fs`);

    return { layout, vs, fs };
}

// The `lines` sear surface: the segment schema, its group-2 layout, and the screen-space quad expansion
// as TGSL. Sibling of `index.ts` because that module holds the ECS/system/plugin half (kernels live
// beside, not inside, registry code); the CPU staging + upload live in `segments.ts`.
//
// `localPos.xy` carries the quad corner: x = t (0 start, 1 end), y = edge (-1, +1). The surface is
// `screen: true`, so its `vs` owns the projection: it projects both endpoints, near-plane-clips, then
// offsets the chosen endpoint perpendicular by a pixel half-width (constant-pixel: pixels → NDC is
// × 2/resolution). 1px of AA pad each side; a sub-pixel width clamps the geometry to 1px and fades the
// alpha to keep its energy. `world` is the chosen endpoint so the (unused, DCE'd) shadow sample reads a
// valid position.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { unpackLdrColor } from "../../engine/utils/core";
import {
    engineLayout,
    fsCtxSchema,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "../../standard/sear/core";

/** one debug segment: two world endpoints + a pixel width + a packed sRGBA color. 32 B, the layout
 *  `segments.ts` stages (`a.xyz` shares its 16-byte slot with `width`, `b.xyz` with `color`). */
export const Segment = d
    .struct({ a: d.vec3f, width: d.f32, b: d.vec3f, color: d.u32 })
    .$name("Segment");

/** the surface's own group-2 layout — one read-only storage array of {@link Segment}, published under
 *  `lineSegments` by `warmSegments`. */
export const lineLayout = surfaceLayout({
    lineSegments: { type: "storage", element: Segment },
});

/**
 * the two custom interpolators the quad crosses (locations 5–6, gpu.md rule 9's 4-slot budget). `edge`
 * packs the AA pair (`x` = signed distance from the segment centreline in pixels, `y` = the half-width the
 * fs compares it against). Two, not one: rgb + alpha + one AA shape parameter is five independent floats —
 * straight-alpha blending can't fold alpha into rgb, and `min(w,1)` / `max(w,1)*0.5` aren't mutually
 * recoverable below `w = 1`. The AA pair itself is already at its floor (`localPos.y` interpolates the
 * `edge` term and smoothstep is invariant under common positive scale), which is why it's 2 and not 3.
 */
export const lineVaryings = { lineRgba: d.vec4f, edge: d.vec2f };

const LinePatch = vsPatchSchema(lineVaryings);
const LineCtx = fsCtxSchema(lineVaryings);

/** what {@link lineQuad} resolves per vertex: the clip position the `screen` surface projects itself,
 *  the AA pair, the per-vertex color modulation (`tint`, which collapses the whole rgba to zero on the
 *  both-behind-camera arm), and which endpoint this corner sits on (`useEnd`, 1 = the segment's `b`). */
const LineQuad = d
    .struct({ clip: d.vec4f, edge: d.vec2f, tint: d.vec4f, useEnd: d.f32 })
    .$name("LineQuad");

// the w below which an endpoint is treated as behind the camera. Both the degenerate test and the
// near-plane interpolation target it, so a clipped endpoint lands exactly on the plane
const NEAR_W = 1e-5;

/**
 * expand one projected segment into this corner's screen-space quad vertex — the whole projection /
 * near-clip / constant-pixel-width / sub-pixel-fade kernel, pure and CPU-callable (its only inputs are
 * the two clip-space endpoints, the viewport resolution, and the corner). Both endpoints behind the
 * camera collapses to an off-screen vertex with a zero tint; otherwise the endpoint nearer than the near
 * plane is pulled onto it before the perpendicular offset.
 *
 * @example const q = lineQuad(sClip, eClip, vec2f(1920, 1080), 0, -1, 2);
 */
export const lineQuad = tgpu
    .fn(
        [d.vec4f, d.vec4f, d.vec2f, d.f32, d.f32, d.f32],
        LineQuad,
    )((sClip, eClip, res, t, edge, widthPx) => {
        "use gpu";
        const nearW = d.f32(NEAR_W);
        let sc = d.vec4f(sClip);
        let ec = d.vec4f(eClip);
        if (sc.w < nearW && ec.w < nearW) {
            // both endpoints behind the camera — collapse offscreen (z < 0 clips) and zero the color
            return LineQuad({
                clip: d.vec4f(0, 0, -1, 1),
                edge: d.vec2f(0),
                tint: d.vec4f(0),
                useEnd: d.f32(0),
            });
        }
        if (sc.w < nearW) {
            const k = (nearW - sc.w) / (ec.w - sc.w);
            sc = d.vec4f(std.mix(sc, ec, k));
        } else {
            if (ec.w < nearW) {
                const k = (nearW - ec.w) / (sc.w - ec.w);
                ec = d.vec4f(std.mix(ec, sc, k));
            }
        }
        const sNdc = std.div(sc.xy, sc.w);
        const eNdc = std.div(ec.xy, ec.w);
        const dirPx = std.mul(std.sub(eNdc, sNdc), res);
        const lenPx = std.length(dirPx);
        const dir = std.select(d.vec2f(1, 0), std.div(dirPx, lenPx), lenPx > d.f32(1e-4));
        const perp = d.vec2f(-dir.y, dir.x);
        const halfW = std.max(widthPx, d.f32(1)) * d.f32(0.5);
        const total = halfW + d.f32(1);
        const useEnd = t > d.f32(0.5);
        const baseNdc = std.select(sNdc, eNdc, useEnd);
        const baseClip = std.select(sc, ec, useEnd);
        const ndc = std.add(baseNdc, std.div(std.mul(std.mul(perp, edge * total), d.f32(2)), res));
        return LineQuad({
            clip: d.vec4f(ndc, baseClip.z / baseClip.w, 1),
            edge: d.vec2f(edge * total, halfW),
            // a sub-pixel segment is drawn 1px wide and its alpha faded to keep the line's energy
            tint: d.vec4f(1, 1, 1, std.min(widthPx, d.f32(1))),
            useEnd: std.select(d.f32(0), d.f32(1), useEnd),
        });
    })
    .$name("lineQuad");

/** the surface's vertex chunk: pull this instance's segment, project both endpoints, and hand the corner
 *  to {@link lineQuad}. `worldNormal` passes through unchanged — nothing in the alpha pass reads it. */
export const lineVs = tgpu
    .fn(
        [VsIn],
        LinePatch,
    )((vsIn) => {
        "use gpu";
        // the array-element read is a pointer, not a copy — wrap in the element schema
        const seg = Segment(lineLayout.$.lineSegments[vsIn.iid]);
        // `view.viewProj` is re-read rather than bound to a local: a uniform-aliased local emits WGSL
        // pointer form, the shape Metal's compiler is watched on
        const q = lineQuad(
            std.mul(engineLayout.$.view.viewProj, d.vec4f(seg.a, 1)),
            std.mul(engineLayout.$.view.viewProj, d.vec4f(seg.b, 1)),
            engineLayout.$.view.resolution,
            vsIn.localPos.x,
            vsIn.localPos.y,
            seg.width,
        );
        return LinePatch({
            world: d.vec4f(std.select(seg.a, seg.b, q.useEnd > d.f32(0.5)), 1),
            worldNormal: vsIn.worldNormal,
            clip: q.clip,
            lineRgba: std.mul(unpackLdrColor(seg.color), q.tint),
            edge: q.edge,
        } as never);
    })
    .$name("lineVs");

/** the surface's fragment chunk: signed-distance edge AA, fading over one screen-space derivative either
 *  side of the half-width. */
export const lineFs = tgpu
    .fn(
        [LineCtx],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        const edgeDist = ctx.edge.x;
        const halfPx = ctx.edge.y;
        const w = std.fwidth(edgeDist);
        const aa = d.f32(1) - std.smoothstep(halfPx - w, halfPx + w, std.abs(edgeDist));
        return d.vec4f(ctx.lineRgba.xyz, ctx.lineRgba.w * aa);
    })
    .$name("lineFs");

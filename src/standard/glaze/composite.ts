// Glaze's composite kernel and the layout it binds: the per-camera postfx chain as pure TGSL math beside
// the schema that defines its uniform, sitting next to the ECS/system/plugin code in `index.ts` (the
// kernel-sibling convention).
//
// The chain: a scene-referred grade (ASC CDL + exposure) on the HDR radiance, then a tonemap operator
// (`Tonemap`, default Khronos Neutral), then a post-tonemap saturation, then OkLab-L posterize/dither
// (dither before posterize so noise breaks bands), then γ-2.2 vignette, then the linear→sRGB encode. The
// grade straddles the tonemap on purpose: CDL + exposure are scene-referred so they sit before the
// operator (a pushed highlight rolls off its shoulder), saturation is perceptual so it sits after
// (display-referred values read predictably).
//
// Each stage takes the config as a parameter rather than reading the binding, so the whole chain runs on
// the CPU in `bun test` with no device. The operand order is the shipped shader's, term for term —
// reassociating a float product changes the last bits, and the bench's framebuffer probes compare against
// the pre-port baseline.

import tgpu, { type TgpuComputePipeline } from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { Compute } from "../../engine";
import { linearToSrgb } from "../render/core";
import { tmLuma, tonemap } from "./tonemap";

/** the compute workgroup edge — one thread per swapchain pixel. @internal */
export const WORKGROUP = 8;

/**
 * one camera's packed postfx config. Every stage gates on its own zero (mode 0 is Neutral, and vignette /
 * posterize / dither are off at 0), so the kernel needs no per-field presence flag; the grade is the one
 * chain that has no meaning at zero, which is why the packer (`index.ts`) seeds its identities — exposure
 * and slope/power 1, saturation 1 — for a camera with no `Glaze`. Named `GlazeConfig` rather than `Glaze`
 * because the component owns that name; nothing splices the struct, so the WGSL name is free.
 * @internal
 */
export const GlazeConfig = d
    .struct({
        exposure: d.f32,
        vignetteStrength: d.f32,
        vignetteInner: d.f32,
        vignetteOuter: d.f32,
        posterizeBands: d.f32,
        ditherStrength: d.f32,
        tonemapMode: d.u32,
        saturation: d.f32,
        slope: d.vec4f,
        offset: d.vec4f,
        power: d.vec4f,
    })
    .$name("GlazeConfig");

/** the OkLab lightness channel — the axis posterize + dither quantize along, so a band boundary lands
 *  where the eye sees one. Hue is preserved by the caller scaling color by newL/oldL.
 *  @example let L = oklabL(displayColor); */
export const oklabL = tgpu.fn(
    [d.vec3f],
    d.f32,
)((c) => {
    "use gpu";
    const lmsL = 0.4122214708 * c.x + 0.5363325363 * c.y + 0.0514459929 * c.z;
    const lmsM = 0.2119034982 * c.x + 0.6806995451 * c.y + 0.1073969566 * c.z;
    const lmsS = 0.0883024619 * c.x + 0.2220049174 * c.y + 0.6896926207 * c.z;
    const l = std.pow(std.max(lmsL, 0), 1 / 3);
    const m = std.pow(std.max(lmsM, 0), 1 / 3);
    const s = std.pow(std.max(lmsS, 0), 1 / 3);
    return 0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s;
});

/** the scene-referred grade: ASC CDL (`(in·slope + offset)^power`) then exposure, on linear radiance.
 *  Pre-tonemap, so the operator rolls any pushed highlight off its shoulder; the HDR offscreen is what
 *  holds radiance >1 for it to roll off.
 *  @example let graded = applyGrade(radiance, glaze); */
export const applyGrade = tgpu.fn(
    [d.vec3f, GlazeConfig],
    d.vec3f,
)((c, g) => {
    "use gpu";
    const cdl = std.pow(
        std.max(std.add(std.mul(c, g.slope.xyz), g.offset.xyz), d.vec3f(0)),
        g.power.xyz,
    );
    return std.mul(cdl, g.exposure);
});

/** perceptual saturation, post-tonemap (Bevy's `post_saturation`): display-referred values read
 *  predictably.
 *  @example let out = applySaturation(display, glaze); */
export const applySaturation = tgpu.fn(
    [d.vec3f, GlazeConfig],
    d.vec3f,
)((c, g) => {
    "use gpu";
    return std.mix(d.vec3f(tmLuma(c)), c, g.saturation);
});

const bayerM = tgpu
    .const(d.arrayOf(d.f32, 16), [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5])
    .$name("bayerM");

/** the ordered 4×4 Bayer dither offset for a pixel, in `[-0.5, 0.4375]`.
 *  @example let noise = bayer4(vec2f(gid.xy)); */
export const bayer4 = tgpu.fn(
    [d.vec2f],
    d.f32,
)((pos) => {
    "use gpu";
    const x = d.u32(pos.x) % 4;
    const y = d.u32(pos.y) % 4;
    return bayerM.$[x + y * 4] / 16 - 0.5;
});

/** dither then posterize the OkLab lightness. Dither runs first so noise pushes adjacent pixels across
 *  band boundaries; the caller scales color by newL/oldL to preserve hue.
 *  @example let newL = ditherPosterizeL(oldL, vec2f(gid.xy), glaze); */
export const ditherPosterizeL = tgpu.fn(
    [d.f32, d.vec2f, GlazeConfig],
    d.f32,
)((L, pos, g) => {
    "use gpu";
    let out = L;
    if (g.ditherStrength > 0) out = out + bayer4(pos) * g.ditherStrength;
    if (g.posterizeBands > 0)
        out = std.floor(std.saturate(out) * g.posterizeBands + 0.5) / g.posterizeBands;
    return out;
});

/** corner darkening between the `vignetteInner` and `vignetteOuter` screen radii. `strength` means
 *  *perceived* darkness: the γ-2.2 power compensates for the sRGB encode that follows, so a 0.5-strength
 *  vignette renders ~0.5 perceived (not ~0.73).
 *  @example let out = applyVignette(display, uv, glaze); */
export const applyVignette = tgpu.fn(
    [d.vec3f, d.vec2f, GlazeConfig],
    d.vec3f,
)((color, uv, g) => {
    "use gpu";
    if (g.vignetteStrength <= 0) return std.copy(color);
    const dist = std.distance(uv, d.vec2f(0.5, 0.5));
    const v = 1 - std.smoothstep(g.vignetteInner, g.vignetteOuter, dist) * g.vignetteStrength;
    return std.mul(color, std.pow(v, 2.2));
});

/**
 * the composite: one thread per swapchain pixel, reading the renderer's offscreen scene color and writing
 * the swapchain through the postfx chain. The layout + kernel + pipeline are per swapchain format —
 * `navigator.gpu.getPreferredCanvasFormat()` is only known at warm and the storage-texture format is part
 * of the WGSL type — so they come from {@link composite}, memoized per format.
 * @internal
 */
export function glazeLayout(format: "bgra8unorm" | "rgba8unorm") {
    // every entry is pinned to `compute`: typegpu's defaults are vertex+fragment+compute for a readonly
    // entry and compute+fragment for a mutable one, so leaving them would claim a fragment storage-texture
    // slot (and vertex uniform/texture slots) for a pipeline that has no such stage — the pre-port layout
    // declared COMPUTE on all three
    return tgpu.bindGroupLayout({
        input: { texture: d.texture2d(d.f32), visibility: ["compute"] },
        glaze: { uniform: GlazeConfig, visibility: ["compute"] },
        output: {
            storageTexture: d.textureStorage2d(format, "write-only"),
            visibility: ["compute"],
        },
    });
}

/** the composite kernel over a format-matched layout. @internal */
export function glazeKernel(layout: ReturnType<typeof glazeLayout>) {
    return tgpu
        .computeFn({
            workgroupSize: [WORKGROUP, WORKGROUP],
            in: { gid: d.builtin.globalInvocationId },
        })((input) => {
            "use gpu";
            const dim = std.textureDimensions(layout.$.output);
            if (input.gid.x >= dim.x || input.gid.y >= dim.y) return;
            const p = input.gid.xy;
            // the offscreen is a float format, so textureLoad returns the linear radiance a renderer wrote
            let color = std.textureLoad(layout.$.input, p, 0).xyz;

            color = tonemap(layout.$.glaze.tonemapMode, applyGrade(color, layout.$.glaze));
            color = applySaturation(color, layout.$.glaze);
            color = std.saturate(color);

            if (layout.$.glaze.posterizeBands > 0 || layout.$.glaze.ditherStrength > 0) {
                const oldL = std.max(oklabL(color), 1e-4);
                const newL = ditherPosterizeL(oldL, d.vec2f(input.gid.xy), layout.$.glaze);
                color = std.max(std.mul(color, newL / oldL), d.vec3f(0));
            }

            const uv = std.div(std.add(d.vec2f(input.gid.xy), 0.5), d.vec2f(dim));
            color = applyVignette(color, uv, layout.$.glaze);

            std.textureStore(
                layout.$.output,
                p,
                d.vec4f(linearToSrgb(std.max(color, d.vec3f(0))), 1),
            );
        })
        .$name("glazeComposite");
}

// pipelines bind to the root that created them (device-scoped, memoized — `engine/runtime/gpu.ts`), so a
// stale entry from a torn-down device must not be reused; keyed by swapchain format plus the device
// identity it was built against (the `render/image.ts` `_blit` shape)
const _cache = new Map<
    string,
    {
        device: GPUDevice;
        layout: ReturnType<typeof glazeLayout>;
        pipeline: TgpuComputePipeline;
    }
>();

/**
 * the layout + compiled pipeline for one swapchain format, memoized against `Compute.device`. The two
 * canvas formats (`bgra8unorm` / `rgba8unorm`) are the only ones `getPreferredCanvasFormat` returns;
 * anything else is a wiring bug and throws rather than emitting `undefined` into the WGSL (typegpu accepts
 * a wrong storage-format string silently).
 * @internal
 */
export function composite(format: GPUTextureFormat) {
    if (format !== "bgra8unorm" && format !== "rgba8unorm")
        throw new Error(
            `[glaze] the swapchain format ${format} is not a storage-writable canvas format — expected bgra8unorm or rgba8unorm`,
        );
    const device = Compute.device;
    const cached = _cache.get(format);
    if (cached && cached.device === device) return cached;
    const layout = glazeLayout(format);
    const pipeline = Compute.root
        .createComputePipeline({ compute: glazeKernel(layout) })
        .$name("glaze");
    const entry = { device, layout, pipeline };
    _cache.set(format, entry);
    return entry;
}

/** the emitted composite WGSL — the device-free structural seam its test resolves.
 *  @internal */
export function glazeWgsl(format: "bgra8unorm" | "rgba8unorm" = "bgra8unorm"): string {
    return tgpu.resolve([glazeKernel(glazeLayout(format))], { names: "strict" });
}

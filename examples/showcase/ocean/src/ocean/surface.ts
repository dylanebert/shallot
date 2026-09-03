import { mesh, PartPlugin, type Plugin, SearPlugin } from "@dylanebert/shallot";
import {
    engineLayout,
    fsCtxSchema,
    registerSurface,
    surfaceLayout,
    VsIn,
    vsPatchSchema,
} from "@dylanebert/shallot/sear/core";
import { Xform } from "@dylanebert/shallot/utils/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { DuskSkyGpu, sampleSky } from "../sky";
import { buildClipmapMesh, OCEAN_CLIP_LEVELS } from "./clipmap";
import { CASCADE_CONFIGS, SLOPE_CASCADE_CONFIGS } from "./spectrum";

/** all resources consumed by the ocean vertex and fragment stages. */
export const oceanSurfaceLayout = surfaceLayout({
    eids: { type: "storage", element: d.u32 },
    transforms: { type: "storage", element: Xform },
    displace0: { type: "texture-2d" },
    displace1: { type: "texture-2d" },
    slope0: { type: "texture-2d" },
    slopeSampler: { type: "sampler" },
    duskSky: { type: "uniform", struct: DuskSkyGpu },
});
// The device oracle executes the exact texture-reading estimator through this layout in compute;
// render keeps the same group and bindings, while the extra visibility changes no shader interface.
for (const entry of Object.values(oceanSurfaceLayout.entries)) {
    entry.visibility = [...entry.visibility, "compute"];
}

export const oceanSurfaceVaryings = { samplePos: d.vec2f, worldPos: d.vec3f };
export const oceanSurfacePatch = vsPatchSchema(oceanSurfaceVaryings);

export const surfaceWrapIndex = tgpu.fn(
    [d.i32, d.i32],
    d.i32,
)((i, n) => {
    "use gpu";
    return ((i % n) + n) % n;
});

/** uniform Catmull-Rom value, re-declared against the consolidated surface product. */
export const surfaceCatmullRom1D = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.f32],
    d.vec4f,
)((p0, p1, p2, p3, t) => {
    "use gpu";
    const a = std.add(
        std.add(std.mul(-0.5, p0), std.mul(1.5, p1)),
        std.add(std.mul(-1.5, p2), std.mul(0.5, p3)),
    );
    const b = std.add(std.add(p0, std.mul(-2.5, p1)), std.add(std.mul(2, p2), std.mul(-0.5, p3)));
    const c = std.add(std.mul(-0.5, p0), std.mul(0.5, p2));
    return std.add(p1, std.mul(t, std.add(c, std.mul(t, std.add(b, std.mul(t, a))))));
});

/** closed-form derivative of {@link surfaceCatmullRom1D}. */
export const surfaceCatmullRomDerivative1D = tgpu.fn(
    [d.vec4f, d.vec4f, d.vec4f, d.vec4f, d.f32],
    d.vec4f,
)((p0, p1, p2, p3, t) => {
    "use gpu";
    const a = std.add(
        std.add(std.mul(-0.5, p0), std.mul(1.5, p1)),
        std.add(std.mul(-1.5, p2), std.mul(0.5, p3)),
    );
    const b = std.add(std.add(p0, std.mul(-2.5, p1)), std.add(std.mul(2, p2), std.mul(-0.5, p3)));
    const c = std.add(std.mul(-0.5, p0), std.mul(0.5, p2));
    return std.add(c, std.mul(t, std.add(std.mul(2, b), std.mul(std.mul(3, t), a))));
});

export const OceanSampleGradient = d
    .struct({ value: d.vec4f, du: d.vec4f, dv: d.vec4f })
    .$name("OceanSampleGradient");
const SampleGradient = OceanSampleGradient;
const sample0 = tgpu.fn(
    [d.i32, d.f32, d.f32],
    SampleGradient,
)((n, u, v) => {
    "use gpu";
    const ix = d.i32(std.floor(u));
    const iy = d.i32(std.floor(v));
    const fx = u - std.floor(u);
    const fy = v - std.floor(v);
    const xs = d.vec4i(
        surfaceWrapIndex(ix - 1, n),
        surfaceWrapIndex(ix, n),
        surfaceWrapIndex(ix + 1, n),
        surfaceWrapIndex(ix + 2, n),
    );
    const ys = d.vec4i(
        surfaceWrapIndex(iy - 1, n),
        surfaceWrapIndex(iy, n),
        surfaceWrapIndex(iy + 1, n),
        surfaceWrapIndex(iy + 2, n),
    );
    const r0a = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.x, ys.x), 0);
    const r0b = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.y, ys.x), 0);
    const r0c = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.z, ys.x), 0);
    const r0d = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.w, ys.x), 0);
    const r1a = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.x, ys.y), 0);
    const r1b = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.y, ys.y), 0);
    const r1c = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.z, ys.y), 0);
    const r1d = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.w, ys.y), 0);
    const r2a = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.x, ys.z), 0);
    const r2b = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.y, ys.z), 0);
    const r2c = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.z, ys.z), 0);
    const r2d = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.w, ys.z), 0);
    const r3a = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.x, ys.w), 0);
    const r3b = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.y, ys.w), 0);
    const r3c = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.z, ys.w), 0);
    const r3d = std.textureLoad(oceanSurfaceLayout.$.displace0, d.vec2i(xs.w, ys.w), 0);
    const row0 = surfaceCatmullRom1D(r0a, r0b, r0c, r0d, fx);
    const row1 = surfaceCatmullRom1D(r1a, r1b, r1c, r1d, fx);
    const row2 = surfaceCatmullRom1D(r2a, r2b, r2c, r2d, fx);
    const row3 = surfaceCatmullRom1D(r3a, r3b, r3c, r3d, fx);
    const dx0 = surfaceCatmullRomDerivative1D(r0a, r0b, r0c, r0d, fx);
    const dx1 = surfaceCatmullRomDerivative1D(r1a, r1b, r1c, r1d, fx);
    const dx2 = surfaceCatmullRomDerivative1D(r2a, r2b, r2c, r2d, fx);
    const dx3 = surfaceCatmullRomDerivative1D(r3a, r3b, r3c, r3d, fx);
    return SampleGradient({
        value: surfaceCatmullRom1D(row0, row1, row2, row3, fy),
        du: surfaceCatmullRom1D(dx0, dx1, dx2, dx3, fy),
        dv: surfaceCatmullRomDerivative1D(row0, row1, row2, row3, fy),
    });
});

const sample1 = tgpu.fn(
    [d.i32, d.f32, d.f32],
    SampleGradient,
)((n, u, v) => {
    "use gpu";
    const ix = d.i32(std.floor(u));
    const iy = d.i32(std.floor(v));
    const fx = u - std.floor(u);
    const fy = v - std.floor(v);
    const xs = d.vec4i(
        surfaceWrapIndex(ix - 1, n),
        surfaceWrapIndex(ix, n),
        surfaceWrapIndex(ix + 1, n),
        surfaceWrapIndex(ix + 2, n),
    );
    const ys = d.vec4i(
        surfaceWrapIndex(iy - 1, n),
        surfaceWrapIndex(iy, n),
        surfaceWrapIndex(iy + 1, n),
        surfaceWrapIndex(iy + 2, n),
    );
    const r0a = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.x, ys.x), 0);
    const r0b = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.y, ys.x), 0);
    const r0c = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.z, ys.x), 0);
    const r0d = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.w, ys.x), 0);
    const r1a = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.x, ys.y), 0);
    const r1b = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.y, ys.y), 0);
    const r1c = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.z, ys.y), 0);
    const r1d = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.w, ys.y), 0);
    const r2a = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.x, ys.z), 0);
    const r2b = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.y, ys.z), 0);
    const r2c = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.z, ys.z), 0);
    const r2d = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.w, ys.z), 0);
    const r3a = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.x, ys.w), 0);
    const r3b = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.y, ys.w), 0);
    const r3c = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.z, ys.w), 0);
    const r3d = std.textureLoad(oceanSurfaceLayout.$.displace1, d.vec2i(xs.w, ys.w), 0);
    const row0 = surfaceCatmullRom1D(r0a, r0b, r0c, r0d, fx);
    const row1 = surfaceCatmullRom1D(r1a, r1b, r1c, r1d, fx);
    const row2 = surfaceCatmullRom1D(r2a, r2b, r2c, r2d, fx);
    const row3 = surfaceCatmullRom1D(r3a, r3b, r3c, r3d, fx);
    const dx0 = surfaceCatmullRomDerivative1D(r0a, r0b, r0c, r0d, fx);
    const dx1 = surfaceCatmullRomDerivative1D(r1a, r1b, r1c, r1d, fx);
    const dx2 = surfaceCatmullRomDerivative1D(r2a, r2b, r2c, r2d, fx);
    const dx3 = surfaceCatmullRomDerivative1D(r3a, r3b, r3c, r3d, fx);
    return SampleGradient({
        value: surfaceCatmullRom1D(row0, row1, row2, row3, fy),
        du: surfaceCatmullRom1D(dx0, dx1, dx2, dx3, fy),
        dv: surfaceCatmullRomDerivative1D(row0, row1, row2, row3, fy),
    });
});
const L0 = d.f32(CASCADE_CONFIGS[0].L);
const L1 = d.f32(CASCADE_CONFIGS[1].L);

export const OceanDisplacementEstimate = d
    .struct({ g0: OceanSampleGradient, g1: OceanSampleGradient, scale0: d.f32, scale1: d.f32 })
    .$name("OceanDisplacementEstimate");

const coordinates = tgpu.fn(
    [d.vec2f, d.f32, d.f32],
    d.vec2f,
)((world, length, n) => {
    "use gpu";
    return std.sub(
        std.mul(std.add(std.div(world, d.vec2f(length)), d.vec2f(0.5)), d.vec2f(n)),
        d.vec2f(0.5),
    );
});

/** samples both published displacement textures and applies their texture-to-world scales. */
export const oceanEstimateDisplacement = tgpu.fn(
    [d.vec2f],
    OceanDisplacementEstimate,
)((s) => {
    "use gpu";
    const n0 = d.i32(std.textureDimensions(oceanSurfaceLayout.$.displace0).x);
    const n1 = d.i32(std.textureDimensions(oceanSurfaceLayout.$.displace1).x);
    const q0 = coordinates(s, L0, d.f32(n0));
    const q1 = coordinates(s, L1, d.f32(n1));
    return OceanDisplacementEstimate({
        g0: sample0(n0, q0.x, q0.y),
        g1: sample1(n1, q1.x, q1.y),
        scale0: d.f32(n0) / L0,
        scale1: d.f32(n1) / L1,
    });
});

export const oceanSurfaceVs = tgpu.fn(
    [VsIn],
    oceanSurfacePatch,
)((input) => {
    "use gpu";
    const s = d.vec2f(input.localPos.x, input.localPos.z);
    const estimate = oceanEstimateDisplacement(s);
    const displacement = std.add(estimate.g0.value, estimate.g1.value);
    const world = d.vec4f(s.x + displacement.x, displacement.y, s.y + displacement.z, 1);
    return oceanSurfacePatch({
        world,
        worldNormal: d.vec3f(0, 1, 0),
        clip: d.vec4f(0),
        samplePos: s,
        worldPos: world.xyz,
    });
});

/** composes both displacement-cascade derivatives with the published slope payload. */
export const oceanFragmentNormal = tgpu.fn(
    [OceanSampleGradient, OceanSampleGradient, d.f32, d.f32, d.vec4f],
    d.vec4f,
)((g0, g1, scale0, scale1, slope) => {
    "use gpu";
    const du = d.vec3f(
        1 + g0.du.x * scale0 + g1.du.x * scale1,
        g0.du.y * scale0 + g1.du.y * scale1,
        g0.du.z * scale0 + g1.du.z * scale1,
    );
    const dv = d.vec3f(
        g0.dv.x * scale0 + g1.dv.x * scale1,
        g0.dv.y * scale0 + g1.dv.y * scale1,
        1 + g0.dv.z * scale0 + g1.dv.z * scale1,
    );
    const displacementNormal = std.normalize(std.cross(dv, du));
    const normal = std.normalize(
        d.vec3f(
            displacementNormal.x - slope.x,
            displacementNormal.y,
            displacementNormal.z - slope.y,
        ),
    );
    return d.vec4f(normal, std.sqrt(std.max(0, slope.w)));
});

/** variance-averaged Schlick factor from Bruneton's ocean lighting model. */
export const meanFresnel = tgpu.fn(
    [d.f32, d.f32],
    d.f32,
)((cosTheta, sigma) => {
    "use gpu";
    const exponent = 5 * std.exp(-2.69 * sigma);
    return std.pow(1 - std.clamp(cosTheta, 0, 1), exponent) / (1 + 22.7 * std.pow(sigma, 1.5));
});

/** displacement normal plus the published high-frequency slope product. */
export const oceanSurfaceFs = tgpu.fn(
    [fsCtxSchema(oceanSurfaceVaryings)],
    d.vec4f,
)((ctx) => {
    "use gpu";
    const s = ctx.samplePos;
    const estimate = oceanEstimateDisplacement(s);
    const slopeUv = std.add(std.div(s, d.vec2f(SLOPE_CASCADE_CONFIGS[0].L)), d.vec2f(0.5));
    const dx = std.dpdx(slopeUv);
    const dy = std.dpdy(slopeUv);
    const slope = std.textureSampleGrad(
        oceanSurfaceLayout.$.slope0,
        oceanSurfaceLayout.$.slopeSampler,
        slopeUv,
        dx,
        dy,
    );
    const normalRoughness = oceanFragmentNormal(
        estimate.g0,
        estimate.g1,
        estimate.scale0,
        estimate.scale1,
        slope,
    );
    const normal = normalRoughness.xyz;
    const sigma = normalRoughness.w;
    const eye = engineLayout.$.view.eye.xyz;
    const view = std.normalize(std.sub(eye, ctx.worldPos));
    const reflected = std.reflect(std.neg(view), normal);
    const skyDirection = std.normalize(d.vec3f(reflected.x, std.max(reflected.y, 0), reflected.z));
    const sky = sampleSky(
        DuskSkyGpu(oceanSurfaceLayout.$.duskSky),
        skyDirection,
        engineLayout.$.lighting.sunDirection.xyz,
    );
    const factor = meanFresnel(std.max(std.dot(normal, view), 0), sigma);
    const fresnel = 0.02 + 0.98 * factor;
    const body = d.vec3f(0.001, 0.006, 0.008);
    return d.vec4f(std.mix(body, sky, fresnel), 1);
});

/** registers the consolidated ocean surface and its clipmap mesh. */
export function registerOceanSurface(state: Parameters<typeof registerSurface>[0]): void {
    const clip = buildClipmapMesh(OCEAN_CLIP_LEVELS);
    mesh({ name: "oceanGrid", vertices: clip.vertices, indices: clip.indices });
    registerSurface(state, {
        name: "ocean",
        layout: oceanSurfaceLayout,
        varyings: oceanSurfaceVaryings,
        vs: oceanSurfaceVs,
        fs: oceanSurfaceFs,
    });
}

/** surface registration dependency bundle for consumers that only need the render product. */
export const OceanSurfacePlugin = {
    name: "OceanSurface",
    dependencies: [PartPlugin, SearPlugin],
    initialize: registerOceanSurface,
} satisfies Plugin;

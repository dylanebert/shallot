import {
    type Background,
    BgCtx,
    backgroundLayout,
    engineLayout,
} from "@dylanebert/shallot/sear/core";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

/** Demo-local dusk sky parameters shared by CPU sampling and the backdrop shader. */
export const DuskSkyGpu = d
    .struct({
        zenith: d.vec4f,
        horizon: d.vec4f,
        haze: d.vec4f,
        cloud: d.vec4f,
        sun: d.vec4f,
        exposure: d.vec4f,
    })
    .$name("OceanDuskSky");

/** Exact byte size of the demo-local dusk sky uniform. */
export const DUSK_SKY_BYTES = d.sizeOf(DuskSkyGpu);
/** Dusk sky uniform size in f32 lanes. */
export const DUSK_SKY_FLOATS = DUSK_SKY_BYTES / 4;

/** The elevation-gradient channel. */
export const sampleElevation = tgpu
    .fn(
        [DuskSkyGpu, d.vec3f],
        d.vec3f,
    )((sky, dir) => {
        "use gpu";
        const elevation = std.smoothstep(0, 1, std.max(dir.y, 0));
        return d.vec3f(std.mix(sky.horizon.xyz, sky.zenith.xyz, elevation));
    })
    .$name("sampleDuskElevation");

/** The horizon haze channel. */
export const sampleHaze = tgpu
    .fn(
        [DuskSkyGpu, d.vec3f],
        d.vec3f,
    )((sky, dir) => {
        "use gpu";
        const band = std.exp(-std.abs(dir.y) * 18) * sky.haze.w;
        return d.vec3f(std.mul(sky.haze.xyz, band));
    })
    .$name("sampleDuskHaze");

/** The broad procedural cloud channel. */
export const sampleCloud = tgpu
    .fn(
        [DuskSkyGpu, d.vec3f],
        d.vec3f,
    )((sky, dir) => {
        "use gpu";
        const wave = 0.5 + 0.5 * std.sin(dir.x * 11 + dir.z * 7 + dir.y * 3);
        const altitude = std.smoothstep(-0.02, 0.18, dir.y);
        return d.vec3f(std.mul(sky.cloud.xyz, wave * altitude * sky.cloud.w));
    })
    .$name("sampleDuskCloud");

/** The sun disk and glow channel. */
export const sampleSun = tgpu
    .fn(
        [DuskSkyGpu, d.vec3f, d.vec3f],
        d.vec3f,
    )((sky, dir, lightDirection) => {
        "use gpu";
        const alignment = std.max(std.dot(dir, std.neg(lightDirection)), 0);
        const glow = std.pow(alignment, 48) * sky.sun.w;
        return d.vec3f(std.mul(sky.sun.xyz, glow));
    })
    .$name("sampleDuskSun");

/** Complete pure CPU/GPU dusk sky recipe. */
export const sampleSky = tgpu
    .fn(
        [DuskSkyGpu, d.vec3f, d.vec3f],
        d.vec3f,
    )((sky, dir, lightDirection) => {
        "use gpu";
        const optional = std.add(
            std.add(sampleHaze(sky, dir), sampleCloud(sky, dir)),
            sampleSun(sky, dir, lightDirection),
        );
        const radiance = std.add(sampleElevation(sky, dir), optional);
        return d.vec3f(std.mul(radiance, sky.exposure.x));
    })
    .$name("sampleDuskSky");

export const skyLayout = backgroundLayout({ duskSky: { type: "uniform", struct: DuskSkyGpu } });

const skyFs = tgpu
    .fn(
        [BgCtx],
        d.vec3f,
    )((ctx) => {
        "use gpu";
        return sampleSky(
            DuskSkyGpu(skyLayout.$.duskSky),
            ctx.dir,
            engineLayout.$.lighting.sunDirection.xyz,
        );
    })
    .$name("oceanDuskSkyFs");

/** Demo-local dusk backdrop registration. */
export const skyBackground: Background<{
    duskSky: { type: "uniform"; struct: typeof DuskSkyGpu };
}> = { name: "sky", layout: skyLayout, fs: skyFs };

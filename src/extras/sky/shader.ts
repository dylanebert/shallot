// The procedural sky shader. The full view-ray recipe is one pure TGSL graph: production calls it from
// the background with the Sky uniform + engine lighting, and unit tests call the same function on
// the CPU. The background wrapper is the only resource-reading leaf.

import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import {
    type Background,
    BackgroundContext,
    backgroundLayout,
    engineLayout,
    LightingGpu,
} from "../../standard/rendering";

/** Procedural sky colors and atmosphere controls; sun disks come from the directional-light table. */
export const SkyGpu = d
    .struct({
        hazeDensity: d.f32,
        horizonBand: d.f32,
        pad0: d.f32,
        pad1: d.f32,
        hazeColor: d.vec4f,
        skyZenith: d.vec4f,
        skyHorizon: d.vec4f,
        starParams: d.vec4f,
        cloudParams: d.vec4f,
        cloudColor: d.vec4f,
    })
    .$name("Sky");

/** exact byte size of the shipped Sky uniform contract. */
export const SKY_BYTES = d.sizeOf(SkyGpu);
/** Sky uniform size in f32 lanes. */
export const SKY_FLOATS = SKY_BYTES / 4;

const at = <T extends d.BaseData>(schema: T, field: (p: d.Infer<T>) => unknown) =>
    d.memoryLayoutOf(schema, field).offset / 4;

/** f32 lane offsets the flat CPU packer writes, derived from {@link SkyGpu}. @internal */
export const SKY_AT = {
    hazeDensity: at(SkyGpu, (sky) => sky.hazeDensity),
    horizonBand: at(SkyGpu, (sky) => sky.horizonBand),
    hazeColor: at(SkyGpu, (sky) => sky.hazeColor),
    skyZenith: at(SkyGpu, (sky) => sky.skyZenith),
    skyHorizon: at(SkyGpu, (sky) => sky.skyHorizon),
    starParams: at(SkyGpu, (sky) => sky.starParams),
    cloudParams: at(SkyGpu, (sky) => sky.cloudParams),
    cloudColor: at(SkyGpu, (sky) => sky.cloudColor),
} as const;

const hash2 = tgpu
    .fn(
        [d.vec2f],
        d.f32,
    )((p) => {
        "use gpu";
        let p3 = d.vec3f(std.fract(std.mul(d.vec3f(p.x, p.y, p.x), 0.1031)));
        p3 = d.vec3f(std.add(p3, std.dot(p3, std.add(p3.yzx, 33.33))));
        return std.fract((p3.x + p3.y) * p3.z);
    })
    .$name("hash2");

const simplex2 = tgpu
    .fn(
        [d.vec2f],
        d.f32,
    )((p) => {
        "use gpu";
        const K1 = d.f32(0.366025404);
        const K2 = d.f32(0.211324865);
        const i = std.floor(std.add(p, (p.x + p.y) * K1));
        const a = std.add(std.sub(p, i), (i.x + i.y) * K2);
        const o = std.select(d.vec2f(0, 1), d.vec2f(1, 0), a.x > a.y);
        const b = std.add(std.sub(a, o), K2);
        const c = std.add(std.sub(a, d.vec2f(1)), 2 * K2);
        const h = std.max(
            std.sub(d.vec3f(0.5), d.vec3f(std.dot(a, a), std.dot(b, b), std.dot(c, c))),
            d.vec3f(0),
        );
        const h4 = std.mul(std.mul(std.mul(h, h), h), h);
        const n = d.vec3f(
            std.dot(a, d.vec2f(hash2(i) * 2 - 1, hash2(std.add(i, d.vec2f(0, 1))) * 2 - 1)),
            std.dot(
                b,
                d.vec2f(
                    hash2(std.add(i, o)) * 2 - 1,
                    hash2(std.add(std.add(i, o), d.vec2f(0, 1))) * 2 - 1,
                ),
            ),
            std.dot(
                c,
                d.vec2f(
                    hash2(std.add(i, d.vec2f(1))) * 2 - 1,
                    hash2(std.add(i, d.vec2f(1, 2))) * 2 - 1,
                ),
            ),
        );
        return std.dot(h4, n) * 70;
    })
    .$name("simplex2");

const fbm2 = tgpu
    .fn(
        [d.vec2f],
        d.f32,
    )((p) => {
        "use gpu";
        let value = d.f32(0);
        let amplitude = d.f32(0.5);
        let frequency = d.f32(1);
        for (let i = d.u32(0); i < 5; i++) {
            value = value + amplitude * simplex2(std.mul(p, frequency));
            amplitude = amplitude * 0.5;
            frequency = frequency * 2;
        }
        return value;
    })
    .$name("fbm2");

const hash2Star = tgpu
    .fn(
        [d.vec2f],
        d.vec2f,
    )((p) => {
        "use gpu";
        let p3 = d.vec3f(
            std.fract(std.mul(d.vec3f(p.x, p.y, p.x), d.vec3f(0.1031, 0.103, 0.0973))),
        );
        p3 = d.vec3f(std.add(p3, std.dot(p3, std.add(p3.yzx, 33.33))));
        return std.fract(std.mul(std.add(p3.xx, p3.yz), p3.zy));
    })
    .$name("hash2Star");

// biome-ignore lint/suspicious/noApproximativeNumericConstant: the shipped hash grid uses this truncated PI
const STAR_PI = 3.14159;
const STAR_HALF_PI = 1.5708;

/** hash-grid stars: one candidate per neighboring azimuth/elevation cell. The nested 3x3 loop is the
 * production loop, including its early `continue`. */
export const sampleStars = tgpu
    .fn(
        [d.vec3f, d.f32, d.f32],
        d.vec3f,
    )((dir, intensity, amount) => {
        "use gpu";
        if (dir.y < 0) return d.vec3f(0);
        const theta = std.atan2(dir.z, dir.x);
        const phi = std.asin(std.clamp(dir.y, -1, 1));
        const gridSize = std.mix(20, 100, amount);
        const cell = d.vec2f((theta * gridSize) / STAR_PI, (phi * gridSize) / STAR_HALF_PI);
        const cellId = std.floor(cell);
        let starColor = d.vec3f(0);
        for (let dy = d.i32(-1); dy <= 1; dy = dy + 1) {
            for (let dx = d.i32(-1); dx <= 1; dx = dx + 1) {
                const neighbor = std.add(cellId, d.vec2f(d.f32(dx), d.f32(dy)));
                const starHash = hash2(neighbor);
                if (starHash > amount * 0.7) continue;
                const starPos = hash2Star(neighbor);
                const starCenter = std.add(neighbor, starPos);
                const dist = std.length(std.sub(cell, starCenter));
                const brightness = hash2(std.add(neighbor, d.vec2f(100)));
                const radius = 0.02 + brightness * 0.03;
                if (dist < radius) {
                    const twinkle = 0.8 + 0.2 * std.sin(brightness * 100);
                    const strength = intensity * brightness * twinkle;
                    const falloff = 1 - std.smoothstep(0, radius, dist);
                    const temp = hash2(std.add(neighbor, d.vec2f(200)));
                    const tint = std.mix(d.vec3f(1, 0.9, 0.8), d.vec3f(0.8, 0.9, 1), temp);
                    starColor = d.vec3f(
                        std.max(starColor, std.mul(std.mul(tint, strength), falloff)),
                    );
                }
            }
        }
        return starColor;
    })
    .$name("sampleStars");

const sampleClouds = tgpu
    .fn(
        [d.vec3f, d.vec4f, d.vec4f],
        d.vec4f,
    )((dir, params, cloudColor) => {
        "use gpu";
        if (dir.y < 0.01) return d.vec4f(0);
        const t = params.z / std.max(dir.y, 0.001);
        const uv = std.mul(dir.xz, t);
        let n = fbm2(uv);
        n = std.smoothstep(1 - params.x, 1, n * 0.5 + 0.5) * params.y;
        n = n * std.smoothstep(0, 0.15, dir.y);
        return d.vec4f(cloudColor.xyz, n);
    })
    .$name("sampleClouds");

/** The view-ray sky. Every directional light carrying SunDisk draws its own light-colored disk and glow. */
export const sampleSky = tgpu
    .fn(
        [SkyGpu, d.vec3f, LightingGpu],
        d.vec3f,
    )((sky, dir, lighting) => {
        "use gpu";
        const t = std.pow(std.clamp(dir.y, 0, 1), 0.25);
        let color = d.vec3f(std.mix(sky.skyHorizon.xyz, sky.skyZenith.xyz, t));
        if (sky.horizonBand > 0) {
            const horizonBlend = std.pow(1 - std.abs(dir.y), 32) * sky.horizonBand;
            const bandColor = std.mul(sky.skyHorizon.xyz, 1.5);
            color = d.vec3f(std.mix(color, bandColor, horizonBlend));
        }

        color = d.vec3f(std.add(color, sampleStars(dir, sky.starParams.x, sky.starParams.y)));
        const clouds = sampleClouds(dir, sky.cloudParams, sky.cloudColor);
        color = d.vec3f(std.mix(color, clouds.xyz, clouds.w));

        for (let i = d.u32(0); i < lighting.directionalCount; i++) {
            const light = lighting.directionalLights[i];
            if (light.params.w > 0) {
                const sunDot = std.dot(dir, std.neg(light.direction.xyz));
                const sunColor = light.color.xyz;
                const glow = light.disk.z;
                if (glow > 0) {
                    const g = d.f32(0.76);
                    const mie = (1 - g * g) / std.pow(1 + g * g - 2 * g * sunDot, 1.5);
                    color = d.vec3f(
                        std.add(color, std.mul(std.mul(std.mul(sunColor, mie), glow), 0.025)),
                    );
                    const angle = std.max(0, sunDot);
                    const corona = std.pow(angle, 512) * 0.4 + std.pow(angle, 128) * 0.06;
                    color = d.vec3f(
                        std.add(
                            color,
                            std.mul(std.mul(std.mul(d.vec3f(1, 0.9, 0.7), sunColor), corona), glow),
                        ),
                    );
                }

                const angularSize = light.disk.x;
                const threshold = std.cos(angularSize * 0.5);
                const span = std.max(1 - threshold, 1e-6);
                const edgeWidth = std.max(span * 0.15, 1e-6);
                if (angularSize > 0 && light.disk.y > 0) {
                    const diskBlend = std.smoothstep(
                        threshold - edgeWidth,
                        threshold + edgeWidth,
                        sunDot,
                    );
                    if (diskBlend > 0) {
                        const radial = std.saturate((sunDot - threshold) / span);
                        const r = 1 - radial;
                        const mu = std.sqrt(std.max(1 - r * r, 0));
                        const limbDarken = 1 - 0.6 * (1 - mu);
                        color = d.vec3f(
                            std.add(
                                color,
                                std.mul(std.mul(sunColor, light.disk.y * limbDarken), diskBlend),
                            ),
                        );
                        const edgeDist = 1 - std.smoothstep(0, 1, radial);
                        const fringe = d.vec3f(
                            std.smoothstep(0.3, 0.7, edgeDist),
                            std.smoothstep(0.5, 0.9, edgeDist),
                            std.smoothstep(0.7, 1, edgeDist),
                        );
                        color = d.vec3f(
                            std.add(
                                color,
                                std.mul(
                                    std.mul(
                                        std.mul(fringe, sunColor),
                                        0.15 * light.disk.y * diskBlend,
                                    ),
                                    1 - radial,
                                ),
                            ),
                        );
                    }
                }
            }
        }

        if (sky.hazeDensity > 0) {
            const horizonFactor = 1 - std.clamp(dir.y, 0, 1);
            const hazeAmount = std.pow(horizonFactor, 2) * std.saturate(sky.hazeDensity * 5);
            color = d.vec3f(std.mix(color, sky.hazeColor.xyz, hazeAmount));
        }
        return color;
    })
    .$name("sampleSky");

export const skyLayout = backgroundLayout({ sky: { type: "uniform", struct: SkyGpu } });

const skyFs = tgpu
    .fn(
        [BackgroundContext],
        d.vec3f,
    )((ctx) => {
        "use gpu";
        return sampleSky(SkyGpu(skyLayout.$.sky), ctx.dir, LightingGpu(engineLayout.$.lighting));
    })
    .$name("skyFs");

/** the `sky` background registration. @internal */
export const skyBackground: Background<{ sky: { type: "uniform"; struct: typeof SkyGpu } }> = {
    name: "sky",
    layout: skyLayout,
    fs: skyFs,
};

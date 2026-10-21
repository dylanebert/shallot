// Bevy b3bdd75be, crates/bevy_post_process/src/bloom/bloom.wesl.
// MIT OR Apache-2.0; see bevy/LICENSE-MIT.
import tgpu from "typegpu";
import * as d from "typegpu/data";

export const Settings = d.struct({
    threshold: d.vec4f,
    scale: d.vec2f,
    blend: d.f32,
    additive: d.u32,
    stacked: d.u32,
});
export const layout = tgpu
    .bindGroupLayout({
        input: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
        sampler: { sampler: "filtering", visibility: ["fragment"] },
        settings: { uniform: Settings, visibility: ["fragment"] },
        original: { texture: d.texture2d(d.f32), visibility: ["fragment"] },
    })
    .$idx(0);
const karis = tgpu.fn(
    [d.vec3f],
    d.f32,
)(`(color: vec3f) -> f32 {
    return 1.0 / (1.0 + dot(color, vec3f(0.2126, 0.7152, 0.0722)) / 4.0);
}`);
const sample13 = tgpu
    .fn(
        [d.vec2f, d.bool],
        d.vec3f,
    )(`(uv: vec2f, first: bool) -> vec3f {
    var a: vec3f; var b: vec3f; var c: vec3f; var e: vec3f;
    var f: vec3f; var g: vec3f; var h: vec3f; var i: vec3f;
    var j: vec3f; var k: vec3f; var l: vec3f; var m: vec3f; var dd: vec3f;
    if (all(bound.settings.scale == vec2f(1.0))) {
        a = textureSample(bound.input, bound.sampler, uv, vec2i(-2, 2)).rgb;
        b = textureSample(bound.input, bound.sampler, uv, vec2i(0, 2)).rgb;
        c = textureSample(bound.input, bound.sampler, uv, vec2i(2, 2)).rgb;
        dd = textureSample(bound.input, bound.sampler, uv, vec2i(-2, 0)).rgb;
        e = textureSample(bound.input, bound.sampler, uv).rgb;
        f = textureSample(bound.input, bound.sampler, uv, vec2i(2, 0)).rgb;
        g = textureSample(bound.input, bound.sampler, uv, vec2i(-2, -2)).rgb;
        h = textureSample(bound.input, bound.sampler, uv, vec2i(0, -2)).rgb;
        i = textureSample(bound.input, bound.sampler, uv, vec2i(2, -2)).rgb;
        j = textureSample(bound.input, bound.sampler, uv, vec2i(-1, 1)).rgb;
        k = textureSample(bound.input, bound.sampler, uv, vec2i(1, 1)).rgb;
        l = textureSample(bound.input, bound.sampler, uv, vec2i(-1, -1)).rgb;
        m = textureSample(bound.input, bound.sampler, uv, vec2i(1, -1)).rgb;
    } else {
        let ps = bound.settings.scale / vec2f(textureDimensions(bound.input));
        let pl = 2.0 * ps; let ns = -1.0 * ps; let nl = -2.0 * ps;
        a = textureSample(bound.input, bound.sampler, uv + vec2f(nl.x, pl.y)).rgb;
        b = textureSample(bound.input, bound.sampler, uv + vec2f(0.0, pl.y)).rgb;
        c = textureSample(bound.input, bound.sampler, uv + vec2f(pl.x, pl.y)).rgb;
        dd = textureSample(bound.input, bound.sampler, uv + vec2f(nl.x, 0.0)).rgb;
        e = textureSample(bound.input, bound.sampler, uv).rgb;
        f = textureSample(bound.input, bound.sampler, uv + vec2f(pl.x, 0.0)).rgb;
        g = textureSample(bound.input, bound.sampler, uv + vec2f(nl.x, nl.y)).rgb;
        h = textureSample(bound.input, bound.sampler, uv + vec2f(0.0, nl.y)).rgb;
        i = textureSample(bound.input, bound.sampler, uv + vec2f(pl.x, nl.y)).rgb;
        j = textureSample(bound.input, bound.sampler, uv + vec2f(ns.x, ps.y)).rgb;
        k = textureSample(bound.input, bound.sampler, uv + vec2f(ps.x, ps.y)).rgb;
        l = textureSample(bound.input, bound.sampler, uv + vec2f(ns.x, ns.y)).rgb;
        m = textureSample(bound.input, bound.sampler, uv + vec2f(ps.x, ns.y)).rgb;
    }
    if (first) {
        var group0 = (a + b + dd + e) * (0.125 / 4.0);
        var group1 = (b + c + e + f) * (0.125 / 4.0);
        var group2 = (dd + e + g + h) * (0.125 / 4.0);
        var group3 = (e + f + h + i) * (0.125 / 4.0);
        var group4 = (j + k + l + m) * (0.5 / 4.0);
        group0 *= karis(group0); group1 *= karis(group1); group2 *= karis(group2);
        group3 *= karis(group3); group4 *= karis(group4);
        return group0 + group1 + group2 + group3 + group4;
    }
    var sample = (a + c + g + i) * 0.03125;
    sample += (b + dd + f + h) * 0.0625;
    sample += (e + j + k + l + m) * 0.125;
    return sample;
}`)
    .$uses({ bound: layout.$, karis });
const tent = tgpu
    .fn(
        [d.vec2f],
        d.vec3f,
    )(`(uv: vec2f) -> vec3f {
    let size = bound.settings.scale / vec2f(textureDimensions(bound.input));
    let x = size.x; let y = size.y;
    let a = textureSample(bound.input, bound.sampler, vec2f(uv.x-x, uv.y+y)).rgb;
    let b = textureSample(bound.input, bound.sampler, vec2f(uv.x, uv.y+y)).rgb;
    let c = textureSample(bound.input, bound.sampler, vec2f(uv.x+x, uv.y+y)).rgb;
    let dd = textureSample(bound.input, bound.sampler, vec2f(uv.x-x, uv.y)).rgb;
    let e = textureSample(bound.input, bound.sampler, uv).rgb;
    let f = textureSample(bound.input, bound.sampler, vec2f(uv.x+x, uv.y)).rgb;
    let g = textureSample(bound.input, bound.sampler, vec2f(uv.x-x, uv.y-y)).rgb;
    let h = textureSample(bound.input, bound.sampler, vec2f(uv.x, uv.y-y)).rgb;
    let i = textureSample(bound.input, bound.sampler, vec2f(uv.x+x, uv.y-y)).rgb;
    var sample = e * 0.25;
    sample += (b + dd + f + h) * 0.125;
    sample += (a + c + g + i) * 0.0625;
    return sample;
}`)
    .$uses({ bound: layout.$ });
const firstSample = tgpu
    .fn(
        [d.vec2f],
        d.vec4f,
    )(`(uv: vec2f) -> vec4f {
    var sample = clamp(sample13(uv, true), vec3f(0.0001), vec3f(3.40282347E+37));
    let t = bound.settings.threshold;
    if (t.x > 0.0) {
        let brightness = max(sample.r, max(sample.g, sample.b));
        var softness = clamp(brightness - t.y, 0.0, t.z);
        softness = softness * softness * t.w;
        let contribution = max(brightness - t.x, softness) / max(brightness, 0.00001);
        sample *= contribution;
    }
    return vec4f(sample, 1.0);
}`)
    .$uses({ sample13, bound: layout.$ });
export const first = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })((input) => {
    "use gpu";
    return firstSample(input.uv);
});
export const down = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })((input) => {
    "use gpu";
    return d.vec4f(sample13(input.uv, false), 1);
});
export const up = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })((input) => {
    "use gpu";
    return d.vec4f(tent(input.uv), 1);
});
const compositeSample = tgpu
    .fn(
        [d.vec2f],
        d.vec4f,
    )(`(uv: vec2f) -> vec4f {
    let original = textureSample(bound.original, bound.sampler, uv);
    let g = bound.settings;
    let coverage = select(1.0, original.a, g.stacked != 0u);
    let destination = select(1.0 - g.blend, 1.0, g.additive != 0u);
    return vec4f(original.rgb * destination + tent(uv) * g.blend * coverage, original.a);
}`)
    .$uses({ bound: layout.$, tent });
export const composite = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })((input) => {
    "use gpu";
    return compositeSample(input.uv);
});

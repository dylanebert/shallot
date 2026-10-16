import { xformNormal, xformPoint } from "@dylanebert/shallot/rendering";
import {
    engineLayout,
    lit,
    MaterialPlugin,
    MaterialVertexInput,
    materialFragmentContext,
    materialLayout,
    materialType,
    materialVertexOutput,
} from "@dylanebert/shallot/standard/rendering";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";

export const AnimationParameters = d.struct({ time: d.f32 });
export const CheckerParameters = d.struct({ scale: d.f32 });
const checkerLayout = materialLayout(CheckerParameters, {});
const Fragment = materialFragmentContext();
export const CheckerMaterial = materialType({
    name: "ScrollingChecker",
    layout: checkerLayout,
    defaults: { scale: 1 },
    fragmentInputs: { uv: true },
    fragment: tgpu.fn(
        [Fragment],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        const time = engineLayout.$.frame.time;
        const scale = checkerLayout.$.materialParameters[ctx.material].scale;
        const cell = std.floor(ctx.uv.x * 12 * scale + time * 2) + std.floor(ctx.uv.y * 6 * scale);
        const value = cell % 2;
        const color = std.mix(d.vec3f(0.06, 0.045, 0.03), d.vec3f(0.95, 0.36, 0.08), value);
        return d.vec4f(lit(color, ctx.worldNormal), 1);
    }),
});
export const waveVector = d.vec3f(3, 14, 0);
const rippleAmplitude = 0.03;
const RippleSurface = d.struct({ position: d.vec3f, normal: d.vec3f });
export const ripple = tgpu.fn(
    [d.vec3f, d.vec3f, d.f32, d.f32],
    RippleSurface,
)((position, normal, phase, time) => {
    "use gpu";
    const angle = phase + time * 3;
    const offset = std.sin(angle) * rippleAmplitude;
    const radius = std.length(position);
    const tangentGradient = std.sub(waveVector, std.mul(normal, std.dot(waveVector, normal)));
    const slope = std.mul(
        tangentGradient,
        (std.cos(angle) * rippleAmplitude * radius) / (radius + offset),
    );
    return RippleSurface({
        position: std.add(position, std.mul(normal, offset)),
        normal: std.normalize(std.sub(normal, slope)),
    });
});
const waveLayout = materialLayout(AnimationParameters, {
    phase: { type: "attribute", element: d.f32 },
});
const Vertex = materialVertexOutput();
export const WaveMaterial = materialType({
    name: "VertexWave",
    layout: waveLayout,
    defaults: { time: 0 },
    vertex: tgpu.fn(
        [MaterialVertexInput],
        Vertex,
    )((input) => {
        "use gpu";
        const time = waveLayout.$.materialParameters[input.material].time;
        const surface = ripple(
            input.localPos,
            input.localNormal,
            waveLayout.$.phase[input.vidx],
            time,
        );
        return Vertex({
            world: d.vec4f(xformPoint(input.xform, surface.position), 1),
            worldNormal: xformNormal(input.xform, surface.normal),
        });
    }),
    fragment: tgpu.fn(
        [Fragment],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        return d.vec4f(lit(d.vec3f(0.62, 0.268, 0.118), ctx.worldNormal), 1);
    }),
});
export const CheckerPlugin = MaterialPlugin(CheckerMaterial);
export const WavePlugin = MaterialPlugin(WaveMaterial);

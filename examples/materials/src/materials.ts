import {
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
const checkerLayout = materialLayout(AnimationParameters, {});
const Fragment = materialFragmentContext();
export const CheckerMaterial = materialType({
    name: "ScrollingChecker",
    parameters: AnimationParameters,
    layout: checkerLayout,
    defaults: { time: 0 },
    fragmentInputs: { uv: true },
    fragment: tgpu.fn(
        [Fragment],
        d.vec4f,
    )((ctx) => {
        "use gpu";
        const time = checkerLayout.$.materialParameters[ctx.material].time;
        const cell = std.floor(ctx.uv.x * 12 + time * 2) + std.floor(ctx.uv.y * 6);
        const value = cell % 2;
        const color = std.mix(d.vec3f(0.06, 0.045, 0.03), d.vec3f(0.95, 0.36, 0.08), value);
        return d.vec4f(lit(color, ctx.worldNormal), 1);
    }),
});
const waveLayout = materialLayout(AnimationParameters, {
    phase: { type: "attribute", element: d.f32 },
});
const Vertex = materialVertexOutput({});
export const WaveMaterial = materialType({
    name: "VertexWave",
    parameters: AnimationParameters,
    layout: waveLayout,
    varyings: {},
    defaults: { time: 0 },
    vertex: tgpu.fn(
        [MaterialVertexInput],
        Vertex,
    )((input) => {
        "use gpu";
        const time = waveLayout.$.materialParameters[input.material].time;
        const offset = std.sin(waveLayout.$.phase[input.vidx] + time * 3) * 0.22;
        return Vertex({
            world: std.add(input.world, d.vec4f(std.mul(input.worldNormal, offset), 0)),
            worldNormal: input.worldNormal,
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

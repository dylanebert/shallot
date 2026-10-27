import { expect, test } from "bun:test";
import * as d from "typegpu/data";
import { materialLayout } from "./material-type";

const Parameters = d.struct({ value: d.f32 });
const uniforms = (count: number) =>
    Object.fromEntries(
        Array.from({ length: count }, (_, i) => [
            `u${i}`,
            { type: "uniform" as const, struct: Parameters },
        ]),
    );
const samplers = (count: number) =>
    Object.fromEntries(
        Array.from({ length: count }, (_, i) => [`s${i}`, { type: "sampler" as const }]),
    );
test("material layouts count engine uniform buffers in each shader stage", () => {
    expect(() => materialLayout(Parameters, uniforms(6))).not.toThrow();
    expect(() => materialLayout(Parameters, uniforms(7))).toThrow(
        "material layout uses 13 fragment uniform buffers; WebGPU's limit is 12",
    );
    expect(() => materialLayout(Parameters, uniforms(8))).toThrow(
        "material layout uses 13 vertex uniform buffers; WebGPU's limit is 12",
    );
});
test("material layouts count engine samplers against the default limit", () => {
    expect(() => materialLayout(Parameters, samplers(15))).not.toThrow();
    expect(() => materialLayout(Parameters, samplers(16))).toThrow(
        "material layout uses 17 vertex samplers; WebGPU's limit is 16",
    );
});

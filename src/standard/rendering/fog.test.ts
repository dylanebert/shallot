import { expect, test } from "bun:test";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import { World } from "../../engine";
import { FogInput, fogSource, registerFogSource, sealFogSource } from "./fog";

const source = tgpu.fn(
    [FogInput],
    d.vec4f,
)(() => {
    "use gpu";
    return d.vec4f(0, 0, 0, 1);
});
const layout = tgpu.bindGroupLayout({}).$idx(3);
const bindGroup = () => {
    throw new Error("registration must not request a bind group");
};
const spec = { source, layout, bindGroup };

test("fog registration is world-local, permits only the same name and source again, and names a conflicting owner", () => {
    const a = new World();
    const b = new World();
    expect(fogSource(a)).toBeNull();
    registerFogSource(a, "haze", spec);
    expect(() =>
        registerFogSource(a, "haze", {
            ...spec,
            bindGroup: () => {
                throw new Error("unused");
            },
        }),
    ).not.toThrow();
    expect(fogSource(b)).toBeNull();
    expect(() => registerFogSource(a, "other", spec)).toThrow(
        'fog source "haze" is already registered',
    );
    const other = tgpu.fn(
        [FogInput],
        d.vec4f,
    )(() => {
        "use gpu";
        return d.vec4f(0, 0, 0, 1);
    });
    expect(() => registerFogSource(a, "haze", { ...spec, source: other })).toThrow(
        'fog source "haze"',
    );
});

test("fog registration requires reserved group 3 and closes at standard warm, retaining idempotence", () => {
    const world = new World();
    expect(() =>
        registerFogSource(world, "bad", { ...spec, layout: tgpu.bindGroupLayout({}).$idx(2) }),
    ).toThrow("reserved bind group 3");
    registerFogSource(world, "haze", spec);
    sealFogSource(world);
    expect(() => registerFogSource(world, "haze", spec)).not.toThrow();
    const empty = new World();
    sealFogSource(empty);
    expect(() => registerFogSource(empty, "late", spec)).toThrow("closed after standard warm");
});

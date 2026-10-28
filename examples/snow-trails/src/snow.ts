import type { Plugin } from "@dylanebert/shallot";
import {
    lit,
    MaterialPlugin,
    materialLayout,
    materialType,
} from "@dylanebert/shallot/standard/rendering";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import * as std from "typegpu/std";
import { FIELD, HILL_SIZE, Hills, HillsPlugin } from "./hills";
import { DEPTH, RESOLUTION, Trail, TrailPlugin } from "./trail";

const layout = materialLayout(d.struct({ sweep: d.vec3f }), {
    image: { type: "texture", filterable: false },
    ground: { type: "texture", filterable: false },
});
/** The trail's press at a world position, bilinear between texel centers. */
const pressAt = tgpu.fn(
    [d.vec3f],
    d.f32,
)((world) => {
    "use gpu";
    const uv = std.clamp(
        d.vec2f(world.x, -world.z).div(FIELD).add(d.vec2f(0.5)).mul(RESOLUTION).sub(d.vec2f(0.5)),
        d.vec2f(0),
        d.vec2f(RESOLUTION - 1.001),
    );
    const xy = d.vec2i(uv);
    const f = std.fract(uv);
    return std.mix(
        std.mix(
            std.textureLoad(layout.$.image, xy, 0).r,
            std.textureLoad(layout.$.image, xy.add(d.vec2i(1, 0)), 0).r,
            f.x,
        ),
        std.mix(
            std.textureLoad(layout.$.image, xy.add(d.vec2i(0, 1)), 0).r,
            std.textureLoad(layout.$.image, xy.add(d.vec2i(1, 1)), 0).r,
            f.x,
        ),
        f.y,
    );
});
const ground = tgpu.fn(
    [d.vec3f],
    d.f32,
)((world) => {
    "use gpu";
    const uv = std.clamp(
        d
            .vec2f(world.x, world.z)
            .div(FIELD)
            .add(d.vec2f(0.5))
            .mul(HILL_SIZE - 1),
        d.vec2f(0),
        d.vec2f(HILL_SIZE - 1.001),
    );
    const xy = d.vec2i(uv);
    const f = std.fract(uv);
    return std.mix(
        std.mix(
            std.textureLoad(layout.$.ground, xy, 0).r,
            std.textureLoad(layout.$.ground, xy.add(d.vec2i(1, 0)), 0).r,
            f.x,
        ),
        std.mix(
            std.textureLoad(layout.$.ground, xy.add(d.vec2i(0, 1)), 0).r,
            std.textureLoad(layout.$.ground, xy.add(d.vec2i(1, 1)), 0).r,
            f.x,
        ),
        f.y,
    );
});
const bindings = (world: import("@dylanebert/shallot").World) => ({
    image: world.resource(Trail).image!,
    ground: world.resource(Hills).image!,
});
export const SnowMaterial = materialType({
    name: "Snow",
    layout,
    defaults: { sweep: d.vec3f() },
    bindings,
    vertex: (input) => {
        "use gpu";
        const p = input.world.xyz;
        const h = FIELD / RESOLUTION;
        const dx = (pressAt(p.add(d.vec3f(h, 0, 0))) - pressAt(p.sub(d.vec3f(h, 0, 0)))) / (2 * h);
        const dz = (pressAt(p.add(d.vec3f(0, 0, h))) - pressAt(p.sub(d.vec3f(0, 0, h)))) / (2 * h);
        return {
            world: d.vec4f(p.x, ground(p) + DEPTH - pressAt(p) * DEPTH, p.z, 1),
            worldNormal: std.normalize(
                d.vec3f(
                    dx * DEPTH -
                        (ground(p.add(d.vec3f(0.05, 0, 0))) - ground(p.sub(d.vec3f(0.05, 0, 0)))) /
                            0.1,
                    1,
                    dz * DEPTH -
                        (ground(p.add(d.vec3f(0, 0, 0.05))) - ground(p.sub(d.vec3f(0, 0, 0.05)))) /
                            0.1,
                ),
            ),
        };
    },
    fragment: (ctx) => {
        "use gpu";
        const press = pressAt(ctx.world);
        const color = std.mix(d.vec3f(0.78, 0.8, 0.79), d.vec3f(0.29, 0.36, 0.39), press);
        return d.vec4f(lit(color, ctx.worldNormal), 1);
    },
});
export const StampMaterial = materialType({
    name: "PressDepth",
    layout,
    bindings,
    defaults: { sweep: d.vec3f() },
    // The half facing back along `sweep` trails by it, so one draw presses the capsule the stamp swept.
    vertex: (input) => {
        "use gpu";
        const back = std.select(
            d.vec3f(0),
            input.parameters.sweep,
            std.dot(input.worldNormal, input.parameters.sweep) < 0,
        );
        return { world: d.vec4f(input.world.xyz.sub(back), 1), worldNormal: input.worldNormal };
    },
    fragment: (ctx) => {
        "use gpu";
        const press = std.clamp((ground(ctx.world) + DEPTH - ctx.world.y) / DEPTH, 0, 1);
        return d.vec4f(d.vec3f(press), 1);
    },
});
export const SnowPlugin: Plugin = {
    name: "Snow",
    dependencies: [
        HillsPlugin,
        TrailPlugin,
        MaterialPlugin(SnowMaterial),
        MaterialPlugin(StampMaterial),
    ],
};

export function snowPlane() {
    const n = RESOLUTION / 2;
    const vertices = new Float32Array((n + 1) * (n + 1) * 8);
    const indices = new Uint32Array(n * n * 6);
    for (let z = 0; z <= n; z++)
        for (let x = 0; x <= n; x++) {
            const i = (z * (n + 1) + x) * 8;
            vertices.set(
                [(x / n - 0.5) * FIELD, DEPTH, (z / n - 0.5) * FIELD, x / n, 0, 1, 0, z / n],
                i,
            );
        }
    for (let z = 0; z < n; z++)
        for (let x = 0; x < n; x++) {
            const a = z * (n + 1) + x,
                b = a + n + 1;
            indices.set([a, b, a + 1, a + 1, b, b + 1], (z * n + x) * 6);
        }
    return { vertices, indices };
}
